"""The native PyTorch port and its parity with the Keras reference."""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from doc_motion.config import TrainConfig  # noqa: E402
from doc_motion.torch_model import DocMotionNet, from_keras  # noqa: E402

HAND_COUNTED_PARAMETERS = 11_889_921  # see test_keras_model.py


def _count(net):
    # Keras counts BatchNorm moving mean and variance as (untrainable)
    # parameters; PyTorch keeps them as buffers.
    running = sum(
        m.running_mean.numel() + m.running_var.numel()
        for m in net.modules()
        if isinstance(m, torch.nn.BatchNorm2d)
    )
    return sum(p.numel() for p in net.parameters()) + running


def test_parameter_count_matches_a_hand_count_of_the_architecture():
    assert _count(DocMotionNet()) == HAND_COUNTED_PARAMETERS


def test_takes_nhwc_frames_and_outputs_one_probability_each():
    net = DocMotionNet().eval()
    X = torch.rand(4, 100, 70, 3)

    with torch.no_grad():
        p = net(X)

    assert p.shape == (4, 1)
    assert torch.all((p >= 0) & (p <= 1))


def test_initialises_like_keras():
    torch.manual_seed(0)
    net = DocMotionNet()
    first, second = net.conv1.weight, net.conv2.weight

    # he_uniform on the first conv: limit sqrt(6 / fan_in), fan_in = 11*11*3.
    assert first.abs().max() <= math.sqrt(6 / (11 * 11 * 3))
    # glorot_uniform elsewhere: limit sqrt(6 / (fan_in + fan_out)).
    assert second.abs().max() <= math.sqrt(6 / (5 * 5 * 32 + 5 * 5 * 256))
    assert all(
        torch.all(m.bias == 0)
        for m in net.modules()
        if isinstance(m, torch.nn.Conv2d | torch.nn.Linear)
    )


def test_batchnorm_matches_keras_epsilon_and_momentum():
    bn = DocMotionNet().bn1

    assert bn.eps == pytest.approx(1e-3)
    assert bn.momentum == pytest.approx(0.01)


def test_matches_the_keras_model_given_the_same_weights():
    pytest.importorskip("keras")
    from doc_motion.keras_model import build_keras_model

    rng = np.random.default_rng(0)
    X = rng.random((16, 100, 70, 3), dtype=np.float32)
    y = rng.integers(0, 2, size=16).astype(np.float32)
    reference = build_keras_model(TrainConfig(seed=0, learning_rate=1e-4))
    # One short fit so the BatchNorm moving statistics are no longer 0 and 1.
    reference.fit(X, y, epochs=1, batch_size=8, verbose=0)

    net = from_keras(reference).eval()
    with torch.no_grad():
        ported = net(torch.from_numpy(X)).numpy()
    expected = np.asarray(reference.predict(X, verbose=0))

    assert _count(net) == reference.count_params() == HAND_COUNTED_PARAMETERS
    np.testing.assert_allclose(ported, expected, atol=1e-5, rtol=0)
