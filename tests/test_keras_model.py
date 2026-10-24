"""The Keras reference model, on whichever backend KERAS_BACKEND selects."""

import numpy as np
import pytest

keras = pytest.importorskip("keras")

from doc_motion.config import TrainConfig  # noqa: E402
from doc_motion.keras_model import build_keras_model  # noqa: E402

# Counted by hand from the architecture's layer list (not a number recorded
# in 2022), BatchNorm moving statistics included: the three conv blocks
# shrink 100x70 to 8x5x256 = 10,240 features before the dense layers.
HAND_COUNTED_PARAMETERS = 11_889_921


def test_parameter_count_matches_a_hand_count_of_the_architecture():
    model = build_keras_model(TrainConfig())

    assert model.count_params() == HAND_COUNTED_PARAMETERS


def test_outputs_one_probability_per_frame():
    model = build_keras_model(TrainConfig())
    X = np.random.default_rng(0).random((4, 100, 70, 3), dtype=np.float32)

    p = np.asarray(model.predict(X, verbose=0))

    assert p.shape == (4, 1)
    assert np.all((p >= 0) & (p <= 1))


def test_is_compiled_with_adam_at_the_configured_learning_rate():
    model = build_keras_model(TrainConfig(learning_rate=0.003))

    assert isinstance(model.optimizer, keras.optimizers.Adam)
    assert float(model.optimizer.learning_rate) == pytest.approx(0.003)
    assert model.loss == "binary_crossentropy"


def test_same_seed_gives_the_same_initial_weights():
    a = build_keras_model(TrainConfig(seed=3))
    b = build_keras_model(TrainConfig(seed=3))
    c = build_keras_model(TrainConfig(seed=4))

    np.testing.assert_array_equal(a.get_weights()[0], b.get_weights()[0])
    assert not np.array_equal(a.get_weights()[0], c.get_weights()[0])
