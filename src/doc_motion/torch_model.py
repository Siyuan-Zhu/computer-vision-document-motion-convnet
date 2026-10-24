"""A hand-written PyTorch port of the 2022 ConvNet, with its own training loop.

The port mirrors the Keras defaults that matter for matching behaviour:

* BatchNorm ``eps=1e-3`` and ``momentum=0.01`` (Keras ``momentum=0.99``).
* He-uniform initialisation on the first convolution, Glorot-uniform on the
  other layers, zero biases.
* Adam with ``eps=1e-7`` and binary cross-entropy on probabilities clipped
  to ``[1e-7, 1 - 1e-7]``.
* Inputs are NHWC like the Keras model; the features are flattened in NHWC
  order, so ``from_keras`` copies the first dense layer without reordering.

One difference remains in training only: PyTorch updates the running
variance with the unbiased batch variance, Keras with the biased one.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from .config import TrainConfig
from .data import IMAGE_SHAPE


class DocMotionNet(nn.Module):
    """The 2022 ConvNet. Takes float32 NHWC frames, returns P(flip)."""

    def __init__(self, input_shape: tuple[int, int, int] = IMAGE_SHAPE) -> None:
        super().__init__()
        height, width, channels = input_shape
        self.input_shape = input_shape

        def bn(n: int) -> nn.BatchNorm2d:
            return nn.BatchNorm2d(n, eps=1e-3, momentum=0.01)

        self.conv1 = nn.Conv2d(channels, 32, 11, padding="same")
        self.bn1 = bn(32)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 256, 5, padding="same")
        self.bn2 = bn(256)
        self.pool2 = nn.MaxPool2d(3)
        self.bn3 = bn(256)
        self.conv3 = nn.Conv2d(256, 256, 1)
        self.bn4 = bn(256)
        self.conv4 = nn.Conv2d(256, 256, 1)
        self.bn5 = bn(256)
        self.pool3 = nn.MaxPool2d(2)

        flat = (height // 2 // 3 // 2) * (width // 2 // 3 // 2) * 256
        self.fc1 = nn.Linear(flat, 1024)
        self.drop1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout(0.2)
        self.out = nn.Linear(1024, 1)
        self._init_like_keras()

    def _init_like_keras(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d | nn.Linear):
                if module is self.conv1:
                    nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                else:
                    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        x = self.pool1(self.bn1(torch.relu(self.conv1(x))))
        x = self.pool2(self.bn2(torch.relu(self.conv2(x))))
        x = self.bn4(torch.relu(self.conv3(self.bn3(x))))
        x = self.pool3(self.bn5(torch.relu(self.conv4(x))))
        x = x.permute(0, 2, 3, 1).flatten(1)  # flatten in Keras (NHWC) order
        x = self.drop1(torch.relu(self.fc1(x)))
        x = self.drop2(torch.relu(self.fc2(x)))
        return torch.sigmoid(self.out(x))


def from_keras(keras_model: Any) -> DocMotionNet:
    """Copy the weights of a ``build_keras_model`` model into a new port."""
    import keras

    convs = [m for m in keras_model.layers if isinstance(m, keras.layers.Conv2D)]
    norms = [
        m for m in keras_model.layers if isinstance(m, keras.layers.BatchNormalization)
    ]
    denses = [m for m in keras_model.layers if isinstance(m, keras.layers.Dense)]
    net = DocMotionNet(tuple(keras_model.input_shape[1:]))

    def copy(target: torch.Tensor | None, value: NDArray[Any]) -> None:
        assert target is not None
        target.copy_(torch.from_numpy(np.ascontiguousarray(value, dtype=np.float32)))

    with torch.no_grad():
        for layer, conv in zip(
            convs, [net.conv1, net.conv2, net.conv3, net.conv4], strict=True
        ):
            kernel, bias = layer.get_weights()
            copy(conv.weight, kernel.transpose(3, 2, 0, 1))  # HWIO -> OIHW
            copy(conv.bias, bias)
        for layer, norm in zip(
            norms, [net.bn1, net.bn2, net.bn3, net.bn4, net.bn5], strict=True
        ):
            gamma, beta, mean, var = layer.get_weights()
            copy(norm.weight, gamma)
            copy(norm.bias, beta)
            copy(norm.running_mean, mean)
            copy(norm.running_var, var)
        for layer, dense in zip(denses, [net.fc1, net.fc2, net.out], strict=True):
            kernel, bias = layer.get_weights()
            copy(dense.weight, kernel.T)
            copy(dense.bias, bias)
    return net


def train(
    net: DocMotionNet,
    X: NDArray[np.float32],
    y: NDArray[Any],
    cfg: TrainConfig,
) -> list[float]:
    """Train ``net`` in place like ``keras.Model.fit``; return each epoch's loss.

    Batches are reshuffled every epoch and the last batch may be short, as
    in Keras. ``cfg.seed`` seeds the shuffling and the dropout masks.
    """
    torch.manual_seed(cfg.seed)
    shuffler = torch.Generator().manual_seed(cfg.seed)
    inputs = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
    targets = torch.from_numpy(np.asarray(y, dtype=np.float32)).reshape(-1, 1)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.learning_rate, eps=1e-7)
    losses = []
    net.train()
    for _ in range(cfg.epochs):
        order = torch.randperm(len(inputs), generator=shuffler)
        total = 0.0
        for start in range(0, len(order), cfg.batch_size):
            batch = order[start : start + cfg.batch_size]
            p = net(inputs[batch]).clamp(1e-7, 1 - 1e-7)
            loss = nn.functional.binary_cross_entropy(p, targets[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        losses.append(total / len(order))
    net.eval()
    return losses


def predict_proba(
    net: DocMotionNet, X: NDArray[np.float32], batch_size: int = 128
) -> NDArray[np.float32]:
    """Return P(flip) for each frame, with dropout off and BatchNorm frozen."""
    net.eval()
    inputs = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
    with torch.no_grad():
        parts = [
            net(inputs[i : i + batch_size]).reshape(-1)
            for i in range(0, len(inputs), batch_size)
        ]
    return torch.cat(parts).numpy().astype(np.float32)
