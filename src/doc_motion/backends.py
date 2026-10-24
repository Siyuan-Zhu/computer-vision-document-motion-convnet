"""One interface over the Keras reference model and the native PyTorch port.

Both backends take the same arrays: float32 NHWC frames of shape
(n, 100, 70, 3) and 0/1 labels. Frameworks are imported only when a
backend is used, so either one works without the other installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from .config import TrainConfig


class Backend(Protocol):
    """What training and prediction need from a framework."""

    name: str
    model_suffix: str

    def framework(self) -> str: ...

    def build(self, cfg: TrainConfig) -> Any: ...

    def fit(
        self, model: Any, X: NDArray[np.float32], y: NDArray[Any], cfg: TrainConfig
    ) -> list[float]:
        """Train ``model`` in place; return the training loss of each epoch."""
        ...

    def predict_proba(self, model: Any, X: NDArray[np.float32]) -> NDArray[np.float32]:
        """Return P(moving) for each frame, shape (n,)."""
        ...

    def save(self, model: Any, path: Path) -> None: ...

    def load(self, path: Path) -> Any: ...


class KerasBackend:
    """The Keras 3 reference model on whichever ``KERAS_BACKEND`` is set."""

    name = "keras"
    model_suffix = ".keras"

    def framework(self) -> str:
        import keras

        return f"keras {keras.__version__} ({keras.backend.backend()})"

    def build(self, cfg: TrainConfig) -> Any:
        from .keras_model import build_keras_model

        return build_keras_model(cfg)

    def fit(
        self, model: Any, X: NDArray[np.float32], y: NDArray[Any], cfg: TrainConfig
    ) -> list[float]:
        history = model.fit(
            X, y, epochs=cfg.epochs, batch_size=cfg.batch_size, verbose=0
        )
        return [float(v) for v in history.history["loss"]]

    def predict_proba(self, model: Any, X: NDArray[np.float32]) -> NDArray[np.float32]:
        p = model.predict(X, batch_size=128, verbose=0)
        return np.asarray(p, dtype=np.float32).reshape(-1)

    def save(self, model: Any, path: Path) -> None:
        model.save(path)

    def load(self, path: Path) -> Any:
        import keras

        return keras.saving.load_model(path)


class TorchBackend:
    """The hand-written PyTorch port."""

    name = "torch"
    model_suffix = ".pt"

    def framework(self) -> str:
        import torch

        return f"torch {torch.__version__}"

    def build(self, cfg: TrainConfig) -> Any:
        import torch

        from .torch_model import DocMotionNet

        torch.manual_seed(cfg.seed)
        return DocMotionNet(cfg.input_shape)

    def fit(
        self, model: Any, X: NDArray[np.float32], y: NDArray[Any], cfg: TrainConfig
    ) -> list[float]:
        from .torch_model import train

        return train(model, X, y, cfg)

    def predict_proba(self, model: Any, X: NDArray[np.float32]) -> NDArray[np.float32]:
        from .torch_model import predict_proba

        return predict_proba(model, X)

    def save(self, model: Any, path: Path) -> None:
        import torch

        torch.save(
            {"input_shape": model.input_shape, "state_dict": model.state_dict()}, path
        )

    def load(self, path: Path) -> Any:
        import torch

        from .torch_model import DocMotionNet

        saved = torch.load(path, map_location="cpu", weights_only=True)
        model = DocMotionNet(tuple(saved["input_shape"]))
        model.load_state_dict(saved["state_dict"])
        return model.eval()


BACKENDS: dict[str, type[KerasBackend] | type[TorchBackend]] = {
    "keras": KerasBackend,
    "torch": TorchBackend,
}


def get_backend(name: str) -> Backend:
    """Return the backend called ``name``: ``"keras"`` or ``"torch"``."""
    try:
        return BACKENDS[name]()
    except KeyError:
        choices = ", ".join(BACKENDS)
        raise ValueError(f"unknown backend {name!r}; choose from {choices}") from None


def backend_for_model(path: str | Path) -> Backend:
    """Pick the backend that saved a model file, from its suffix."""
    suffix = Path(path).suffix
    for backend in BACKENDS.values():
        if backend.model_suffix == suffix:
            return backend()
    known = ", ".join(b.model_suffix for b in BACKENDS.values())
    raise ValueError(f"unknown model file type {suffix!r}; expected {known}")
