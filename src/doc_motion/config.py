"""Training settings, read from and written to YAML."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml

from .data import IMAGE_SHAPE


@dataclass(frozen=True)
class TrainConfig:
    """Settings for one training run. The defaults are the 2022 values.

    Attributes
    ----------
    image_height, image_width : int
        Input size of the network. Preprocessing always produces
        ``data.IMAGE_SHAPE`` (100 x 70), so change these only together with
        your own preprocessing.
    folds : int
        Number of cross-validation folds on the training set.
    shuffle_seed : int
        ``random_state`` of the shuffled ``KFold`` split.
    epochs, batch_size : int
        Used for every cross-validation fit and for the final fit.
    learning_rate : float
        Adam learning rate.
    threshold : float
        A frame is labelled ``moving`` when its probability is above this.
    seed : int
        Seeds weight initialisation, dropout and batch order. The 2022 run
        was unseeded; this is added so packaged runs are repeatable.
    """

    image_height: int = IMAGE_SHAPE[0]
    image_width: int = IMAGE_SHAPE[1]
    folds: int = 5
    shuffle_seed: int = 1
    epochs: int = 25
    batch_size: int = 128
    learning_rate: float = 0.01
    threshold: float = 0.5
    seed: int = 0

    @property
    def input_shape(self) -> tuple[int, int, int]:
        """Network input as (height, width, channels)."""
        return (self.image_height, self.image_width, 3)


def load_config(path: str | Path) -> TrainConfig:
    """Read a YAML file into a ``TrainConfig``; missing keys keep defaults."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a mapping of settings")
    known = {f.name for f in fields(TrainConfig)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"unknown settings in {path}: {', '.join(unknown)}")
    return TrainConfig(**raw)


def save_config(cfg: TrainConfig, path: str | Path) -> None:
    """Write ``cfg`` to a YAML file that ``load_config`` reads back."""
    Path(path).write_text(yaml.safe_dump(asdict(cfg), sort_keys=False))
