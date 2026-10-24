"""Batch prediction for a folder of images."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple

import numpy as np

from .backends import backend_for_model
from .data import list_images, read_image


class Prediction(NamedTuple):
    """One image's result."""

    path: Path
    p_moving: float
    label: str


def predict_folder(
    model_path: str | Path, folder: str | Path, threshold: float = 0.5
) -> list[Prediction]:
    """Classify every image in ``folder`` with a saved model.

    The backend is chosen from the model file's suffix (``.keras`` or
    ``.pt``). Images are read in name order and preprocessed exactly as in
    training; a frame is ``moving`` when its probability is above
    ``threshold``.
    """
    paths = list_images(folder)
    if not paths:
        raise FileNotFoundError(f"no images (.jpg, .jpeg, .png) found in {folder}")
    backend = backend_for_model(model_path)
    model = backend.load(Path(model_path))
    X = np.stack([read_image(p) for p in paths])
    probabilities = backend.predict_proba(model, X)
    return [
        Prediction(path, float(p), "moving" if p > threshold else "still")
        for path, p in zip(paths, probabilities, strict=True)
    ]


def write_csv(rows: Iterable[Prediction], path: str | Path) -> None:
    """Write predictions as CSV with columns path, p_moving, label."""
    with Path(path).open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(Prediction._fields)
        for row in rows:
            writer.writerow([str(row.path), f"{row.p_moving:.6f}", row.label])
