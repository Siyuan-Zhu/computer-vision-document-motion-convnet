"""Turn camera frames into model input, and load a labelled image folder.

The preprocessing is the 2022 OpenCV recipe, step for step. Its quirks are
kept on purpose: the resize distorts the aspect ratio, the crop drops the
bottom 40 of 140 rows, and the brightening saturates light pixels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

IMAGE_SHAPE = (100, 70, 3)
"""Height, width and channels of one preprocessed frame."""

CLASSES = {"moving": 1, "still": 0}
"""Folder name and label of each class."""

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def preprocess(image: NDArray[Any]) -> NDArray[np.float32]:
    """Preprocess one 8-bit frame into a (100, 70, 3) float32 array in [0, 1].

    Parameters
    ----------
    image : ndarray of uint8
        A BGR frame of shape (H, W, 3) as ``cv2.imread`` returns it, or a
        grayscale frame of shape (H, W) or (H, W, 1), which is first copied
        into three channels.

    Returns
    -------
    ndarray of float32, shape (100, 70, 3)
        The frame resized to 70 x 140, cropped to its top 100 rows,
        brightened (``1.5 * x + 0.5``, saturating at 255), inverted and
        scaled to [0, 1].
    """
    image = np.asarray(image)
    if image.dtype != np.uint8:
        raise ValueError(f"expected a uint8 image, got {image.dtype}")
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected 1 or 3 channels, got image shape {image.shape}")

    resized = cv2.resize(image, dsize=(70, 140), interpolation=cv2.INTER_CUBIC)
    cropped = resized[0:100, 0:70]
    brightened = cv2.addWeighted(cropped, 1.5, np.zeros_like(cropped), 0, 0.5)
    inverted = cv2.bitwise_not(brightened)
    return (inverted / 255).astype(np.float32)


def read_image(path: str | Path) -> NDArray[np.float32]:
    """Read an image file with OpenCV and preprocess it."""
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"could not read an image from {path}")
    return preprocess(image)


def list_images(folder: str | Path) -> list[Path]:
    """Return the image files directly inside ``folder``, sorted by name."""
    return sorted(
        p
        for p in Path(folder).iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def load_split(
    root: str | Path, split: str
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    """Load ``<root>/<split>/{moving,still}/*.jpg`` as preprocessed arrays.

    Returns
    -------
    X : ndarray of float32, shape (n, 100, 70, 3)
    y : ndarray of int64, shape (n,)
        1 for ``moving`` and 0 for ``still``. Moving frames come first, as
        in the 2022 notebook.
    """
    images: list[NDArray[np.float32]] = []
    labels: list[int] = []
    for name, label in CLASSES.items():
        folder = Path(root) / split / name
        paths = list_images(folder) if folder.is_dir() else []
        images.extend(read_image(p) for p in paths)
        labels.extend([label] * len(paths))
    if not images:
        raise FileNotFoundError(
            f"no images found under {Path(root) / split}/{{moving,still}}"
        )
    return np.stack(images), np.asarray(labels, dtype=np.int64)
