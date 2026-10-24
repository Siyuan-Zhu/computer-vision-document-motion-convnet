"""Shared test fixtures: backends that skip cleanly, and synthetic JPEGs."""

import cv2
import numpy as np
import pytest

FRAMEWORK = {"keras": "keras", "torch": "torch"}


@pytest.fixture
def backend_named():
    """Return a function giving the named backend, skipping if its framework
    is not installed."""

    def get(name):
        pytest.importorskip(FRAMEWORK[name])
        from doc_motion.backends import get_backend

        return get_backend(name)

    return get


@pytest.fixture
def write_jpgs():
    """Return a function writing ``n`` small JPEGs of grey level ``value``
    (plus uniform noise of +-``noise``) into a new ``folder``."""

    def write(folder, n, value, noise=0, seed=0):
        folder.mkdir(parents=True)
        rng = np.random.default_rng(seed)
        for i in range(n):
            pixels = value + rng.integers(-noise, noise + 1, size=(80, 60, 3))
            image = np.clip(pixels, 0, 255).astype(np.uint8)
            cv2.imwrite(str(folder / f"{i}.jpg"), image)

    return write
