"""End-to-end training runs on tiny synthetic data, for both backends."""

import datetime as dt
import json

import numpy as np
import pytest

from doc_motion.config import TrainConfig, load_config


def _backend(name):
    pytest.importorskip("keras" if name == "keras" else "torch")
    from doc_motion.backends import get_backend

    return get_backend(name)


BACKENDS = ["keras", "torch"]
TINY = TrainConfig(folds=2, epochs=1, batch_size=4, seed=0)


def _frames(n_per_class, seed):
    # Moving frames are dark, still frames bright, with a little noise:
    # an easy problem, so a few epochs are enough to separate them.
    rng = np.random.default_rng(seed)
    shape = (n_per_class, 100, 70, 3)
    moving = 0.2 + 0.05 * rng.standard_normal(shape)
    still = 0.8 + 0.05 * rng.standard_normal(shape)
    X = np.concatenate([moving, still]).astype(np.float32)
    y = np.array([1] * n_per_class + [0] * n_per_class)
    return X, y


@pytest.mark.parametrize("name", BACKENDS)
def test_writes_a_dated_run_folder_with_config_metrics_and_model(name, tmp_path):
    from doc_motion.training import run

    backend = _backend(name)
    record = run(TINY, _frames(4, 0), _frames(2, 1), backend, tmp_path)

    assert record.run_dir.parent == tmp_path
    assert record.run_dir.name.startswith(dt.date.today().isoformat())
    assert load_config(record.run_dir / "config.yaml") == TINY
    metrics = json.loads((record.run_dir / "metrics.json").read_text())
    assert metrics["backend"] == name
    assert metrics["n_train"] == 8
    assert metrics["n_test"] == 4
    assert len(metrics["fold_accuracies"]) == 2
    assert all(0 <= a <= 1 for a in metrics["fold_accuracies"])
    assert {"moving", "still", "accuracy"} <= set(metrics["test_report"])
    assert record.model_path.parent == record.run_dir
    reloaded = backend.load(record.model_path)
    assert backend.predict_proba(reloaded, _frames(1, 2)[0]).shape == (2,)


@pytest.mark.parametrize("name", BACKENDS)
def test_training_loss_falls_on_an_easy_split(name, tmp_path):
    # Only the training loss: after a handful of steps the BatchNorm moving
    # averages (momentum 0.99) are still near their start, so eval-mode
    # predictions on tiny data say little.
    from doc_motion.training import run

    cfg = TrainConfig(folds=2, epochs=4, batch_size=8, learning_rate=1e-3, seed=0)
    record = run(cfg, _frames(8, 0), _frames(4, 1), _backend(name), tmp_path)
    metrics = json.loads((record.run_dir / "metrics.json").read_text())

    losses = metrics["final_fit_loss"]
    assert len(losses) == 4
    assert losses[-1] < losses[0]


@pytest.mark.parametrize("name", BACKENDS)
def test_same_seed_gives_the_same_run(name, tmp_path):
    from doc_motion.training import run

    backend = _backend(name)
    first = run(TINY, _frames(4, 0), _frames(2, 1), backend, tmp_path)
    second = run(TINY, _frames(4, 0), _frames(2, 1), backend, tmp_path)

    assert first.run_dir != second.run_dir
    assert first.fold_accuracies == second.fold_accuracies
    np.testing.assert_allclose(
        backend.predict_proba(backend.load(first.model_path), _frames(2, 3)[0]),
        backend.predict_proba(backend.load(second.model_path), _frames(2, 3)[0]),
        rtol=1e-5,
    )


def test_rejects_frames_of_the_wrong_size(tmp_path):
    from doc_motion.training import run

    X = np.zeros((4, 50, 35, 3), np.float32)
    y = np.array([0, 1, 0, 1])

    with pytest.raises(ValueError, match="100, 70, 3"):
        run(TINY, (X, y), (X, y), _backend("torch"), tmp_path)


def test_unknown_backend_name_is_rejected():
    from doc_motion.backends import get_backend

    with pytest.raises(ValueError, match="keras"):
        get_backend("caffe")
