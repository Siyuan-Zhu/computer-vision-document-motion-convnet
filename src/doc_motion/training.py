"""The 2022 training procedure, recorded as a dated run folder.

1. Shuffled k-fold cross-validation on the training set: a fresh model per
   fold, scored by accuracy at the threshold on the held-out fold.
2. A final model fitted on the whole training set.
3. A classification report for that model on the separate test set.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold

from .backends import Backend
from .config import TrainConfig, save_config
from .data import CLASS_NAMES

Split = tuple[NDArray[np.float32], NDArray[Any]]


@dataclass(frozen=True)
class RunRecord:
    """Where a run was written and its headline numbers."""

    run_dir: Path
    model_path: Path
    fold_accuracies: list[float]
    cv_accuracy_mean: float
    cv_accuracy_std: float
    test_report: dict[str, Any]


def run(
    cfg: TrainConfig,
    train: Split,
    test: Split,
    backend: Backend,
    out_dir: str | Path,
) -> RunRecord:
    """Cross-validate, fit the final model, evaluate it and save the run.

    The run folder ``<out_dir>/<date>_<time>_<backend>`` holds the
    ``config.yaml`` used, a ``metrics.json`` and the final model file.
    The cross-validation standard deviation is the population one
    (``np.std``), as in 2022.
    """
    X_train, y_train = _check(train, cfg, "train")
    X_test, y_test = _check(test, cfg, "test")

    fold_accuracies = []
    kfold = KFold(cfg.folds, shuffle=True, random_state=cfg.shuffle_seed)
    for fit_idx, val_idx in kfold.split(X_train):
        model = backend.build(cfg)
        backend.fit(model, X_train[fit_idx], y_train[fit_idx], cfg)
        predicted = _label(backend.predict_proba(model, X_train[val_idx]), cfg)
        fold_accuracies.append(float(np.mean(predicted == y_train[val_idx])))

    model = backend.build(cfg)
    final_fit_loss = backend.fit(model, X_train, y_train, cfg)
    predicted = _label(backend.predict_proba(model, X_test), cfg)
    report: dict[str, Any] = classification_report(
        y_test,
        predicted,
        labels=list(range(len(CLASS_NAMES))),
        target_names=list(CLASS_NAMES),
        output_dict=True,
        zero_division=0,
    )

    run_dir = _new_run_dir(Path(out_dir), backend.name)
    model_path = run_dir / f"model{backend.model_suffix}"
    backend.save(model, model_path)
    save_config(cfg, run_dir / "config.yaml")
    record = RunRecord(
        run_dir=run_dir,
        model_path=model_path,
        fold_accuracies=fold_accuracies,
        cv_accuracy_mean=float(np.mean(fold_accuracies)),
        cv_accuracy_std=float(np.std(fold_accuracies)),
        test_report=report,
    )
    metrics = {
        "backend": backend.name,
        "framework": backend.framework(),
        "n_train": len(y_train),
        "n_test": len(y_test),
        "fold_accuracies": record.fold_accuracies,
        "cv_accuracy_mean": record.cv_accuracy_mean,
        "cv_accuracy_std": record.cv_accuracy_std,
        "final_fit_loss": final_fit_loss,
        "test_accuracy": report["accuracy"],
        "test_report": report,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return record


def _check(split: Split, cfg: TrainConfig, name: str) -> Split:
    X, y = np.asarray(split[0], dtype=np.float32), np.asarray(split[1])
    if X.shape[1:] != cfg.input_shape:
        raise ValueError(
            f"{name} frames have shape {X.shape[1:]}, expected {cfg.input_shape}"
        )
    if len(X) != len(y):
        raise ValueError(f"{name} has {len(X)} frames but {len(y)} labels")
    return X, y


def _label(p: NDArray[np.float32], cfg: TrainConfig) -> NDArray[np.int64]:
    return (p > cfg.threshold).astype(np.int64)


def _new_run_dir(out_dir: Path, backend_name: str) -> Path:
    stem = f"{dt.datetime.now():%Y-%m-%d_%H%M%S}_{backend_name}"
    candidate, n = out_dir / stem, 1
    while candidate.exists():
        n += 1
        candidate = out_dir / f"{stem}_{n}"
    candidate.mkdir(parents=True)
    return candidate
