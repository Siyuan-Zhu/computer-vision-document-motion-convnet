"""The doc-motion command line, on tiny synthetic JPEGs."""

import csv
import json

import cv2
import numpy as np
import pytest

from doc_motion.cli import main
from doc_motion.config import TrainConfig, save_config


def _backend(name):
    pytest.importorskip("keras" if name == "keras" else "torch")
    from doc_motion.backends import get_backend

    return get_backend(name)


def _write_jpgs(folder, n, value, seed=0):
    folder.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    for i in range(n):
        noise = rng.integers(-20, 20, size=(80, 60, 3))
        image = np.clip(value + noise, 0, 255).astype(np.uint8)
        cv2.imwrite(str(folder / f"{i}.jpg"), image)


@pytest.mark.parametrize("name", ["keras", "torch"])
def test_predict_writes_one_csv_row_per_image(name, tmp_path):
    backend = _backend(name)
    model_path = tmp_path / f"model{backend.model_suffix}"
    backend.save(backend.build(TrainConfig()), model_path)
    _write_jpgs(tmp_path / "frames", 3, 60)
    (tmp_path / "frames" / "notes.txt").write_text("not an image")
    out = tmp_path / "predictions.csv"

    code = main(
        [
            "predict",
            "--model",
            str(model_path),
            "--images",
            str(tmp_path / "frames"),
            "--out",
            str(out),
        ]
    )

    assert code == 0
    with out.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["path"].rsplit("/", 1)[-1] for r in rows] == ["0.jpg", "1.jpg", "2.jpg"]
    for row in rows:
        p = float(row["p_moving"])
        assert 0 <= p <= 1
        assert row["label"] == ("moving" if p > 0.5 else "still")


def test_predict_fails_cleanly_on_a_folder_with_no_images(tmp_path, capsys):
    backend = _backend("torch")
    model_path = tmp_path / "model.pt"
    backend.save(backend.build(TrainConfig()), model_path)
    (tmp_path / "empty").mkdir()

    code = main(
        [
            "predict",
            "--model",
            str(model_path),
            "--images",
            str(tmp_path / "empty"),
            "--out",
            str(tmp_path / "out.csv"),
        ]
    )

    assert code == 1
    assert "no images" in capsys.readouterr().err
    assert not (tmp_path / "out.csv").exists()


def test_train_writes_a_run_folder_from_an_image_folder(tmp_path, capsys):
    _backend("torch")
    for split, n in [("train", 4), ("test", 2)]:
        _write_jpgs(tmp_path / "data" / split / "moving", n, 40, seed=1)
        _write_jpgs(tmp_path / "data" / split / "still", n, 200, seed=2)
    config = tmp_path / "tiny.yaml"
    save_config(TrainConfig(folds=2, epochs=1, batch_size=4), config)

    code = main(
        [
            "train",
            "--config",
            str(config),
            "--data",
            str(tmp_path / "data"),
            "--backend",
            "torch",
            "--out",
            str(tmp_path / "runs"),
        ]
    )

    assert code == 0
    (run_dir,) = (tmp_path / "runs").iterdir()
    assert str(run_dir) in capsys.readouterr().out
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert (metrics["n_train"], metrics["n_test"]) == (8, 4)
    assert (run_dir / "model.pt").is_file()
