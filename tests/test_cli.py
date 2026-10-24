"""The doc-motion command line, on tiny synthetic JPEGs."""

import csv
import json

import numpy as np
import pytest

from doc_motion.cli import main
from doc_motion.config import TrainConfig, save_config


@pytest.mark.parametrize("name", ["keras", "torch"])
def test_predict_writes_one_csv_row_per_image(
    name, tmp_path, backend_named, write_jpgs
):
    backend = backend_named(name)
    model = backend.build(TrainConfig())
    # One training step, so the saved optimizer state is complete.
    frames = np.zeros((2, 100, 70, 3), np.float32)
    backend.fit(model, frames, np.array([0, 1]), TrainConfig(epochs=1))
    model_path = tmp_path / f"model{backend.model_suffix}"
    backend.save(model, model_path)
    write_jpgs(tmp_path / "frames", 3, 60, noise=20)
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
        p = float(row["p_flip"])
        assert 0 <= p <= 1
        assert row["label"] == ("flip" if p > 0.5 else "notflip")


def test_predict_fails_cleanly_on_a_folder_with_no_images(
    tmp_path, capsys, backend_named
):
    backend = backend_named("torch")
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


def test_train_writes_a_run_folder_from_an_image_folder(
    tmp_path, capsys, backend_named, write_jpgs
):
    backend_named("torch")
    for split, n in [("train", 4), ("test", 2)]:
        write_jpgs(tmp_path / "data" / split / "flip", n, 40, noise=20, seed=1)
        write_jpgs(tmp_path / "data" / split / "notflip", n, 200, noise=20, seed=2)
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


def test_predict_rejects_an_unknown_model_file_type(tmp_path, capsys, write_jpgs):
    model_path = tmp_path / "model.h5"
    model_path.write_bytes(b"")
    write_jpgs(tmp_path / "frames", 1, 60)

    code = main(
        [
            "predict",
            "--model",
            str(model_path),
            "--images",
            str(tmp_path / "frames"),
            "--out",
            str(tmp_path / "out.csv"),
        ]
    )

    assert code == 1
    assert ".h5" in capsys.readouterr().err
