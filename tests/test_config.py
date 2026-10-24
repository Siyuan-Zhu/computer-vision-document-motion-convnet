"""Training configuration: 2022 defaults and YAML round trip."""

import pytest

from doc_motion.config import TrainConfig, load_config, save_config


def test_defaults_are_the_2022_training_settings():
    cfg = TrainConfig()

    assert (cfg.image_height, cfg.image_width) == (100, 70)
    assert cfg.folds == 5
    assert cfg.shuffle_seed == 1
    assert cfg.epochs == 25
    assert cfg.batch_size == 128
    assert cfg.learning_rate == 0.01
    assert cfg.threshold == 0.5


def test_round_trips_through_yaml(tmp_path):
    cfg = TrainConfig(folds=3, epochs=2, batch_size=16, learning_rate=0.001, seed=7)
    path = tmp_path / "config.yaml"

    save_config(cfg, path)

    assert load_config(path) == cfg


def test_keys_missing_from_the_file_keep_their_defaults(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("epochs: 3\n")

    assert load_config(path) == TrainConfig(epochs=3)


def test_unknown_keys_are_rejected(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("epochs: 3\nlearning_rat: 0.1\n")

    with pytest.raises(ValueError, match="learning_rat"):
        load_config(path)
