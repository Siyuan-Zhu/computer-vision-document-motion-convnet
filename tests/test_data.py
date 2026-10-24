"""Preprocessing and folder loading, on synthetic images only."""

import numpy as np
import pytest

from doc_motion.data import load_split, preprocess


def test_output_is_a_100_by_70_float32_image_in_the_unit_range():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(1920, 1080, 3), dtype=np.uint8)

    out = preprocess(image)

    assert out.shape == (100, 70, 3)
    assert out.dtype == np.float32
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_flat_grey_frame_is_brightened_then_inverted():
    # 101 * 1.5 + 0.5 = 152 after brightening; inverted, 255 - 152 = 103.
    image = np.full((300, 200, 3), 101, dtype=np.uint8)

    out = preprocess(image)

    np.testing.assert_allclose(out, 103 / 255, rtol=1e-6)


def test_bright_pixels_saturate_to_white_then_invert_to_black():
    # 200 * 1.5 + 0.5 = 300.5 saturates at 255, which inverts to 0.
    image = np.full((140, 70, 3), 200, dtype=np.uint8)

    out = preprocess(image)

    assert np.all(out == 0.0)


def test_keeps_only_the_top_100_of_140_resized_rows():
    # Top half black, bottom half white: after resizing to 140 rows the
    # kept rows 0:100 are mostly black, so mostly white after inversion,
    # and the bottom rows of the crop are the white half, now black.
    image = np.zeros((140, 70, 3), dtype=np.uint8)
    image[70:] = 255

    out = preprocess(image)

    # Black brightens to 0.5, which rounds to 0 or 1 before inversion.
    np.testing.assert_allclose(out[:65], 1.0, atol=1 / 255)
    np.testing.assert_allclose(out[75:], 0.0)


@pytest.mark.parametrize("shape", [(37, 500, 3), (1, 1, 3), (999, 3, 3)])
def test_odd_sized_frames_still_come_out_100_by_70(shape):
    image = np.full(shape, 50, dtype=np.uint8)

    assert preprocess(image).shape == (100, 70, 3)


def test_grayscale_frame_matches_the_same_frame_in_colour():
    rng = np.random.default_rng(1)
    gray = rng.integers(0, 256, size=(240, 160), dtype=np.uint8)
    colour = np.repeat(gray[:, :, None], 3, axis=2)

    np.testing.assert_array_equal(preprocess(gray), preprocess(colour))
    np.testing.assert_array_equal(preprocess(gray[:, :, None]), preprocess(colour))


def test_rejects_images_that_are_not_8_bit():
    with pytest.raises(ValueError, match="uint8"):
        preprocess(np.zeros((100, 70, 3), dtype=np.float32))


def test_rejects_images_with_an_unsupported_channel_count():
    with pytest.raises(ValueError, match="channels"):
        preprocess(np.zeros((100, 70, 2), dtype=np.uint8))


def test_load_split_labels_flip_as_1_and_notflip_as_0(tmp_path, write_jpgs):
    write_jpgs(tmp_path / "train" / "flip", 3, 10)
    write_jpgs(tmp_path / "train" / "notflip", 2, 240)

    X, y = load_split(tmp_path, "train")

    assert X.shape == (5, 100, 70, 3)
    assert X.dtype == np.float32
    np.testing.assert_array_equal(y, [1, 1, 1, 0, 0])
    # The dark flip frames invert to bright, the bright notflip frames to dark.
    assert X[:3].mean() > 0.9
    assert X[3:].mean() < 0.1


def test_load_split_fails_clearly_when_the_folder_is_empty(tmp_path):
    (tmp_path / "test" / "flip").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="test"):
        load_split(tmp_path, "test")


def test_load_split_names_a_file_it_cannot_read(tmp_path, write_jpgs):
    write_jpgs(tmp_path / "train" / "notflip", 1, 100)
    (tmp_path / "train" / "notflip" / "broken.jpg").write_bytes(b"not a jpeg")

    with pytest.raises(ValueError, match=r"broken\.jpg"):
        load_split(tmp_path, "train")
