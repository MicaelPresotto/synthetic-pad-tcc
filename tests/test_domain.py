import numpy as np
import pytest

from synthetic_pad.domain import RgbImage, ensure_rgb_uint8


def test_ensure_rgb_uint8_accepts_valid_image(face_like_image: RgbImage) -> None:
    assert ensure_rgb_uint8(face_like_image) is face_like_image


@pytest.mark.parametrize(
    ("array", "message"),
    [
        (np.zeros((8, 8), dtype=np.uint8), "RGB image"),
        (np.zeros((8, 8, 4), dtype=np.uint8), "RGB image"),
        (np.zeros((8, 8, 3), dtype=np.float32), "uint8"),
    ],
)
def test_ensure_rgb_uint8_rejects_malformed_arrays(array: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ensure_rgb_uint8(array)
