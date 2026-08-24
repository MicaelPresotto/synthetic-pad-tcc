import numpy as np
import pytest

from synthetic_pad.domain import RgbImage


@pytest.fixture
def face_like_image() -> RgbImage:
    rng = np.random.default_rng(20260823)
    height, width = 192, 224
    base = np.linspace(40, 210, width, dtype=np.float64)
    image = np.repeat(base[None, :], height, axis=0)
    stacked = np.stack([image, image * 0.85, image * 0.7], axis=-1)
    stacked[60:130, 50:160] += 30.0
    noise = rng.normal(0.0, 3.0, stacked.shape)
    return np.clip(stacked + noise, 0, 255).astype(np.uint8)
