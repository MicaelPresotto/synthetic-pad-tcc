"""The one image format shared across every layer: RGB, uint8, (H, W, 3)."""

from typing import Any

import numpy as np

type RgbImage = np.ndarray[Any, np.dtype[np.uint8]]

LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114])
"""ITU-R BT.601 weights that turn RGB into perceived brightness."""


def ensure_rgb_uint8(image: np.ndarray[Any, Any]) -> RgbImage:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) RGB image, got shape {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"expected dtype uint8, got {image.dtype}")
    return image
