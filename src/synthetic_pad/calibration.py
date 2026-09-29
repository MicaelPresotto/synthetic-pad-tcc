"""Image statistics for matching generated attacks to real ones."""

from collections.abc import Sequence
from itertools import pairwise
from typing import Any

import cv2
import numpy as np

from synthetic_pad.domain import RgbImage, ensure_rgb_uint8

type BandPowers = np.ndarray[Any, np.dtype[np.float64]]

LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114])
BLACK_QUANTILE = 0.01
WHITE_QUANTILE = 0.99
CHROMA_BANDS = (4, 8, 12, 16, 24, 32, 48, 64, 80, 96, 112)


def tone_statistics(image: RgbImage) -> dict[str, float]:
    """Saturation, contrast, black and white levels in [0, 1]; sharpness as the variance
    of the Laplacian of luma (ITU-R BT.601 weights)."""
    ensure_rgb_uint8(image)
    unit = image.astype(np.float64) / 255.0
    luma = unit @ LUMA_WEIGHTS
    brightest, darkest = unit.max(axis=2), unit.min(axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        saturation = np.where(brightest > 0.0, (brightest - darkest) / brightest, 0.0)
    return {
        "saturation": float(saturation.mean()),
        "contrast": float(luma.std()),
        "black_level": float(np.quantile(luma, BLACK_QUANTILE)),
        "white_level": float(np.quantile(luma, WHITE_QUANTILE)),
        "sharpness": float(cv2.Laplacian(luma, cv2.CV_64F).var()),
    }


def chroma_power(image: RgbImage, bands: Sequence[int] = CHROMA_BANDS) -> BandPowers:
    """Mean spectral power of the colour left once each pixel's grey is removed, in bands
    of cycles per image width; the part of an image moire fringes add to."""
    ensure_rgb_uint8(image)
    height, width = image.shape[:2]
    unit = image.astype(np.float64) / 255.0
    chroma = unit - unit.mean(axis=2, keepdims=True)
    window = np.outer(np.hanning(height), np.hanning(width))
    power = np.sum(np.abs(np.fft.fft2(chroma * window[..., None], axes=(0, 1))) ** 2, axis=2)
    cycles_y = np.fft.fftfreq(height) * width
    cycles_x = np.fft.fftfreq(width) * width
    radius = np.hypot(*np.meshgrid(cycles_y, cycles_x, indexing="ij"))
    return np.array(
        [power[(radius >= low) & (radius < high)].mean() for low, high in pairwise(bands)]
    )
