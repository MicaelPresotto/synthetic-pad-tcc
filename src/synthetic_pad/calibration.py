"""Image statistics for matching generated attacks to real ones."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import cv2
import numpy as np

from synthetic_pad.domain import LUMA_WEIGHTS, RgbImage, ensure_rgb_uint8

type BandPowers = np.ndarray[Any, np.dtype[np.float64]]
type Crops = np.ndarray[Any, np.dtype[np.uint8]]

BLACK_QUANTILE = 0.01
WHITE_QUANTILE = 0.99
CHROMA_BANDS = (4, 8, 12, 16, 24, 32, 48, 64, 80, 96, 112)
LEVELS = ("black_level", "white_level")
LEVEL_STEP = 0.05


@dataclass(frozen=True, slots=True)
class Appearance:
    """Mean tone statistics and colour-fringe power of a set of crops."""

    tone: Mapping[str, float]
    chroma_power: BandPowers


def tone_statistics(image: RgbImage) -> dict[str, float]:
    """Saturation, contrast, black and white levels in [0, 1]; sharpness as the variance
    of the Laplacian of luma."""
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


def appearance(images: Crops) -> Appearance:
    """Crops stacked as (N, H, W, 3)."""
    tones = [tone_statistics(image) for image in images]
    return Appearance(
        {name: float(np.mean([tone[name] for tone in tones])) for name in tones[0]},
        np.mean([chroma_power(image) for image in images], axis=0),
    )


def appearance_distance(generated: Appearance, real: Appearance) -> float:
    """How far one set of crops looks from another, tone and colour weighted equally.

    Levels differ in steps of LEVEL_STEP; every other statistic by its log ratio.
    """
    tone_gaps = [
        (generated.tone[name] - real.tone[name]) / LEVEL_STEP
        if name in LEVELS
        else np.log(generated.tone[name] / real.tone[name])
        for name in real.tone
    ]
    colour_gaps = np.log(generated.chroma_power / real.chroma_power)
    return float((np.mean(np.square(tone_gaps)) + np.mean(np.square(colour_gaps))) / 2.0)
