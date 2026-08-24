"""Replay-attack generator: a face shown on a screen and photographed again.

Stages run in physical light-path order, and the order changes the result — glare lies on
the glass so the lens that follows blurs it, whereas moiré is born at the sensor and
therefore stays crisp. The README documents the full chain.
"""

import hashlib
import math
import warnings
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np

from synthetic_pad.domain.image import RgbImage, ensure_rgb_uint8
from synthetic_pad.domain.ports import AttackGenerator

type FloatImage = np.ndarray[Any, np.dtype[np.float64]]

MIN_RESOLVABLE_PERIOD_PX = 2.0

GLARE_CENTRE_X_RANGE = (0.15, 0.85)
GLARE_CENTRE_Y_RANGE = (0.05, 0.55)
GLARE_ASPECT = 0.32
GLARE_ANGLE_DEG = -35.0

MOIRE_HARMONIC_FALLOFF = 1.6
MOIRE_BEAT_FREQUENCY = 2.5
MOIRE_BEAT_DEPTH = 0.6


@dataclass(frozen=True, slots=True)
class MoireConfig:
    """Parameters of the display-recapture simulation.

    Units the names cannot carry: ``tilt_x`` and ``tilt_y`` are fractions of the
    perpendicular dimension, ``moire_frequency`` counts cycles across the image width,
    and ``noise_sigma`` is on the 0-255 scale.
    """

    tilt_x: float = 0.05
    tilt_y: float = 0.03
    rotation_deg: float = 0.8
    contrast: float = 0.82
    black_level: float = 0.06
    white_level: float = 0.93
    glare_intensity: float = 0.20
    glare_extent: float = 0.26
    blur_sigma: float = 0.7
    moire_amplitude: float = 0.10
    moire_frequency: float = 20.0
    moire_angle_deg: float = 12.0
    moire_harmonics: int = 3
    moire_chroma_shift: float = 0.9
    moire_cross_weight: float = 0.45
    noise_sigma: float = 2.0
    variation: float = 0.25
    seed: int = 20260823

    def __post_init__(self) -> None:
        if not 0.0 <= self.black_level < self.white_level <= 1.0:
            raise ValueError(
                "require 0 <= black_level < white_level <= 1, got "
                f"black_level={self.black_level}, white_level={self.white_level}"
            )
        if self.contrast <= 0.0:
            raise ValueError(f"contrast must be positive, got {self.contrast}")
        if self.moire_harmonics < 1:
            raise ValueError(f"moire_harmonics must be at least 1, got {self.moire_harmonics}")
        if self.moire_frequency <= 0.0:
            raise ValueError(f"moire_frequency must be positive, got {self.moire_frequency}")
        for field_name in ("moire_amplitude", "blur_sigma", "noise_sigma", "variation"):
            value: float = getattr(self, field_name)
            if value < 0.0:
                raise ValueError(f"{field_name} must be non-negative, got {value}")
        if not 0.0 <= self.glare_intensity <= 1.0:
            raise ValueError(f"glare_intensity must lie in [0, 1], got {self.glare_intensity}")
        if self.glare_extent <= 0.0:
            raise ValueError(f"glare_extent must be positive, got {self.glare_extent}")
        if self.moire_cross_weight < 0.0:
            raise ValueError(
                f"moire_cross_weight must be non-negative, got {self.moire_cross_weight}"
            )


class MoireRecaptureGenerator(AttackGenerator):
    name = "moire"

    def __init__(self, config: MoireConfig | None = None) -> None:
        self._config = config if config is not None else MoireConfig()

    def generate(self, image: RgbImage) -> RgbImage:
        ensure_rgb_uint8(image)
        rng = np.random.default_rng([self._config.seed, _content_digest(image)])
        canvas: FloatImage = image.astype(np.float64) / 255.0
        canvas = self._apply_geometry(canvas, rng)
        canvas = self._apply_tone(canvas)
        canvas = self._apply_glare(canvas, rng)
        canvas = self._apply_lens(canvas)
        canvas = self._apply_moire(canvas, rng)
        canvas = self._apply_noise(canvas, rng)
        return _quantise_to_uint8(canvas)

    def describe(self) -> Mapping[str, object]:
        return {"generator": self.name, **asdict(self._config)}

    def _apply_geometry(self, canvas: FloatImage, rng: np.random.Generator) -> FloatImage:
        """The target quad over-covers the frame on purpose. Left inset, the keystone
        would strand border pixels that no source pixel maps to, and that rim would
        become a cue in itself rather than part of the recapture.
        """
        cfg = self._config
        tilt_x = cfg.tilt_x * self._jitter(rng)
        tilt_y = cfg.tilt_y * self._jitter(rng)
        roll = math.radians(cfg.rotation_deg * self._jitter(rng))
        if tilt_x == 0.0 and tilt_y == 0.0 and roll == 0.0:
            return canvas

        height, width = canvas.shape[:2]
        source = np.array(
            [[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]], dtype=np.float32
        )
        far_vertical_edge_pinch = tilt_x * height / 2.0
        far_horizontal_edge_pinch = tilt_y * width / 2.0
        target = np.array(
            [
                [far_horizontal_edge_pinch, far_vertical_edge_pinch],
                [width - far_horizontal_edge_pinch, 0.0],
                [width, height],
                [0.0, height - far_vertical_edge_pinch],
            ],
            dtype=np.float64,
        )
        centre = np.array([width / 2.0, height / 2.0])
        overscan_to_cover_frame = (1.0 + max(abs(tilt_x), abs(tilt_y))) * (
            1.0 + 2.0 * abs(math.sin(roll))
        )
        rotation = np.array([[math.cos(roll), -math.sin(roll)], [math.sin(roll), math.cos(roll)]])
        target = centre + ((target - centre) * overscan_to_cover_frame) @ rotation.T

        matrix = cv2.getPerspectiveTransform(source, target.astype(np.float32))
        warped = cv2.warpPerspective(
            canvas,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return _as_float(warped)

    def _apply_tone(self, canvas: FloatImage) -> FloatImage:
        cfg = self._config
        toned = np.clip((canvas - 0.5) * cfg.contrast + 0.5, 0.0, 1.0)
        return cfg.black_level + toned * (cfg.white_level - cfg.black_level)

    def _apply_glare(self, canvas: FloatImage, rng: np.random.Generator) -> FloatImage:
        """The highlight is biased to the upper half, where the lights of a room
        actually reflect off a screen.
        """
        cfg = self._config
        if cfg.glare_intensity == 0.0:
            return canvas

        height, width = canvas.shape[:2]
        rows, cols = np.mgrid[0:height, 0:width].astype(np.float64)
        centre_x = rng.uniform(*GLARE_CENTRE_X_RANGE) * width
        centre_y = rng.uniform(*GLARE_CENTRE_Y_RANGE) * height
        angle = math.radians(GLARE_ANGLE_DEG * self._jitter(rng))
        offset_x, offset_y = cols - centre_x, rows - centre_y
        major = offset_x * math.cos(angle) + offset_y * math.sin(angle)
        minor = -offset_x * math.sin(angle) + offset_y * math.cos(angle)
        sigma_major = cfg.glare_extent * width
        sigma_minor = cfg.glare_extent * GLARE_ASPECT * width
        glare = cfg.glare_intensity * np.exp(
            -0.5 * ((major / sigma_major) ** 2 + (minor / sigma_minor) ** 2)
        )
        return _screen_blend(canvas, glare[..., None])

    def _apply_lens(self, canvas: FloatImage) -> FloatImage:
        if self._config.blur_sigma == 0.0:
            return canvas
        blurred = cv2.GaussianBlur(
            canvas, (0, 0), sigmaX=self._config.blur_sigma, borderType=cv2.BORDER_REPLICATE
        )
        return _as_float(blurred)

    def _apply_moire(self, canvas: FloatImage, rng: np.random.Generator) -> FloatImage:
        """Modulation is multiplicative because moiré modulates emitted light. An
        additive pattern would lift the black regions that a real screen leaves black.
        """
        cfg = self._config
        if cfg.moire_amplitude == 0.0:
            return canvas

        height, width = canvas.shape[:2]
        rows, cols = np.mgrid[0:height, 0:width].astype(np.float64)
        x_in_widths, y_in_widths = cols / width, rows / width
        angle = math.radians(cfg.moire_angle_deg * self._jitter(rng))
        along_grating = x_in_widths * math.cos(angle) + y_in_widths * math.sin(angle)
        across_grating = -x_in_widths * math.sin(angle) + y_in_widths * math.cos(angle)

        frequency = cfg.moire_frequency * self._jitter(rng)
        self._warn_if_unresolvable(frequency, width)

        channel_phases = np.arange(3) * cfg.moire_chroma_shift
        pattern = np.zeros((height, width, 3), dtype=np.float64)
        weight_total = 0.0
        for axis, axis_weight in ((along_grating, 1.0), (across_grating, cfg.moire_cross_weight)):
            if axis_weight == 0.0:
                continue
            for harmonic in range(1, cfg.moire_harmonics + 1):
                amplitude = axis_weight / harmonic**MOIRE_HARMONIC_FALLOFF
                phase = rng.uniform(0.0, 2.0 * math.pi)
                carrier = 2.0 * math.pi * frequency * harmonic * axis + phase
                pattern += amplitude * np.sin(carrier[..., None] + channel_phases)
                weight_total += amplitude

        pattern /= weight_total
        beat_phase = rng.uniform(0.0, 2.0 * math.pi)
        beat_envelope = 1.0 - MOIRE_BEAT_DEPTH * (
            0.5 - 0.5 * np.sin(2.0 * math.pi * MOIRE_BEAT_FREQUENCY * across_grating + beat_phase)
        )
        pattern *= beat_envelope[..., None]
        return canvas * (1.0 + cfg.moire_amplitude * pattern)

    def _warn_if_unresolvable(self, frequency: float, width: int) -> None:
        period_px = width / frequency
        if period_px >= MIN_RESOLVABLE_PERIOD_PX:
            return
        warnings.warn(
            f"moire_frequency={frequency:.1f} yields a {period_px:.2f} px fringe period at "
            f"width {width}px. Below {MIN_RESOLVABLE_PERIOD_PX} px the pattern aliases and no "
            "longer represents the requested frequency: lower moire_frequency or work on "
            "larger crops.",
            RuntimeWarning,
            stacklevel=4,
        )

    def _apply_noise(self, canvas: FloatImage, rng: np.random.Generator) -> FloatImage:
        if self._config.noise_sigma == 0.0:
            return canvas
        noise = rng.normal(0.0, self._config.noise_sigma / 255.0, canvas.shape)
        return canvas + noise

    def _jitter(self, rng: np.random.Generator) -> float:
        if self._config.variation == 0.0:
            return 1.0
        return 1.0 + float(rng.uniform(-self._config.variation, self._config.variation))


def _as_float(array: Any) -> FloatImage:
    return np.asarray(array, dtype=np.float64)


def _screen_blend(base: FloatImage, layer: FloatImage) -> FloatImage:
    """Combine two normalised images so highlights saturate instead of clipping."""
    return _as_float(1.0 - (1.0 - base) * (1.0 - layer))


def _quantise_to_uint8(canvas: FloatImage) -> RgbImage:
    """Round rather than truncate: truncation would darken every generated sample by
    half a level, a bias present in all of them.
    """
    return np.rint(np.clip(canvas * 255.0, 0.0, 255.0)).astype(np.uint8)


def _content_digest(image: RgbImage) -> int:
    """Folded into the seed, this gives both properties at once: the same image always
    yields the same attack, while different images get different moiré phases and
    highlight positions.
    """
    digest = hashlib.blake2b(image.tobytes(), digest_size=8).digest()
    return int.from_bytes(digest, "big")
