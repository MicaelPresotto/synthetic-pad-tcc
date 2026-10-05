"""Replay attacks rendered by carrying the low-frequency amplitude spectrum of a real replay onto
a bona fide face, keeping the face's phase: the Fourier domain adaptation of Yang and Soatto
(2020), used here as a generator that learns the look of a recapture from data instead of
simulating it."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from synthetic_pad.domain.image import RgbImage, content_digest, ensure_rgb_uint8
from synthetic_pad.domain.ports import AttackGenerator

NYQUIST_CYCLES_PER_PX = 0.5

type Crops = np.ndarray[Any, np.dtype[np.uint8]]


@dataclass(frozen=True, slots=True)
class SpectralConfig:
    band_cycles_per_px: float = 0.07
    seed: int = 20261005

    def __post_init__(self) -> None:
        if not 0.0 < self.band_cycles_per_px <= NYQUIST_CYCLES_PER_PX:
            raise ValueError(
                f"band_cycles_per_px must lie in (0, {NYQUIST_CYCLES_PER_PX}], "
                f"got {self.band_cycles_per_px}"
            )


class SpectralTransferGenerator(AttackGenerator):
    name = "spectral"

    def __init__(self, replays: Crops, config: SpectralConfig | None = None) -> None:
        if len(replays) == 0:
            raise ValueError("the generator needs at least one real replay to take a spectrum from")
        self._replays = replays
        self._config = config if config is not None else SpectralConfig()

    def generate(self, image: RgbImage) -> RgbImage:
        ensure_rgb_uint8(image)
        rng = np.random.default_rng([self._config.seed, content_digest(image)])
        replay = self._replays[rng.integers(len(self._replays))]
        if replay.shape != image.shape:
            raise ValueError(f"replay shape {replay.shape} does not match image {image.shape}")
        return transfer_amplitude(image, replay, self._config.band_cycles_per_px)

    def describe(self) -> Mapping[str, object]:
        return {"generator": self.name, **asdict(self._config), "replays": len(self._replays)}


def transfer_amplitude(image: RgbImage, replay: RgbImage, band_cycles_per_px: float) -> RgbImage:
    """The image with the amplitude of every frequency up to the band, in both directions and
    each colour channel, taken from the replay; the phase, which holds the layout, is kept."""
    spectrum = np.fft.fft2(image.astype(np.float64), axes=(0, 1))
    replay_amplitude = np.abs(np.fft.fft2(replay.astype(np.float64), axes=(0, 1)))
    height, width = image.shape[:2]
    low_rows = np.abs(np.fft.fftfreq(height)) <= band_cycles_per_px
    low_cols = np.abs(np.fft.fftfreq(width)) <= band_cycles_per_px
    within_band = (low_rows[:, None] & low_cols[None, :])[..., None]
    amplitude = np.where(within_band, replay_amplitude, np.abs(spectrum))
    transferred = np.fft.ifft2(amplitude * np.exp(1j * np.angle(spectrum)), axes=(0, 1)).real
    attack: RgbImage = np.rint(np.clip(transferred, 0.0, 255.0)).astype(np.uint8)
    return attack
