import numpy as np
import pytest

from synthetic_pad.calibration import CHROMA_BANDS, chroma_power, tone_statistics
from synthetic_pad.domain import RgbImage

SIZE = 224


def grating(cycles_per_width: float, chromatic: bool) -> RgbImage:
    phase = 2.0 * np.pi * cycles_per_width * np.arange(SIZE) / SIZE
    offsets = np.array([0.0, 2.1, 4.2]) if chromatic else np.zeros(3)
    rows = 128.0 + 60.0 * np.sin(phase[:, None] + offsets)
    return np.repeat(rows[None], SIZE, axis=0).astype(np.uint8)


class TestToneStatistics:
    def test_a_grey_image_has_no_saturation(self) -> None:
        assert tone_statistics(np.full((16, 16, 3), 90, dtype=np.uint8))["saturation"] == 0.0

    def test_a_flat_image_has_neither_contrast_nor_sharpness(self) -> None:
        statistics = tone_statistics(np.full((16, 16, 3), 90, dtype=np.uint8))
        assert statistics["contrast"] == pytest.approx(0.0, abs=1e-12)
        assert statistics["sharpness"] == pytest.approx(0.0, abs=1e-12)

    def test_levels_follow_the_darkest_and_brightest_pixels(self) -> None:
        halves = np.zeros((16, 16, 3), dtype=np.uint8)
        halves[:, 8:] = 255
        statistics = tone_statistics(halves)
        assert statistics["black_level"] == 0.0
        assert statistics["white_level"] == pytest.approx(1.0)


class TestChromaPower:
    def test_gives_one_value_per_band(self) -> None:
        assert chroma_power(grating(20.0, chromatic=True)).shape == (len(CHROMA_BANDS) - 1,)

    def test_puts_a_chromatic_fringe_in_the_band_of_its_frequency(self) -> None:
        powers = chroma_power(grating(20.0, chromatic=True))
        assert int(np.argmax(powers)) == CHROMA_BANDS.index(16)

    def test_ignores_a_grey_fringe(self) -> None:
        assert chroma_power(grating(20.0, chromatic=False)).max() == pytest.approx(0.0, abs=1e-6)
