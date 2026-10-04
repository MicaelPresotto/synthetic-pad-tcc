import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

from synthetic_pad.calibration import (
    CHROMA_BANDS,
    LEVEL_STEP,
    Appearance,
    appearance,
    appearance_distance,
    chroma_power,
    noise_level,
    read_fitted_config,
    tone_statistics,
)
from synthetic_pad.domain import RgbImage
from synthetic_pad.generators.moire import MoireConfig

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


def looks(black_level: float = 0.05, contrast: float = 0.2, fringe: float = 1.0) -> Appearance:
    tone = {"saturation": 0.3, "contrast": contrast, "black_level": black_level}
    return Appearance(tone | {"white_level": 0.9, "sharpness": 0.01}, np.full(10, fringe))


class TestAppearance:
    def test_averages_over_the_crops(self) -> None:
        crops = np.stack([grating(20.0, chromatic=True), np.full((SIZE, SIZE, 3), 90, np.uint8)])
        measured = appearance(crops)
        assert measured.tone["saturation"] == pytest.approx(
            np.mean([tone_statistics(crop)["saturation"] for crop in crops])
        )
        np.testing.assert_allclose(
            measured.chroma_power, np.mean([chroma_power(crop) for crop in crops], axis=0)
        )


class TestAppearanceDistance:
    def test_is_zero_between_identical_appearances(self) -> None:
        assert appearance_distance(looks(), looks()) == 0.0

    def test_counts_a_level_gap_in_steps(self) -> None:
        one_step = appearance_distance(looks(black_level=0.05 + LEVEL_STEP), looks())
        assert one_step == pytest.approx(1.0 / 5 / 2)

    def test_counts_doubling_and_halving_alike(self) -> None:
        doubled = appearance_distance(looks(contrast=0.4), looks())
        halved = appearance_distance(looks(contrast=0.1), looks())
        assert doubled == pytest.approx(halved)

    def test_weighs_tone_and_colour_equally(self) -> None:
        reference = looks()
        every_tone_one_unit_off = Appearance(
            {
                name: value + LEVEL_STEP if name in ("black_level", "white_level") else value * np.e
                for name, value in reference.tone.items()
            },
            reference.chroma_power,
        )
        every_band_one_unit_off = Appearance(reference.tone, reference.chroma_power * np.e)
        assert appearance_distance(every_tone_one_unit_off, reference) == pytest.approx(0.5)
        assert appearance_distance(every_band_one_unit_off, reference) == pytest.approx(0.5)


class TestReadFittedConfig:
    def test_reads_back_the_settings_a_fit_wrote(self, tmp_path: Path) -> None:
        fitted = replace(MoireConfig(), saturation=1.3, moire_amplitude=0.07)
        path = tmp_path / "fitted.json"
        path.write_text(json.dumps({"distance": 0.1, "config": asdict(fitted)}))
        assert read_fitted_config(path) == fitted


class TestNoiseLevel:
    def test_a_flat_image_has_no_noise(self) -> None:
        assert noise_level(np.full((32, 32, 3), 90, dtype=np.uint8)) == 0.0

    def test_ignores_a_linear_ramp(self) -> None:
        ramp = np.repeat(np.linspace(20.0, 220.0, SIZE)[None, :, None], SIZE, axis=0)
        image = np.repeat(ramp, 3, axis=2).astype(np.uint8)
        assert noise_level(image) < 0.5

    def test_recovers_the_deviation_of_added_grey_noise(self) -> None:
        rng = np.random.default_rng(20261004)
        noise = rng.normal(0.0, 5.0, (SIZE, SIZE, 1))
        image = np.clip(np.rint(128.0 + noise), 0, 255).repeat(3, axis=2).astype(np.uint8)
        assert noise_level(image) == pytest.approx(5.0, rel=0.05)
