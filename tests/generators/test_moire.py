import dataclasses
import warnings
from typing import Any

import numpy as np
import pytest

from synthetic_pad.domain import RgbImage
from synthetic_pad.generators.moire import MoireConfig, MoireRecaptureGenerator

NEUTRAL = MoireConfig(
    tilt_x=0.0,
    tilt_y=0.0,
    rotation_deg=0.0,
    contrast=1.0,
    black_level=0.0,
    white_level=1.0,
    glare_intensity=0.0,
    blur_sigma_px=0.0,
    moire_amplitude=0.0,
    noise_sigma_levels=0.0,
    variation=0.0,
)

EVERY_STAGE = [
    {"tilt_x": 0.08},
    {"tilt_y": 0.08},
    {"rotation_deg": 2.0},
    {"contrast": 0.7},
    {"black_level": 0.1},
    {"white_level": 0.85},
    {"glare_intensity": 0.3},
    {"blur_sigma_px": 1.5},
    {"moire_amplitude": 0.15},
    {"noise_sigma_levels": 4.0},
]

MEANINGLESS_PARAMETERS = [
    ({"black_level": 0.9, "white_level": 0.5}, "black_level < white_level"),
    ({"contrast": 0.0}, "contrast must be positive"),
    ({"moire_cycles_per_width": 0.0}, "moire_cycles_per_width must be positive"),
    ({"moire_amplitude": -0.1}, "non-negative"),
    ({"noise_sigma_levels": -1.0}, "non-negative"),
    ({"glare_intensity": 2.0}, r"\[0, 1\]"),
    ({"glare_extent": 0.0}, "must be positive"),
    ({"moire_cross_weight": -0.5}, "non-negative"),
]


def high_frequency_energy(image: RgbImage) -> float:
    grey = image.astype(np.float64).mean(axis=2)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(grey)))
    rows, cols = grey.shape
    centre_r, centre_c = rows // 2, cols // 2
    keep = min(rows, cols) // 4
    spectrum[centre_r - keep : centre_r + keep, centre_c - keep : centre_c + keep] = 0.0
    return float(spectrum.mean())


class TestOutputContract:
    def test_preserves_shape_and_dtype(self, face_like_image: RgbImage) -> None:
        attack = MoireRecaptureGenerator().generate(face_like_image)
        assert attack.shape == face_like_image.shape
        assert attack.dtype == np.uint8

    def test_stays_within_the_uint8_range(self, face_like_image: RgbImage) -> None:
        config = dataclasses.replace(MoireConfig(), moire_amplitude=0.9, glare_intensity=1.0)
        attack = MoireRecaptureGenerator(config).generate(face_like_image)
        assert attack.min() >= 0
        assert attack.max() <= 255

    def test_is_deterministic(self, face_like_image: RgbImage) -> None:
        first = MoireRecaptureGenerator().generate(face_like_image)
        second = MoireRecaptureGenerator().generate(face_like_image)
        np.testing.assert_array_equal(first, second)

    def test_a_different_seed_changes_the_result(self, face_like_image: RgbImage) -> None:
        default = MoireRecaptureGenerator().generate(face_like_image)
        reseeded = MoireRecaptureGenerator(MoireConfig(seed=7)).generate(face_like_image)
        assert not np.array_equal(default, reseeded)

    def test_different_images_get_different_patterns(self, face_like_image: RgbImage) -> None:
        generator = MoireRecaptureGenerator(dataclasses.replace(NEUTRAL, moire_amplitude=0.25))
        flat = np.full_like(face_like_image, 128)
        shifted = flat.copy()
        shifted[0, 0, 0] = 129

        first = generator.generate(flat).astype(np.int16)
        second = generator.generate(shifted).astype(np.int16)
        assert np.abs(first - second).mean() > 1.0


class TestArtifacts:
    def test_the_neutral_config_is_the_identity(self, face_like_image: RgbImage) -> None:
        attack = MoireRecaptureGenerator(NEUTRAL).generate(face_like_image)
        np.testing.assert_array_equal(attack, face_like_image)

    @pytest.mark.parametrize("override", EVERY_STAGE)
    def test_every_stage_changes_the_image(
        self, face_like_image: RgbImage, override: dict[str, Any]
    ) -> None:
        baseline = MoireRecaptureGenerator(NEUTRAL).generate(face_like_image)
        config = dataclasses.replace(NEUTRAL, **override)
        attack = MoireRecaptureGenerator(config).generate(face_like_image)
        assert not np.array_equal(attack, baseline)

    def test_moire_adds_high_frequency_energy(self, face_like_image: RgbImage) -> None:
        without = MoireRecaptureGenerator(NEUTRAL).generate(face_like_image)
        config = dataclasses.replace(NEUTRAL, moire_amplitude=0.2, moire_cycles_per_width=60.0)
        with_moire = MoireRecaptureGenerator(config).generate(face_like_image)
        assert high_frequency_energy(with_moire) > high_frequency_energy(without)

    def test_moire_is_chromatic(self, face_like_image: RgbImage) -> None:
        config = dataclasses.replace(NEUTRAL, moire_amplitude=0.25, moire_chroma_shift_rad=1.2)
        attack = MoireRecaptureGenerator(config).generate(face_like_image).astype(np.float64)
        baseline = MoireRecaptureGenerator(NEUTRAL).generate(face_like_image).astype(np.float64)
        assert attack.std(axis=2).mean() > baseline.std(axis=2).mean()

    def test_cross_weight_zero_leaves_a_single_grating(self, face_like_image: RgbImage) -> None:
        stripes_only = dataclasses.replace(NEUTRAL, moire_amplitude=0.2, moire_cross_weight=0.0)
        both = dataclasses.replace(stripes_only, moire_cross_weight=0.45)
        baseline = MoireRecaptureGenerator(NEUTRAL).generate(face_like_image)

        single = MoireRecaptureGenerator(stripes_only).generate(face_like_image)
        crossed = MoireRecaptureGenerator(both).generate(face_like_image)
        assert not np.array_equal(single, baseline)
        assert not np.array_equal(single, crossed)

    def test_contrast_reduction_flattens_the_histogram(self, face_like_image: RgbImage) -> None:
        config = dataclasses.replace(NEUTRAL, contrast=0.6)
        attack = MoireRecaptureGenerator(config).generate(face_like_image)
        assert attack.std() < face_like_image.std()

    def test_dynamic_range_compression_lifts_black_and_lowers_white(self) -> None:
        extremes = np.zeros((8, 8, 3), dtype=np.uint8)
        extremes[:, 4:] = 255
        config = dataclasses.replace(NEUTRAL, black_level=0.1, white_level=0.85)
        attack = MoireRecaptureGenerator(config).generate(extremes)
        assert attack.min() > 0
        assert attack.max() < 255

    def test_blur_reduces_local_gradients(self) -> None:
        edges = np.zeros((32, 32, 3), dtype=np.uint8)
        edges[:, ::2] = 255
        config = dataclasses.replace(NEUTRAL, blur_sigma_px=1.5)
        attack = MoireRecaptureGenerator(config).generate(edges)
        assert (
            np.abs(np.diff(attack.astype(np.float64), axis=1)).mean()
            < np.abs(np.diff(edges.astype(np.float64), axis=1)).mean()
        )

    def test_geometry_leaves_no_unmapped_border(self, face_like_image: RgbImage) -> None:
        config = dataclasses.replace(NEUTRAL, tilt_x=0.12, tilt_y=0.1, rotation_deg=3.0)
        attack = MoireRecaptureGenerator(config).generate(face_like_image)
        border = np.concatenate(
            [attack[0].ravel(), attack[-1].ravel(), attack[:, 0].ravel(), attack[:, -1].ravel()]
        )
        assert border.min() > 0


class TestConfig:
    @pytest.mark.parametrize(("override", "message"), MEANINGLESS_PARAMETERS)
    def test_rejects_meaningless_parameters(self, override: dict[str, Any], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            MoireConfig(**override)

    def test_describe_carries_every_parameter(self) -> None:
        config = MoireConfig(moire_cycles_per_width=123.0)
        described = MoireRecaptureGenerator(config).describe()
        assert described["generator"] == "moire"
        assert described["moire_cycles_per_width"] == 123.0
        assert set(described) == {"generator"} | {f.name for f in dataclasses.fields(MoireConfig)}


class TestFringeResolution:
    def test_warns_when_the_period_falls_below_nyquist(self) -> None:
        tiny = np.full((48, 48, 3), 128, dtype=np.uint8)
        config = dataclasses.replace(NEUTRAL, moire_amplitude=0.1, moire_cycles_per_width=60.0)
        with pytest.warns(RuntimeWarning, match="fringe period"):
            MoireRecaptureGenerator(config).generate(tiny)

    def test_stays_silent_at_the_default_on_dataset_sized_crops(
        self, face_like_image: RgbImage
    ) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            MoireRecaptureGenerator().generate(face_like_image)
