import dataclasses
import warnings
from typing import Any

import numpy as np
import pytest

from synthetic_pad.domain import RgbImage
from synthetic_pad.generators.moire import MoireConfig, MoireRecaptureGenerator, noise_level

NEUTRAL = MoireConfig(
    tilt_x=0.0,
    tilt_y=0.0,
    rotation_deg=0.0,
    contrast=1.0,
    saturation=1.0,
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
    {"saturation": 1.5},
    {"black_level": 0.1},
    {"white_level": 0.85},
    {"glare_intensity": 0.3},
    {"blur_sigma_px": 1.5},
    {"moire_amplitude": 0.15},
    {"noise_sigma_levels": 4.0},
    {"screen_min_pitch_px": 1.2, "screen_max_pitch_px": 1.6},
]

MEANINGLESS_PARAMETERS = [
    ({"black_level": 0.9, "white_level": 0.5}, "black_level < white_level"),
    ({"contrast": 0.0}, "contrast must be positive"),
    ({"moire_min_cycles_per_width": 0.0}, "moire_min_cycles_per_width must be positive"),
    (
        {"moire_min_cycles_per_width": 40.0, "moire_max_cycles_per_width": 20.0},
        "moire_min_cycles_per_width <= moire_max_cycles_per_width",
    ),
    ({"moire_max_angle_deg": -5.0}, "non-negative"),
    ({"saturation": -0.5}, "non-negative"),
    ({"moire_amplitude": -0.1}, "non-negative"),
    ({"noise_sigma_levels": -1.0}, "non-negative"),
    ({"noise_to_bona_fide": -1.0}, "non-negative"),
    (
        {"noise_sigma_levels": 2.0, "noise_to_bona_fide": 1.2},
        "either noise_sigma_levels or noise_to_bona_fide",
    ),
    ({"glare_intensity": 2.0}, r"\[0, 1\]"),
    ({"glare_extent": 0.0}, "must be positive"),
    ({"moire_cross_weight": -0.5}, "non-negative"),
    (
        {"moire_amplitude": 0.0, "screen_min_pitch_px": 2.0, "screen_max_pitch_px": 1.0},
        "screen_min_pitch_px <= screen_max_pitch_px",
    ),
    ({"moire_amplitude": 0.0, "screen_max_pitch_px": 1.0}, "screen_min_pitch_px > 0"),
    ({"screen_min_pitch_px": 1.0, "screen_max_pitch_px": 1.5}, "either by the screen"),
    ({"screen_black_matrix": 1.5}, r"\[0, 1\]"),
    ({"camera_px_per_output_px": 0.0}, "must be positive"),
]


def grey_with_noise(sigma_levels: float, size: int = 224) -> RgbImage:
    noise = np.random.default_rng(20261004).normal(0.0, sigma_levels, (size, size, 1))
    return np.clip(np.rint(128.0 + noise), 0, 255).repeat(3, axis=2).astype(np.uint8)


def at_fixed_frequency(config: MoireConfig, cycles_per_width: float) -> MoireConfig:
    return dataclasses.replace(
        config,
        moire_min_cycles_per_width=cycles_per_width,
        moire_max_cycles_per_width=cycles_per_width,
    )


def upright_fringe_frequency(image: RgbImage) -> int:
    row = image[0].astype(np.float64).mean(axis=1)
    spectrum = np.abs(np.fft.rfft(row - row.mean()))
    return int(np.argmax(spectrum))


def fringe_handedness(image: RgbImage) -> float:
    gradient_y, gradient_x = np.gradient(image.astype(np.float64).mean(axis=2))
    return float(np.mean(gradient_x * gradient_y))


def variants_of(image: RgbImage, count: int) -> list[RgbImage]:
    seeded = []
    for index in range(count):
        variant = image.copy()
        variant[0, 0] = index % 256
        seeded.append(variant)
    return seeded


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

    def test_the_neutral_config_stays_the_identity_with_variation(
        self, face_like_image: RgbImage
    ) -> None:
        config = dataclasses.replace(NEUTRAL, variation=0.25)
        attack = MoireRecaptureGenerator(config).generate(face_like_image)
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
        config = at_fixed_frequency(dataclasses.replace(NEUTRAL, moire_amplitude=0.2), 60.0)
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


class TestOrientationSymmetry:
    def test_moire_leans_both_ways_across_images(self, face_like_image: RgbImage) -> None:
        config = dataclasses.replace(NEUTRAL, moire_amplitude=0.25, variation=0.25)
        generator = MoireRecaptureGenerator(config)
        leans = {
            fringe_handedness(generator.generate(variant)) > 0.0
            for variant in variants_of(np.full_like(face_like_image, 128), 40)
        }
        assert leans == {True, False}

    def test_rotation_turns_both_ways_across_images(self) -> None:
        ramp = np.repeat(np.linspace(0, 255, 96, dtype=np.uint8)[None, :, None], 96, axis=0).repeat(
            3, axis=2
        )
        config = dataclasses.replace(NEUTRAL, rotation_deg=6.0, variation=0.25)
        generator = MoireRecaptureGenerator(config)
        turns = {
            float(
                np.gradient(generator.generate(variant).astype(np.float64).mean(axis=2))[0].mean()
            )
            > 0.0
            for variant in variants_of(ramp, 40)
        }
        assert turns == {True, False}

    def test_a_zero_angle_spread_keeps_the_fringes_upright(self, face_like_image: RgbImage) -> None:
        config = dataclasses.replace(
            NEUTRAL, moire_amplitude=0.25, moire_cross_weight=0.0, moire_max_angle_deg=0.0
        )
        generator = MoireRecaptureGenerator(config)
        for variant in variants_of(np.full_like(face_like_image, 128), 10):
            attack = generator.generate(variant)
            assert np.array_equal(attack[1], attack[-1])


class TestFringeFrequency:
    SPREAD = dataclasses.replace(
        NEUTRAL,
        moire_amplitude=0.25,
        moire_cross_weight=0.0,
        moire_max_angle_deg=0.0,
        moire_min_cycles_per_width=8.0,
        moire_max_cycles_per_width=64.0,
    )

    def frequencies(self, config: MoireConfig, count: int) -> list[int]:
        flat = np.full((224, 224, 3), 128, dtype=np.uint8)
        generator = MoireRecaptureGenerator(config)
        return [upright_fringe_frequency(generator.generate(v)) for v in variants_of(flat, count)]

    def test_stays_inside_the_configured_range(self) -> None:
        assert all(8 <= cycles <= 64 for cycles in self.frequencies(self.SPREAD, 30))

    def test_spreads_evenly_on_a_logarithmic_scale(self) -> None:
        geometric_middle = (8.0 * 64.0) ** 0.5
        below = sum(cycles < geometric_middle for cycles in self.frequencies(self.SPREAD, 60))
        assert 18 <= below <= 42

    def test_a_range_of_one_value_fixes_it(self) -> None:
        fixed = at_fixed_frequency(self.SPREAD, 20.0)
        assert set(self.frequencies(fixed, 10)) == {20}


class TestNoise:
    def test_leaves_a_grey_image_grey(self) -> None:
        config = dataclasses.replace(NEUTRAL, noise_sigma_levels=4.0)
        attack = MoireRecaptureGenerator(config).generate(np.full((32, 32, 3), 128, np.uint8))
        assert not np.all(attack == 128)
        assert np.array_equal(attack[..., 0], attack[..., 1])
        assert np.array_equal(attack[..., 1], attack[..., 2])


class TestScreen:
    GREY = np.full((224, 224, 3), 150, dtype=np.uint8)

    def photographed(self, image: RgbImage, pitch: float) -> np.ndarray:
        config = dataclasses.replace(NEUTRAL, screen_min_pitch_px=pitch, screen_max_pitch_px=pitch)
        return MoireRecaptureGenerator(config).generate(image).astype(np.float64)

    def test_keeps_the_mean_colour(self) -> None:
        tinted = np.zeros_like(self.GREY)
        tinted[...] = (180, 90, 40)
        means = self.photographed(tinted, 1.2).reshape(-1, 3).mean(axis=0)
        assert np.allclose(means, (180, 90, 40), atol=3.0)

    def test_turns_a_uniform_grey_into_colour_fringes(self) -> None:
        attack = self.photographed(self.GREY, 1.7)
        assert (attack[..., 0] - attack[..., 1]).std() > 5.0

    def test_fringes_fade_as_the_screen_pixels_shrink_below_the_sensor_pixels(self) -> None:
        coarse = (self.photographed(self.GREY, 1.7)[..., 0]).std()
        fine = (self.photographed(self.GREY, 0.8)[..., 0]).std()
        assert fine < coarse / 3.0


class TestNoiseMatchedToBonaFide:
    def matched(self, ratio: float) -> MoireRecaptureGenerator:
        return MoireRecaptureGenerator(dataclasses.replace(NEUTRAL, noise_to_bona_fide=ratio))

    def test_brings_the_noise_of_the_attack_to_the_ratio_of_the_bona_fide(self) -> None:
        bona_fide = grey_with_noise(3.0)
        attack = self.matched(1.5).generate(bona_fide)
        assert noise_level(attack) == pytest.approx(1.5 * noise_level(bona_fide), rel=0.1)

    def test_adds_nothing_when_the_attack_is_already_noisy_enough(self) -> None:
        bona_fide = grey_with_noise(3.0)
        assert np.array_equal(self.matched(0.5).generate(bona_fide), bona_fide)

    def test_applies_to_real_crops_through_the_sensor_noise_alone(self) -> None:
        crop = grey_with_noise(3.0)
        noisy = self.matched(2.0).add_sensor_noise(crop)
        assert noise_level(noisy) == pytest.approx(2.0 * noise_level(crop), rel=0.1)


class TestNoiseLevel:
    def test_a_flat_image_has_no_noise(self) -> None:
        assert noise_level(np.full((32, 32, 3), 90, dtype=np.uint8)) == 0.0

    def test_ignores_a_linear_ramp(self) -> None:
        ramp = np.repeat(np.linspace(20.0, 220.0, 224)[None, :, None], 224, axis=0)
        image = np.repeat(ramp, 3, axis=2).astype(np.uint8)
        assert noise_level(image) < 0.5

    def test_recovers_the_deviation_of_added_grey_noise(self) -> None:
        assert noise_level(grey_with_noise(5.0)) == pytest.approx(5.0, rel=0.05)


class TestSensorNoise:
    GREY = np.full((64, 64, 3), 128, dtype=np.uint8)

    def noisy(self, sigma_levels: float, image: RgbImage | None = None) -> RgbImage:
        config = dataclasses.replace(MoireConfig(), noise_sigma_levels=sigma_levels)
        return MoireRecaptureGenerator(config).add_sensor_noise(
            self.GREY if image is None else image
        )

    def test_spreads_the_levels_by_the_configured_sigma(self) -> None:
        deviation = self.noisy(4.0).astype(np.float64) - 128.0
        assert abs(deviation.mean()) < 0.5
        assert 3.5 < deviation.std() < 4.5

    def test_applies_no_other_stage(self) -> None:
        assert np.array_equal(self.noisy(0.0), self.GREY)

    def test_adds_the_same_noise_to_every_channel(self) -> None:
        noisy = self.noisy(4.0)
        assert np.array_equal(noisy[..., 0], noisy[..., 1])
        assert np.array_equal(noisy[..., 1], noisy[..., 2])

    def test_repeats_itself_on_the_same_image(self) -> None:
        assert np.array_equal(self.noisy(4.0), self.noisy(4.0))

    def test_draws_new_noise_for_a_different_image(self) -> None:
        darker = np.full((64, 64, 3), 100, dtype=np.uint8)
        assert not np.array_equal(self.noisy(4.0) - 128, self.noisy(4.0, darker) - 100)


class TestConfig:
    @pytest.mark.parametrize(("override", "message"), MEANINGLESS_PARAMETERS)
    def test_rejects_meaningless_parameters(self, override: dict[str, Any], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            MoireConfig(**override)

    def test_describe_carries_every_parameter(self) -> None:
        config = MoireConfig(saturation=1.23)
        described = MoireRecaptureGenerator(config).describe()
        assert described["generator"] == "moire"
        assert described["saturation"] == 1.23
        assert set(described) == {"generator"} | {f.name for f in dataclasses.fields(MoireConfig)}


class TestFringeResolution:
    def test_warns_when_the_period_falls_below_nyquist(self) -> None:
        tiny = np.full((48, 48, 3), 128, dtype=np.uint8)
        config = at_fixed_frequency(dataclasses.replace(NEUTRAL, moire_amplitude=0.1), 60.0)
        with pytest.warns(RuntimeWarning, match="fringe period"):
            MoireRecaptureGenerator(config).generate(tiny)

    def test_stays_silent_at_the_default_on_dataset_sized_crops(
        self, face_like_image: RgbImage
    ) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            MoireRecaptureGenerator().generate(face_like_image)
