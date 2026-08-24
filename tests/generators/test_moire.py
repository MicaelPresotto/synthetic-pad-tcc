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
    blur_sigma=0.0,
    moire_amplitude=0.0,
    noise_sigma=0.0,
    variation=0.0,
)


def high_frequency_energy(image: RgbImage) -> float:
    grey = image.astype(np.float64).mean(axis=2)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(grey)))
    rows, cols = grey.shape
    centre_r, centre_c = rows // 2, cols // 2
    keep = min(rows, cols) // 4
    spectrum[centre_r - keep : centre_r + keep, centre_c - keep : centre_c + keep] = 0.0
    return float(spectrum.mean())


def test_output_preserves_shape_and_dtype(face_like_image: RgbImage) -> None:
    attack = MoireRecaptureGenerator().generate(face_like_image)
    assert attack.shape == face_like_image.shape
    assert attack.dtype == np.uint8


def test_output_stays_within_the_uint8_range(face_like_image: RgbImage) -> None:
    config = dataclasses.replace(MoireConfig(), moire_amplitude=0.9, glare_intensity=1.0)
    attack = MoireRecaptureGenerator(config).generate(face_like_image)
    assert attack.min() >= 0
    assert attack.max() <= 255


def test_generation_is_deterministic(face_like_image: RgbImage) -> None:
    first = MoireRecaptureGenerator().generate(face_like_image)
    second = MoireRecaptureGenerator().generate(face_like_image)
    np.testing.assert_array_equal(first, second)


def test_seed_changes_the_result(face_like_image: RgbImage) -> None:
    default = MoireRecaptureGenerator().generate(face_like_image)
    reseeded = MoireRecaptureGenerator(MoireConfig(seed=7)).generate(face_like_image)
    assert not np.array_equal(default, reseeded)


def test_different_images_get_different_patterns(face_like_image: RgbImage) -> None:
    """Samples must be decorrelated by the content digest, otherwise a detector could
    memorise one fixed moiré pattern instead of learning the artifact.
    """
    generator = MoireRecaptureGenerator(dataclasses.replace(NEUTRAL, moire_amplitude=0.25))
    flat = np.full_like(face_like_image, 128)
    shifted = flat.copy()
    shifted[0, 0, 0] = 129

    first = generator.generate(flat).astype(np.int16)
    second = generator.generate(shifted).astype(np.int16)
    assert np.abs(first - second).mean() > 1.0


def test_neutral_config_is_the_identity(face_like_image: RgbImage) -> None:
    """Every stage must be individually switchable, so ablations are possible."""
    attack = MoireRecaptureGenerator(NEUTRAL).generate(face_like_image)
    np.testing.assert_array_equal(attack, face_like_image)


@pytest.mark.parametrize(
    "override",
    [
        {"tilt_x": 0.08},
        {"tilt_y": 0.08},
        {"rotation_deg": 2.0},
        {"contrast": 0.7},
        {"black_level": 0.1},
        {"white_level": 0.85},
        {"glare_intensity": 0.3},
        {"blur_sigma": 1.5},
        {"moire_amplitude": 0.15},
        {"noise_sigma": 4.0},
    ],
)
def test_each_stage_changes_the_image(face_like_image: RgbImage, override: dict[str, Any]) -> None:
    baseline = MoireRecaptureGenerator(NEUTRAL).generate(face_like_image)
    config = dataclasses.replace(NEUTRAL, **override)
    assert not np.array_equal(MoireRecaptureGenerator(config).generate(face_like_image), baseline)


def test_moire_adds_high_frequency_energy(face_like_image: RgbImage) -> None:
    without = MoireRecaptureGenerator(NEUTRAL).generate(face_like_image)
    with_moire = dataclasses.replace(NEUTRAL, moire_amplitude=0.2, moire_frequency=60.0)
    assert high_frequency_energy(
        MoireRecaptureGenerator(with_moire).generate(face_like_image)
    ) > high_frequency_energy(without)


def test_moire_is_chromatic(face_like_image: RgbImage) -> None:
    """A real recapture aliases per subpixel, so the fringes must be coloured."""
    config = dataclasses.replace(NEUTRAL, moire_amplitude=0.25, moire_chroma_shift=1.2)
    attack = MoireRecaptureGenerator(config).generate(face_like_image).astype(np.float64)
    baseline = MoireRecaptureGenerator(NEUTRAL).generate(face_like_image).astype(np.float64)
    assert attack.std(axis=2).mean() > baseline.std(axis=2).mean()


def test_contrast_reduction_flattens_the_histogram(face_like_image: RgbImage) -> None:
    config = dataclasses.replace(NEUTRAL, contrast=0.6)
    attack = MoireRecaptureGenerator(config).generate(face_like_image)
    assert attack.std() < face_like_image.std()


def test_dynamic_range_compression_lifts_black_and_lowers_white() -> None:
    extremes = np.zeros((8, 8, 3), dtype=np.uint8)
    extremes[:, 4:] = 255
    config = dataclasses.replace(NEUTRAL, black_level=0.1, white_level=0.85)
    attack = MoireRecaptureGenerator(config).generate(extremes)
    assert attack.min() > 0
    assert attack.max() < 255


def test_blur_reduces_local_gradients() -> None:
    edges = np.zeros((32, 32, 3), dtype=np.uint8)
    edges[:, ::2] = 255
    config = dataclasses.replace(NEUTRAL, blur_sigma=1.5)
    attack = MoireRecaptureGenerator(config).generate(edges)
    assert (
        np.abs(np.diff(attack.astype(np.float64), axis=1)).mean()
        < np.abs(np.diff(edges.astype(np.float64), axis=1)).mean()
    )


def test_geometry_leaves_no_unmapped_border(face_like_image: RgbImage) -> None:
    """A black rim would be a cue in itself, rather than a property of the recapture
    the detector is supposed to pick up on.
    """
    config = dataclasses.replace(NEUTRAL, tilt_x=0.12, tilt_y=0.1, rotation_deg=3.0)
    attack = MoireRecaptureGenerator(config).generate(face_like_image)
    border = np.concatenate(
        [attack[0].ravel(), attack[-1].ravel(), attack[:, 0].ravel(), attack[:, -1].ravel()]
    )
    assert border.min() > 0


def test_describe_carries_every_parameter() -> None:
    config = MoireConfig(moire_frequency=123.0)
    described = MoireRecaptureGenerator(config).describe()
    assert described["generator"] == "moire"
    assert described["moire_frequency"] == 123.0
    assert set(described) == {"generator"} | {f.name for f in dataclasses.fields(MoireConfig)}


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"black_level": 0.9, "white_level": 0.5}, "black_level < white_level"),
        ({"contrast": 0.0}, "contrast must be positive"),
        ({"moire_harmonics": 0}, "at least 1"),
        ({"moire_frequency": 0.0}, "moire_frequency must be positive"),
        ({"moire_amplitude": -0.1}, "non-negative"),
        ({"noise_sigma": -1.0}, "non-negative"),
        ({"glare_intensity": 2.0}, r"\[0, 1\]"),
        ({"glare_extent": 0.0}, "must be positive"),
        ({"moire_cross_weight": -0.5}, "non-negative"),
    ],
)
def test_config_rejects_meaningless_parameters(override: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        MoireConfig(**override)


def test_warns_when_the_fringe_period_falls_below_nyquist() -> None:
    """A crop too small for the requested frequency must not fail silently."""
    tiny = np.full((48, 48, 3), 128, dtype=np.uint8)
    config = dataclasses.replace(NEUTRAL, moire_amplitude=0.1, moire_frequency=60.0)
    with pytest.warns(RuntimeWarning, match="fringe period"):
        MoireRecaptureGenerator(config).generate(tiny)


def test_no_warning_at_the_default_frequency_on_dataset_sized_crops(
    face_like_image: RgbImage,
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        MoireRecaptureGenerator().generate(face_like_image)


def test_cross_weight_zero_leaves_a_single_grating(face_like_image: RgbImage) -> None:
    """The screen grid is two dimensional, so the cross grating is on by default, but
    switching it off must still produce plain stripes rather than no moire at all.
    """
    stripes_only = dataclasses.replace(NEUTRAL, moire_amplitude=0.2, moire_cross_weight=0.0)
    both = dataclasses.replace(stripes_only, moire_cross_weight=0.45)
    baseline = MoireRecaptureGenerator(NEUTRAL).generate(face_like_image)

    single = MoireRecaptureGenerator(stripes_only).generate(face_like_image)
    crossed = MoireRecaptureGenerator(both).generate(face_like_image)
    assert not np.array_equal(single, baseline)
    assert not np.array_equal(single, crossed)
