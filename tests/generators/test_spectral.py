import numpy as np
import pytest

from synthetic_pad.domain import RgbImage
from synthetic_pad.generators.spectral import SpectralConfig, SpectralTransferGenerator

SIZE = 64


def uniform(colour: tuple[int, int, int]) -> RgbImage:
    image = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    image[...] = colour
    return image


def textured(seed: int) -> RgbImage:
    rng = np.random.default_rng(seed)
    return rng.integers(60, 200, (SIZE, SIZE, 3), dtype=np.uint8)


def correlation(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.corrcoef(first.ravel(), second.ravel())[0, 1])


class TestTransfer:
    def test_returns_an_rgb_image_of_the_same_size(self) -> None:
        attack = SpectralTransferGenerator(uniform((120, 120, 120))[None]).generate(textured(0))
        assert attack.shape == (SIZE, SIZE, 3)
        assert attack.dtype == np.uint8

    def test_takes_the_mean_colour_of_the_replay(self) -> None:
        attack = SpectralTransferGenerator(uniform((160, 120, 100))[None]).generate(textured(0))
        assert np.allclose(attack.reshape(-1, 3).mean(axis=0), (160, 120, 100), atol=1.0)

    def test_keeps_the_layout_of_the_face(self) -> None:
        face = textured(0)
        generator = SpectralTransferGenerator(
            uniform((150, 150, 150))[None], SpectralConfig(band_cycles_per_px=0.02)
        )
        attack = generator.generate(face).astype(np.float64)
        assert correlation(attack, face.astype(np.float64)) > 0.9

    def test_a_wider_band_takes_more_of_the_replay(self) -> None:
        face, replay = textured(0), textured(1)
        attacks = [
            SpectralTransferGenerator(replay[None], SpectralConfig(band_cycles_per_px=band))
            .generate(face)
            .astype(np.float64)
            for band in (0.02, 0.3)
        ]
        face_float = face.astype(np.float64)
        assert correlation(attacks[1], face_float) < correlation(attacks[0], face_float)

    def test_repeats_itself_on_the_same_image(self) -> None:
        generator = SpectralTransferGenerator(np.stack([textured(1), textured(2)]))
        assert np.array_equal(generator.generate(textured(0)), generator.generate(textured(0)))

    def test_draws_its_replay_by_the_content_of_the_image(self) -> None:
        replays = np.stack([uniform((40, 40, 40)), uniform((210, 210, 210))])
        generator = SpectralTransferGenerator(replays)
        means = {round(float(generator.generate(textured(seed)).mean())) for seed in range(20)}
        assert len(means) > 1


class TestContract:
    @pytest.mark.parametrize("band", [0.0, -0.1, 0.6])
    def test_rejects_a_band_outside_the_spectrum(self, band: float) -> None:
        with pytest.raises(ValueError, match="band_cycles_per_px"):
            SpectralConfig(band_cycles_per_px=band)

    def test_needs_at_least_one_replay(self) -> None:
        with pytest.raises(ValueError, match="at least one real replay"):
            SpectralTransferGenerator(np.zeros((0, SIZE, SIZE, 3), dtype=np.uint8))

    def test_rejects_a_replay_of_another_size(self) -> None:
        small = np.zeros((1, SIZE // 2, SIZE // 2, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="does not match"):
            SpectralTransferGenerator(small).generate(textured(0))

    def test_describe_carries_the_settings_and_the_number_of_replays(self) -> None:
        described = SpectralTransferGenerator(np.stack([textured(1), textured(2)])).describe()
        assert described == {
            "generator": "spectral",
            "band_cycles_per_px": SpectralConfig().band_cycles_per_px,
            "seed": SpectralConfig().seed,
            "replays": 2,
        }
