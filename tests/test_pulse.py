import numpy as np
import pytest

from synthetic_pad.pulse import (
    PULSE_FREQUENCIES,
    WINDOW_SECONDS,
    band_power,
    pos_signal,
    pulse_score,
)

FPS = 30.0
TIMES = np.arange(round(WINDOW_SECONDS * FPS)) / FPS
HEART_RATE_HZ = 1.2


def wave(frequency: float) -> np.ndarray:
    return np.sin(2 * np.pi * frequency * TIMES)


class TestPulseScore:
    def test_is_low_when_one_frequency_holds_the_band(self) -> None:
        assert pulse_score(band_power(wave(HEART_RATE_HZ), FPS)) < 0.2

    def test_is_high_for_noise(self) -> None:
        noise = np.random.default_rng(20261031).normal(size=TIMES.size)
        assert pulse_score(band_power(noise, FPS)) > 0.5


class TestPosSignal:
    def test_keeps_the_pulse_and_drops_a_change_of_brightness(self) -> None:
        brightness = 1 + 0.05 * wave(2.5)
        colors = np.outer(brightness, [180.0, 130.0, 110.0])
        colors[:, 1] += 0.3 * wave(HEART_RATE_HZ)
        power = band_power(pos_signal(colors, FPS), FPS)
        assert PULSE_FREQUENCIES[power.argmax()] == pytest.approx(
            HEART_RATE_HZ, abs=1 / WINDOW_SECONDS
        )
