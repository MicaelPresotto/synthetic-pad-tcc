"""The pulse read from the color of the skin, remote photoplethysmography (rPPG), by the
plane-orthogonal-to-skin (POS) method of Wang et al. (2017), and an attack score per video from
the share of the pulse band its strongest peak holds."""

import math

import numpy as np

WINDOW_SECONDS = 9.0
LOWEST_PULSE_HZ = 0.7
HIGHEST_PULSE_HZ = 4.0
POS_WINDOW_SECONDS = 1.6
PEAK_HALF_WIDTH = 1
PULSE_FREQUENCIES = (
    np.arange(
        math.ceil(LOWEST_PULSE_HZ * WINDOW_SECONDS),
        math.floor(HIGHEST_PULSE_HZ * WINDOW_SECONDS) + 1,
    )
    / WINDOW_SECONDS
)
SKIN_PLANE = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])


def pos_signal(colors: np.ndarray, fps: float) -> np.ndarray:
    """Pulse signal from the mean skin RGB of each frame, an (N, 3) array."""
    window = math.ceil(POS_WINDOW_SECONDS * fps)
    pulse = np.zeros(len(colors))
    for start in range(len(colors) - window + 1):
        segment = colors[start : start + window]
        first, second = SKIN_PLANE @ (segment / segment.mean(axis=0)).T
        combined = first + first.std() / second.std() * second
        pulse[start : start + window] += combined - combined.mean()
    return pulse


def band_power(signal: np.ndarray, fps: float) -> np.ndarray:
    """Power at each of PULSE_FREQUENCIES, which are spaced for a WINDOW_SECONDS-long signal."""
    tapered = (signal - signal.mean()) * np.hanning(len(signal))
    times = np.arange(len(signal)) / fps
    return np.abs(np.exp(-2j * np.pi * np.outer(PULSE_FREQUENCIES, times)) @ tapered) ** 2


def pulse_score(power: np.ndarray) -> float:
    """One minus the share of the band power at the strongest peak and beside it: low on a pulse."""
    peak = int(power.argmax())
    around_peak = power[max(peak - PEAK_HALF_WIDTH, 0) : peak + PEAK_HALF_WIDTH + 1]
    return float(1 - around_peak.sum() / power.sum())
