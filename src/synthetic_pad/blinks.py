"""Eye blinks as a liveness cue: the eye aspect ratio (EAR) of Soukupová and Čech (2016), from
the 68 face landmarks of the iBUG layout, and an attack score per video from its deepest drop."""

import numpy as np

LEFT_EYE = slice(36, 42)
RIGHT_EYE = slice(42, 48)
HALF_OF_MEDIAN = 0.5


def eye_aspect_ratio(landmarks: np.ndarray) -> float:
    """Mean over both eyes of the two eyelid gaps over twice the eye width; near 0 when shut."""
    ratios = []
    for eye in (landmarks[LEFT_EYE, :2], landmarks[RIGHT_EYE, :2]):
        outer_corner, upper_outer, upper_inner, inner_corner, lower_inner, lower_outer = eye
        eyelid_gaps = np.linalg.norm(upper_outer - lower_outer) + np.linalg.norm(
            upper_inner - lower_inner
        )
        ratios.append(eyelid_gaps / (2 * np.linalg.norm(outer_corner - inner_corner)))
    return float(np.mean(ratios))


def blink_score(ratios: np.ndarray) -> float:
    """Lowest EAR of a video over its median: near 1 when the eyes never close, an attack."""
    return float(ratios.min() / np.median(ratios))


def frames_below_half(ratios: np.ndarray) -> int:
    """Frames whose EAR is below half the median: a blink spans a few, a stray landmark one."""
    return int(np.sum(ratios < HALF_OF_MEDIAN * np.median(ratios)))
