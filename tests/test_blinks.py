import numpy as np
import pytest

from synthetic_pad.blinks import blink_score, eye_aspect_ratio, frames_below_half


def eyes(gap: float) -> np.ndarray:
    """68 landmarks whose two eyes are one unit wide, both eyelid pairs `gap` apart."""
    landmarks = np.zeros((68, 3))
    for first, left in ((36, 0.0), (42, 3.0)):
        landmarks[first : first + 6, :2] = [
            (left, 0.0),
            (left + 0.3, gap / 2),
            (left + 0.7, gap / 2),
            (left + 1.0, 0.0),
            (left + 0.7, -gap / 2),
            (left + 0.3, -gap / 2),
        ]
    return landmarks


class TestEyeAspectRatio:
    def test_is_the_eyelid_gap_over_the_eye_width(self) -> None:
        assert eye_aspect_ratio(eyes(0.3)) == pytest.approx(0.3)

    def test_is_zero_with_the_eyes_shut(self) -> None:
        assert eye_aspect_ratio(eyes(0.0)) == pytest.approx(0.0)


class TestBlinkScore:
    def test_is_one_when_the_eyes_never_close(self) -> None:
        assert blink_score(np.full(30, 0.3)) == pytest.approx(1.0)

    def test_is_the_deepest_drop_over_the_median(self) -> None:
        ratios = np.array([0.3, 0.3, 0.06, 0.3, 0.3])
        assert blink_score(ratios) == pytest.approx(0.2)


class TestFramesBelowHalf:
    def test_counts_the_frames_of_a_blink(self) -> None:
        ratios = np.array([0.3, 0.3, 0.1, 0.12, 0.3, 0.3])
        assert frames_below_half(ratios) == 2

    def test_is_zero_without_a_blink(self) -> None:
        assert frames_below_half(np.full(30, 0.3)) == 0
