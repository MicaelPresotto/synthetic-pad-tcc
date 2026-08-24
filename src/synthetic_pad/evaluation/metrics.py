"""PAD metrics as defined in ISO/IEC 30107-3.

Scores run higher for attacks, and a sample counts as an attack at or above the
threshold.
"""

from typing import Any

import numpy as np

type FloatArray = np.ndarray[Any, np.dtype[np.floating[Any]]]


def apcer(attack_scores: FloatArray, threshold: float) -> float:
    """Fraction of attacks that were accepted as genuine."""
    raise NotImplementedError


def bpcer(bonafide_scores: FloatArray, threshold: float) -> float:
    """Fraction of bona fide samples that were rejected."""
    raise NotImplementedError


def acer(attack_scores: FloatArray, bonafide_scores: FloatArray, threshold: float) -> float:
    """Mean of APCER and BPCER at a given threshold."""
    raise NotImplementedError


def eer(attack_scores: FloatArray, bonafide_scores: FloatArray) -> float:
    """Error rate at the threshold where APCER equals BPCER."""
    raise NotImplementedError
