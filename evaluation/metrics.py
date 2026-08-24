import numpy as np


def apcer(scores: np.ndarray, threshold: float) -> float:
    """Fraction of attacks that were accepted as genuine."""
    raise NotImplementedError


def bpcer(scores: np.ndarray, threshold: float) -> float:
    """Fraction of bona fide samples that were rejected."""
    raise NotImplementedError


def acer(attack_scores: np.ndarray, bonafide_scores: np.ndarray, threshold: float) -> float:
    """Mean of APCER and BPCER at a given threshold."""
    raise NotImplementedError


def eer(attack_scores: np.ndarray, bonafide_scores: np.ndarray) -> float:
    """Error rate at the threshold where APCER equals BPCER."""
    raise NotImplementedError
