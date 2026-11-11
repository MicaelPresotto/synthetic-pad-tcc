"""PAD metrics as defined in ISO/IEC 30107-3, and the general classification metrics.

Scores run higher for attacks, and a sample counts as an attack at or above the
threshold.
"""

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

type FloatArray = np.ndarray[Any, np.dtype[np.floating[Any]]]


def apcer(attack_scores: FloatArray, threshold: float) -> float:
    """Fraction of attacks that were accepted as genuine."""
    _require_scores(attack_scores=attack_scores)
    return float(np.mean(attack_scores < threshold))


def bpcer(bonafide_scores: FloatArray, threshold: float) -> float:
    """Fraction of bona fide samples that were rejected."""
    _require_scores(bonafide_scores=bonafide_scores)
    return float(np.mean(bonafide_scores >= threshold))


def acer(attack_scores: FloatArray, bonafide_scores: FloatArray, threshold: float) -> float:
    """Mean of APCER and BPCER at a given threshold."""
    return (apcer(attack_scores, threshold) + bpcer(bonafide_scores, threshold)) / 2.0


def eer(attack_scores: FloatArray, bonafide_scores: FloatArray) -> float:
    """Error rate at the threshold where APCER equals BPCER."""
    _, apcer_at_each, bpcer_at_each = _operating_curve(attack_scores, bonafide_scores)
    crossing = _crossing(apcer_at_each, bpcer_at_each)
    return float((apcer_at_each[crossing] + bpcer_at_each[crossing]) / 2.0)


def eer_threshold(attack_scores: FloatArray, bonafide_scores: FloatArray) -> float:
    """Threshold at which APCER and BPCER meet.

    When a whole interval of thresholds meets equally well, as it does whenever the two
    classes separate perfectly, the midpoint of that interval is taken: the widest margin.
    """
    thresholds, apcer_at_each, bpcer_at_each = _operating_curve(attack_scores, bonafide_scores)
    gap = np.abs(apcer_at_each - bpcer_at_each)
    tied = np.flatnonzero(gap == gap.min())
    highest, lowest = tied[0], tied[-1]
    if lowest + 1 == thresholds.size:
        return float(thresholds[highest])
    return float((thresholds[highest] + thresholds[lowest + 1]) / 2.0)


def bpcer_threshold(bonafide_scores: FloatArray, target_bpcer: float) -> float:
    """Lowest threshold whose BPCER stays at or below the target, a fraction in [0, 1].

    Set from bona fide scores alone, so it does not depend on which attacks a development
    set happens to hold. It sits midway to the next lower score, as eer_threshold does.
    """
    _require_scores(bonafide_scores=bonafide_scores)
    if not 0.0 <= target_bpcer <= 1.0:
        raise ValueError(f"target_bpcer must lie in [0, 1], got {target_bpcer}")
    observed = np.unique(bonafide_scores)
    within = np.flatnonzero(
        [bpcer(bonafide_scores, threshold) <= target_bpcer for threshold in observed]
    )
    if within.size == 0:
        return float(np.inf)
    lowest = within[0]
    if lowest == 0:
        return float(observed[0])
    return float((observed[lowest - 1] + observed[lowest]) / 2.0)


def hter(
    development_attack_scores: FloatArray,
    development_bonafide_scores: FloatArray,
    attack_scores: FloatArray,
    bonafide_scores: FloatArray,
) -> float:
    """Mean of APCER and BPCER, at the threshold a development set fixed beforehand.

    The cross-dataset measure: the threshold never sees the evaluation set.
    """
    threshold = eer_threshold(development_attack_scores, development_bonafide_scores)
    return acer(attack_scores, bonafide_scores, threshold)


def auc(attack_scores: FloatArray, bonafide_scores: FloatArray) -> float:
    """Area under the ROC curve, which no choice of threshold influences."""
    _, apcer_at_each, bpcer_at_each = _operating_curve(attack_scores, bonafide_scores)
    return float(np.trapezoid(1.0 - apcer_at_each, bpcer_at_each))


def classification_metrics(
    attack_scores: FloatArray, bonafide_scores: FloatArray, threshold: float
) -> dict[str, float]:
    """Accuracy, precision, recall and F1, with the attack as the positive class."""
    _require_scores(attack_scores=attack_scores, bonafide_scores=bonafide_scores)
    is_attack = np.concatenate([np.ones(attack_scores.size), np.zeros(bonafide_scores.size)])
    flagged = np.concatenate([attack_scores, bonafide_scores]) >= threshold
    return {
        "accuracy": float(accuracy_score(is_attack, flagged)),
        "precision": float(precision_score(is_attack, flagged, zero_division=0.0)),
        "recall": float(recall_score(is_attack, flagged, zero_division=0.0)),
        "f1": float(f1_score(is_attack, flagged, zero_division=0.0)),
    }


def _operating_curve(
    attack_scores: FloatArray, bonafide_scores: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    thresholds = _candidate_thresholds(attack_scores, bonafide_scores)
    apcer_at_each = np.array([apcer(attack_scores, threshold) for threshold in thresholds])
    bpcer_at_each = np.array([bpcer(bonafide_scores, threshold) for threshold in thresholds])
    return thresholds, apcer_at_each, bpcer_at_each


def _crossing(apcer_at_each: FloatArray, bpcer_at_each: FloatArray) -> int:
    return int(np.argmin(np.abs(apcer_at_each - bpcer_at_each)))


def _candidate_thresholds(attack_scores: FloatArray, bonafide_scores: FloatArray) -> FloatArray:
    _require_scores(attack_scores=attack_scores, bonafide_scores=bonafide_scores)
    observed = np.unique(np.concatenate([attack_scores, bonafide_scores]))
    rejects_nothing = np.array([np.inf])
    return np.concatenate([rejects_nothing, observed[::-1]])


def _require_scores(**fields: FloatArray) -> None:
    for name, scores in fields.items():
        if scores.size == 0:
            raise ValueError(f"{name} must hold at least one score, got an empty array")
