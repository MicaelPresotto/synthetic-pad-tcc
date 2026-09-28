from collections.abc import Mapping
from typing import Any

import numpy as np

from synthetic_pad.evaluation.metrics import FloatArray, apcer, auc, bpcer, eer, eer_threshold, hter


def pad_report(
    development_attack_scores: FloatArray,
    development_bonafide_scores: FloatArray,
    attack_scores_by_species: Mapping[str, FloatArray],
    bonafide_scores: FloatArray,
) -> dict[str, Any]:
    """Evaluation metrics at the threshold the development set fixes, overall and per species."""
    threshold = eer_threshold(development_attack_scores, development_bonafide_scores)
    every_attack = np.concatenate(list(attack_scores_by_species.values()))

    def at_threshold(attack_scores: FloatArray) -> dict[str, float]:
        return {
            "apcer": apcer(attack_scores, threshold),
            "hter": hter(
                development_attack_scores,
                development_bonafide_scores,
                attack_scores,
                bonafide_scores,
            ),
            "auc": auc(attack_scores, bonafide_scores),
        }

    return {
        "threshold": threshold,
        "development_eer": eer(development_attack_scores, development_bonafide_scores),
        "bpcer": bpcer(bonafide_scores, threshold),
        "all_attacks": at_threshold(every_attack),
        "by_species": {
            species: at_threshold(scores) for species, scores in attack_scores_by_species.items()
        },
    }
