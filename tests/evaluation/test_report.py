import numpy as np

from synthetic_pad.evaluation.report import pad_report


def scores(*values: float) -> np.ndarray:
    return np.array(values, dtype=np.float64)


class TestPadReport:
    def test_fixes_the_threshold_on_the_development_set(self) -> None:
        report = pad_report(
            scores(0.6, 0.7), scores(0.1, 0.2), {"replay": scores(0.9)}, scores(0.3)
        )
        assert 0.2 < report["threshold"] <= 0.6

    def test_reports_every_species_and_their_union(self) -> None:
        report = pad_report(
            scores(0.6, 0.7),
            scores(0.1, 0.2),
            {"replay": scores(0.9, 0.8), "print": scores(0.1, 0.05)},
            scores(0.3, 0.1),
        )
        assert report["by_species"]["replay"]["apcer"] == 0.0
        assert report["by_species"]["print"]["apcer"] == 1.0
        assert report["all_attacks"]["apcer"] == 0.5

    def test_keeps_the_species_breakdown_consistent_with_bpcer(self) -> None:
        report = pad_report(scores(0.6), scores(0.1), {"replay": scores(0.9)}, scores(0.95, 0.1))
        assert report["bpcer"] == 0.5
        assert report["by_species"]["replay"]["hter"] == (0.0 + 0.5) / 2
