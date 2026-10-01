import numpy as np
import pytest

from synthetic_pad.evaluation.metrics import (
    FloatArray,
    acer,
    apcer,
    auc,
    bpcer,
    classification_metrics,
    eer,
    eer_threshold,
    hter,
)


def scores(*values: float) -> FloatArray:
    return np.array(values, dtype=np.float64)


def rank_based_auc(attack_scores: FloatArray, bonafide_scores: FloatArray) -> float:
    pairs = attack_scores[:, None] - bonafide_scores[None, :]
    return float(np.mean(pairs > 0.0) + np.mean(pairs == 0.0) / 2.0)


class TestApcer:
    def test_counts_only_the_attacks_that_fell_below_the_threshold(self) -> None:
        assert apcer(scores(0.1, 0.2, 0.9, 0.95), 0.5) == 0.5

    def test_a_score_on_the_threshold_is_an_attack_not_an_acceptance(self) -> None:
        assert apcer(scores(0.5), 0.5) == 0.0

    def test_reaches_one_when_every_attack_slipped_through(self) -> None:
        assert apcer(scores(0.1, 0.2), 0.9) == 1.0

    def test_rejects_an_empty_array(self) -> None:
        with pytest.raises(ValueError, match="attack_scores"):
            apcer(scores(), 0.5)


class TestBpcer:
    def test_counts_the_bona_fide_that_reached_the_threshold(self) -> None:
        assert bpcer(scores(0.1, 0.2, 0.9, 0.95), 0.5) == 0.5

    def test_a_score_on_the_threshold_is_a_rejection(self) -> None:
        assert bpcer(scores(0.5), 0.5) == 1.0

    def test_reaches_zero_when_no_bona_fide_was_rejected(self) -> None:
        assert bpcer(scores(0.1, 0.2), 0.9) == 0.0

    def test_rejects_an_empty_array(self) -> None:
        with pytest.raises(ValueError, match="bonafide_scores"):
            bpcer(scores(), 0.5)


class TestAcer:
    def test_averages_the_two_error_rates(self) -> None:
        attacks, bonafide = scores(0.1, 0.2, 0.3, 0.9), scores(0.8, 0.1, 0.1, 0.1)
        assert acer(attacks, bonafide, 0.5) == (apcer(attacks, 0.5) + bpcer(bonafide, 0.5)) / 2.0


class TestEer:
    def test_is_zero_when_the_two_distributions_do_not_overlap(self) -> None:
        assert eer(scores(0.7, 0.8, 0.9), scores(0.1, 0.2, 0.3)) == 0.0

    def test_is_one_half_when_the_scores_carry_no_information(self) -> None:
        assert eer(scores(0.0, 1.0), scores(0.0, 1.0)) == 0.5

    def test_survives_a_monotonic_rescaling_of_the_scores(self) -> None:
        attacks, bonafide = scores(0.4, 0.6, 0.8), scores(0.2, 0.5, 0.7)
        assert eer(attacks, bonafide) == eer(attacks * 2.0 + 1.0, bonafide * 2.0 + 1.0)

    def test_rejects_an_empty_array(self) -> None:
        with pytest.raises(ValueError, match="attack_scores"):
            eer(scores(), scores(0.5))


class TestEerThreshold:
    def test_centres_itself_between_two_distributions_that_do_not_overlap(self) -> None:
        assert eer_threshold(scores(0.7, 0.8, 0.9), scores(0.1, 0.2, 0.3)) == pytest.approx(0.5)

    def test_centres_itself_in_the_equal_error_interval_when_the_classes_overlap(self) -> None:
        attacks, bonafide = scores(0.2, 0.7, 0.8), scores(0.1, 0.3, 0.6)
        assert eer_threshold(attacks, bonafide) == pytest.approx(0.45)

    def test_reproduces_the_equal_error_rate_it_was_derived_from(self) -> None:
        attacks, bonafide = scores(0.3, 0.55, 0.6, 0.9), scores(0.1, 0.4, 0.5, 0.7)
        assert acer(attacks, bonafide, eer_threshold(attacks, bonafide)) == eer(attacks, bonafide)


class TestHter:
    def test_collapses_onto_the_eer_when_the_development_set_is_the_evaluation_set(self) -> None:
        attacks, bonafide = scores(0.3, 0.55, 0.6, 0.9), scores(0.1, 0.4, 0.5, 0.7)
        assert hter(attacks, bonafide, attacks, bonafide) == eer(attacks, bonafide)

    def test_holds_the_development_threshold_against_a_shifted_evaluation_set(self) -> None:
        development_attacks, development_bonafide = scores(0.6, 0.7, 0.8), scores(0.1, 0.2, 0.3)
        shifted_attacks, shifted_bonafide = scores(0.1, 0.2, 0.3), scores(0.6, 0.7, 0.8)
        threshold = eer_threshold(development_attacks, development_bonafide)
        assert hter(
            development_attacks, development_bonafide, shifted_attacks, shifted_bonafide
        ) == acer(shifted_attacks, shifted_bonafide, threshold)

    def test_reaches_one_when_the_evaluation_set_inverts_the_development_set(self) -> None:
        development_attacks, development_bonafide = scores(0.6, 0.7, 0.8), scores(0.1, 0.2, 0.3)
        assert (
            hter(
                development_attacks,
                development_bonafide,
                scores(0.1, 0.2, 0.3),
                scores(0.6, 0.7, 0.8),
            )
            == 1.0
        )


class TestAuc:
    def test_is_one_when_every_attack_outscores_every_bona_fide(self) -> None:
        assert auc(scores(0.7, 0.8), scores(0.1, 0.2)) == 1.0

    def test_is_zero_when_the_ranking_is_exactly_inverted(self) -> None:
        assert auc(scores(0.1, 0.2), scores(0.7, 0.8)) == 0.0

    def test_is_one_half_when_every_score_is_tied(self) -> None:
        assert auc(scores(1.0, 1.0), scores(1.0, 1.0)) == 0.5

    def test_gives_a_tie_half_the_credit_of_a_win(self) -> None:
        assert auc(scores(1.0, 2.0), scores(0.0, 1.0)) == 0.875

    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_matches_the_rank_based_definition(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        attacks = rng.uniform(0.3, 1.0, 40)
        bonafide = rng.uniform(0.0, 0.7, 25)
        assert auc(attacks, bonafide) == pytest.approx(rank_based_auc(attacks, bonafide))

    def test_matches_the_rank_based_definition_when_scores_are_heavily_tied(self) -> None:
        rng = np.random.default_rng(20260914)
        attacks = rng.integers(0, 4, 30).astype(np.float64)
        bonafide = rng.integers(0, 4, 30).astype(np.float64)
        assert auc(attacks, bonafide) == pytest.approx(rank_based_auc(attacks, bonafide))

    def test_rejects_an_empty_array(self) -> None:
        with pytest.raises(ValueError, match="bonafide_scores"):
            auc(scores(0.5), scores())


class TestClassificationMetrics:
    def test_treats_the_attack_as_the_positive_class(self) -> None:
        attacks, bonafide = scores(0.9, 0.8, 0.2), scores(0.1, 0.7)
        assert classification_metrics(attacks, bonafide, 0.5) == pytest.approx(
            {"accuracy": 3 / 5, "precision": 2 / 3, "recall": 2 / 3, "f1": 2 / 3}
        )

    def test_a_score_on_the_threshold_is_flagged_as_an_attack(self) -> None:
        assert classification_metrics(scores(0.5), scores(0.1), 0.5)["recall"] == 1.0

    def test_precision_is_zero_when_nothing_is_flagged(self) -> None:
        assert classification_metrics(scores(0.1), scores(0.2), 0.9)["precision"] == 0.0

    def test_rejects_an_empty_array(self) -> None:
        with pytest.raises(ValueError, match="bonafide_scores"):
            classification_metrics(scores(0.9), scores(), 0.5)
