from pathlib import Path

import numpy as np
import pytest

from synthetic_pad.evaluation.metrics import auc
from synthetic_pad.judge.resnet import ResNetLivenessDetector, TrainingConfig, train_resnet

TINY = TrainingConfig(pretrained=False, epochs=6, batch_size=8, learning_rate=1e-3)


def separable_set(count: int = 32, size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3)
    is_attack = np.arange(count) % 2 == 0
    images = rng.integers(0, 60, size=(count, size, size, 3)).astype(np.uint8)
    images[is_attack] += 150
    return images, is_attack


@pytest.fixture(scope="module")
def trained() -> tuple[ResNetLivenessDetector, list[float]]:
    images, is_attack = separable_set()
    return train_resnet(images, is_attack, TINY, device="cpu")


class TestTrainingConfig:
    def test_rejects_an_unknown_architecture(self) -> None:
        with pytest.raises(ValueError, match="architecture"):
            TrainingConfig(architecture="vgg")

    def test_rejects_zero_epochs(self) -> None:
        with pytest.raises(ValueError, match="epochs"):
            TrainingConfig(epochs=0)


class TestTrainResnet:
    def test_records_one_loss_per_epoch(
        self, trained: tuple[ResNetLivenessDetector, list[float]]
    ) -> None:
        assert len(trained[1]) == TINY.epochs

    def test_learns_to_separate_an_easy_set(
        self, trained: tuple[ResNetLivenessDetector, list[float]]
    ) -> None:
        images, is_attack = separable_set()
        scores = trained[0].score_batch(images)
        assert auc(scores[is_attack], scores[~is_attack]) > 0.9


class TestResNetLivenessDetector:
    def test_scores_lie_in_the_unit_interval(
        self, trained: tuple[ResNetLivenessDetector, list[float]]
    ) -> None:
        scores = trained[0].score_batch(separable_set()[0])
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0

    def test_a_single_score_matches_the_batch(
        self, trained: tuple[ResNetLivenessDetector, list[float]]
    ) -> None:
        images = separable_set()[0]
        assert trained[0].score(images[3]) == pytest.approx(
            trained[0].score_batch(images)[3], rel=1e-4
        )

    def test_a_saved_detector_scores_the_same_after_loading(
        self, trained: tuple[ResNetLivenessDetector, list[float]], tmp_path: Path
    ) -> None:
        images = separable_set()[0]
        trained[0].save(tmp_path / "judge.pt")
        reloaded = ResNetLivenessDetector.load(tmp_path / "judge.pt", device="cpu")
        np.testing.assert_allclose(reloaded.score_batch(images), trained[0].score_batch(images))
