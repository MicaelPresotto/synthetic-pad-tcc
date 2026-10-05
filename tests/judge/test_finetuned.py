from pathlib import Path

import numpy as np
import pytest
import torch

from synthetic_pad.evaluation.metrics import auc
from synthetic_pad.judge.finetuned import (
    FineTunedDetector,
    TrainingConfig,
    build_model,
    train_detector,
)

TINY = TrainingConfig(pretrained=False, epochs=6, batch_size=8, learning_rate=1e-3)


def separable_set(count: int = 32, size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3)
    is_attack = np.arange(count) % 2 == 0
    images = rng.integers(0, 60, size=(count, size, size, 3)).astype(np.uint8)
    images[is_attack] += 150
    return images, is_attack


@pytest.fixture(scope="module")
def trained() -> tuple[FineTunedDetector, list[float]]:
    images, is_attack = separable_set()
    return train_detector(images, is_attack, TINY, device="cpu")


class TestTrainingConfig:
    def test_rejects_an_unknown_architecture(self) -> None:
        with pytest.raises(ValueError, match="architecture"):
            TrainingConfig(architecture="vgg")

    def test_rejects_zero_epochs(self) -> None:
        with pytest.raises(ValueError, match="epochs"):
            TrainingConfig(epochs=0)


class TestTrainDetector:
    def test_records_one_loss_per_epoch(
        self, trained: tuple[FineTunedDetector, list[float]]
    ) -> None:
        assert len(trained[1]) == TINY.epochs

    def test_learns_to_separate_an_easy_set(
        self, trained: tuple[FineTunedDetector, list[float]]
    ) -> None:
        images, is_attack = separable_set()
        scores = trained[0].score_batch(images)
        assert auc(scores[is_attack], scores[~is_attack]) > 0.9


class TestFineTunedDetector:
    def test_scores_lie_in_the_unit_interval(
        self, trained: tuple[FineTunedDetector, list[float]]
    ) -> None:
        scores = trained[0].score_batch(separable_set()[0])
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0

    def test_a_single_score_matches_the_batch(
        self, trained: tuple[FineTunedDetector, list[float]]
    ) -> None:
        images = separable_set()[0]
        assert trained[0].score(images[3]) == pytest.approx(
            trained[0].score_batch(images)[3], rel=1e-4
        )

    def test_a_saved_detector_scores_the_same_after_loading(
        self, trained: tuple[FineTunedDetector, list[float]], tmp_path: Path
    ) -> None:
        images = separable_set()[0]
        trained[0].save(tmp_path / "judge.pt")
        reloaded = FineTunedDetector.load(tmp_path / "judge.pt", device="cpu")
        np.testing.assert_allclose(reloaded.score_batch(images), trained[0].score_batch(images))


class TestGradCam:
    def test_gives_one_map_per_image_at_the_last_stage_resolution(
        self, trained: tuple[FineTunedDetector, list[float]]
    ) -> None:
        assert trained[0].grad_cam(separable_set(count=5, size=64)[0]).shape == (5, 2, 2)

    def test_maps_are_never_negative(self, trained: tuple[FineTunedDetector, list[float]]) -> None:
        assert trained[0].grad_cam(separable_set(size=64)[0]).min() >= 0.0

    def test_leaves_the_scores_untouched(
        self, trained: tuple[FineTunedDetector, list[float]]
    ) -> None:
        images = separable_set()[0]
        before = trained[0].score_batch(images)
        trained[0].grad_cam(images)
        np.testing.assert_array_equal(trained[0].score_batch(images), before)

    def test_reduces_to_the_class_activation_map_of_a_global_average_pooled_network(
        self,
    ) -> None:
        torch.manual_seed(0)
        model = build_model("resnet18", pretrained=False)
        detector = FineTunedDetector(model, "resnet18", device="cpu")
        images = separable_set(count=4, size=96)[0]
        captured: list[torch.Tensor] = []
        hook = model.get_submodule("layer4").register_forward_hook(
            lambda _module, _inputs, out: captured.append(out.detach())
        )
        detector.score_batch(images)
        hook.remove()
        (activations,) = captured
        classifier_weights = model.state_dict()["fc.weight"][0]
        pooled_area = activations.shape[2] * activations.shape[3]
        expected = torch.relu(torch.einsum("c,nchw->nhw", classifier_weights, activations))
        np.testing.assert_allclose(
            detector.grad_cam(images), expected.numpy() / pooled_area, rtol=1e-4, atol=1e-8
        )


@pytest.fixture(scope="module")
def vision_transformer() -> FineTunedDetector:
    return FineTunedDetector(build_model("vit_b_32", pretrained=False), "vit_b_32", "cpu")


class TestVisionTransformer:
    def test_scores_one_value_per_image(self, vision_transformer: FineTunedDetector) -> None:
        images = separable_set(count=2, size=224)[0]
        assert vision_transformer.score_batch(images).shape == (2,)

    def test_has_no_convolutional_stage_for_grad_cam(
        self, vision_transformer: FineTunedDetector
    ) -> None:
        with pytest.raises(ValueError, match="no convolutional stage"):
            vision_transformer.grad_cam(separable_set(count=1, size=224)[0])
