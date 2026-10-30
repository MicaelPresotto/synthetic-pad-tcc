from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from synthetic_pad.evaluation.metrics import auc
from synthetic_pad.judge.finetuned import (
    DEPTH_HEAD,
    FineTunedDetector,
    TrainingConfig,
    build_model,
    depth_loss,
    mix_styles,
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

    def test_rejects_mixstyle_on_a_vision_transformer(self) -> None:
        with pytest.raises(ValueError, match="MixStyle"):
            TrainingConfig(architecture="vit_b_16", mixstyle=True)

    def test_rejects_depth_supervision_on_a_vision_transformer(self) -> None:
        with pytest.raises(ValueError, match="depth map"):
            TrainingConfig(architecture="vit_b_16", depth_supervision=True)

    def test_rejects_a_depth_auxiliary_task_on_a_vision_transformer(self) -> None:
        with pytest.raises(ValueError, match="depth map"):
            TrainingConfig(architecture="vit_b_16", depth_auxiliary=True)

    def test_rejects_a_depth_map_as_both_the_only_and_an_auxiliary_target(self) -> None:
        with pytest.raises(ValueError, match="either"):
            TrainingConfig(depth_supervision=True, depth_auxiliary=True)

    def test_rejects_an_unknown_depth_stage(self) -> None:
        with pytest.raises(ValueError, match="depth_stage"):
            TrainingConfig(depth_auxiliary=True, depth_stage="layer2")

    def test_rejects_a_depth_weight_without_a_depth_auxiliary_task(self) -> None:
        with pytest.raises(ValueError, match="auxiliary task only"):
            TrainingConfig(depth_supervision=True, depth_weight=10.0)

    def test_rejects_a_zero_depth_weight(self) -> None:
        with pytest.raises(ValueError, match="depth_weight"):
            TrainingConfig(depth_auxiliary=True, depth_weight=0.0)


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


class TestMixStyle:
    @staticmethod
    def features() -> torch.Tensor:
        generator = torch.Generator().manual_seed(0)
        scale = torch.tensor([1.0, 3.0]).view(2, 1, 1, 1)
        return torch.randn(2, 4, 6, 6, generator=generator) * scale + scale

    def test_a_full_weight_on_itself_leaves_the_features_unchanged(self) -> None:
        features = self.features()
        mixed = mix_styles(features, np.ones(2), np.array([1, 0]))
        torch.testing.assert_close(mixed, features, rtol=1e-4, atol=1e-4)

    def test_a_zero_weight_takes_on_the_partner_statistics(self) -> None:
        features = self.features()
        mixed = mix_styles(features, np.zeros(2), np.array([1, 0]))
        torch.testing.assert_close(
            mixed.mean(dim=(2, 3)), features.mean(dim=(2, 3)).flip(0), rtol=1e-4, atol=1e-4
        )
        torch.testing.assert_close(
            mixed.std(dim=(2, 3)), features.std(dim=(2, 3)).flip(0), rtol=1e-3, atol=1e-3
        )

    def test_keeps_the_spatial_pattern_of_each_map(self) -> None:
        features = self.features()
        mixed = mix_styles(features, np.full(2, 0.3), np.array([1, 0]))
        correlation = np.corrcoef(features[0, 0].flatten(), mixed[0, 0].flatten())[0, 1]
        assert correlation == pytest.approx(1.0)

    def test_a_judge_trained_with_it_still_learns_an_easy_set(self) -> None:
        images, is_attack = separable_set()
        detector, _ = train_detector(images, is_attack, replace(TINY, mixstyle=True), device="cpu")
        scores = detector.score_batch(images)
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


DEPTH = replace(TINY, depth_supervision=True)
DEPTH_SIDE = 64


def relief_targets(is_attack: np.ndarray, side: int = 2) -> np.ndarray:
    """A raised map for every bona fide image and a flat one for every attack, by default on
    the 2 x 2 grid that a 64 px image leaves at the last stage."""
    raised = np.where(is_attack[:, None, None], 0.0, 1.0)
    return raised.repeat(side, 1).repeat(side, 2).astype(np.float32)


@pytest.fixture(scope="module")
def depth_trained() -> FineTunedDetector:
    images, is_attack = separable_set(size=DEPTH_SIDE)
    return train_detector(images, is_attack, DEPTH, "cpu", relief_targets(is_attack))[0]


class TestDepthSupervision:
    def test_draws_one_depth_logit_per_cell_of_the_last_stage(self) -> None:
        model = build_model("resnet18", pretrained=False, depth_supervision=True)
        images = separable_set(count=3, size=DEPTH_SIDE)[0]
        assert FineTunedDetector(model, "resnet18", "cpu", True).score_batch(images).shape == (3,)
        with torch.no_grad():
            assert model(torch.zeros(3, 3, DEPTH_SIDE, DEPTH_SIDE)).shape == (3, 4)

    def test_learns_to_separate_an_easy_set(self, depth_trained: FineTunedDetector) -> None:
        images, is_attack = separable_set(size=DEPTH_SIDE)
        scores = depth_trained.score_batch(images)
        assert auc(scores[is_attack], scores[~is_attack]) > 0.9
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0

    def test_scores_the_same_after_saving_and_loading(
        self, depth_trained: FineTunedDetector, tmp_path: Path
    ) -> None:
        images = separable_set(size=DEPTH_SIDE)[0]
        depth_trained.save(tmp_path / "judge.pt")
        reloaded = FineTunedDetector.load(tmp_path / "judge.pt", device="cpu")
        np.testing.assert_allclose(reloaded.score_batch(images), depth_trained.score_batch(images))

    def test_has_no_attack_logit_for_grad_cam(self, depth_trained: FineTunedDetector) -> None:
        with pytest.raises(ValueError, match="Grad-CAM"):
            depth_trained.grad_cam(separable_set(count=1, size=DEPTH_SIDE)[0])

    def test_needs_depth_targets(self) -> None:
        images, is_attack = separable_set(size=DEPTH_SIDE)
        with pytest.raises(ValueError, match="depth targets"):
            train_detector(images, is_attack, DEPTH, "cpu")

    def test_refuses_depth_targets_without_it(self) -> None:
        images, is_attack = separable_set(size=DEPTH_SIDE)
        with pytest.raises(ValueError, match="depth targets"):
            train_detector(images, is_attack, TINY, "cpu", relief_targets(is_attack))


AUXILIARY = replace(TINY, depth_auxiliary=True)


@pytest.fixture(scope="module")
def auxiliary_trained() -> FineTunedDetector:
    images, is_attack = separable_set(size=DEPTH_SIDE)
    return train_detector(images, is_attack, AUXILIARY, "cpu", relief_targets(is_attack))[0]


class TestDepthAuxiliary:
    def test_keeps_one_attack_logit_beside_a_depth_head(self) -> None:
        model = build_model("resnet18", pretrained=False, depth_auxiliary=True)
        with torch.no_grad():
            assert model(torch.zeros(3, 3, DEPTH_SIDE, DEPTH_SIDE)).shape == (3, 1)
        assert isinstance(model.get_submodule(DEPTH_HEAD), torch.nn.Conv2d)

    def test_learns_to_separate_an_easy_set(self, auxiliary_trained: FineTunedDetector) -> None:
        images, is_attack = separable_set(size=DEPTH_SIDE)
        scores = auxiliary_trained.score_batch(images)
        assert auc(scores[is_attack], scores[~is_attack]) > 0.9
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0

    def test_scores_the_same_after_saving_and_loading(
        self, auxiliary_trained: FineTunedDetector, tmp_path: Path
    ) -> None:
        images = separable_set(size=DEPTH_SIDE)[0]
        auxiliary_trained.save(tmp_path / "judge.pt")
        reloaded = FineTunedDetector.load(tmp_path / "judge.pt", device="cpu")
        np.testing.assert_allclose(
            reloaded.score_batch(images), auxiliary_trained.score_batch(images)
        )

    def test_still_gives_a_grad_cam_of_the_attack_logit(
        self, auxiliary_trained: FineTunedDetector
    ) -> None:
        maps = auxiliary_trained.grad_cam(separable_set(count=2, size=DEPTH_SIDE)[0])
        assert maps.shape == (2, 2, 2)

    def test_needs_depth_targets(self) -> None:
        images, is_attack = separable_set(size=DEPTH_SIDE)
        with pytest.raises(ValueError, match="depth targets"):
            train_detector(images, is_attack, AUXILIARY, "cpu")

    def test_reads_the_chosen_stage(self) -> None:
        model = build_model("resnet18", False, depth_auxiliary=True, depth_stage="layer3")
        head = model.get_submodule(DEPTH_HEAD)
        assert isinstance(head, torch.nn.Conv2d)
        assert head.in_channels == 256

    def test_a_layer3_head_learns_the_finer_map_and_survives_saving_and_loading(
        self, tmp_path: Path
    ) -> None:
        images, is_attack = separable_set(size=DEPTH_SIDE)
        config = replace(AUXILIARY, depth_stage="layer3", depth_weight=10.0)
        detector = train_detector(images, is_attack, config, "cpu", relief_targets(is_attack, 4))[0]
        scores = detector.score_batch(images)
        assert auc(scores[is_attack], scores[~is_attack]) > 0.9
        detector.save(tmp_path / "judge.pt")
        reloaded = FineTunedDetector.load(tmp_path / "judge.pt", device="cpu")
        np.testing.assert_allclose(reloaded.score_batch(images), scores)


class TestDepthLoss:
    def test_is_zero_for_a_perfect_map(self) -> None:
        logits = torch.tensor([[20.0, 20.0], [-20.0, -20.0]])
        targets = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
        assert float(depth_loss(logits, targets, torch.tensor([0.0, 1.0]), 3.0)) < 1e-6

    def test_weighs_an_attack_by_the_attack_weight(self) -> None:
        logits = torch.zeros(2, 1)
        targets = torch.ones(2, 1)
        bona_fide_only = depth_loss(logits, targets, torch.tensor([0.0, 0.0]), 3.0)
        one_attack = depth_loss(logits, targets, torch.tensor([0.0, 1.0]), 3.0)
        assert float(one_attack) == pytest.approx(2.0 * float(bona_fide_only))
