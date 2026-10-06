"""An ImageNet-pretrained network, convolutional or a vision Transformer, fine-tuned as a
liveness detector."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torchvision import models

from synthetic_pad.domain import LivenessDetector, RgbImage, ensure_rgb_uint8
from synthetic_pad.domain.validation import require_positive

type Batch = np.ndarray[Any, np.dtype[np.uint8]]
type Labels = np.ndarray[Any, np.dtype[np.bool_]]
type Scores = np.ndarray[Any, np.dtype[np.float64]]
type AttentionMaps = np.ndarray[Any, np.dtype[np.float64]]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SCORING_BATCH = 128
MIXSTYLE_PROBABILITY = 0.5
MIXSTYLE_ALPHA = 0.1
MIXSTYLE_EPSILON = 1e-6


@dataclass(frozen=True, slots=True)
class Architecture:
    """A torchvision network, its ImageNet weights, the classifier replaced by a single
    attack logit, the stage Grad-CAM reads and the early stages MixStyle follows, if the
    network has convolutional ones."""

    build: Callable[..., nn.Module]
    weights: models.WeightsEnum
    head: str
    grad_cam_stage: str | None
    mixstyle_stages: tuple[str, ...]


RESNET_EARLY_STAGES = ("layer1", "layer2", "layer3")
ARCHITECTURES = {
    "resnet18": Architecture(
        models.resnet18,
        models.ResNet18_Weights.IMAGENET1K_V1,
        "fc",
        "layer4",
        RESNET_EARLY_STAGES,
    ),
    "resnet50": Architecture(
        models.resnet50,
        models.ResNet50_Weights.IMAGENET1K_V2,
        "fc",
        "layer4",
        RESNET_EARLY_STAGES,
    ),
    "vit_b_16": Architecture(
        models.vit_b_16, models.ViT_B_16_Weights.IMAGENET1K_V1, "heads.head", None, ()
    ),
    "vit_b_32": Architecture(
        models.vit_b_32, models.ViT_B_32_Weights.IMAGENET1K_V1, "heads.head", None, ()
    ),
}


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    architecture: str = "resnet18"
    pretrained: bool = True
    epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    mixstyle: bool = False
    seed: int = 20260928

    def __post_init__(self) -> None:
        if self.architecture not in ARCHITECTURES:
            raise ValueError(f"architecture must be one of {sorted(ARCHITECTURES)}")
        if self.mixstyle and not ARCHITECTURES[self.architecture].mixstyle_stages:
            raise ValueError(f"{self.architecture} has no convolutional stages for MixStyle")
        require_positive(
            epochs=self.epochs, batch_size=self.batch_size, learning_rate=self.learning_rate
        )


class FineTunedDetector(LivenessDetector):
    name = "finetuned"

    def __init__(self, model: nn.Module, architecture: str, device: str) -> None:
        self._model = model.to(device).eval()
        self._architecture = architecture
        self._device = device

    def score(self, image: RgbImage) -> float:
        ensure_rgb_uint8(image)
        return float(self.score_batch(image[None])[0])

    def score_batch(self, images: Batch) -> Scores:
        scores = []
        with torch.no_grad():
            for start in range(0, len(images), SCORING_BATCH):
                logits = self._model(
                    _to_tensor(images[start : start + SCORING_BATCH], self._device)
                )
                scores.append(torch.sigmoid(logits.squeeze(1)).cpu().numpy())
        return np.concatenate(scores).astype(np.float64)

    def grad_cam(self, images: Batch) -> AttentionMaps:
        """Grad-CAM (Selvaraju et al., 2017) of the attack logit at the last convolutional stage.

        One non-negative map per image, at that stage's resolution: 7 x 7 for a 224 px crop.
        """
        stage_name = ARCHITECTURES[self._architecture].grad_cam_stage
        if stage_name is None:
            raise ValueError(f"{self._architecture} has no convolutional stage for Grad-CAM")
        captured: list[torch.Tensor] = []
        stage = self._model.get_submodule(stage_name)
        hook = stage.register_forward_hook(lambda _module, _inputs, out: captured.append(out))
        maps = []
        try:
            for start in range(0, len(images), SCORING_BATCH):
                logits = self._model(
                    _to_tensor(images[start : start + SCORING_BATCH], self._device)
                )
                activations = captured.pop()
                (gradients,) = torch.autograd.grad(logits.sum(), activations)
                channel_weights = gradients.mean(dim=(2, 3), keepdim=True)
                cam = torch.relu((channel_weights * activations).sum(dim=1))
                maps.append(cam.detach().cpu().numpy())
        finally:
            hook.remove()
        return np.concatenate(maps).astype(np.float64)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"architecture": self._architecture, "state": self._model.state_dict()}, path)

    @classmethod
    def load(cls, path: Path, device: str) -> "FineTunedDetector":
        saved = torch.load(path, map_location=device, weights_only=True)
        model = build_model(saved["architecture"], pretrained=False)
        model.load_state_dict(saved["state"])
        return cls(model, saved["architecture"], device)


def build_model(architecture: str, pretrained: bool) -> nn.Module:
    chosen = ARCHITECTURES[architecture]
    model = chosen.build(weights=chosen.weights if pretrained else None)
    parent_name, _, attribute = chosen.head.rpartition(".")
    parent = model.get_submodule(parent_name)
    classifier = parent.get_submodule(attribute)
    if not isinstance(classifier, nn.Linear):
        raise TypeError(f"{architecture}.{chosen.head} is not a linear classifier")
    setattr(parent, attribute, nn.Linear(classifier.in_features, 1))
    return model


def train_detector(
    images: Batch, is_attack: Labels, config: TrainingConfig, device: str
) -> tuple[FineTunedDetector, list[float]]:
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    model = build_model(config.architecture, config.pretrained).to(device)
    attack_share = float(is_attack.mean())
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor((1.0 - attack_share) / attack_share, device=device)
    )
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    stages = ARCHITECTURES[config.architecture].mixstyle_stages if config.mixstyle else ()
    hooks = [
        model.get_submodule(stage).register_forward_hook(_mixstyle_hook(rng)) for stage in stages
    ]

    epoch_losses = []
    for _ in range(config.epochs):
        model.train()
        order = rng.permutation(len(images))
        batch_losses = []
        for start in range(0, len(order), config.batch_size):
            batch = order[start : start + config.batch_size]
            mirrored = rng.random(len(batch)) < 0.5
            inputs = _to_tensor(images[batch], device, mirrored)
            targets = torch.as_tensor(is_attack[batch], dtype=torch.float32, device=device)
            optimiser.zero_grad()
            loss = loss_function(model(inputs).squeeze(1), targets)
            loss.backward()
            optimiser.step()
            batch_losses.append(float(loss.detach().cpu()))
        epoch_losses.append(float(np.mean(batch_losses)))
    for hook in hooks:
        hook.remove()
    return FineTunedDetector(model, config.architecture, device), epoch_losses


def mix_styles(features: torch.Tensor, weights: np.ndarray, partners: np.ndarray) -> torch.Tensor:
    """MixStyle (Zhou et al., 2021): each feature map is renormalised to a convex mix, with
    its weight in [0, 1], of its own channel mean and standard deviation and its partner's."""
    mean = features.mean(dim=(2, 3), keepdim=True).detach()
    std = (features.var(dim=(2, 3), keepdim=True) + MIXSTYLE_EPSILON).sqrt().detach()
    weight = torch.as_tensor(weights, dtype=features.dtype, device=features.device)
    weight = weight.view(-1, 1, 1, 1)
    partner = torch.as_tensor(partners, device=features.device)
    mixed_mean = weight * mean + (1 - weight) * mean[partner]
    mixed_std = weight * std + (1 - weight) * std[partner]
    return (features - mean) / std * mixed_std + mixed_mean


def _mixstyle_hook(
    rng: np.random.Generator,
) -> Callable[[nn.Module, object, torch.Tensor], torch.Tensor]:
    def hook(module: nn.Module, _inputs: object, features: torch.Tensor) -> torch.Tensor:
        if not module.training or rng.random() >= MIXSTYLE_PROBABILITY:
            return features
        count = len(features)
        weights = rng.beta(MIXSTYLE_ALPHA, MIXSTYLE_ALPHA, count)
        return mix_styles(features, weights, rng.permutation(count))

    return hook


def _to_tensor(images: Batch, device: str, mirrored: Labels | None = None) -> torch.Tensor:
    batch = images.astype(np.float32) / 255.0
    if mirrored is not None:
        batch[mirrored] = batch[mirrored, :, ::-1]
    normalised = (batch - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(normalised.transpose(0, 3, 1, 2))).to(device)
