"""Depth Anything V2 Small (Yang et al., 2024) at a pinned revision, run locally, so that no crop
of the corpora leaves the machine."""

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForDepthEstimation, DPTImageProcessor

DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
DEPTH_MODEL_REVISION = "5426e4f0f36572d16453bbda7a8389317b1bef99"
ESTIMATION_BATCH = 32


class DepthAnything:
    def __init__(self, device: str) -> None:
        self._processor: Callable[..., Any] = DPTImageProcessor.from_pretrained(
            DEPTH_MODEL, revision=DEPTH_MODEL_REVISION
        )
        model = AutoModelForDepthEstimation.from_pretrained(
            DEPTH_MODEL, revision=DEPTH_MODEL_REVISION
        )
        self._model = model.to(device).eval()
        self._device = device

    def relative_depth(self, images: np.ndarray) -> np.ndarray:
        """Relative depth of one batch of crops, larger values nearer, at their resolution."""
        inputs = self._processor(images=list(images), return_tensors="pt").to(self._device)
        with torch.no_grad():
            predicted = self._model(**inputs).predicted_depth.cpu()
        resized = torch.nn.functional.interpolate(
            predicted[:, None], size=images.shape[1:3], mode="bilinear", align_corners=False
        )
        return np.asarray(resized[:, 0].numpy())
