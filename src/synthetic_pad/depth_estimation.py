"""Two estimators of the relative depth of a face crop, run locally, so that no crop of the
corpora leaves the machine: Depth Anything V2 Small (Yang et al., 2024), a monocular depth
estimator at a pinned revision, and 3DFAN-4 (Bulat and Tzimiropoulos, 2017), a 3D face model
of 68 landmarks at the weights that its package version pins."""

from collections.abc import Callable
from typing import Any

import face_alignment
import numpy as np
import torch
from transformers import AutoModelForDepthEstimation, DPTImageProcessor

from synthetic_pad.depth import landmark_relief
from synthetic_pad.frames import FACE_CROP_SCALE

DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
DEPTH_MODEL_REVISION = "5426e4f0f36572d16453bbda7a8389317b1bef99"
ESTIMATION_BATCH = 32


class DepthAnything:
    name = DEPTH_MODEL
    revision = DEPTH_MODEL_REVISION

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


class FaceModel:
    name = "3DFAN-4"
    revision = f"face-alignment {face_alignment.__version__}"

    def __init__(self, device: str) -> None:
        self._aligner = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.THREE_D,
            device=device,
            face_detector="folder",
            compile=False,
        )

    def relative_depth(self, images: np.ndarray) -> np.ndarray:
        """Relative depth of one batch of crops, larger values nearer, at their resolution, from
        the landmarks found inside the face box that each crop centres: no face detector runs."""
        side = images.shape[1]
        margin = side * (1 - 1 / FACE_CROP_SCALE) / 2
        face_box = np.array([margin, margin, side - margin, side - margin])
        depths = []
        for image in images:
            (landmarks,) = self._aligner.get_landmarks_from_image(image, detected_faces=[face_box])
            depths.append(landmark_relief(landmarks, side))
        return np.stack(depths)
