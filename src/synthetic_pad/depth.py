"""Pseudo-depth maps of bona fide face crops, the targets of a depth-supervised detector
(Liu et al., 2018): the relief of a live face, against the flat map of every attack."""

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from synthetic_pad.frames import FACE_CROP_SCALE, CropSample, face_cells

DEPTH_GRID = 7


def depth_target(relative_depth: np.ndarray, grid: int = DEPTH_GRID) -> np.ndarray:
    """A grid x grid map in [0, 1] from the relative depth of one square crop, larger values
    nearer: rescaled to span [0, 1] inside the face box, zero outside it, then averaged over
    each cell. A face box of constant depth gives a flat map."""
    face = face_cells(relative_depth.shape[0], FACE_CROP_SCALE)
    inside = relative_depth[face]
    spread = inside.max() - inside.min()
    relief = np.zeros(relative_depth.shape, dtype=np.float64)
    if spread > 0:
        relief[face] = (inside - inside.min()) / spread
    return cv2.resize(relief, (grid, grid), interpolation=cv2.INTER_AREA).astype(np.float32)


def face_prominence(relative_depth: np.ndarray) -> float:
    """How far the face box of a square crop stands out from the rest of it in relative depth,
    larger values nearer: the difference of their means over the range of the whole crop, in
    [-1, 1]. A flat crop gives zero."""
    face = face_cells(relative_depth.shape[0], FACE_CROP_SCALE)
    spread = relative_depth.max() - relative_depth.min()
    if spread == 0:
        return 0.0
    return float((relative_depth[face].mean() - relative_depth[~face].mean()) / spread)


def shape_similarity(maps: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Pearson correlation of each target map with a reference map over the cells of the face
    box. A map with no relief has no shape and gives zero."""
    face = face_cells(reference.shape[0], FACE_CROP_SCALE)
    centred = maps[:, face] - maps[:, face].mean(axis=1, keepdims=True)
    reference_centred = reference[face] - reference[face].mean()
    norms = np.linalg.norm(centred, axis=1) * np.linalg.norm(reference_centred)
    products = centred @ reference_centred
    return np.asarray(np.divide(products, norms, out=np.zeros(len(maps)), where=norms > 0))


def crop_key(sample: CropSample, corpus_frames: Path) -> str:
    return sample.image.relative_to(corpus_frames).as_posix()


def write_depth_maps(path: Path, keys: Sequence[str], maps: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, keys=np.array(keys), maps=maps)


def read_depth_maps(path: Path, samples: Sequence[CropSample], corpus_frames: Path) -> np.ndarray:
    with np.load(path) as saved:
        row_of = {str(key): row for row, key in enumerate(saved["keys"])}
        keys = [crop_key(sample, corpus_frames) for sample in samples]
        missing = [key for key in keys if key not in row_of]
        if missing:
            raise KeyError(f"{path} has no depth map for {len(missing)} crops, e.g. {missing[0]}")
        return np.asarray(saved["maps"][[row_of[key] for key in keys]], dtype=np.float32)
