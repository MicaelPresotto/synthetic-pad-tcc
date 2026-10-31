import argparse
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import torch
from matplotlib.figure import Figure
from skimage.data import lbp_frontal_face_cascade_filename
from skimage.feature import Cascade

from synthetic_pad.calibration import read_fitted_config
from synthetic_pad.dataset import WORKING_RESOLUTION, read_image
from synthetic_pad.depth import DEPTH_GRID, depth_target, face_prominence, shape_similarity
from synthetic_pad.depth_estimation import DepthAnything, FaceModel
from synthetic_pad.domain import RgbImage
from synthetic_pad.frames import FACE_CROP_SCALE, square_face_crop
from synthetic_pad.generators.moire import MoireRecaptureGenerator

FINE_GRID = 14
DETECTION_SCALE_FACTOR = 1.2
SMALLEST_FACE_PX = 60
FIGURE_WIDTH_IN = 6.3
TITLE_HEIGHT_IN = 0.2
ROW_HEIGHT_IN = 1.2
COLOUR_MAP = "viridis"
INK = "#52514e"
COLUMN_TITLES = (
    "Input crop",
    "Depth Anything V2",
    r"7$\times$7",
    r"14$\times$14",
    r"3DFAN-4, 7$\times$7",
    "Training target",
)


def face_crop(image: RgbImage) -> RgbImage:
    """The largest face found, cut and resized as the crops of the corpora were."""
    faces = Cascade(lbp_frontal_face_cascade_filename()).detect_multi_scale(
        img=image,
        scale_factor=DETECTION_SCALE_FACTOR,
        step_ratio=1,
        min_size=(SMALLEST_FACE_PX, SMALLEST_FACE_PX),
        max_size=image.shape[:2],
    )
    if not faces:
        raise ValueError("no face found in the input image")
    face = max(faces, key=lambda found: found["width"] * found["height"])
    box = (face["c"], face["r"], face["width"], face["height"])
    crop = square_face_crop(image, box, FACE_CROP_SCALE)
    size = (WORKING_RESOLUTION, WORKING_RESOLUTION)
    return np.asarray(cv2.resize(crop, size, interpolation=cv2.INTER_AREA))


def unit_range(depth: np.ndarray) -> np.ndarray:
    return np.asarray((depth - depth.min()) / np.ptp(depth))


def print_similarity(estimator: str, depths: np.ndarray, names: Sequence[str]) -> None:
    maps = np.stack([depth_target(depth) for depth in depths])
    similarities = shape_similarity(maps, maps[0])
    for name, depth, similarity in zip(names, depths, similarities, strict=True):
        print(
            f"{estimator}, {name}: shape similarity to the bona fide map {similarity:.2f}, "
            f"face prominence {face_prominence(depth):.2f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Relative depth of a bona fide portrait, of a synthetic replay of it and, if "
        "given, of a photograph of it on a screen, the maps estimated from each, and the target a "
        "depth-supervised judge trains each on: the estimated map for the bona fide crop, a flat "
        "one for every replay."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="bona fide portrait; its largest face is cut as the crops of the corpora were",
    )
    parser.add_argument(
        "--generator-config",
        type=Path,
        default=Path("experiments/moire-fitted-to-msu-matched-noise.json"),
    )
    parser.add_argument(
        "--replay-photo",
        type=Path,
        help="photograph of the portrait shown on a screen, drawn as a third row: a real replay",
    )
    parser.add_argument("--output", type=Path, default=Path("results/depth_preview.pdf"))
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    bona_fide = face_crop(read_image(args.input))
    generator = MoireRecaptureGenerator(read_fitted_config(args.generator_config))
    rows = {"Bona fide": bona_fide, "Synthetic replay": generator.generate(bona_fide)}
    if args.replay_photo:
        rows["Real replay"] = face_crop(read_image(args.replay_photo))
    crops = np.stack(list(rows.values()))
    estimated = DepthAnything(args.device).relative_depth(crops)
    from_landmarks = FaceModel(args.device).relative_depth(crops)
    print_similarity("Depth Anything V2", estimated, list(rows))
    print_similarity("3DFAN-4", from_landmarks, list(rows))

    height = TITLE_HEIGHT_IN + ROW_HEIGHT_IN * len(rows)
    figure = Figure(figsize=(FIGURE_WIDTH_IN, height), layout="constrained")
    axes = figure.subplots(len(rows), len(COLUMN_TITLES))
    flat = np.zeros((DEPTH_GRID, DEPTH_GRID))
    targets = [depth_target(estimated[0])] + [flat] * (len(rows) - 1)
    for row_axes, crop, depth, landmarks, target, title in zip(
        axes, crops, estimated, from_landmarks, targets, rows, strict=True
    ):
        row_axes[0].imshow(crop)
        maps = (
            unit_range(depth),
            depth_target(depth),
            depth_target(depth, FINE_GRID),
            depth_target(landmarks),
            target,
        )
        for axis, relief in zip(row_axes[1:], maps, strict=True):
            axis.imshow(relief, cmap=COLOUR_MAP, vmin=0, vmax=1, interpolation="nearest")
        for axis in row_axes:
            axis.set_xticks([])
            axis.set_yticks([])
        row_axes[0].set_ylabel(title, fontsize=8, color=INK)
    for axis, title in zip(axes[0], COLUMN_TITLES, strict=True):
        axis.set_title(title, fontsize=8, color=INK)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
