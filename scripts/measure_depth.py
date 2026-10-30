import argparse
from pathlib import Path

import numpy as np
import torch

from synthetic_pad.corpora import BONA_FIDE
from synthetic_pad.depth import depth_target, face_prominence, shape_similarity
from synthetic_pad.depth_estimation import ESTIMATION_BATCH, DepthAnything
from synthetic_pad.frames import CORPORA, CropSample, load_partitions, read_crops, select


def estimate(estimator: DepthAnything, samples: list[CropSample]) -> tuple[np.ndarray, np.ndarray]:
    """Target maps and face prominence of each crop."""
    maps, prominence = [], []
    for start in range(0, len(samples), ESTIMATION_BATCH):
        for depth in estimator.relative_depth(
            read_crops(samples[start : start + ESTIMATION_BATCH])
        ):
            maps.append(depth_target(depth))
            prominence.append(face_prominence(depth))
    return np.stack(maps), np.array(prominence)


def spread(values: np.ndarray) -> str:
    low, median, high = np.quantile(values, [0.25, 0.5, 0.75])
    return f"{median:5.2f} ({low:5.2f} to {high:5.2f})"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Depth Anything V2 relative depth of the training crops of each corpus, by "
        "species: how close the relief of each crop is in shape to the mean bona fide face, and "
        "how far its face box stands out from the rest of the crop. A screen or a print measured "
        "by a depth sensor would score zero on both."
    )
    parser.add_argument("--frames", type=Path, default=Path("data/frames"))
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    estimator = DepthAnything(args.device)
    partitions = load_partitions(args.frames)
    for corpus in CORPORA:
        training = select(partitions[corpus], "train")
        species = sorted({sample.species for sample in training}, key=lambda s: s != BONA_FIDE)
        estimated = {
            name: estimate(estimator, [sample for sample in training if sample.species == name])
            for name in species
        }
        reference = estimated[BONA_FIDE][0].mean(axis=0)
        print(f"\n{corpus}, training partition, median (interquartile range)")
        print(f"  {'species':10s} {'shape similarity':24s} {'face prominence':24s} crops")
        for name, (maps, prominence) in estimated.items():
            similarity = shape_similarity(maps, reference)
            print(f"  {name:10s} {spread(similarity):24s} {spread(prominence):24s} {len(maps)}")


if __name__ == "__main__":
    main()
