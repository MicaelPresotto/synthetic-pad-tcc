import argparse
from pathlib import Path

import numpy as np
import torch

from synthetic_pad.dataset import write_manifest
from synthetic_pad.depth import DEPTH_GRID, crop_key, depth_target, write_depth_maps
from synthetic_pad.depth_estimation import (
    DEPTH_MODEL,
    DEPTH_MODEL_REVISION,
    ESTIMATION_BATCH,
    DepthAnything,
)
from synthetic_pad.frames import CORPORA, FACE_CROP_SCALE, load_crops, read_crops


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate the pseudo-depth of every bona fide crop of both corpora with "
        "Depth Anything V2, as the target maps of a depth-supervised judge."
    )
    parser.add_argument("--frames", type=Path, default=Path("data/frames"))
    parser.add_argument("--output", type=Path, default=Path("data/depth"))
    parser.add_argument(
        "--grid",
        type=int,
        default=DEPTH_GRID,
        help="cells per side of each map: 7 for a depth head on layer4, 14 for one on layer3",
    )
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()

    estimator = DepthAnything(args.device)
    counts = {}
    for corpus in CORPORA:
        corpus_frames = args.frames / corpus
        bona_fide = [sample for sample in load_crops(corpus_frames) if not sample.is_attack]
        maps: list[np.ndarray] = []
        for start in range(0, len(bona_fide), ESTIMATION_BATCH):
            images = read_crops(bona_fide[start : start + ESTIMATION_BATCH])
            maps.extend(
                depth_target(depth, args.grid) for depth in estimator.relative_depth(images)
            )
        stacked = np.stack(maps)
        keys = [crop_key(sample, corpus_frames) for sample in bona_fide]
        write_depth_maps(args.output / f"{corpus}.npz", keys, stacked)
        counts[corpus] = len(bona_fide)
        print(
            f"{corpus}: {len(bona_fide)} bona fide crops, mean map\n{stacked.mean(axis=0).round(2)}"
        )
    write_manifest(
        args.output,
        {
            "model": DEPTH_MODEL,
            "revision": DEPTH_MODEL_REVISION,
            "grid": args.grid,
            "face_crop_scale": FACE_CROP_SCALE,
            "bona_fide_crops": counts,
        },
    )


if __name__ == "__main__":
    main()
