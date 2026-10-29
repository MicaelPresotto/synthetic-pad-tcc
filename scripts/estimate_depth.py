import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForDepthEstimation, DPTImageProcessor

from synthetic_pad.dataset import write_manifest
from synthetic_pad.depth import DEPTH_GRID, crop_key, depth_target, write_depth_maps
from synthetic_pad.frames import CORPORA, FACE_CROP_SCALE, load_crops, read_crops

DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
DEPTH_MODEL_REVISION = "5426e4f0f36572d16453bbda7a8389317b1bef99"
ESTIMATION_BATCH = 32


def relative_depth(
    images: np.ndarray,
    processor: Callable[..., Any],
    model: torch.nn.Module,
    device: str,
) -> np.ndarray:
    """Depth Anything V2 relative depth, larger values nearer, at the resolution of the crops."""
    inputs = processor(images=list(images), return_tensors="pt").to(device)
    with torch.no_grad():
        predicted = model(**inputs).predicted_depth.cpu()
    resized = torch.nn.functional.interpolate(
        predicted[:, None], size=images.shape[1:3], mode="bilinear", align_corners=False
    )
    return resized[:, 0].numpy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate the pseudo-depth of every bona fide crop of both corpora with "
        "Depth Anything V2, as the target maps of a depth-supervised judge."
    )
    parser.add_argument("--frames", type=Path, default=Path("data/frames"))
    parser.add_argument("--output", type=Path, default=Path("data/depth"))
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()

    processor = DPTImageProcessor.from_pretrained(DEPTH_MODEL, revision=DEPTH_MODEL_REVISION)
    model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL, revision=DEPTH_MODEL_REVISION)
    model = model.to(args.device).eval()
    counts = {}
    for corpus in CORPORA:
        corpus_frames = args.frames / corpus
        bona_fide = [sample for sample in load_crops(corpus_frames) if not sample.is_attack]
        maps: list[np.ndarray] = []
        for start in range(0, len(bona_fide), ESTIMATION_BATCH):
            images = read_crops(bona_fide[start : start + ESTIMATION_BATCH])
            maps.extend(
                depth_target(depth)
                for depth in relative_depth(images, processor, model, args.device)
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
            "grid": DEPTH_GRID,
            "face_crop_scale": FACE_CROP_SCALE,
            "bona_fide_crops": counts,
        },
    )


if __name__ == "__main__":
    main()
