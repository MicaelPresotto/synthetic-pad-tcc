import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from synthetic_pad.corpora import BONA_FIDE
from synthetic_pad.dataset import write_image
from synthetic_pad.frames import (
    CORPORA,
    FACE_CROP_SCALE,
    CropSample,
    load_partitions,
    read_crops,
    select,
)
from synthetic_pad.generators.moire import MoireConfig, MoireRecaptureGenerator
from synthetic_pad.judge.finetuned import AttentionMaps, FineTunedDetector

SYNTHETIC_REPLAY = "synthetic replay"
OVERLAY_OPACITY = 0.45


def face_cells(grid: int, scale: float) -> np.ndarray:
    """Cells of the attention grid whose centre falls inside the face box of the crop."""
    centres = (np.arange(grid) + 0.5) / grid
    inside = np.abs(centres - 0.5) <= 0.5 / scale
    return np.asarray(inside[:, None] & inside[None, :])


def face_share(maps: AttentionMaps, scale: float) -> np.ndarray:
    """Fraction of each map's mass on the face; NaN where the map is empty."""
    totals = maps.sum(axis=(1, 2))
    on_face = maps[:, face_cells(maps.shape[1], scale)].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(totals > 0.0, on_face / totals, np.nan)


def summarise(scores: np.ndarray, maps: AttentionMaps) -> dict[str, object]:
    """Grad-CAM keeps only evidence for an attack, so a crop judged plainly genuine has an
    empty map; those are counted and left out of the attention figures."""
    shares = face_share(maps, FACE_CROP_SCALE)
    attended = ~np.isnan(shares)
    summary: dict[str, object] = {
        "crops": len(maps),
        "attended_crops": int(attended.sum()),
        "median_score": float(np.median(scores)),
        "uniform_face_share": float(face_cells(maps.shape[1], FACE_CROP_SCALE).mean()),
    }
    if attended.any():
        masses = maps[attended].sum(axis=(1, 2), keepdims=True)
        summary |= {
            "face_share_mean": float(shares[attended].mean()),
            "face_share_std": float(shares[attended].std()),
            "mean_map": (maps[attended] / masses).mean(axis=0).round(4).tolist(),
        }
    return summary


def overlay(image: np.ndarray, attention: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    peak = attention.max()
    scaled = attention / peak if peak > 0.0 else attention
    upsampled = cv2.resize(scaled.astype(np.float32), (width, height))
    heat = cv2.applyColorMap((255 * upsampled).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    blended = (1.0 - OVERLAY_OPACITY) * image + OVERLAY_OPACITY * heat
    return np.asarray(blended.round(), dtype=np.uint8)


def groups(
    partitions: dict[str, list[CropSample]], corpus: str, generator: MoireRecaptureGenerator
) -> dict[str, np.ndarray]:
    found: dict[str, np.ndarray] = {}
    for scenario, tested in (("intra", corpus), ("cross", _other(corpus))):
        samples = select(partitions[tested], "test")
        for species in sorted({sample.species for sample in samples}):
            chosen = [sample for sample in samples if sample.species == species]
            found[f"{scenario} {tested} {species}"] = read_crops(chosen)
        bona_fide = found[f"{scenario} {tested} {BONA_FIDE}"]
        found[f"{scenario} {tested} {SYNTHETIC_REPLAY}"] = np.stack(
            [generator.generate(image) for image in bona_fide]
        )
    return found


def _other(corpus: str) -> str:
    return next(other for other in CORPORA if other != corpus)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Where a trained judge looks: Grad-CAM over the test crops of both corpora."
    )
    parser.add_argument("--run", type=Path, required=True, help="folder train_judge.py wrote")
    parser.add_argument("--frames", type=Path, default=Path("data/frames"))
    parser.add_argument("--overlays", type=int, default=0, help="heat-map examples per group")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    training = json.loads((args.run / "report.json").read_text())["training"]
    detector = FineTunedDetector.load(args.run / "judge.pt", args.device)
    generator = MoireRecaptureGenerator(MoireConfig(seed=training["config"]["seed"]))

    summary = {}
    for name, images in groups(load_partitions(args.frames), training["corpus"], generator).items():
        maps = detector.grad_cam(images)
        summary[name] = summarise(detector.score_batch(images), maps)
        for index in np.linspace(0, len(images) - 1, args.overlays).round().astype(int):
            path = args.run / "grad-cam" / name.replace(" ", "-") / f"{index:05d}.png"
            write_image(path, overlay(images[index], maps[index]))
        print(
            f"{name:40s} median score {summary[name]['median_score']:.4f}  "
            f"attended {summary[name]['attended_crops']:4d}/{len(images):4d}  "
            f"face share {summary[name].get('face_share_mean', float('nan')):.3f}"
        )
    (args.run / "attention.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
