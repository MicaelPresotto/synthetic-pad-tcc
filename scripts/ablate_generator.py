import argparse
import json
from pathlib import Path

import numpy as np
import torch

from synthetic_pad.corpora import BONA_FIDE, TEST
from synthetic_pad.frames import load_partitions, read_crops, select
from synthetic_pad.generators.moire import (
    STAGE_OFF,
    MoireConfig,
    MoireRecaptureGenerator,
    switched_off,
)
from synthetic_pad.judge.finetuned import FineTunedDetector


def variants(config: MoireConfig) -> dict[str, MoireConfig]:
    found = {"every stage": config}
    for stage in STAGE_OFF:
        others = [other for other in STAGE_OFF if other != stage]
        found[f"every stage but {stage}"] = switched_off(config, [stage])
        found[f"{stage} alone"] = switched_off(config, others)
    return found


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Median score a trained judge gives to synthetic attacks rendered with "
        "each stage of its own generator switched on or off. A judge trained on real attacks "
        "only has no generator of its own and is shown the default one."
    )
    parser.add_argument("--run", type=Path, required=True, help="folder train_judge.py wrote")
    parser.add_argument("--frames", type=Path, default=Path("data/frames"))
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    training = json.loads((args.run / "report.json").read_text())["training"]
    recorded = training.get("generator", {})
    settings = {name: value for name, value in recorded.items() if name != "generator"}
    detector = FineTunedDetector.load(args.run / "judge.pt", args.device)
    tested = select(load_partitions(args.frames)[training["corpus"]], TEST)
    bona_fide = read_crops([sample for sample in tested if sample.species == BONA_FIDE])

    medians = {}
    for name, config in variants(MoireConfig(**settings)).items():
        generator = MoireRecaptureGenerator(config)
        attacks = np.stack([generator.generate(image) for image in bona_fide])
        medians[name] = float(np.median(detector.score_batch(attacks)))
        print(f"{name:28s} median score {medians[name]:.4f}")
    summary = {"crops": len(bona_fide), "median_score": medians}
    (args.run / "ablation.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
