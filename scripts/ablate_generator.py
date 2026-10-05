import argparse
import dataclasses
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from synthetic_pad.corpora import BONA_FIDE
from synthetic_pad.frames import load_partitions, read_crops, select
from synthetic_pad.generators.moire import MoireConfig, MoireRecaptureGenerator
from synthetic_pad.judge.finetuned import FineTunedDetector

STAGE_OFF = {
    "geometry": {"tilt_x": 0.0, "tilt_y": 0.0, "rotation_deg": 0.0},
    "tone": {"contrast": 1.0, "saturation": 1.0, "black_level": 0.0, "white_level": 1.0},
    "glare": {"glare_intensity": 0.0},
    "lens": {"blur_sigma_px": 0.0},
    "moire": {"moire_amplitude": 0.0},
    "screen": {"screen_min_pitch_px": 0.0, "screen_max_pitch_px": 0.0},
    "noise": {"noise_sigma_levels": 0.0, "noise_to_bona_fide": 0.0},
}


def switched_off(config: MoireConfig, stages: Iterable[str]) -> MoireConfig:
    """The configuration with the given stages disabled. A field the configuration does
    not have is skipped, so the first generator, which had no saturation control, works too."""
    known = {field.name for field in dataclasses.fields(config)}
    neutral: dict[str, Any] = {
        name: value for stage in stages for name, value in STAGE_OFF[stage].items() if name in known
    }
    return dataclasses.replace(config, **neutral)


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
    tested = select(load_partitions(args.frames)[training["corpus"]], "test")
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
