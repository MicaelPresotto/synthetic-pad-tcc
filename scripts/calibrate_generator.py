import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from synthetic_pad.calibration import CHROMA_BANDS, chroma_power, tone_statistics
from synthetic_pad.corpora import BONA_FIDE
from synthetic_pad.frames import CORPORA, load_partitions, read_crops, select
from synthetic_pad.generators.moire import MoireRecaptureGenerator

GENERATED = "generated"


def measure(images: np.ndarray) -> dict[str, object]:
    tones = [tone_statistics(image) for image in images]
    return {
        **{name: float(np.mean([tone[name] for tone in tones])) for name in tones[0]},
        "chroma_power": np.mean([chroma_power(image) for image in images], axis=0).tolist(),
    }


def relative_to(measured: dict[str, object], reference: dict[str, object]) -> dict[str, Any]:
    return {
        name: (np.array(value) / np.array(reference[name])).round(3).tolist()
        for name, value in measured.items()
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare generated attacks with the real ones of each training partition."
    )
    parser.add_argument("--frames", type=Path, default=Path("data/frames"))
    parser.add_argument("--output", type=Path, default=Path("results/calibration.json"))
    parser.add_argument("--crops", type=int, default=300, help="crops sampled per species")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rng = np.random.default_rng(args.seed)
    generator = MoireRecaptureGenerator()
    partitions = load_partitions(args.frames)
    report: dict[str, object] = {"generator": dict(generator.describe()), "bands": CHROMA_BANDS}
    for corpus in CORPORA:
        training = select(partitions[corpus], "train")
        groups = {}
        for species in sorted({sample.species for sample in training}):
            of_species = [sample for sample in training if sample.species == species]
            chosen = rng.choice(len(of_species), min(args.crops, len(of_species)), replace=False)
            groups[species] = read_crops([of_species[index] for index in sorted(chosen)])
        groups[GENERATED] = np.stack([generator.generate(image) for image in groups[BONA_FIDE]])

        measured = {group: measure(images) for group, images in groups.items()}
        relative = {group: relative_to(m, measured[BONA_FIDE]) for group, m in measured.items()}
        report[corpus] = {"measured": measured, "relative_to_bona_fide": relative}

        print(f"\n{corpus}, training partition, relative to bona fide")
        for group, ratios in relative.items():
            tones = "  ".join(
                f"{name} {value:.2f}" for name, value in ratios.items() if name != "chroma_power"
            )
            print(f"  {group:10s} {tones}")
        print(f"  chroma power by band of cycles per width, from {CHROMA_BANDS}")
        for group, ratios in relative.items():
            print(f"  {group:10s} " + " ".join(f"{value:5.2f}" for value in ratios["chroma_power"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
