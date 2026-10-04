import argparse
from pathlib import Path

import numpy as np

from synthetic_pad.calibration import noise_level, read_fitted_config
from synthetic_pad.frames import CORPORA, load_partitions, read_crops, select
from synthetic_pad.generators.moire import MoireRecaptureGenerator


def median_noise(images: np.ndarray) -> float:
    return float(np.median([noise_level(image) for image in images]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Median sensor noise of the training crops of each corpus, by species, and "
        "of what each fitted generator renders from its bona fide crops."
    )
    parser.add_argument("--frames", type=Path, default=Path("data/frames"))
    parser.add_argument("--experiments", type=Path, default=Path("experiments"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    partitions = load_partitions(args.frames)
    for corpus in CORPORA:
        training = select(partitions[corpus], "train")
        print(f"\n{corpus}, training partition, median noise in grey levels")
        for species in sorted({sample.species for sample in training}):
            crops = read_crops([sample for sample in training if sample.species == species])
            print(f"  real {species:31s} {median_noise(crops):5.2f}  ({len(crops)} crops)")
        bona_fide = read_crops([sample for sample in training if not sample.is_attack])
        for name in (f"moire-fitted-to-{corpus}", f"moire-fitted-to-{corpus}-without-noise"):
            generator = MoireRecaptureGenerator(
                read_fitted_config(args.experiments / f"{name}.json")
            )
            synthetic = np.stack([generator.generate(image) for image in bona_fide])
            noisy = np.stack([generator.add_sensor_noise(image) for image in bona_fide])
            print(
                f"  {name}: synthetic {median_noise(synthetic):5.2f}, bona fide with its noise "
                f"{median_noise(noisy):5.2f}"
            )


if __name__ == "__main__":
    main()
