import argparse
import json
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from synthetic_pad.calibration import read_fitted_config
from synthetic_pad.evaluation.report import pad_report
from synthetic_pad.frames import CORPORA, CropSample, load_partitions, read_crops, select
from synthetic_pad.generators.moire import MoireConfig, MoireRecaptureGenerator
from synthetic_pad.judge.resnet import (
    ARCHITECTURES,
    Scores,
    TrainingConfig,
    train_resnet,
)

TRAINING_DATA = ("real", "synthetic", "mixed")
COPY_SEED_STRIDE = 1000
DEFAULTS = TrainingConfig()


def labels(samples: list[CropSample]) -> np.ndarray:
    return np.array([sample.is_attack for sample in samples])


def training_set(
    samples: list[CropSample],
    data: str,
    generators: list[MoireRecaptureGenerator],
    noise_real_crops: bool,
) -> tuple[np.ndarray, np.ndarray]:
    bona_fide = read_crops([sample for sample in samples if not sample.is_attack])
    real = [bona_fide]
    if data in ("real", "mixed"):
        real.append(read_crops([sample for sample in samples if sample.is_attack]))
    synthetic = []
    if data in ("synthetic", "mixed"):
        synthetic = [
            np.stack([generator.generate(image) for image in bona_fide]) for generator in generators
        ]
    if noise_real_crops:
        noise = generators[0].add_sensor_noise
        real = [np.stack([noise(image) for image in group]) for group in real]
    images = np.concatenate([*real, *synthetic])
    is_attack = np.arange(len(images)) >= len(bona_fide)
    return images, is_attack


def report_on(
    development: tuple[list[CropSample], Scores], evaluation: tuple[list[CropSample], Scores]
) -> dict[str, object]:
    development_samples, development_scores = development
    evaluation_samples, evaluation_scores = evaluation
    development_is_attack = labels(development_samples)
    by_species: dict[str, list[float]] = defaultdict(list)
    for sample, score in zip(evaluation_samples, evaluation_scores, strict=True):
        if sample.is_attack:
            by_species[sample.species].append(float(score))
    return pad_report(
        development_scores[development_is_attack],
        development_scores[~development_is_attack],
        {species: np.array(values) for species, values in sorted(by_species.items())},
        evaluation_scores[~labels(evaluation_samples)],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a ResNet judge on the crops of one corpus; test it on both."
    )
    parser.add_argument("--train-corpus", choices=CORPORA, required=True)
    parser.add_argument(
        "--data",
        choices=TRAINING_DATA,
        default="real",
        help="attacks the judge trains on: the corpus's own, synthetic moire, or both",
    )
    parser.add_argument(
        "--generator-config",
        type=Path,
        help="fitted moire settings from calibrate_generator.py --fit; defaults otherwise",
    )
    parser.add_argument(
        "--noise-real-crops",
        action="store_true",
        help="add the generator's sensor noise to every real training crop as well, so that "
        "the noise level cannot tell a synthetic attack from a real crop",
    )
    parser.add_argument(
        "--synthetic-per-face",
        type=int,
        default=1,
        help="synthetic attacks rendered from each bona fide training crop, each with its own "
        "realization of the artifacts",
    )
    parser.add_argument("--frames", type=Path, default=Path("data/frames"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--architecture", choices=sorted(ARCHITECTURES), default="resnet18")
    parser.add_argument("--epochs", type=int, default=DEFAULTS.epochs)
    parser.add_argument("--seed", type=int, default=DEFAULTS.seed)
    parser.add_argument("--no-pretrained", action="store_true", help="random initial weights")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.synthetic_per_face < 1:
        parser.error(f"--synthetic-per-face must be at least 1, got {args.synthetic_per_face}")
    config = TrainingConfig(
        architecture=args.architecture,
        pretrained=not args.no_pretrained,
        epochs=args.epochs,
        seed=args.seed,
    )
    partitions = load_partitions(args.frames)
    source = partitions[args.train_corpus]
    target_corpus = next(corpus for corpus in CORPORA if corpus != args.train_corpus)
    fitted = read_fitted_config(args.generator_config) if args.generator_config else MoireConfig()
    generators = [
        MoireRecaptureGenerator(replace(fitted, seed=args.seed + copy * COPY_SEED_STRIDE))
        for copy in range(args.synthetic_per_face)
    ]
    images, is_attack = training_set(
        select(source, "train"), args.data, generators, args.noise_real_crops
    )

    detector, losses = train_resnet(images, is_attack, config, args.device)
    detector.save(args.output / "judge.pt")

    scored: dict[str, tuple[list[CropSample], Scores]] = {}
    for name, samples in (
        ("development", select(source, "devel")),
        ("intra", select(source, "test")),
        ("cross", select(partitions[target_corpus], "test")),
    ):
        scored[name] = (samples, detector.score_batch(read_crops(samples)))

    report = {
        "training": {
            "corpus": args.train_corpus,
            "data": args.data,
            "generator": (
                dict(generators[0].describe())
                if args.data != "real" or args.noise_real_crops
                else None
            ),
            "noise_real_crops": args.noise_real_crops,
            "synthetic_per_face": args.synthetic_per_face,
            "config": asdict(config),
            "device": args.device,
            "torch": torch.__version__,
            "epoch_losses": losses,
            "bona_fide_crops": int(np.sum(~is_attack)),
            "attack_crops": int(np.sum(is_attack)),
            "msu_development_subjects": sorted(
                {s.subject for s in partitions["msu"] if s.partition == "devel"}
            ),
        },
        "intra": {"corpus": args.train_corpus, **report_on(scored["development"], scored["intra"])},
        "cross": {"corpus": target_corpus, **report_on(scored["development"], scored["cross"])},
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "scores.json").write_text(
        json.dumps(
            {
                name: [
                    {"image": str(s.image), "species": s.species, "score": float(score)}
                    for s, score in zip(samples, scores, strict=True)
                ]
                for name, (samples, scores) in scored.items()
            }
        )
        + "\n"
    )
    for name in ("intra", "cross"):
        overall = report[name]["all_attacks"]
        print(f"{name:5s} HTER {overall['hter']:.3f}  AUC {overall['auc']:.3f}")


if __name__ == "__main__":
    main()
