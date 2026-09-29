import argparse
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from synthetic_pad.dataset import read_image
from synthetic_pad.evaluation.report import pad_report
from synthetic_pad.frames import CropSample, hold_out_subjects, load_crops
from synthetic_pad.generators.moire import MoireConfig, MoireRecaptureGenerator
from synthetic_pad.judge.resnet import (
    ARCHITECTURES,
    Scores,
    TrainingConfig,
    train_resnet,
)

CORPORA = ("msu", "replay-mobile")
TRAINING_DATA = ("real", "synthetic", "mixed")
MSU_DEVELOPMENT_SUBJECTS = 5
MSU_SPLIT_SEED = 20260928
DEFAULTS = TrainingConfig()


def load_partitions(frames_root: Path) -> dict[str, list[CropSample]]:
    msu = hold_out_subjects(
        load_crops(frames_root / "msu"), "train", "devel", MSU_DEVELOPMENT_SUBJECTS, MSU_SPLIT_SEED
    )
    return {"msu": msu, "replay-mobile": load_crops(frames_root / "replay-mobile")}


def select(samples: list[CropSample], partition: str) -> list[CropSample]:
    return [sample for sample in samples if sample.partition == partition]


def pixels(samples: list[CropSample]) -> np.ndarray:
    return np.stack([read_image(sample.image) for sample in samples])


def labels(samples: list[CropSample]) -> np.ndarray:
    return np.array([sample.is_attack for sample in samples])


def training_set(
    samples: list[CropSample], data: str, generator: MoireRecaptureGenerator
) -> tuple[np.ndarray, np.ndarray]:
    bona_fide = pixels([sample for sample in samples if not sample.is_attack])
    attacks = []
    if data in ("real", "mixed"):
        attacks.append(pixels([sample for sample in samples if sample.is_attack]))
    if data in ("synthetic", "mixed"):
        attacks.append(np.stack([generator.generate(image) for image in bona_fide]))
    images = np.concatenate([bona_fide, *attacks])
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
    parser.add_argument("--frames", type=Path, default=Path("data/frames"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--architecture", choices=sorted(ARCHITECTURES), default="resnet18")
    parser.add_argument("--epochs", type=int, default=DEFAULTS.epochs)
    parser.add_argument("--seed", type=int, default=DEFAULTS.seed)
    parser.add_argument("--no-pretrained", action="store_true", help="random initial weights")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = TrainingConfig(
        architecture=args.architecture,
        pretrained=not args.no_pretrained,
        epochs=args.epochs,
        seed=args.seed,
    )
    partitions = load_partitions(args.frames)
    source = partitions[args.train_corpus]
    target_corpus = next(corpus for corpus in CORPORA if corpus != args.train_corpus)
    generator = MoireRecaptureGenerator(MoireConfig(seed=args.seed))
    images, is_attack = training_set(select(source, "train"), args.data, generator)

    detector, losses = train_resnet(images, is_attack, config, args.device)
    detector.save(args.output / "judge.pt")

    scored: dict[str, tuple[list[CropSample], Scores]] = {}
    for name, samples in (
        ("development", select(source, "devel")),
        ("intra", select(source, "test")),
        ("cross", select(partitions[target_corpus], "test")),
    ):
        scored[name] = (samples, detector.score_batch(pixels(samples)))

    report = {
        "training": {
            "corpus": args.train_corpus,
            "data": args.data,
            "generator": dict(generator.describe()) if args.data != "real" else None,
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
