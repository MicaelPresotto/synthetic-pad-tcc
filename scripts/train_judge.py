import argparse
import json
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from synthetic_pad.calibration import read_fitted_config
from synthetic_pad.corpora import REPLAY
from synthetic_pad.depth import read_depth_maps
from synthetic_pad.domain import AttackGenerator, RgbImage
from synthetic_pad.evaluation.report import pad_report
from synthetic_pad.frames import (
    CORPORA,
    MSU_SPLIT_SEED,
    CropSample,
    load_partitions,
    read_crops,
    select,
)
from synthetic_pad.generators.moire import MoireConfig, MoireRecaptureGenerator
from synthetic_pad.generators.spectral import SpectralConfig, SpectralTransferGenerator
from synthetic_pad.judge.finetuned import (
    ARCHITECTURES,
    DEPTH_STAGES,
    Scores,
    TrainingConfig,
    train_detector,
)

TRAINING_DATA = ("real", "synthetic", "mixed")
COPY_SEED_STRIDE = 1000
DEFAULTS = TrainingConfig()


def labels(samples: list[CropSample]) -> np.ndarray:
    return np.array([sample.is_attack for sample in samples])


def fraction_of(count: int, fraction: float, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).choice(count, round(fraction * count), replace=False)


def training_set(
    samples: list[CropSample],
    data: str,
    generators: Sequence[AttackGenerator],
    real_crop_noise: Callable[[RgbImage], RgbImage] | None,
    synthetic_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    bona_fide = read_crops([sample for sample in samples if not sample.is_attack])
    real = [bona_fide]
    if data in ("real", "mixed"):
        real.append(read_crops([sample for sample in samples if sample.is_attack]))
    synthetic = []
    if data in ("synthetic", "mixed"):
        sources = bona_fide[sorted(fraction_of(len(bona_fide), synthetic_fraction, seed))]
        synthetic = [
            np.stack([generator.generate(image) for image in sources]) for generator in generators
        ]
    if real_crop_noise is not None:
        real = [np.stack([real_crop_noise(image) for image in group]) for group in real]
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
        description="Fine-tune a judge on the crops of one corpus; test it on both."
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
        "--spectral-band",
        type=float,
        help="render the synthetic attacks by carrying the low-frequency amplitude spectrum of "
        "the corpus's real training replays, up to this many cycles per pixel, onto the bona "
        "fide crops, instead of with the moire generator",
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
    parser.add_argument(
        "--synthetic-fraction",
        type=float,
        default=1.0,
        help="fraction of the bona fide training crops, drawn at random, that synthetic "
        "attacks are rendered from; below 1 the judge sees fewer synthetic than bona fide",
    )
    parser.add_argument(
        "--depth-maps",
        type=Path,
        help="supervise the judge with a depth map instead of one label per crop: the "
        "pseudo-depth that estimate_depth.py wrote to this directory for each bona fide crop, "
        "and a flat map for every attack",
    )
    parser.add_argument(
        "--depth-auxiliary",
        action="store_true",
        help="with --depth-maps, keep the single attack label and learn the depth map beside it "
        "as an auxiliary task; the score still comes from the label",
    )
    parser.add_argument(
        "--depth-stage",
        choices=DEPTH_STAGES,
        default=DEFAULTS.depth_stage,
        help="with --depth-auxiliary, the ResNet stage the depth head reads: layer4 draws a "
        "7 x 7 map of a 224 px crop, layer3 a 14 x 14 one; the depth maps must match it",
    )
    parser.add_argument(
        "--depth-weight",
        type=float,
        default=DEFAULTS.depth_weight,
        help="with --depth-auxiliary, the weight of the depth loss against the binary one",
    )
    parser.add_argument("--frames", type=Path, default=Path("data/frames"))
    parser.add_argument(
        "--msu-split-seed",
        type=int,
        default=MSU_SPLIT_SEED,
        help="seed that draws the MSU-MFSD training subjects held out as its development set",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--architecture", choices=sorted(ARCHITECTURES), default="resnet18")
    parser.add_argument("--epochs", type=int, default=DEFAULTS.epochs)
    parser.add_argument("--seed", type=int, default=DEFAULTS.seed)
    parser.add_argument("--no-pretrained", action="store_true", help="random initial weights")
    parser.add_argument(
        "--mixstyle",
        action="store_true",
        help="mix the channel statistics of pairs of training crops after the early "
        "convolutional stages (MixStyle), so that the judge cannot rely on their style",
    )
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.synthetic_per_face < 1:
        parser.error(f"--synthetic-per-face must be at least 1, got {args.synthetic_per_face}")
    if not 0 < args.synthetic_fraction <= 1:
        parser.error(f"--synthetic-fraction must be in (0, 1], got {args.synthetic_fraction}")
    if args.depth_auxiliary and args.depth_maps is None:
        parser.error("--depth-auxiliary needs --depth-maps")
    if args.spectral_band is not None and (args.generator_config or args.noise_real_crops):
        parser.error("--spectral-band replaces the moire generator and its sensor noise")
    config = TrainingConfig(
        architecture=args.architecture,
        pretrained=not args.no_pretrained,
        epochs=args.epochs,
        mixstyle=args.mixstyle,
        depth_supervision=args.depth_maps is not None and not args.depth_auxiliary,
        depth_auxiliary=args.depth_auxiliary,
        depth_stage=args.depth_stage,
        depth_weight=args.depth_weight,
        seed=args.seed,
    )
    partitions = load_partitions(args.frames, args.msu_split_seed)
    source = partitions[args.train_corpus]
    target_corpus = next(corpus for corpus in CORPORA if corpus != args.train_corpus)
    copy_seeds = [args.seed + copy * COPY_SEED_STRIDE for copy in range(args.synthetic_per_face)]
    generators: list[AttackGenerator]
    real_crop_noise = None
    if args.spectral_band is None:
        fitted = (
            read_fitted_config(args.generator_config) if args.generator_config else MoireConfig()
        )
        moire = [MoireRecaptureGenerator(replace(fitted, seed=seed)) for seed in copy_seeds]
        generators = list(moire)
        real_crop_noise = moire[0].add_sensor_noise if args.noise_real_crops else None
    else:
        replays = read_crops([s for s in select(source, "train") if s.species == REPLAY])
        generators = [
            SpectralTransferGenerator(replays, SpectralConfig(args.spectral_band, seed))
            for seed in copy_seeds
        ]
    images, is_attack = training_set(
        select(source, "train"),
        args.data,
        generators,
        real_crop_noise,
        args.synthetic_fraction,
        args.seed,
    )

    depth_targets = None
    if args.depth_maps is not None:
        bona_fide = [sample for sample in select(source, "train") if not sample.is_attack]
        maps = read_depth_maps(
            args.depth_maps / f"{args.train_corpus}.npz", bona_fide, args.frames / args.train_corpus
        )
        flat = np.zeros((int(is_attack.sum()), *maps.shape[1:]), dtype=np.float32)
        depth_targets = np.concatenate([maps, flat])

    detector, losses = train_detector(images, is_attack, config, args.device, depth_targets)
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
            "synthetic_fraction": args.synthetic_fraction,
            "depth_maps": str(args.depth_maps) if args.depth_maps else None,
            "config": asdict(config),
            "device": args.device,
            "torch": torch.__version__,
            "epoch_losses": losses,
            "bona_fide_crops": int(np.sum(~is_attack)),
            "attack_crops": int(np.sum(is_attack)),
            "msu_split_seed": args.msu_split_seed,
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
