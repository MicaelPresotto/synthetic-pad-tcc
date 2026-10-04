import argparse
import json
import math
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from synthetic_pad.calibration import (
    CHROMA_BANDS,
    appearance,
    appearance_distance,
    read_fitted_config,
)
from synthetic_pad.corpora import BONA_FIDE, REPLAY
from synthetic_pad.frames import CORPORA, CropSample, load_partitions, read_crops, select
from synthetic_pad.generators.moire import MoireConfig, MoireRecaptureGenerator

GENERATED = "generated"
SEARCH_SPACE = {
    "contrast": (0.8, 2.0),
    "saturation": (0.8, 1.6),
    "black_level": (0.0, 0.1),
    "white_level": (0.85, 1.0),
    "blur_sigma_px": (0.0, 1.0),
    "moire_amplitude": (0.0, 0.15),
    "noise_sigma_levels": (0.0, 4.0),
}
LOWEST_FRINGE_RANGE = (2.0, 16.0)
FRINGE_SPAN = MoireConfig().moire_max_cycles_per_width / MoireConfig().moire_min_cycles_per_width
SCREEN_SEARCH_SPACE = {
    "camera_psf_sigma_px": (0.3, 1.2),
    "camera_px_per_output_px": (1.0, 3.0),
    "camera_sharpening": (0.0, 2.5),
    "screen_black_matrix": (0.0, 0.6),
    "screen_max_rotation_deg": (0.0, 10.0),
}
FINEST_PITCH_RANGE = (0.7, 2.5)
PITCH_SPAN = 1.5

type Drawer = Callable[[np.random.Generator, MoireConfig], MoireConfig]


def sample_by_species(
    samples: list[CropSample], crops: int, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    grouped = {}
    for species in sorted({sample.species for sample in samples}):
        of_species = [sample for sample in samples if sample.species == species]
        chosen = rng.choice(len(of_species), min(crops, len(of_species)), replace=False)
        grouped[species] = read_crops([of_species[index] for index in sorted(chosen)])
    return grouped


def render(config: MoireConfig, bona_fide: np.ndarray) -> np.ndarray:
    generator = MoireRecaptureGenerator(config)
    return np.stack([generator.generate(image) for image in bona_fide])


def draw_candidate(rng: np.random.Generator, base: MoireConfig) -> MoireConfig:
    drawn = {name: float(rng.uniform(low, high)) for name, (low, high) in SEARCH_SPACE.items()}
    lowest = math.exp(rng.uniform(*np.log(LOWEST_FRINGE_RANGE)))
    highest = lowest * FRINGE_SPAN
    fields: dict[str, Any] = asdict(base)
    fields |= drawn
    fields |= {"moire_min_cycles_per_width": lowest, "moire_max_cycles_per_width": highest}
    return MoireConfig(**fields)


def draw_screen_candidate(rng: np.random.Generator, base: MoireConfig) -> MoireConfig:
    """The screen and camera settings, on top of a configuration whose other stages are kept."""
    drawn: dict[str, Any] = {
        name: float(rng.uniform(low, high)) for name, (low, high) in SCREEN_SEARCH_SPACE.items()
    }
    finest = math.exp(rng.uniform(*np.log(FINEST_PITCH_RANGE)))
    return replace(
        base,
        moire_amplitude=0.0,
        screen_min_pitch_px=finest,
        screen_max_pitch_px=finest * PITCH_SPAN,
        **drawn,
    )


def fit(
    corpus: str,
    groups: dict[str, np.ndarray],
    trials: int,
    rng: np.random.Generator,
    base: MoireConfig,
    draw: Drawer,
    base_competes: bool,
) -> dict[str, object]:
    """Random search for the settings whose attacks look most like the corpus's replays.

    The fringe band moves but keeps its width: the statistics compared are means, which a
    single fringe frequency can match, and a single frequency is what a judge learns as a
    shortcut.
    """
    target = appearance(groups[REPLAY])
    default_distance = appearance_distance(appearance(render(base, groups[BONA_FIDE])), target)
    best, best_distance = (base, default_distance) if base_competes else (base, math.inf)
    for trial in range(trials):
        candidate = draw(rng, base)
        distance = appearance_distance(appearance(render(candidate, groups[BONA_FIDE])), target)
        if distance < best_distance:
            best, best_distance = candidate, distance
            print(f"  trial {trial:3d}  distance {distance:.4f}")
    return {
        "fitted_to": f"{corpus} training replays",
        "trials": trials,
        "distance": best_distance,
        "default_distance": default_distance,
        "config": asdict(best),
    }


def report(corpus: str, groups: dict[str, np.ndarray], config: MoireConfig) -> dict[str, object]:
    groups = {**groups, GENERATED: render(config, groups[BONA_FIDE])}
    measured = {group: appearance(images) for group, images in groups.items()}
    reference = measured[BONA_FIDE]
    tone = {
        group: {name: m.tone[name] / reference.tone[name] for name in reference.tone}
        for group, m in measured.items()
    }
    colour = {
        group: (m.chroma_power / reference.chroma_power).round(3).tolist()
        for group, m in measured.items()
    }
    print(f"\n{corpus}, training partition, relative to bona fide")
    for group, ratios in tone.items():
        print(f"  {group:10s} " + "  ".join(f"{name} {v:.2f}" for name, v in ratios.items()))
    print(f"  chroma power by band of cycles per width, from {CHROMA_BANDS}")
    for group, powers in colour.items():
        print(f"  {group:10s} " + " ".join(f"{v:5.2f}" for v in powers))
    distance = appearance_distance(measured[GENERATED], measured[REPLAY])
    print(f"  distance from generated to real replay {distance:.4f}")
    return {"tone": tone, "chroma_power": colour, "distance_to_replay": distance}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare generated attacks with the real ones of each training partition, "
        "or fit the generator to the replays of one."
    )
    parser.add_argument("--frames", type=Path, default=Path("data/frames"))
    parser.add_argument("--output", type=Path, default=Path("results/calibration.json"))
    parser.add_argument("--crops", type=int, default=300, help="crops sampled per species")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--generator-config", type=Path, help="fitted settings to report on")
    parser.add_argument("--fit", choices=CORPORA, help="search settings for this corpus alone")
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument(
        "--screen",
        action="store_true",
        help="with --fit, search the screen and camera settings in place of the sinusoidal "
        "fringes, keeping every other setting of --generator-config",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rng = np.random.default_rng(args.seed)
    partitions = load_partitions(args.frames)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    base = read_fitted_config(args.generator_config) if args.generator_config else MoireConfig()
    if args.fit:
        groups = sample_by_species(select(partitions[args.fit], "train"), args.crops, rng)
        draw = draw_screen_candidate if args.screen else draw_candidate
        fitted = fit(args.fit, groups, args.trials, rng, base, draw, base_competes=not args.screen)
        if args.screen:
            fitted["fitted_to"] = f"{fitted['fitted_to']}, screen and camera stage"
        args.output.write_text(json.dumps(fitted, indent=2) + "\n")
        print(f"distance {fitted['distance']:.4f}, defaults {fitted['default_distance']:.4f}")
        return

    summary: dict[str, object] = {"generator": asdict(base), "bands": CHROMA_BANDS}
    for corpus in CORPORA:
        groups = sample_by_species(select(partitions[corpus], "train"), args.crops, rng)
        summary[corpus] = report(corpus, groups, base)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
