import argparse
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
from matplotlib.figure import Figure

from synthetic_pad.dataset import read_image
from synthetic_pad.domain import RgbImage
from synthetic_pad.frames import face_crop
from synthetic_pad.generators.moire import (
    DRAWN_FRINGE_STAGES,
    MoireRecaptureGenerator,
    noise_level,
    read_fitted_config,
    switched_off,
)
from synthetic_pad.generators.spectral import SpectralTransferGenerator

SCENE = 1
DRAWS = 3
STAGE_TITLES = ("Geometry", "+ Tone", "+ Glare", "+ Lens blur", "+ Moiré", "+ Noise")
ZOOM_TOP = 80
ZOOM_LEFT = 56
ZOOM_SIDE = 56
ZOOM_FACTOR = 4
DETAIL_BLUR_SIGMA_PX = 1.0
DETAIL_GAIN = 8.0
MID_GREY = 128.0
FIGURE_WIDTH_IN = 6.3
TITLE_HEIGHT_IN = 0.25
INK = "#52514e"

type Panel = tuple[str, RgbImage]


def grey_levels(image: RgbImage) -> np.ndarray:
    return np.asarray(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), dtype=np.float32)


def aligned(image: RgbImage, reference: RgbImage) -> RgbImage:
    """The image shifted onto the reference, by phase correlation of their grey levels."""
    (shift_x, shift_y), _ = cv2.phaseCorrelate(grey_levels(reference), grey_levels(image))
    height, width = image.shape[:2]
    shift = np.array([[1, 0, -shift_x], [0, 1, -shift_y]], dtype=np.float32)
    moved = cv2.warpAffine(image, shift, (width, height), borderMode=cv2.BORDER_REFLECT)
    return np.asarray(moved, dtype=np.uint8)


def zoomed(image: RgbImage) -> RgbImage:
    patch = image[ZOOM_TOP : ZOOM_TOP + ZOOM_SIDE, ZOOM_LEFT : ZOOM_LEFT + ZOOM_SIDE]
    size = (ZOOM_SIDE * ZOOM_FACTOR, ZOOM_SIDE * ZOOM_FACTOR)
    return np.asarray(cv2.resize(patch, size, interpolation=cv2.INTER_NEAREST), dtype=np.uint8)


def fine_detail(image: RgbImage) -> RgbImage:
    """Grey levels minus a blur of them, amplified around mid-grey, so noise shows as grain."""
    grey = grey_levels(image).astype(np.float64)
    detail = grey - cv2.GaussianBlur(grey, (0, 0), DETAIL_BLUR_SIGMA_PX)
    shown = np.clip(MID_GREY + DETAIL_GAIN * detail, 0, 255).astype(np.uint8)
    return np.repeat(shown[..., None], 3, axis=2)


def first_generator_draws(crop: RgbImage, checkout: Path) -> list[RgbImage]:
    """Draws of the first generator, rendered by its own code at the tag first-generator."""
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is needed to run the first generator from its checkout")
    script = Path(__file__).with_name("first_generator_draws.py")
    with tempfile.TemporaryDirectory() as folder:
        source, drawn = Path(folder) / "crop.png", Path(folder) / "draws.npy"
        cv2.imwrite(str(source), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        command = [uv, "run", "--project", str(checkout), "python", str(script)]
        arguments = ["--input", str(source), "--draws", str(DRAWS), "--output", str(drawn)]
        subprocess.run([*command, *arguments], check=True)
        return list(np.load(drawn))


def draw_grid(rows: Sequence[Sequence[Panel]], path: Path) -> None:
    columns = len(rows[0])
    side = FIGURE_WIDTH_IN / columns
    height = len(rows) * (side + TITLE_HEIGHT_IN)
    figure = Figure(figsize=(FIGURE_WIDTH_IN, height), layout="constrained")
    axes = figure.subplots(len(rows), columns, squeeze=False)
    for row_axes, row in zip(axes, rows, strict=True):
        for axis, (title, image) in zip(row_axes, row, strict=True):
            axis.imshow(image, interpolation="nearest")
            axis.set_title(title, fontsize=8, color=INK)
            axis.set_xticks([])
            axis.set_yticks([])
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    print(f"wrote {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Example figures of the generator on the face of a bona fide portrait: its "
        "stages one by one, the first generator against the calibrated one, the camera noise "
        "they add, and the alternatives of the third iteration, beside a photograph of the "
        "portrait on a screen as a real replay."
    )
    parser.add_argument("--input", type=Path, required=True, help="bona fide portrait")
    parser.add_argument(
        "--replay-photo",
        type=Path,
        required=True,
        help="photograph of the portrait shown on a screen; also the replay the spectral "
        "transfer takes its spectrum from",
    )
    parser.add_argument(
        "--mirror-replay-photo",
        action="store_true",
        help="flip the photograph left to right, as a front camera takes it",
    )
    parser.add_argument(
        "--first-generator-checkout",
        type=Path,
        default=Path("../first-generator"),
        help="worktree of the tag first-generator",
    )
    parser.add_argument(
        "--fringes-config",
        type=Path,
        default=Path("experiments/moire-fitted-to-replay-mobile-matched-noise.json"),
    )
    parser.add_argument(
        "--fixed-noise-config",
        type=Path,
        default=Path("experiments/moire-fitted-to-replay-mobile.json"),
    )
    parser.add_argument(
        "--screen-config",
        type=Path,
        default=Path("experiments/moire-screen-fitted-to-replay-mobile.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/generator-preview"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    crop = face_crop(read_image(args.input))
    photo = face_crop(read_image(args.replay_photo))
    if args.mirror_replay_photo:
        photo = np.ascontiguousarray(photo[:, ::-1])
    photo = aligned(photo, crop)
    fringes = read_fitted_config(args.fringes_config)

    stages: list[Panel] = [("Bona fide", crop)]
    for index, title in enumerate(STAGE_TITLES):
        kept = switched_off(fringes, DRAWN_FRINGE_STAGES[index + 1 :])
        stages.append((title, MoireRecaptureGenerator(kept).generate(crop, scene=SCENE)))
    stages.append(("+ Noise, zoomed", zoomed(stages[-1][1])))
    draw_grid([stages[:4], stages[4:]], args.output_dir / "fig-generator-stages.pdf")

    first = first_generator_draws(crop, args.first_generator_checkout)
    calibrated = MoireRecaptureGenerator(fringes)
    evolution = [
        [("Bona fide", crop), *((f"First generator ({k})", d) for k, d in enumerate(first, 1))],
        [
            ("Real replay", photo),
            *(
                (f"Calibrated ({k})", calibrated.generate(crop, scene=k))
                for k in range(1, DRAWS + 1)
            ),
        ],
    ]
    draw_grid(evolution, args.output_dir / "fig-generator-evolution.pdf")

    fixed_noise = read_fitted_config(args.fixed_noise_config)
    noise_alone = switched_off(fixed_noise, DRAWN_FRINGE_STAGES[:-1])
    noisy: list[Panel] = [
        ("Bona fide", crop),
        ("Noise alone", MoireRecaptureGenerator(noise_alone).generate(crop, scene=SCENE)),
        ("Calibrated noise", MoireRecaptureGenerator(fixed_noise).generate(crop, scene=SCENE)),
        ("Matched noise", calibrated.generate(crop, scene=SCENE)),
    ]
    for title, image in noisy:
        print(f"{title}: noise level {noise_level(image):.2f} grey levels")
    noise_rows = [
        [(title, zoomed(image)) for title, image in noisy],
        [(f"{title}, detail", zoomed(fine_detail(image))) for title, image in noisy],
    ]
    draw_grid(noise_rows, args.output_dir / "fig-noise-shortcut.pdf")

    screen = MoireRecaptureGenerator(read_fitted_config(args.screen_config))
    alternatives: list[Panel] = [
        ("Bona fide", crop),
        ("Drawn fringes", calibrated.generate(crop, scene=SCENE)),
        ("Simulated screen", screen.generate(crop, scene=SCENE)),
        ("Spectral transfer", SpectralTransferGenerator(np.stack([photo])).generate(crop)),
    ]
    alternative_rows = [
        alternatives,
        [(f"{title}, zoomed", zoomed(image)) for title, image in alternatives],
    ]
    draw_grid(alternative_rows, args.output_dir / "fig-generator-alternatives.pdf")


if __name__ == "__main__":
    main()
