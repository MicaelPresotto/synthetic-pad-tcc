import argparse
import zlib
from pathlib import Path

import cv2
import numpy as np
from figure_style import FIGURE_WIDTH_IN, GRID, INK, SERIES_COLOURS
from matplotlib.figure import Figure

from synthetic_pad.dataset import read_image
from synthetic_pad.domain import RgbImage
from synthetic_pad.generators.moire import MoireRecaptureGenerator, read_fitted_config

SWAY_PX = 4.0
SHOWN_FRAMES = 4
TRACE_HEIGHT_IN = 1.6
SERIES = (
    ("Bona fide", INK),
    ("One screen", SERIES_COLOURS[0]),
    ("Screen per frame", SERIES_COLOURS[1]),
)


def swayed(image: RgbImage, frames: int) -> list[RgbImage]:
    """The image shifted sideways by up to SWAY_PX pixels, one sway over the clip, as a head
    that moves a little in front of the camera."""
    height, width = image.shape[:2]
    clip = []
    for index in range(frames):
        shift = SWAY_PX * np.sin(2.0 * np.pi * index / frames)
        moved = cv2.warpAffine(
            image,
            np.array([[1, 0, shift], [0, 1, 0]], dtype=np.float32),
            (width, height),
            borderMode=cv2.BORDER_REFLECT,
        )
        clip.append(np.asarray(moved, dtype=np.uint8))
    return clip


def centre_brightness(clip: list[RgbImage]) -> np.ndarray:
    """Mean grey level of the central third of each frame."""
    height, width = clip[0].shape[:2]
    centre = (slice(height // 3, 2 * height // 3), slice(width // 3, 2 * width // 3))
    return np.array([frame[centre].mean() for frame in clip])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="A bona fide portrait swayed into a short clip and recaptured as a video "
        "replay in two ways: through one screen for the whole clip, and through a new screen in "
        "every frame. Draws the first frames of each and the brightness of every frame."
    )
    parser.add_argument("--input", type=Path, required=True, help="bona fide portrait")
    parser.add_argument(
        "--generator-config",
        type=Path,
        default=Path("experiments/moire-fitted-to-replay-mobile-matched-noise.json"),
    )
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--output", type=Path, default=Path("results/video_replay_preview.pdf"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    generator = MoireRecaptureGenerator(read_fitted_config(args.generator_config))
    bona_fide = swayed(read_image(args.input), args.frames)
    scene = zlib.crc32(args.input.name.encode())
    clips = (
        bona_fide,
        [generator.generate(frame, scene=scene) for frame in bona_fide],
        [generator.generate(frame) for frame in bona_fide],
    )

    row_height = FIGURE_WIDTH_IN / SHOWN_FRAMES
    figure = Figure(
        figsize=(FIGURE_WIDTH_IN, 2 * row_height + TRACE_HEIGHT_IN), layout="constrained"
    )
    grid = figure.add_gridspec(3, SHOWN_FRAMES, height_ratios=(row_height, row_height, 1.4))
    for row, ((title, _), clip) in enumerate(zip(SERIES[1:], clips[1:], strict=True)):
        for column in range(SHOWN_FRAMES):
            axis = figure.add_subplot(grid[row, column])
            axis.imshow(clip[column])
            axis.set_xticks([])
            axis.set_yticks([])
            if column == 0:
                axis.set_ylabel(title, fontsize=8, color=INK)
            if row == 0:
                axis.set_title(f"Frame {column + 1}", fontsize=8, color=INK)
    trace = figure.add_subplot(grid[2, :])
    for (title, colour), clip in zip(SERIES, clips, strict=True):
        brightness = centre_brightness(clip)
        trace.plot(np.arange(1, args.frames + 1), brightness, color=colour, linewidth=1.5)
        trace.annotate(
            title,
            (args.frames, brightness[-1]),
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=7,
            color=colour,
            va="center",
        )
        steps = np.abs(np.diff(brightness))
        print(f"{title}: mean change between frames {steps.mean():.2f} grey levels")
    trace.set_xlabel("Frame", fontsize=8, color=INK)
    trace.set_ylabel("Center brightness", fontsize=8, color=INK)
    trace.tick_params(labelsize=7, colors=INK)
    trace.grid(color=GRID, linewidth=0.5)
    for spine in trace.spines.values():
        spine.set_color(GRID)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
