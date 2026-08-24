#!/usr/bin/env python3
"""Render a before/after view of the moiré recapture generator, for visual tuning.

Two modes:

* Pair (default) — the bona fide image beside its recapture.
* Sweep (--sweep FIELD=START:STOP:STEPS) — a contact sheet walking one MoireConfig
  field across a range with everything else held fixed, which is the fast way to
  find a plausible value for a single artifact.

With no --input, a synthetic test chart is used. Its zone plate makes moiré immediately
visible, which is far more informative for tuning than a face, and it means the script
runs before any dataset is in place.

Examples:
    uv run python scripts/preview_moire.py
    uv run python scripts/preview_moire.py --input data/msu/real/001.png
    uv run python scripts/preview_moire.py --sweep moire_frequency=8:70:8
    uv run python scripts/preview_moire.py --moire-amplitude 0.2 --blur-sigma 1.4
"""

import argparse
import dataclasses
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from synthetic_pad.domain.image import RgbImage
from synthetic_pad.generators.moire import MoireConfig, MoireRecaptureGenerator

DEFAULT_OUTPUT = Path("results/moire_preview.png")
DEFAULT_CHART_SIZE = 224
LABEL_HEIGHT = 34
GUTTER = 12
BACKGROUND = 24


def build_test_chart(size: int = DEFAULT_CHART_SIZE) -> RgbImage:
    """Each region answers a different tuning question: the zone plate rises in
    frequency with radius, so it beats against any sampling grid and shows exactly
    where moiré sets in; the smooth lobe has no detail, so only the highlight and the
    lens blur show there; the grey ramp reveals the contrast squeeze and the lifted
    black; and the colour bars reveal the chromatic fringing.
    """
    height = width = size
    rows, cols = np.mgrid[0:height, 0:width].astype(np.float64)
    split = int(height * 0.68)
    midline = width // 2

    centre_y, centre_x = split / 2.0, midline / 2.0
    radius_sq = (rows - centre_y) ** 2 + (cols - centre_x) ** 2
    zone_plate = 0.5 + 0.5 * np.cos(radius_sq * (math.pi / (0.30 * width)))

    lobe_y, lobe_x = split * 0.45, midline * 1.5
    lobe_sigma = width * 0.16
    lobe = 0.30 + 0.55 * np.exp(
        -0.5 * (((rows - lobe_y) ** 2 + (cols - lobe_x) ** 2) / lobe_sigma**2)
    )

    ramp = np.linspace(0.0, 1.0, width, dtype=np.float64)
    bars = np.array(
        [
            (0.85, 0.20, 0.20),
            (0.20, 0.80, 0.30),
            (0.25, 0.35, 0.90),
            (0.90, 0.85, 0.25),
            (0.92, 0.92, 0.92),
            (0.10, 0.10, 0.10),
        ]
    )

    chart = np.zeros((height, width, 3), dtype=np.float64)
    chart[:split, :midline] = zone_plate[:split, :midline, None]
    chart[:split, midline:] = lobe[:split, midline:, None]

    ramp_end = split + int((height - split) * 0.5)
    chart[split:ramp_end] = ramp[None, :, None]
    bar_width = width // len(bars)
    for index, colour in enumerate(bars):
        start = index * bar_width
        stop = width if index == len(bars) - 1 else start + bar_width
        chart[ramp_end:, start:stop] = colour

    return (chart * 255.0).astype(np.uint8)


def load_image(path: Path) -> RgbImage:
    raw = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if raw is None:
        raise FileNotFoundError(f"could not read an image from {path}")
    return np.asarray(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB), dtype=np.uint8)


def label_panel(image: RgbImage, text: str) -> RgbImage:
    bar = np.full((LABEL_HEIGHT, image.shape[1], 3), BACKGROUND, dtype=np.uint8)
    cv2.putText(
        bar,
        text,
        (8, LABEL_HEIGHT - 11),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    return np.asarray(np.vstack([bar, image]), dtype=np.uint8)


def contact_sheet(panels: list[RgbImage], columns: int) -> RgbImage:
    panel_h, panel_w = panels[0].shape[:2]
    rows = math.ceil(len(panels) / columns)
    sheet = np.full(
        (rows * panel_h + (rows + 1) * GUTTER, columns * panel_w + (columns + 1) * GUTTER, 3),
        BACKGROUND,
        dtype=np.uint8,
    )
    for index, panel in enumerate(panels):
        row, column = divmod(index, columns)
        top = GUTTER + row * (panel_h + GUTTER)
        left = GUTTER + column * (panel_w + GUTTER)
        sheet[top : top + panel_h, left : left + panel_w] = panel
    return sheet


def parse_sweep(spec: str) -> tuple[str, list[float]]:
    field_name, _, span = spec.partition("=")
    parts = span.split(":")
    if not field_name or len(parts) != 3:
        raise ValueError(f"expected FIELD=START:STOP:STEPS, got {spec!r}")

    known = {f.name for f in dataclasses.fields(MoireConfig)} - {"seed"}
    if field_name not in known:
        raise ValueError(f"{field_name!r} is not a sweepable MoireConfig field")

    start, stop, steps = float(parts[0]), float(parts[1]), int(parts[2])
    if steps < 2:
        raise ValueError(f"a sweep needs at least 2 steps, got {steps}")
    is_integer_field = field_name == "moire_harmonics"
    values = [start + (stop - start) * i / (steps - 1) for i in range(steps)]
    return field_name, [round(v) if is_integer_field else v for v in values]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input", type=Path, help="bona fide image; omit to use the synthetic test chart"
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="where to write the PNG"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=DEFAULT_CHART_SIZE,
        help=f"test chart side in pixels (default: {DEFAULT_CHART_SIZE}, ignored with --input)",
    )
    parser.add_argument(
        "--sweep", help="walk one field: FIELD=START:STOP:STEPS, e.g. moire_frequency=8:70:8"
    )
    parser.add_argument("--columns", type=int, default=3, help="panels per row in sweep mode")

    tuning = parser.add_argument_group("MoireConfig overrides")
    defaults = MoireConfig()
    for field in dataclasses.fields(MoireConfig):
        tuning.add_argument(
            f"--{field.name.replace('_', '-')}",
            type=int if field.type is int else float,
            help=f"default: {getattr(defaults, field.name)}",
        )
    return parser


def config_from_args(args: argparse.Namespace) -> MoireConfig:
    overrides: dict[str, Any] = {
        field.name: getattr(args, field.name)
        for field in dataclasses.fields(MoireConfig)
        if getattr(args, field.name, None) is not None
    }
    return MoireConfig(**overrides)


def main() -> None:
    args = build_parser().parse_args()
    base_config = config_from_args(args)
    source = load_image(args.input) if args.input else build_test_chart(args.size)
    origin = args.input.name if args.input else f"test chart {args.size}px"

    if args.sweep:
        field_name, values = parse_sweep(args.sweep)
        panels = [label_panel(source, f"bona fide  ({origin})")]
        for value in values:
            override: dict[str, Any] = {field_name: value}
            config = dataclasses.replace(base_config, **override)
            attack = MoireRecaptureGenerator(config).generate(source)
            panels.append(label_panel(attack, f"{field_name} = {value:g}"))
        columns = max(1, min(args.columns, len(panels)))
        sheet = contact_sheet(panels, columns)
        summary = f"sweep {field_name}: {values[0]:g} -> {values[-1]:g} over {len(values)} steps"
    else:
        attack = MoireRecaptureGenerator(base_config).generate(source)
        sheet = contact_sheet(
            [label_panel(source, f"bona fide  ({origin})"), label_panel(attack, "moiré recapture")],
            columns=2,
        )
        summary = "before / after"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print(f"{summary}\nwrote {args.output}  ({sheet.shape[1]}x{sheet.shape[0]})")


if __name__ == "__main__":
    main()
