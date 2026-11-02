import argparse
from pathlib import Path

import cv2
import numpy as np

from synthetic_pad.generators.moire import MoireConfig, MoireRecaptureGenerator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Draws of the first generator, tuned by eye, on one image, one per seed. Run "
        "it with the package at the tag first-generator, as preview_generator.py does: it uses "
        "only what that package and the current one share."
    )
    parser.add_argument("--input", type=Path, required=True, help="RGB image, read as PNG")
    parser.add_argument("--draws", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True, help=".npy file of the draws")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    bgr = cv2.imread(str(args.input))
    if bgr is None:
        raise FileNotFoundError(args.input)
    image = np.asarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)
    draws = [
        MoireRecaptureGenerator(MoireConfig(seed=seed)).generate(image)
        for seed in range(1, args.draws + 1)
    ]
    np.save(args.output, np.stack(draws))


if __name__ == "__main__":
    main()
