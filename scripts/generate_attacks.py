import argparse
from pathlib import Path

from synthetic_pad.dataset import WORKING_RESOLUTION, build_attack_set, write_manifest
from synthetic_pad.generators.moire import MoireRecaptureGenerator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a moire recapture attack for every bona fide image in a folder."
    )
    parser.add_argument("--input", type=Path, required=True, help="folder of bona fide images")
    parser.add_argument("--output", type=Path, required=True, help="where the attacks are written")
    parser.add_argument(
        "--size",
        type=int,
        default=WORKING_RESOLUTION,
        help=f"working resolution in pixels (default: {WORKING_RESOLUTION})",
    )
    parser.add_argument("--limit", type=int, help="stop after this many images")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    attack_set = build_attack_set(
        MoireRecaptureGenerator(), args.input, args.output, args.size, args.limit
    )
    path = write_manifest(args.output, attack_set)
    print(f"wrote {len(attack_set.attacks)} attacks to {args.output}\nprovenance in {path}")


if __name__ == "__main__":
    main()
