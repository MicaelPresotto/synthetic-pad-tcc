import argparse
from dataclasses import asdict
from pathlib import Path

from synthetic_pad.corpora import read_msu, read_replay_mobile
from synthetic_pad.dataset import WORKING_RESOLUTION, write_manifest
from synthetic_pad.frames import FACE_CROP_SCALE, FRAMES_PER_VIDEO, extract_corpus

READERS = {"msu": read_msu, "replay-mobile": read_replay_mobile}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cut face crops from the videos of a PAD corpus, keeping its partitions."
    )
    parser.add_argument("--corpus", choices=sorted(READERS), required=True)
    parser.add_argument(
        "--root", type=Path, required=True, help="folder the corpus was extracted to"
    )
    parser.add_argument("--output", type=Path, required=True, help="where the crops are written")
    parser.add_argument(
        "--frames-per-video",
        type=int,
        default=FRAMES_PER_VIDEO,
        help=f"evenly spaced frames per video (default: {FRAMES_PER_VIDEO})",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=WORKING_RESOLUTION,
        help=f"crop resolution in pixels (default: {WORKING_RESOLUTION})",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=FACE_CROP_SCALE,
        help=f"crop side over the longer side of the face box (default: {FACE_CROP_SCALE})",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    recordings = READERS[args.corpus](args.root)
    frame_set = extract_corpus(
        recordings, args.root, args.output, args.frames_per_video, args.size, args.scale
    )
    path = write_manifest(args.output, asdict(frame_set))
    print(f"{len(frame_set.frames)} crops from {len(recordings)} videos\nprovenance in {path}")


if __name__ == "__main__":
    main()
