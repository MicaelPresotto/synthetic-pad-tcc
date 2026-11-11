import argparse
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from synthetic_pad.corpora import CORPORA
from synthetic_pad.frames import (
    DAMAGE_DISTANCE_RATIO,
    DAMAGE_FLAT_EXCESS,
    CropSample,
    damage_measures,
    is_damaged,
    load_crops,
    read_crops,
)


def by_video(samples: Sequence[CropSample]) -> dict[str, list[CropSample]]:
    videos: dict[str, list[CropSample]] = defaultdict(list)
    for sample in samples:
        videos[sample.video].append(sample)
    return videos


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Flag face crops that a decoding fault damaged: crops far from the median "
        "crop of their video and flatter than it, which motion alone does not produce."
    )
    parser.add_argument("--frames", type=Path, default=Path("data/frames"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for corpus in CORPORA:
        videos = by_video(load_crops(args.frames / corpus))
        damaged: list[Path] = []
        far_flat_excess: list[float] = []
        largest_ratio = 0.0
        for samples in videos.values():
            distance_ratio, flat_excess = damage_measures(read_crops(samples))
            largest_ratio = max(largest_ratio, float(distance_ratio.max()))
            far_flat_excess += flat_excess[distance_ratio > DAMAGE_DISTANCE_RATIO].tolist()
            flags = is_damaged(distance_ratio, flat_excess)
            damaged += [sample.image for sample, flag in zip(samples, flags, strict=True) if flag]
        crops = sum(len(samples) for samples in videos.values())
        print(f"\n{corpus}: {len(videos)} videos, {crops} crops, {len(damaged)} damaged")
        print(
            f"  largest distance to the median crop: {largest_ratio:.2f} x the typical one of "
            f"its video (limit {DAMAGE_DISTANCE_RATIO})"
        )
        print(
            f"  crops beyond that limit: {len(far_flat_excess)}, largest flat-block excess "
            f"among them: {max(far_flat_excess, default=0.0):.3f} (limit {DAMAGE_FLAT_EXCESS})"
        )
        for image in damaged:
            print(f"  damaged: {image.relative_to(args.frames)}")


if __name__ == "__main__":
    main()
