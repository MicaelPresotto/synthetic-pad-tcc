import argparse
import json
from collections import defaultdict
from collections.abc import Iterable
from functools import partial
from pathlib import Path

import cv2
import numpy as np

from synthetic_pad.corpora import FaceBox, Recording
from synthetic_pad.domain import RgbImage
from synthetic_pad.frames import CORPORA, load_partitions
from synthetic_pad.pulse import (
    PULSE_FREQUENCIES,
    WINDOW_SECONDS,
    band_power,
    pos_signal,
    pulse_score,
)
from synthetic_pad.video_scores import (
    Footage,
    Recapture,
    Video,
    cached_videos,
    recorded,
    report_every_direction,
    synthetic_replays,
)

SKIN_SHARE = 0.6
MEASURED = ("train", "devel", "test")
RESTING_HEART_RATE_HZ = (1.0, 1.7)


def skin_color(frame: RgbImage, box: FaceBox) -> np.ndarray:
    """Mean RGB of the central SKIN_SHARE of the face box, away from hair and background."""
    x, y, width, height = box
    margin_x = round(width * (1 - SKIN_SHARE) / 2)
    margin_y = round(height * (1 - SKIN_SHARE) / 2)
    skin = frame[
        max(y + margin_y, 0) : y + height - margin_y,
        max(x + margin_x, 0) : x + width - margin_x,
    ]
    return skin.mean(axis=(0, 1))


def skin_colors(recording: Recording, recapture: Recapture) -> tuple[np.ndarray, float]:
    """Mean skin RGB per frame of the first WINDOW_SECONDS, under the last box seen; and the fps."""
    capture = cv2.VideoCapture(str(recording.video))
    fps = capture.get(cv2.CAP_PROP_FPS)
    box = recording.face_boxes[min(recording.face_boxes)]
    colors = []
    try:
        for index in range(round(WINDOW_SECONDS * fps)):
            ok, frame = capture.read()
            if not ok:
                break
            box = recording.face_boxes.get(index, box)
            rgb = np.asarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), dtype=np.uint8)
            colors.append(skin_color(recapture(rgb), box))
    finally:
        capture.release()
    return np.array(colors), fps


def measure(corpus: str, footage: Iterable[Footage]) -> list[Video]:
    videos = []
    for labels, recording, recapture in footage:
        colors, fps = skin_colors(recording, recapture)
        power = band_power(pos_signal(colors, fps), fps)
        video = {
            **labels,
            "score": pulse_score(power),
            "peak_hz": float(PULSE_FREQUENCIES[power.argmax()]),
            "band_share": (power / power.sum()).round(5).tolist(),
        }
        videos.append(video)
        print(
            f"{corpus} {video['partition']:5s} {video['kind']:27s} {video['score']:.2f}", flush=True
        )
    return videos


def print_peak_frequencies(videos: dict[str, list[Video]]) -> None:
    low, high = RESTING_HEART_RATE_HZ
    at_rest = (low <= PULSE_FREQUENCIES) & (high >= PULSE_FREQUENCIES)
    print(f"\nShare of videos peaking at {low}-{high} Hz ({at_rest.mean():.0%} by chance):")
    for corpus, measured in videos.items():
        by_kind: dict[str, list[bool]] = defaultdict(list)
        for video in measured:
            by_kind[video["kind"]].append(low <= video["peak_hz"] <= high)
        for kind, peaks in sorted(by_kind.items()):
            print(f"  {corpus:13s} {kind:27s} {np.mean(peaks):.0%} of {len(peaks)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score every video of both corpora by its pulse, the share of the 0.7-4 Hz "
        "band its strongest peak holds in the POS signal of the skin color, with no training, "
        "together with two synthetic replays of each bona fide test video, and report the PAD "
        "metrics at the threshold each development set fixes, within and across corpora."
    )
    parser.add_argument(
        "--roots",
        type=Path,
        nargs=2,
        metavar=("MSU", "REPLAY_MOBILE"),
        default=[Path("data/msu"), Path("data/replay-mobile")],
        help="folders the corpora were extracted to, as for extract_frames.py",
    )
    parser.add_argument("--frames", type=Path, default=Path("data/frames"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pulse"),
        help="per-corpus band spectra are kept here and reused on the next run",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    partitions = load_partitions(args.frames)
    videos = {}
    for corpus, root in zip(CORPORA, args.roots, strict=True):
        partition_of = {sample.video: sample.partition for sample in partitions[corpus]}
        videos[corpus] = cached_videos(
            args.output / f"{corpus}.json",
            partial(measure, corpus, recorded(corpus, root, partition_of, MEASURED)),
        ) + cached_videos(
            args.output / f"{corpus}-synthetic.json",
            partial(measure, corpus, synthetic_replays(corpus, root, partition_of)),
        )

    reports = report_every_direction(videos)
    (args.output / "report.json").write_text(json.dumps(reports, indent=2) + "\n")
    print_peak_frequencies(videos)


if __name__ == "__main__":
    main()
