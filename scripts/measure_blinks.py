import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import face_alignment
import numpy as np
import torch

from synthetic_pad.blinks import blink_score, eye_aspect_ratio, frames_below_half
from synthetic_pad.corpora import BONA_FIDE, REPLAY, Recording, read_msu, read_replay_mobile
from synthetic_pad.evaluation.report import pad_report
from synthetic_pad.frames import CORPORA, load_partitions

READERS = {"msu": read_msu, "replay-mobile": read_replay_mobile}
FRAME_STEP = 3
EVALUATED = ("devel", "test")

type Video = dict[str, Any]


def attack_kind(recording: Recording) -> str:
    """Print, or replay of a photo or of a video: only a replayed video moves like a face."""
    if recording.species != REPLAY:
        return recording.species
    shown = "video" if "_video" in recording.video.stem else "photo"
    return f"replay of a {shown}"


def eye_aspect_ratios(aligner: Any, recording: Recording) -> np.ndarray:
    """EAR of every FRAME_STEP-th frame that has a face box."""
    ratios = []
    capture = cv2.VideoCapture(str(recording.video))
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            box = recording.face_boxes.get(index)
            if index % FRAME_STEP == 0 and box is not None:
                x, y, width, height = box
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face = np.array([x, y, x + width, y + height])
                (landmarks,) = aligner.get_landmarks_from_image(rgb, detected_faces=[face])
                ratios.append(eye_aspect_ratio(landmarks))
            index += 1
    finally:
        capture.release()
    return np.array(ratios)


def measure(corpus: str, root: Path, partition_of: dict[str, str], aligner: Any) -> list[Video]:
    videos = []
    for recording in READERS[corpus](root):
        name = recording.video.relative_to(root).as_posix()
        partition = partition_of.get(name)
        if partition not in EVALUATED:
            continue
        ratios = eye_aspect_ratios(aligner, recording)
        if len(ratios) == 0:
            print(f"{name}: no frame with a face box, left out")
            continue
        video = {
            "video": name,
            "partition": partition,
            "subject": recording.subject,
            "kind": attack_kind(recording),
            "score": blink_score(ratios),
            "frames_below_half": frames_below_half(ratios),
            "ratios": ratios.round(4).tolist(),
        }
        videos.append(video)
        print(f"{corpus} {partition:5s} {video['kind']:18s} {video['score']:.2f}")
    return videos


def scores(videos: list[Video], partition: str, attack: bool) -> np.ndarray:
    return np.array(
        [
            video["score"]
            for video in videos
            if video["partition"] == partition and (video["kind"] != BONA_FIDE) == attack
        ]
    )


def report(source: list[Video], tested: list[Video]) -> dict[str, Any]:
    by_kind: dict[str, list[float]] = defaultdict(list)
    for video in tested:
        if video["partition"] == "test" and video["kind"] != BONA_FIDE:
            by_kind[video["kind"]].append(video["score"])
    return pad_report(
        scores(source, "devel", attack=True),
        scores(source, "devel", attack=False),
        {kind: np.array(values) for kind, values in sorted(by_kind.items())},
        scores(tested, "test", attack=False),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score the development and test videos of both corpora by their deepest "
        "blink, the lowest eye aspect ratio over the median, with no training, and report the "
        "PAD metrics at the threshold each development set fixes, within and across corpora."
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
        default=Path("results/blinks"),
        help="per-corpus EAR series are kept here and reused on the next run",
    )
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    partitions = load_partitions(args.frames)
    aligner = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.THREE_D,
        device=args.device,
        face_detector="folder",
        compile=False,
    )
    videos = {}
    for corpus, root in zip(CORPORA, args.roots, strict=True):
        path = args.output / f"{corpus}.json"
        if path.exists():
            videos[corpus] = json.loads(path.read_text())
            continue
        partition_of = {sample.video: sample.partition for sample in partitions[corpus]}
        videos[corpus] = measure(corpus, root, partition_of, aligner)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(videos[corpus]) + "\n")

    reports = {}
    for source in CORPORA:
        target = next(corpus for corpus in CORPORA if corpus != source)
        for protocol, tested in (("intra", source), ("cross", target)):
            result = report(videos[source], videos[tested])
            reports[f"{source} {protocol}"] = {"tested": tested, **result}
            print(f"\n{source} -> {tested} ({protocol}), BPCER {result['bpcer']:.3f}")
            for kind, metrics in {
                "all attacks": result["all_attacks"],
                **result["by_species"],
            }.items():
                print(
                    f"  {kind:18s} HTER {metrics['hter']:.3f}  AUC {metrics['auc']:.3f}  "
                    f"APCER {metrics['apcer']:.3f}"
                )
    (args.output / "report.json").write_text(json.dumps(reports, indent=2) + "\n")

    for corpus, measured in videos.items():
        lone = sum(1 for video in measured if video["frames_below_half"] == 1)
        print(f"{corpus}: {lone} of {len(measured)} videos fall below half the median in one frame")


if __name__ == "__main__":
    main()
