"""Attack scores of whole videos from a cue with no training, reported by attack kind within and
across corpora at the threshold each development set fixes."""

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import numpy as np

from synthetic_pad.corpora import BONA_FIDE, REPLAY, Recording
from synthetic_pad.evaluation.report import pad_report
from synthetic_pad.frames import CORPORA

type Video = dict[str, Any]


def attack_kind(recording: Recording) -> str:
    """Print, or replay of a photo or of a video: only a replayed video moves like a face."""
    if recording.species != REPLAY:
        return recording.species
    shown = "video" if "_video" in recording.video.stem else "photo"
    return f"replay of a {shown}"


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


def report_every_direction(videos: Mapping[str, list[Video]]) -> dict[str, Any]:
    """Reports within and across corpora, printed as they are computed."""
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
    return reports
