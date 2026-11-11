"""Attack scores of whole videos from a cue with no training, reported by attack kind within and
across corpora at the threshold each development set fixes, with synthetic replays of the bona
fide test videos beside the recorded attacks."""

import json
import zlib
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from synthetic_pad.corpora import BONA_FIDE, CORPORA, DEVELOPMENT, READERS, REPLAY, TEST, Recording
from synthetic_pad.domain import RgbImage
from synthetic_pad.evaluation.report import pad_report
from synthetic_pad.generators.moire import MoireRecaptureGenerator, read_fitted_config

SYNTHETIC_REPLAYS = ("synthetic, one screen", "synthetic, screen per frame")
GENERATOR_CONFIGS = {
    corpus: Path(f"experiments/moire-fitted-to-{corpus}-matched-noise.json") for corpus in CORPORA
}

type Video = dict[str, Any]
type Recapture = Callable[[RgbImage], RgbImage]
type Footage = tuple[Video, Recording, Recapture]


def attack_kind(recording: Recording) -> str:
    """Print, or replay of a photo or of a video: only a replayed video moves like a face."""
    if recording.species != REPLAY:
        return recording.species
    shown = "video" if "_video" in recording.video.stem else "photo"
    return f"replay of a {shown}"


def as_recorded(frame: RgbImage) -> RgbImage:
    return frame


def recorded(
    corpus: str, root: Path, partition_of: Mapping[str, str], partitions: tuple[str, ...]
) -> Iterator[Footage]:
    for recording in READERS[corpus](root):
        name = recording.video.relative_to(root).as_posix()
        partition = partition_of.get(name)
        if partition in partitions:
            video = {
                "video": name,
                "partition": partition,
                "subject": recording.subject,
                "kind": attack_kind(recording),
            }
            yield video, recording, as_recorded


def synthetic_replays(
    corpus: str, root: Path, partition_of: Mapping[str, str]
) -> Iterator[Footage]:
    """Each bona fide test video recaptured by the generator fitted to its corpus, once through
    one screen for the whole video and once through a new screen in every frame."""
    generator = MoireRecaptureGenerator(read_fitted_config(GENERATOR_CONFIGS[corpus]))
    one_screen, screen_per_frame = SYNTHETIC_REPLAYS
    for video, recording, _ in recorded(corpus, root, partition_of, (TEST,)):
        if video["kind"] == BONA_FIDE:
            scene = zlib.crc32(video["video"].encode())
            yield {**video, "kind": one_screen}, recording, partial(generator.generate, scene=scene)
            yield {**video, "kind": screen_per_frame}, recording, generator.generate


def cached_videos(path: Path, measure: Callable[[], list[Video]]) -> list[Video]:
    if path.exists():
        videos: list[Video] = json.loads(path.read_text())
        return videos
    videos = measure()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(videos) + "\n")
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
    """The synthetic replays get rows of their own but stay out of the "all attacks" figures."""
    by_kind: dict[str, list[float]] = defaultdict(list)
    for video in tested:
        if video["partition"] == TEST and video["kind"] != BONA_FIDE:
            by_kind[video["kind"]].append(video["score"])
    development = (
        scores(source, DEVELOPMENT, attack=True),
        scores(source, DEVELOPMENT, attack=False),
    )
    bona_fide = scores(tested, TEST, attack=False)
    recorded_attacks = {
        kind: np.array(values)
        for kind, values in sorted(by_kind.items())
        if kind not in SYNTHETIC_REPLAYS
    }
    synthetic_attacks = {kind: np.array(by_kind[kind]) for kind in SYNTHETIC_REPLAYS}
    result = pad_report(*development, recorded_attacks, bona_fide)
    result["by_species"] |= pad_report(*development, synthetic_attacks, bona_fide)["by_species"]
    return result


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
                    f"  {kind:27s} HTER {metrics['hter']:.3f}  AUC {metrics['auc']:.3f}  "
                    f"APCER {metrics['apcer']:.3f}"
                )
    return reports
