import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

type FaceBox = tuple[int, int, int, int]

BONA_FIDE = "bona fide"
REPLAY = "replay"
PRINT = "print"

MSU = "msu"
REPLAY_MOBILE = "replay-mobile"
CORPORA = (MSU, REPLAY_MOBILE)

TRAIN = "train"
DEVELOPMENT = "devel"
TEST = "test"

MSU_ATTACK_SPECIES = {"ipad_video": REPLAY, "iphone_video": REPLAY, "printed_photo": PRINT}
REPLAY_MOBILE_PARTITIONS = (TRAIN, DEVELOPMENT, TEST)


@dataclass(frozen=True, slots=True)
class Recording:
    """One video of a PAD corpus; face boxes are (x, y, width, height), keyed by frame index."""

    corpus: str
    video: Path
    partition: str
    subject: str
    species: str
    face_boxes: Mapping[int, FaceBox]

    @property
    def is_attack(self) -> bool:
        return self.species != BONA_FIDE


def read_msu(root: Path) -> list[Recording]:
    partition_of = {
        **dict.fromkeys(_subject_list(root / "train_sub_list.txt"), TRAIN),
        **dict.fromkeys(_subject_list(root / "test_sub_list.txt"), TEST),
    }
    recordings = []
    for face_file in sorted((root / "scene01").glob("*/*.face")):
        subject = _client_id(face_file.stem)
        recordings.append(
            Recording(
                corpus=MSU,
                video=_msu_video(face_file),
                partition=partition_of[subject],
                subject=subject,
                species=_msu_species(face_file.stem),
                face_boxes=_read_msu_boxes(face_file),
            )
        )
    return recordings


def read_replay_mobile(root: Path) -> list[Recording]:
    database = root / "database"
    recordings = []
    for partition in REPLAY_MOBILE_PARTITIONS:
        for video in sorted((database / partition).glob("*/*.mov")):
            annotation = (database / "faceloc" / "rect" / video.relative_to(database)).with_suffix(
                ".face"
            )
            recordings.append(
                Recording(
                    corpus=REPLAY_MOBILE,
                    video=video,
                    partition=partition,
                    subject=_client_id(video.stem),
                    species=_replay_mobile_species(video.stem),
                    face_boxes=_read_replay_mobile_boxes(annotation),
                )
            )
    return recordings


READERS = {MSU: read_msu, REPLAY_MOBILE: read_replay_mobile}


def _subject_list(path: Path) -> list[str]:
    return [f"{int(token):03d}" for token in path.read_text().split()]


def _client_id(stem: str) -> str:
    found = re.search(r"client(\d+)", stem)
    if found is None:
        raise ValueError(f"no client id in {stem!r}")
    return f"{int(found.group(1)):03d}"


def _msu_video(face_file: Path) -> Path:
    for suffix in (".mov", ".mp4"):
        candidate = face_file.with_suffix(suffix)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no video next to {face_file}")


def _msu_species(stem: str) -> str:
    if stem.startswith("real_"):
        return BONA_FIDE
    for marker, species in MSU_ATTACK_SPECIES.items():
        if marker in stem:
            return species
    raise ValueError(f"unknown MSU attack type in {stem!r}")


def _replay_mobile_species(stem: str) -> str:
    if not stem.startswith("attack_"):
        return BONA_FIDE
    return REPLAY if "_mattescreen_" in stem else PRINT


def _read_msu_boxes(path: Path) -> dict[int, FaceBox]:
    rows = np.loadtxt(path, delimiter=",", ndmin=2)
    return {
        int(frame): (round(x1), round(y1), round(x2 - x1), round(y2 - y1))
        for frame, x1, y1, x2, y2 in rows[:, :5]
    }


def _read_replay_mobile_boxes(path: Path) -> dict[int, FaceBox]:
    rows = np.loadtxt(path, ndmin=2)
    return {
        frame: (round(x), round(y), round(width), round(height))
        for frame, (x, y, width, height) in enumerate(rows)
    }
