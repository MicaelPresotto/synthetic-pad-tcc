"""Face crops sampled from PAD corpus videos, at the working resolution."""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from synthetic_pad.corpora import FaceBox, Recording
from synthetic_pad.dataset import WORKING_RESOLUTION, write_image
from synthetic_pad.domain import RgbImage

FRAMES_PER_VIDEO = 10
FACE_CROP_SCALE = 1.4


@dataclass(frozen=True, slots=True)
class FrameSet:
    """Every crop written, the settings that cut them and the video each came from."""

    corpus: str
    source_directory: str
    frames_per_video: int
    working_resolution: int
    face_crop_scale: float
    frames: Sequence[Mapping[str, object]]


def square_face_crop(frame: RgbImage, box: FaceBox, scale: float) -> RgbImage:
    height, width = frame.shape[:2]
    x, y, box_width, box_height = box
    side = min(round(max(box_width, box_height) * scale), height, width)
    left = round(x + box_width / 2 - side / 2)
    top = round(y + box_height / 2 - side / 2)
    left = min(max(left, 0), width - side)
    top = min(max(top, 0), height - side)
    return np.ascontiguousarray(frame[top : top + side, left : left + side])


def evenly_spaced(indices: Sequence[int], count: int) -> list[int]:
    if count >= len(indices):
        return list(indices)
    positions = np.linspace(0, len(indices) - 1, count).round().astype(int)
    return [indices[position] for position in positions]


def readable_frame_count(video: Path) -> int:
    """Frames that actually decode; container headers overstate it for truncated videos."""
    capture = cv2.VideoCapture(str(video))
    count = 0
    try:
        while capture.grab():
            count += 1
    finally:
        capture.release()
    return count


def sample_face_crops(
    recording: Recording, count: int, size: int, scale: float
) -> Iterator[tuple[int, RgbImage]]:
    readable = readable_frame_count(recording.video)
    annotated = sorted(index for index in recording.face_boxes if index < readable)
    wanted = set(evenly_spaced(annotated, count))
    capture = cv2.VideoCapture(str(recording.video))
    index = 0
    try:
        while wanted:
            ok, frame = capture.read()
            if not ok:
                break
            if index in wanted:
                wanted.discard(index)
                rgb = np.asarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), dtype=np.uint8)
                crop = square_face_crop(rgb, recording.face_boxes[index], scale)
                resized = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
                yield index, np.asarray(resized, dtype=np.uint8)
            index += 1
    finally:
        capture.release()


def extract_corpus(
    recordings: Sequence[Recording],
    source_directory: Path,
    output_directory: Path,
    count: int = FRAMES_PER_VIDEO,
    size: int = WORKING_RESOLUTION,
    scale: float = FACE_CROP_SCALE,
) -> FrameSet:
    rows: list[Mapping[str, object]] = []
    for recording in recordings:
        label = "attack" if recording.is_attack else "real"
        for index, crop in sample_face_crops(recording, count, size, scale):
            relative = (
                Path(recording.partition) / label / f"{recording.video.stem}_f{index:04d}.png"
            )
            write_image(output_directory / relative, crop)
            rows.append(
                {
                    "partition": recording.partition,
                    "subject": recording.subject,
                    "species": recording.species,
                    "video": recording.video.relative_to(source_directory).as_posix(),
                    "frame": index,
                    "image": relative.as_posix(),
                }
            )
    corpus = recordings[0].corpus if recordings else ""
    return FrameSet(corpus, str(source_directory), count, size, scale, rows)
