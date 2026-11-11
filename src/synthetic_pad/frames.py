"""Face crops sampled from PAD corpus videos, at the working resolution."""

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np
from skimage.data import lbp_frontal_face_cascade_filename
from skimage.feature import Cascade

from synthetic_pad.corpora import (
    BONA_FIDE,
    DEVELOPMENT,
    MSU,
    REPLAY_MOBILE,
    TRAIN,
    FaceBox,
    Recording,
)
from synthetic_pad.dataset import MANIFEST_NAME, WORKING_RESOLUTION, read_image, write_image
from synthetic_pad.domain import RgbImage

FRAMES_PER_VIDEO = 10
FACE_CROP_SCALE = 1.4
DETECTION_SCALE_FACTOR = 1.2
SMALLEST_FACE_PX = 60
MSU_DEVELOPMENT_SUBJECTS = 5
MSU_SPLIT_SEED = 20260928
FLAT_BLOCK_SIDE = 8
FLAT_BLOCK_SPREAD = 0.5
BLACK_CLIP_LEVEL = 5.0
WHITE_CLIP_LEVEL = 250.0
DAMAGE_DISTANCE_RATIO = 3.0
DAMAGE_FLAT_EXCESS = 0.10


@dataclass(frozen=True, slots=True)
class CropSample:
    image: Path
    partition: str
    subject: str
    species: str
    video: str

    @property
    def is_attack(self) -> bool:
        return self.species != BONA_FIDE


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


def face_crop(image: RgbImage) -> RgbImage:
    """The largest face found, cut and resized as the crops of the corpora were."""
    faces = Cascade(lbp_frontal_face_cascade_filename()).detect_multi_scale(
        img=image,
        scale_factor=DETECTION_SCALE_FACTOR,
        step_ratio=1,
        min_size=(SMALLEST_FACE_PX, SMALLEST_FACE_PX),
        max_size=image.shape[:2],
    )
    if not faces:
        raise ValueError("no face found in the input image")
    face = max(faces, key=lambda found: found["width"] * found["height"])
    box = (face["c"], face["r"], face["width"], face["height"])
    crop = square_face_crop(image, box, FACE_CROP_SCALE)
    size = (WORKING_RESOLUTION, WORKING_RESOLUTION)
    return np.asarray(cv2.resize(crop, size, interpolation=cv2.INTER_AREA))


def face_cells(grid: int, scale: float) -> np.ndarray:
    """Cells of a grid laid over a square crop whose centre falls inside its face box, the
    central square that square_face_crop leaves at 1 / scale of the crop side."""
    centres = (np.arange(grid) + 0.5) / grid
    inside = np.abs(centres - 0.5) <= 0.5 / scale
    return np.asarray(inside[:, None] & inside[None, :])


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
            decoded, frame = capture.read()
            if not decoded:
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


def load_crops(frames_directory: Path) -> list[CropSample]:
    manifest = json.loads((frames_directory / MANIFEST_NAME).read_text())
    return [
        CropSample(
            frames_directory / row["image"],
            row["partition"],
            row["subject"],
            row["species"],
            row["video"],
        )
        for row in manifest["frames"]
    ]


def hold_out_subjects(
    samples: Sequence[CropSample], source: str, target: str, count: int, seed: int
) -> list[CropSample]:
    """Move `count` randomly chosen subjects of one partition into a new one, whole."""
    subjects = sorted({sample.subject for sample in samples if sample.partition == source})
    if not 0 < count < len(subjects):
        raise ValueError(f"cannot hold out {count} of the {len(subjects)} subjects in {source!r}")
    held = set(np.random.default_rng(seed).choice(subjects, size=count, replace=False).tolist())
    return [
        replace(sample, partition=target)
        if sample.partition == source and sample.subject in held
        else sample
        for sample in samples
    ]


def load_partitions(
    frames_root: Path, msu_split_seed: int = MSU_SPLIT_SEED
) -> dict[str, list[CropSample]]:
    """Crops of every corpus; MSU-MFSD has no development partition, so five of its
    training subjects, drawn with the split seed, are held out as one."""
    msu = hold_out_subjects(
        load_crops(frames_root / MSU), TRAIN, DEVELOPMENT, MSU_DEVELOPMENT_SUBJECTS, msu_split_seed
    )
    return {MSU: msu, REPLAY_MOBILE: load_crops(frames_root / REPLAY_MOBILE)}


def select(samples: Sequence[CropSample], partition: str) -> list[CropSample]:
    return [sample for sample in samples if sample.partition == partition]


def read_crops(samples: Sequence[CropSample]) -> np.ndarray:
    return np.stack([read_image(sample.image) for sample in samples])


def flat_block_fraction(crop: RgbImage) -> float:
    """Share of the square luminance blocks whose standard deviation, in grey levels, is below
    FLAT_BLOCK_SPREAD, leaving out blocks an over- or underexposed capture clips to white or
    black."""
    luminance = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float64)
    rows, columns = luminance.shape[0] // FLAT_BLOCK_SIDE, luminance.shape[1] // FLAT_BLOCK_SIDE
    blocks = luminance[: rows * FLAT_BLOCK_SIDE, : columns * FLAT_BLOCK_SIDE].reshape(
        rows, FLAT_BLOCK_SIDE, columns, FLAT_BLOCK_SIDE
    )
    level = blocks.mean(axis=(1, 3))
    unclipped = (level > BLACK_CLIP_LEVEL) & (level < WHITE_CLIP_LEVEL)
    return float(((blocks.std(axis=(1, 3)) < FLAT_BLOCK_SPREAD) & unclipped).mean())


def damage_measures(crops: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per crop of one video: its distance to the median crop over the typical distance, and
    its share of flat blocks above the typical share."""
    frames = crops.astype(np.float64)
    distances = np.abs(frames - np.median(frames, axis=0)).mean(axis=(1, 2, 3))
    flatness = np.array([flat_block_fraction(crop) for crop in crops])
    return distances / np.median(distances), flatness - np.median(flatness)


def is_damaged(distance_ratio: np.ndarray, flat_excess: np.ndarray) -> np.ndarray:
    """Far from the median crop and flatter than it; motion alone only does the first."""
    return (distance_ratio > DAMAGE_DISTANCE_RATIO) & (flat_excess > DAMAGE_FLAT_EXCESS)
