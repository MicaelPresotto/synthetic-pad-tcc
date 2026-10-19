from pathlib import Path

import cv2
import numpy as np
import pytest

from synthetic_pad.corpora import BONA_FIDE, REPLAY, Recording
from synthetic_pad.domain import RgbImage
from synthetic_pad.frames import (
    DAMAGE_DISTANCE_RATIO,
    CropSample,
    damage_measures,
    evenly_spaced,
    extract_corpus,
    flat_block_fraction,
    hold_out_subjects,
    is_damaged,
    load_crops,
    readable_frame_count,
    sample_face_crops,
    square_face_crop,
)


def write_video(path: Path, frames: int, height: int = 120, width: int = 160) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"MJPG"), 10.0, (width, height))
    for index in range(frames):
        writer.write(np.full((height, width, 3), index * 20 % 256, dtype=np.uint8))
    writer.release()
    return path


def recording(video: Path, frames: int, species: str = BONA_FIDE) -> Recording:
    boxes = dict.fromkeys(range(frames), (40, 30, 60, 60))
    return Recording("test", video, "train", "001", species, boxes)


class TestSquareFaceCrop:
    def test_is_square_and_scaled_from_the_longer_side(self) -> None:
        frame = np.zeros((200, 300, 3), dtype=np.uint8)
        assert square_face_crop(frame, (100, 60, 40, 50), 1.4).shape == (70, 70, 3)

    def test_centres_on_the_face(self) -> None:
        frame = np.zeros((200, 300, 3), dtype=np.uint8)
        frame[80:90, 115:125] = 255
        crop = square_face_crop(frame, (100, 65, 40, 40), 1.0)
        assert crop[20, 20].tolist() == [255, 255, 255]

    def test_shifts_inside_the_frame_instead_of_padding(self) -> None:
        frame = np.full((100, 100, 3), 7, dtype=np.uint8)
        crop = square_face_crop(frame, (0, 0, 40, 40), 1.5)
        assert crop.shape == (60, 60, 3)
        assert (crop == 7).all()

    def test_never_exceeds_the_frame(self) -> None:
        frame = np.zeros((80, 120, 3), dtype=np.uint8)
        assert square_face_crop(frame, (10, 10, 100, 70), 2.0).shape == (80, 80, 3)


class TestEvenlySpaced:
    def test_keeps_both_ends(self) -> None:
        assert evenly_spaced(list(range(100)), 3) == [0, 50, 99]

    def test_returns_everything_when_there_are_too_few(self) -> None:
        assert evenly_spaced([2, 5], 10) == [2, 5]


class TestReadableFrameCount:
    def test_counts_the_frames_that_decode(self, tmp_path: Path) -> None:
        assert readable_frame_count(write_video(tmp_path / "clip.avi", frames=7)) == 7

    def test_is_zero_for_a_file_that_is_not_a_video(self, tmp_path: Path) -> None:
        (tmp_path / "broken.avi").write_bytes(b"not a video")
        assert readable_frame_count(tmp_path / "broken.avi") == 0


class TestSampleFaceCrops:
    def test_yields_the_requested_number_at_the_working_resolution(self, tmp_path: Path) -> None:
        video = write_video(tmp_path / "clip.avi", frames=12)
        crops = list(sample_face_crops(recording(video, 12), count=4, size=32, scale=1.2))
        assert [index for index, _ in crops] == [0, 4, 7, 11]
        assert all(crop.shape == (32, 32, 3) for _, crop in crops)

    def test_skips_annotations_past_the_end_of_the_video(self, tmp_path: Path) -> None:
        video = write_video(tmp_path / "clip.avi", frames=5)
        crops = list(sample_face_crops(recording(video, 20), count=20, size=16, scale=1.0))
        assert [index for index, _ in crops] == [0, 1, 2, 3, 4]

    def test_spreads_over_the_frames_that_exist_when_annotations_run_past_them(
        self, tmp_path: Path
    ) -> None:
        video = write_video(tmp_path / "clip.avi", frames=5)
        crops = list(sample_face_crops(recording(video, 300), count=3, size=16, scale=1.0))
        assert [index for index, _ in crops] == [0, 2, 4]


class TestExtractCorpus:
    def test_writes_each_crop_under_its_partition_and_label(self, tmp_path: Path) -> None:
        source = tmp_path / "corpus"
        video = write_video(source / "attack" / "clip.avi", frames=6)
        frame_set = extract_corpus(
            [recording(video, 6, REPLAY)], source, tmp_path / "out", count=2, size=16
        )
        assert [row["image"] for row in frame_set.frames] == [
            "train/attack/clip_f0000.png",
            "train/attack/clip_f0005.png",
        ]
        assert (tmp_path / "out" / "train" / "attack" / "clip_f0000.png").exists()

    def test_records_where_every_crop_came_from(self, tmp_path: Path) -> None:
        source = tmp_path / "corpus"
        video = write_video(source / "real" / "clip.avi", frames=3)
        frame_set = extract_corpus(
            [recording(video, 3)], source, tmp_path / "out", count=1, size=16
        )
        assert frame_set.frames[0] == {
            "partition": "train",
            "subject": "001",
            "species": BONA_FIDE,
            "video": "real/clip.avi",
            "frame": 0,
            "image": "train/real/clip_f0000.png",
        }
        assert frame_set.frames_per_video == 1

    @pytest.mark.parametrize("count", [1, 3])
    def test_writes_one_png_per_manifest_row(self, tmp_path: Path, count: int) -> None:
        source = tmp_path / "corpus"
        video = write_video(source / "real" / "clip.avi", frames=4)
        frame_set = extract_corpus([recording(video, 4)], source, tmp_path / "out", count, 16)
        assert len(list((tmp_path / "out").rglob("*.png"))) == len(frame_set.frames) == count


def samples_for(subjects: range, partition: str = "train") -> list[CropSample]:
    return [
        CropSample(
            Path(f"{subject}_{frame}.png"), partition, f"{subject:03d}", BONA_FIDE, f"{subject}"
        )
        for subject in subjects
        for frame in range(3)
    ]


class TestLoadCrops:
    def test_reads_back_what_extraction_wrote(self, tmp_path: Path) -> None:
        from dataclasses import asdict

        from synthetic_pad.dataset import write_manifest

        source = tmp_path / "corpus"
        video = write_video(source / "attack" / "clip.avi", frames=2)
        frame_set = extract_corpus([recording(video, 2, REPLAY)], source, tmp_path / "out", 2, 16)
        write_manifest(tmp_path / "out", asdict(frame_set))
        crops = load_crops(tmp_path / "out")
        assert [crop.image.name for crop in crops] == ["clip_f0000.png", "clip_f0001.png"]
        assert all(crop.is_attack and crop.image.exists() for crop in crops)


class TestHoldOutSubjects:
    def test_moves_whole_subjects(self) -> None:
        moved = hold_out_subjects(samples_for(range(10)), "train", "devel", count=3, seed=1)
        held = {s.subject for s in moved if s.partition == "devel"}
        kept = {s.subject for s in moved if s.partition == "train"}
        assert len(held) == 3
        assert not held & kept

    def test_leaves_other_partitions_alone(self) -> None:
        samples = samples_for(range(6)) + samples_for(range(10, 14), "test")
        moved = hold_out_subjects(samples, "train", "devel", count=2, seed=1)
        assert [s for s in moved if s.partition == "test"] == samples_for(range(10, 14), "test")

    def test_is_reproducible_from_the_seed(self) -> None:
        samples = samples_for(range(10))
        assert hold_out_subjects(samples, "train", "devel", 3, seed=5) == hold_out_subjects(
            samples, "train", "devel", 3, seed=5
        )

    def test_refuses_to_empty_the_partition(self) -> None:
        with pytest.raises(ValueError, match="cannot hold out"):
            hold_out_subjects(samples_for(range(3)), "train", "devel", count=3, seed=1)


def noisy_copies(image: RgbImage, count: int) -> np.ndarray:
    rng = np.random.default_rng(7)
    noisy = image[None].astype(np.float64) + rng.normal(0.0, 2.0, (count, *image.shape))
    return np.clip(noisy, 0, 255).astype(np.uint8)


class TestDamagedCrops:
    def test_a_uniform_crop_is_all_flat(self) -> None:
        assert flat_block_fraction(np.full((64, 64, 3), 90, dtype=np.uint8)) == 1.0

    @pytest.mark.parametrize("clipped", [0, 255])
    def test_a_crop_clipped_to_black_or_white_is_not_flat(self, clipped: int) -> None:
        assert flat_block_fraction(np.full((64, 64, 3), clipped, dtype=np.uint8)) == 0.0

    def test_motion_is_not_damage(self, face_like_image: RgbImage) -> None:
        crops = noisy_copies(face_like_image, 10)
        crops[4] = np.roll(crops[4], 40, axis=1)
        distance_ratio, flat_excess = damage_measures(crops)
        assert distance_ratio[4] > DAMAGE_DISTANCE_RATIO
        assert not is_damaged(distance_ratio, flat_excess).any()

    def test_an_overexposed_crop_is_not_damage(self, face_like_image: RgbImage) -> None:
        crops = noisy_copies(face_like_image, 10)
        crops[4, :96, :96] = 255
        assert not is_damaged(*damage_measures(crops)).any()

    def test_a_flat_block_is_damage(self, face_like_image: RgbImage) -> None:
        crops = noisy_copies(face_like_image, 10)
        crops[4, :96, :96] = 128
        assert is_damaged(*damage_measures(crops)).tolist() == [index == 4 for index in range(10)]
