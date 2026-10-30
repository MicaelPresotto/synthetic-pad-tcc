from pathlib import Path

import numpy as np
import pytest

from synthetic_pad.corpora import BONA_FIDE
from synthetic_pad.depth import (
    crop_key,
    depth_target,
    face_prominence,
    landmark_relief,
    read_depth_maps,
    shape_similarity,
    write_depth_maps,
)
from synthetic_pad.frames import FACE_CROP_SCALE, CropSample, face_cells

SIDE = 224


def face_relief() -> np.ndarray:
    rows, columns = np.mgrid[0:SIDE, 0:SIDE] / (SIDE - 1) - 0.5
    return np.asarray(np.exp(-(rows**2 + columns**2) / 0.05))


class TestDepthTarget:
    def test_is_a_seven_by_seven_map_in_the_unit_interval(self) -> None:
        target = depth_target(face_relief())
        assert target.shape == (7, 7)
        assert target.min() >= 0.0
        assert target.max() <= 1.0

    def test_is_highest_at_the_centre_of_the_face(self) -> None:
        target = depth_target(face_relief())
        assert target[3, 3] == target.max()

    def test_is_zero_on_cells_wholly_outside_the_face_box(self) -> None:
        target = depth_target(face_relief())
        assert np.all(target[0] == 0.0)
        assert np.all(target[:, -1] == 0.0)

    def test_ignores_how_near_the_face_is_overall(self) -> None:
        np.testing.assert_allclose(
            depth_target(face_relief() * 3.0 + 10.0), depth_target(face_relief()), atol=1e-6
        )

    def test_a_face_box_of_constant_depth_gives_a_flat_map(self) -> None:
        assert np.all(depth_target(np.full((SIDE, SIDE), 5.0)) == 0.0)


class TestDepthMapStore:
    @staticmethod
    def samples(root: Path, names: list[str]) -> list[CropSample]:
        return [CropSample(root / name, "train", "001", BONA_FIDE, "v") for name in names]

    def test_reads_the_maps_back_in_the_order_of_the_samples(self, tmp_path: Path) -> None:
        written = self.samples(tmp_path, ["a/1.png", "a/2.png", "b/1.png"])
        maps = np.arange(3 * 4, dtype=np.float32).reshape(3, 2, 2)
        write_depth_maps(tmp_path / "msu.npz", [crop_key(s, tmp_path) for s in written], maps)
        wanted = self.samples(tmp_path, ["b/1.png", "a/1.png"])
        read = read_depth_maps(tmp_path / "msu.npz", wanted, tmp_path)
        np.testing.assert_array_equal(read, maps[[2, 0]])

    def test_refuses_a_crop_without_a_map(self, tmp_path: Path) -> None:
        written = self.samples(tmp_path, ["a/1.png"])
        write_depth_maps(
            tmp_path / "msu.npz", [crop_key(written[0], tmp_path)], np.zeros((1, 2, 2))
        )
        with pytest.raises(KeyError, match="no depth map"):
            read_depth_maps(tmp_path / "msu.npz", self.samples(tmp_path, ["a/2.png"]), tmp_path)


def test_the_face_box_covers_the_central_cells_of_the_grid() -> None:
    expected = np.zeros((7, 7), dtype=bool)
    expected[1:6, 1:6] = True
    np.testing.assert_array_equal(face_cells(7, FACE_CROP_SCALE), expected)


class TestFaceProminence:
    def test_is_positive_for_a_face_nearer_than_its_background(self) -> None:
        assert face_prominence(face_relief()) > 0.0

    def test_is_negative_for_a_face_farther_than_its_background(self) -> None:
        assert face_prominence(-face_relief()) < 0.0

    def test_is_zero_for_a_flat_crop(self) -> None:
        assert face_prominence(np.full((SIDE, SIDE), 2.0)) == 0.0

    def test_ignores_the_scale_and_offset_of_the_relative_depth(self) -> None:
        assert face_prominence(face_relief() * 4.0 - 1.0) == pytest.approx(
            face_prominence(face_relief())
        )


class TestShapeSimilarity:
    def test_is_one_for_the_same_relief_at_another_strength(self) -> None:
        reference = depth_target(face_relief())
        similarity = shape_similarity(np.stack([reference * 0.5]), reference)
        assert similarity[0] == pytest.approx(1.0)

    def test_is_negative_for_an_inverted_relief(self) -> None:
        reference = depth_target(face_relief())
        inverted = depth_target(-face_relief())
        assert shape_similarity(np.stack([inverted]), reference)[0] < 0.0

    def test_is_zero_for_a_map_with_no_relief(self) -> None:
        reference = depth_target(face_relief())
        assert shape_similarity(np.zeros((1, 7, 7)), reference)[0] == 0.0


def cone_landmarks() -> np.ndarray:
    """A ring of far landmarks around one near landmark at the centre of a 64 px crop."""
    angles = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    ring = np.stack([32 + 20 * np.cos(angles), 32 + 20 * np.sin(angles), np.zeros(12)], axis=1)
    return np.vstack([ring, [[32.0, 32.0, 1.0]]])


class TestLandmarkRelief:
    def test_is_nearest_at_the_nearest_landmark(self) -> None:
        relief = landmark_relief(cone_landmarks(), 64)
        assert relief[32, 32] == pytest.approx(1.0)
        assert relief[32, 42] == pytest.approx(0.5, abs=0.05)

    def test_takes_the_depth_of_the_nearest_landmark_outside_their_hull(self) -> None:
        landmarks = cone_landmarks()
        landmarks[0, 2] = 0.3
        relief = landmark_relief(landmarks, 64)
        assert not np.isnan(relief).any()
        assert relief[32, 63] == pytest.approx(0.3)
        assert relief[0, 0] == pytest.approx(0.0)
