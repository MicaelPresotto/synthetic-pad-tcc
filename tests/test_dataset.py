import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

from synthetic_pad.dataset import (
    AttackSet,
    build_attack_set,
    find_bona_fide_images,
    fit_to_working_resolution,
    read_image,
    write_image,
    write_manifest,
)
from synthetic_pad.domain import AttackGenerator, RgbImage


class InvertingGenerator(AttackGenerator):
    name = "inverting"

    def generate(self, image: RgbImage) -> RgbImage:
        return np.asarray(255 - image, dtype=np.uint8)

    def describe(self) -> Mapping[str, object]:
        return {"generator": self.name, "strength": 1.0}


def solid_image(height: int, width: int, value: int = 120) -> RgbImage:
    return np.full((height, width, 3), value, dtype=np.uint8)


def populate(directory: Path, names: tuple[str, ...]) -> None:
    for name in names:
        write_image(directory / name, solid_image(40, 60))


class TestImageIo:
    def test_a_written_image_reads_back_unchanged(self, tmp_path: Path) -> None:
        image = solid_image(8, 12, 200)
        write_image(tmp_path / "nested" / "frame.png", image)
        assert np.array_equal(read_image(tmp_path / "nested" / "frame.png"), image)

    def test_reading_a_missing_file_says_which_one(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match=r"absent\.png"):
            read_image(tmp_path / "absent.png")


class TestFitToWorkingResolution:
    def test_squares_a_landscape_image(self) -> None:
        assert fit_to_working_resolution(solid_image(80, 200), 32).shape == (32, 32, 3)

    def test_squares_a_portrait_image(self) -> None:
        assert fit_to_working_resolution(solid_image(200, 80), 32).shape == (32, 32, 3)

    def test_keeps_the_centre_of_the_frame(self) -> None:
        image = solid_image(64, 256, 0)
        image[:, 112:144] = 255
        fitted = fit_to_working_resolution(image, 64)
        assert fitted[32, 32].tolist() == [255, 255, 255]


class TestFindBonaFideImages:
    def test_walks_subfolders_in_a_stable_order(self, tmp_path: Path) -> None:
        populate(tmp_path, ("b.png", "a.png", "session/c.jpg"))
        found = [path.relative_to(tmp_path).as_posix() for path in find_bona_fide_images(tmp_path)]
        assert found == ["a.png", "b.png", "session/c.jpg"]

    def test_ignores_files_that_are_not_images(self, tmp_path: Path) -> None:
        populate(tmp_path, ("a.png",))
        (tmp_path / "notes.txt").write_text("not an image")
        assert len(find_bona_fide_images(tmp_path)) == 1

    def test_refuses_a_folder_with_nothing_to_attack(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match=r"no .* image to attack"):
            find_bona_fide_images(tmp_path)


class TestBuildAttackSet:
    def test_writes_one_attack_per_bona_fide_image(self, tmp_path: Path) -> None:
        source, output = tmp_path / "real", tmp_path / "attack"
        populate(source, ("a.png", "session/b.png"))
        attack_set = build_attack_set(InvertingGenerator(), source, output, size=16)
        assert len(attack_set.attacks) == 2
        assert (output / "a.png").exists()
        assert (output / "session" / "b.png").exists()

    def test_puts_every_attack_at_the_working_resolution(self, tmp_path: Path) -> None:
        source, output = tmp_path / "real", tmp_path / "attack"
        populate(source, ("a.png",))
        build_attack_set(InvertingGenerator(), source, output, size=16)
        assert read_image(output / "a.png").shape == (16, 16, 3)

    def test_runs_the_generator_rather_than_copying(self, tmp_path: Path) -> None:
        source, output = tmp_path / "real", tmp_path / "attack"
        populate(source, ("a.png",))
        build_attack_set(InvertingGenerator(), source, output, size=16)
        assert read_image(output / "a.png")[0, 0].tolist() == [135, 135, 135]

    def test_stops_at_the_limit(self, tmp_path: Path) -> None:
        source, output = tmp_path / "real", tmp_path / "attack"
        populate(source, ("a.png", "b.png", "c.png"))
        assert len(build_attack_set(InvertingGenerator(), source, output, 16, limit=2).attacks) == 2

    def test_normalises_jpeg_sources_to_png_attacks(self, tmp_path: Path) -> None:
        source, output = tmp_path / "real", tmp_path / "attack"
        populate(source, ("a.jpg",))
        build_attack_set(InvertingGenerator(), source, output, size=16)
        assert (output / "a.png").exists()

    def test_records_the_generator_settings_that_produced_it(self, tmp_path: Path) -> None:
        source, output = tmp_path / "real", tmp_path / "attack"
        populate(source, ("a.png",))
        attack_set = build_attack_set(InvertingGenerator(), source, output, size=16)
        assert attack_set.generator == {"generator": "inverting", "strength": 1.0}
        assert attack_set.working_resolution == 16

    def test_names_the_bona_fide_each_attack_came_from(self, tmp_path: Path) -> None:
        source, output = tmp_path / "real", tmp_path / "attack"
        populate(source, ("a.png",))
        attack_set = build_attack_set(InvertingGenerator(), source, output, size=16)
        assert attack_set.attacks[0] == {"bona_fide": "a.png", "attack": "a.png"}
        assert attack_set.source_directory == str(source)

    def test_keeps_manifest_paths_relative_to_the_folders_it_was_given(
        self, tmp_path: Path
    ) -> None:
        source, output = tmp_path / "real", tmp_path / "attack"
        populate(source, ("session/deep/a.jpg",))
        attack_set = build_attack_set(InvertingGenerator(), source, output, size=16)
        assert attack_set.attacks[0] == {
            "bona_fide": "session/deep/a.jpg",
            "attack": "session/deep/a.png",
        }


class TestWriteManifest:
    def test_writes_json_that_reads_back_as_the_attack_set(self, tmp_path: Path) -> None:
        attack_set = AttackSet(
            {"generator": "inverting"}, "data/msu/real", 224, [{"bona_fide": "a", "attack": "b"}]
        )
        path = write_manifest(tmp_path, attack_set)
        assert json.loads(path.read_text()) == {
            "generator": {"generator": "inverting"},
            "source_directory": "data/msu/real",
            "working_resolution": 224,
            "attacks": [{"bona_fide": "a", "attack": "b"}],
        }
