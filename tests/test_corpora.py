from pathlib import Path

import pytest

from synthetic_pad.corpora import BONA_FIDE, PRINT, REPLAY, read_msu, read_replay_mobile


def touch(path: Path, content: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


@pytest.fixture
def msu_root(tmp_path: Path) -> Path:
    touch(tmp_path / "train_sub_list.txt", "02\r\n")
    touch(tmp_path / "test_sub_list.txt", "03\r\n")
    scene = tmp_path / "scene01"
    touch(scene / "real" / "real_client002_android_SD_scene01.mov")
    touch(
        scene / "real" / "real_client002_android_SD_scene01.face",
        "0, 10, 20, 110, 140, 1, 1, 2, 2\n",
    )
    touch(scene / "attack" / "attack_client003_laptop_SD_ipad_video_scene01.mp4")
    touch(
        scene / "attack" / "attack_client003_laptop_SD_ipad_video_scene01.face",
        "0, 0, 0, 5, 5, 0, 0, 0, 0\n",
    )
    touch(scene / "attack" / "attack_client003_laptop_SD_printed_photo_scene01.mov")
    touch(
        scene / "attack" / "attack_client003_laptop_SD_printed_photo_scene01.face",
        "4, 0, 0, 5, 5, 0, 0, 0, 0\n",
    )
    return tmp_path


@pytest.fixture
def replay_mobile_root(tmp_path: Path) -> Path:
    database = tmp_path / "database"
    for relative, boxes in (
        (
            "train/real/client001_session01_authenticate_mobile_adverse",
            "10 20 100 120\n11 21 100 120\n",
        ),
        (
            "devel/attack/attack_client015_session01_mattescreen_fixed_mobile_photo_lighton",
            "0 0 5 5\n",
        ),
        ("test/attack/attack_client040_session01_print_hand_tablet_photo_lightoff", "0 0 5 5\n"),
    ):
        touch(database / f"{relative}.mov")
        touch(database / "faceloc" / "rect" / f"{relative}.face", boxes)
    touch(database / "enroll" / "train" / "client001_session01_enroll_mobile_adverse.mov")
    return tmp_path


class TestReadMsu:
    def test_places_each_subject_in_its_official_partition(self, msu_root: Path) -> None:
        partitions = {r.video.name: r.partition for r in read_msu(msu_root)}
        assert partitions["real_client002_android_SD_scene01.mov"] == "train"
        assert partitions["attack_client003_laptop_SD_ipad_video_scene01.mp4"] == "test"

    def test_tells_replay_from_print_from_bona_fide(self, msu_root: Path) -> None:
        species = sorted(r.species for r in read_msu(msu_root))
        assert species == sorted([BONA_FIDE, REPLAY, PRINT])

    def test_converts_corner_boxes_to_origin_and_size(self, msu_root: Path) -> None:
        real = next(r for r in read_msu(msu_root) if not r.is_attack)
        assert real.face_boxes == {0: (10, 20, 100, 120)}

    def test_keys_boxes_by_the_annotated_frame(self, msu_root: Path) -> None:
        printed = next(r for r in read_msu(msu_root) if r.species == PRINT)
        assert list(printed.face_boxes) == [4]


class TestReadReplayMobile:
    def test_reads_the_three_partitions_and_skips_enrollment(
        self, replay_mobile_root: Path
    ) -> None:
        partitions = sorted(r.partition for r in read_replay_mobile(replay_mobile_root))
        assert partitions == ["devel", "test", "train"]

    def test_calls_mattescreen_replay_and_print_print(self, replay_mobile_root: Path) -> None:
        by_partition = {r.partition: r.species for r in read_replay_mobile(replay_mobile_root)}
        assert by_partition == {"train": BONA_FIDE, "devel": REPLAY, "test": PRINT}

    def test_keys_boxes_by_line_number(self, replay_mobile_root: Path) -> None:
        real = next(r for r in read_replay_mobile(replay_mobile_root) if r.partition == "train")
        assert real.face_boxes == {0: (10, 20, 100, 120), 1: (11, 21, 100, 120)}
        assert real.subject == "001"
