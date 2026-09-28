"""Turning a folder of bona fide images into a folder of attacks, with provenance."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from synthetic_pad.domain import AttackGenerator, RgbImage

WORKING_RESOLUTION = 224
READABLE_SUFFIXES = (".png", ".jpg", ".jpeg")
MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True, slots=True)
class AttackSet:
    """Every attack written, and the exact generator settings that produced them."""

    generator: Mapping[str, object]
    source_directory: str
    working_resolution: int
    attacks: Sequence[Mapping[str, str]]


def read_image(path: Path) -> RgbImage:
    raw = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if raw is None:
        raise FileNotFoundError(f"could not read an image from {path}")
    return np.asarray(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB), dtype=np.uint8)


def write_image(path: Path, image: RgbImage) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def fit_to_working_resolution(image: RgbImage, size: int) -> RgbImage:
    height, width = image.shape[:2]
    scale = size / min(height, width)
    resized = cv2.resize(
        image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA
    )
    top = (resized.shape[0] - size) // 2
    left = (resized.shape[1] - size) // 2
    return np.asarray(resized[top : top + size, left : left + size], dtype=np.uint8)


def find_bona_fide_images(source_directory: Path) -> Sequence[Path]:
    found = sorted(
        path
        for path in source_directory.rglob("*")
        if path.suffix.lower() in READABLE_SUFFIXES and path.is_file()
    )
    if not found:
        raise ValueError(
            f"{source_directory} holds no {'/'.join(READABLE_SUFFIXES)} image to attack"
        )
    return found


def build_attack_set(
    generator: AttackGenerator,
    source_directory: Path,
    output_directory: Path,
    size: int = WORKING_RESOLUTION,
    limit: int | None = None,
) -> AttackSet:
    written = []
    for source in find_bona_fide_images(source_directory)[:limit]:
        relative_source = source.relative_to(source_directory)
        relative_attack = relative_source.with_suffix(".png")
        bona_fide = fit_to_working_resolution(read_image(source), size)
        write_image(output_directory / relative_attack, generator.generate(bona_fide))
        written.append(
            {"bona_fide": relative_source.as_posix(), "attack": relative_attack.as_posix()}
        )
    return AttackSet(dict(generator.describe()), str(source_directory), size, written)


def write_manifest(output_directory: Path, manifest: Mapping[str, object]) -> Path:
    path = output_directory / MANIFEST_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path
