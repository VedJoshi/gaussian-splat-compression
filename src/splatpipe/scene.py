"""Validate an input scene directory before spending GPU minutes on it.

gsplat's colmap parser raises deep inside its own loading code when something
is missing. Checking up front costs milliseconds and turns a confused traceback
into a list of exactly what to fix.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from splatpipe.errors import SceneError

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
COLMAP_FILES = ("cameras.bin", "images.bin", "points3D.bin")
MIN_IMAGES = 8


@dataclass(frozen=True)
class SceneLayout:
    root: Path
    images_dir: Path
    sparse_dir: Path
    image_count: int

    @classmethod
    def discover(cls, root: Path | str, data_factor: int = 1) -> SceneLayout:
        root = Path(root)
        problems: list[str] = []

        # gsplat needs `images/` regardless, because COLMAP's filenames are
        # recorded against it, plus `images_<factor>/` when downscaling. Its
        # colmap loader zips sorted(images/) with sorted(images_<factor>/) to
        # map between the two (_gsplat_repo/examples/datasets/colmap.py,
        # colmap_to_image), so a thin `images/` truncates that mapping and a
        # later lookup by a COLMAP image name raises a bare KeyError. The
        # minimum image count therefore applies to both folders, not only the
        # one actually used for training.
        colmap_images = root / "images"
        images_dir = colmap_images if data_factor == 1 else root / f"images_{data_factor}"

        image_count = 0
        for directory in dict.fromkeys((colmap_images, images_dir)):
            if not directory.is_dir():
                problems.append(f"missing image folder {directory}")
                continue
            count = sum(1 for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
            if directory == images_dir:
                image_count = count
            if count < MIN_IMAGES:
                problems.append(f"{directory} holds {count} images, need at least {MIN_IMAGES}")

        sparse_dir = root / "sparse" / "0"
        for name in COLMAP_FILES:
            if not (sparse_dir / name).is_file():
                problems.append(f"missing COLMAP file {sparse_dir / name}")

        if problems:
            raise SceneError(f"{root} is not a usable scene:\n  - " + "\n  - ".join(problems))

        return cls(
            root=root, images_dir=images_dir, sparse_dir=sparse_dir, image_count=image_count
        )
