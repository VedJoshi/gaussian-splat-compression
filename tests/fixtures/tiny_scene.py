"""A synthetic scene small enough to train in seconds.

Cameras sit on a ring looking at the origin, and the images are the 3D points
projected into each view as coloured dots. It is not a good reconstruction and
it does not need to be: it exists so the end-to-end test exercises the real
trainer without a 7 minute wait.

The defaults are chosen to clear gsplat's minimums: at least 8 images so that
test_every=8 leaves a non-empty test split, and enough points for the k=4
nearest-neighbour initialisation in examples/utils.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from tests.fixtures.colmap_bin import (
    PINHOLE,
    look_at_quaternion,
    write_cameras_bin,
    write_images_bin,
    write_points3D_bin,
)


def make_tiny_scene(
    root: Path,
    n_images: int = 24,
    width: int = 96,
    height: int = 72,
    n_points: int = 512,
    seed: int = 0,
) -> Path:
    """Write a complete COLMAP scene under `root` and return `root`."""
    root = Path(root)
    images_dir = root / "images"
    sparse_dir = root / "sparse" / "0"
    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    points = rng.uniform(-1.0, 1.0, size=(n_points, 3))
    colors = rng.integers(0, 256, size=(n_points, 3), dtype=np.uint8)

    focal = float(width)
    cx, cy = width / 2.0, height / 2.0
    intrinsics = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]])

    camera_records = [(1, PINHOLE, width, height, (focal, focal, cx, cy))]
    image_records = []
    tracks: dict[int, list[tuple[int, int]]] = {i: [] for i in range(1, n_points + 1)}

    for index in range(n_images):
        angle = 2 * np.pi * index / n_images
        position = np.array([4.0 * np.cos(angle), 4.0 * np.sin(angle), 1.5])
        qvec, tvec = look_at_quaternion(position, np.zeros(3))

        rotation = _quaternion_to_matrix(qvec)
        camera_points = points @ rotation.T + tvec
        in_front = camera_points[:, 2] > 1e-3
        projected = (camera_points[in_front] @ intrinsics.T)
        pixels = projected[:, :2] / projected[:, 2:3]

        on_screen = (
            (pixels[:, 0] >= 0) & (pixels[:, 0] < width)
            & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
        )
        visible_ids = np.flatnonzero(in_front)[on_screen]
        pixels = pixels[on_screen]

        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        for (x, y), point_index in zip(pixels, visible_ids):
            canvas[int(y), int(x)] = colors[point_index]
        name = f"{index:04d}.png"
        Image.fromarray(canvas).save(images_dir / name)

        for slot, point_index in enumerate(visible_ids):
            tracks[int(point_index) + 1].append((index + 1, slot))

        image_records.append(
            (index + 1, qvec, tvec, 1, name, pixels, visible_ids + 1)
        )

    point_records = [
        (i + 1, points[i], colors[i], 0.5, tracks[i + 1]) for i in range(n_points)
    ]

    write_cameras_bin(sparse_dir / "cameras.bin", camera_records)
    write_images_bin(sparse_dir / "images.bin", image_records)
    write_points3D_bin(sparse_dir / "points3D.bin", point_records)
    return root


def _quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
