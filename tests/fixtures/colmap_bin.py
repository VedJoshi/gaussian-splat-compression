"""Write COLMAP sparse model binaries.

Every format string is explicitly little-endian with explicit widths. pycolmap's
own write path (scene_manager.py lines 313-421) still uses native 'L' and is
broken on Windows, so it cannot be used as a reference implementation here.

Layouts taken from pycolmap's reader in scene_manager.py.
"""

from __future__ import annotations

import struct
from collections.abc import Sequence
from pathlib import Path

import numpy as np

PINHOLE = 1  # model id; 4 params: fx, fy, cx, cy


def write_cameras_bin(path: Path, cameras: Sequence[tuple[int, int, int, int, Sequence[float]]]) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(cameras)))
        for camera_id, model_id, width, height, params in cameras:
            f.write(struct.pack("<IiQQ", camera_id, model_id, width, height))
            f.write(struct.pack(f"<{len(params)}d", *params))


def write_images_bin(path: Path, images: Sequence[tuple]) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for image_id, qvec, tvec, camera_id, name, xys, point3D_ids in images:
            # count2D is written from len(xys) and the loop below zips, so a
            # caller passing lists of different lengths would produce a
            # structurally valid file whose header contradicts its contents.
            # That is silent corruption in the fixture the end-to-end test
            # depends on, and it fails far away from its cause.
            if len(xys) != len(point3D_ids):
                raise ValueError(
                    f"image {name}: {len(xys)} xys but {len(point3D_ids)} point3D_ids"
                )
            f.write(struct.pack("<I4d3dI", image_id, *qvec, *tvec, camera_id))
            f.write(name.encode("utf-8") + b"\x00")
            f.write(struct.pack("<Q", len(xys)))
            for (x, y), point_id in zip(xys, point3D_ids):
                # Two doubles then a raw uint64. The reader slurps three doubles
                # and reinterprets the third one's bits as the point id.
                f.write(struct.pack("<2d", float(x), float(y)))
                f.write(struct.pack("<Q", int(point_id)))


def write_points3D_bin(path: Path, points: Sequence[tuple]) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(points)))
        for point_id, xyz, rgb, error, track in points:
            f.write(
                struct.pack(
                    "<Q3d3BdQ",
                    point_id,
                    *(float(v) for v in xyz),
                    *(int(v) for v in rgb),
                    float(error),
                    len(track),
                )
            )
            for image_id, point2D_idx in track:
                f.write(struct.pack("<II", int(image_id), int(point2D_idx)))


def look_at_quaternion(camera_position: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (qvec, tvec) in COLMAP's world-to-camera convention.

    qvec is (w, x, y, z). COLMAP stores the rotation that takes world points
    into the camera frame, and tvec is that same transform's translation, so
    tvec = -R @ camera_position rather than the camera position itself.
    """
    forward = target - camera_position
    forward = forward / np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)

    # Rows are the camera axes expressed in world coordinates: x right, y down,
    # z forward, which is COLMAP's image convention.
    rotation = np.stack([right, down, forward], axis=0)
    translation = -rotation @ camera_position

    # w = sqrt(1 + trace) / 2 divides by a value that goes to zero near a
    # 180-degree rotation. That is not a remote corner case for this ring: at
    # n_images=24 the quarter-turn camera sits exactly on the +y axis with
    # forward's x-component zero to machine precision, which drives
    # trace to exactly -1.0 (verified bit-exact, not merely close). Extract
    # the quaternion from whichever of w, x, y, z has the largest magnitude
    # instead, the standard robust matrix-to-quaternion construction.
    trace = np.trace(rotation)
    if trace > 0:
        w = np.sqrt(1.0 + trace) / 2.0
        x = (rotation[2, 1] - rotation[1, 2]) / (4 * w)
        y = (rotation[0, 2] - rotation[2, 0]) / (4 * w)
        z = (rotation[1, 0] - rotation[0, 1]) / (4 * w)
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        w = (rotation[2, 1] - rotation[1, 2]) / s
        x = s / 4.0
        y = (rotation[0, 1] + rotation[1, 0]) / s
        z = (rotation[0, 2] + rotation[2, 0]) / s
    elif rotation[1, 1] > rotation[2, 2]:
        s = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        w = (rotation[0, 2] - rotation[2, 0]) / s
        x = (rotation[0, 1] + rotation[1, 0]) / s
        y = s / 4.0
        z = (rotation[1, 2] + rotation[2, 1]) / s
    else:
        s = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        w = (rotation[1, 0] - rotation[0, 1]) / s
        x = (rotation[0, 2] + rotation[2, 0]) / s
        y = (rotation[1, 2] + rotation[2, 1]) / s
        z = s / 4.0
    return np.array([w, x, y, z]), translation
