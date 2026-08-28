"""Encode Gaussian clouds in the 32-byte web .splat format.

Each record stores three little-endian float32 positions, three float32
exponentiated scales, four uint8 RGBA values, and four uint8 normalised
quaternion components. This totals 32 bytes per Gaussian. The format discards
the view-dependent ``f_rest`` spherical harmonic coefficients and retains only
the DC term used for RGB.
"""

from __future__ import annotations

import numpy as np

from splatpipe.errors import ArtifactError, ConfigError
from splatpipe.gaussians import SH_C0, GaussianCloud

BYTES_PER_GAUSSIAN = 32
_FLOAT32_LOG_MAX = np.log(np.finfo(np.float32).max)
_CLOUD_ARRAY_NAMES = ("means", "scales", "quats", "opacities", "sh0", "shN")


def _part1by2(x: np.ndarray) -> np.ndarray:
    """Spread the low 10 bits of x out with two zero bits between each."""
    x = x.astype(np.uint64) & np.uint64(0x000003FF)
    x = (x ^ (x << np.uint64(16))) & np.uint64(0xFF0000FF)
    x = (x ^ (x << np.uint64(8))) & np.uint64(0x0300F00F)
    x = (x ^ (x << np.uint64(4))) & np.uint64(0x030C30C3)
    x = (x ^ (x << np.uint64(2))) & np.uint64(0x09249249)
    return x


def morton_order(means: np.ndarray) -> np.ndarray:
    """Return indices sorted by Morton code for spatial locality.

    This preserves gsplat's maximum-value quirk: scaling the largest coordinate
    produces 1024, and _part1by2 truncates it to the low 10 bits.
    """
    lo = means.min(axis=0)
    lengths = means.max(axis=0) - lo
    lengths[lengths == 0] = 1
    scaled = np.floor((means - lo) / lengths * 1024).astype(np.int64)
    x, y, z = scaled[:, 0], scaled[:, 1], scaled[:, 2]
    codes = (_part1by2(z) << np.uint64(2)) + (_part1by2(y) << np.uint64(1)) + _part1by2(x)
    return np.argsort(codes)


def size_opacity_order(scales: np.ndarray, opacities: np.ndarray) -> np.ndarray:
    """Return the biggest visible splats first for progressive loading."""
    log_weight = scales.sum(axis=1) - np.logaddexp(0, -opacities)
    return np.argsort(-log_weight)


def _validate_splat_inputs(cloud: GaussianCloud) -> None:
    cloud.validate()

    for name in _CLOUD_ARRAY_NAMES:
        if not np.isfinite(getattr(cloud, name)).all():
            raise ArtifactError(f"{name} contains non-finite values")

    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        norms = np.linalg.norm(cloud.quats, axis=1)
    invalid_norm_rows = np.flatnonzero((norms == 0) | ~np.isfinite(norms))
    if len(invalid_norm_rows):
        row = invalid_norm_rows[0]
        if norms[row] == 0:
            raise ArtifactError(f"quats contains zero-length quaternion at row {row}")
        raise ArtifactError(f"quats contains non-normalizable quaternion at row {row}")

    if np.any(cloud.scales >= _FLOAT32_LOG_MAX):
        raise ArtifactError("scales contains values that overflow float32 when exponentiated")


def encode_splat(cloud: GaussianCloud, order: str = "morton") -> bytes:
    """Pack a GaussianCloud into little-endian 32-byte .splat records."""
    _validate_splat_inputs(cloud)

    if order == "morton":
        cloud = cloud.take(morton_order(cloud.means))
    elif order == "size_opacity":
        cloud = cloud.take(size_opacity_order(cloud.scales, cloud.opacities))
    elif order != "none":
        raise ConfigError(f"unknown order {order!r}, expected morton, size_opacity or none")

    n = len(cloud)
    positions = np.ascontiguousarray(cloud.means, dtype="<f4")
    scales = np.ascontiguousarray(np.exp(cloud.scales, dtype=np.float64), dtype="<f4")

    rgb = cloud.sh0 * SH_C0 + 0.5
    with np.errstate(over="ignore"):
        alpha = 1.0 / (1.0 + np.exp(-cloud.opacities))
        color_values = np.concatenate([rgb, alpha[:, None]], axis=1) * 255
    color = np.clip(color_values, 0, 255).astype(np.uint8)

    quats = cloud.quats / np.linalg.norm(cloud.quats, axis=1, keepdims=True)
    rotation = np.clip(quats * 128 + 128, 0, 255).astype(np.uint8)

    buffer = np.empty((n, BYTES_PER_GAUSSIAN), dtype=np.uint8)
    buffer[:, 0:12] = positions.view(np.uint8).reshape(n, 12)
    buffer[:, 12:24] = scales.view(np.uint8).reshape(n, 12)
    buffer[:, 24:28] = color
    buffer[:, 28:32] = rotation
    return buffer.tobytes()
