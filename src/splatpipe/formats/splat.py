"""Encode Gaussian clouds in the 32-byte web .splat format.

Each record stores three little-endian float32 positions, three float32
exponentiated scales, four uint8 RGBA values, and four uint8 normalised
quaternion components. This totals 32 bytes per Gaussian. The format discards
the view-dependent ``f_rest`` spherical harmonic coefficients and retains only
the DC term used for RGB.
"""

from __future__ import annotations

import numpy as np

from splatpipe.config import ORDERS
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

    # An empty cloud is caught here rather than at each ordering, so that all
    # three orders agree. Left to themselves, "none" and "size_opacity" write a
    # zero-byte artifact while "morton" raises NumPy's raw zero-size reduction
    # ValueError from min(axis=0). A prune that removes every Gaussian is a real
    # milestone 4 ablation, and it should say so rather than emit an empty file.
    if len(cloud) == 0:
        raise ArtifactError("cloud is empty, there is nothing to encode")

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

    # ORDERS is the single vocabulary, shared with ExportConfig, so adding a
    # fourth ordering cannot leave the validator and the encoder disagreeing.
    if order not in ORDERS:
        raise ConfigError(f"unknown order {order!r}, expected one of {', '.join(ORDERS)}")

    if order == "morton":
        cloud = cloud.take(morton_order(cloud.means))
    elif order == "size_opacity":
        cloud = cloud.take(size_opacity_order(cloud.scales, cloud.opacities))

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


# The midpoint of the quantisation bucket an 8-bit alpha came from. Inverting
# logit at the bucket edge diverges; the midpoint is the best estimate the
# stored byte supports. Bounds the recovered logit at about plus or minus 6.23.
_ALPHA_MIN = 0.5 / 255
_ALPHA_MAX = 254.5 / 255
# exp() of a sufficiently negative log scale underflows to zero, and log(0) is
# negative infinity. The smallest positive normal float32 bounds it at -87.34.
_FLOAT32_TINY = float(np.finfo(np.float32).tiny)


def decode_splat(data: bytes) -> GaussianCloud:
    """Decode 32-byte .splat records back into a GaussianCloud.

    Lossy by construction, and deliberately so: the format quantises colour and
    rotation to 8 bits and discards f_rest entirely. The returned cloud has
    sh_degree 0. Nothing here special-cases the loss away, because the
    rate-distortion point has to reflect what the format actually costs.
    """
    if len(data) % BYTES_PER_GAUSSIAN:
        raise ArtifactError(
            f"buffer of {len(data)} bytes is not a multiple of {BYTES_PER_GAUSSIAN}"
        )
    n = len(data) // BYTES_PER_GAUSSIAN
    if n == 0:
        raise ArtifactError("buffer holds no gaussians")

    buffer = np.frombuffer(data, dtype=np.uint8).reshape(n, BYTES_PER_GAUSSIAN)

    means = buffer[:, 0:12].tobytes()
    means = np.frombuffer(means, dtype="<f4").reshape(n, 3).astype(np.float32)

    stored = np.frombuffer(buffer[:, 12:24].tobytes(), dtype="<f4").reshape(n, 3)
    scales = np.log(np.maximum(stored, _FLOAT32_TINY)).astype(np.float32)

    color = buffer[:, 24:28].astype(np.float64) / 255.0
    sh0 = ((color[:, :3] - 0.5) / SH_C0).astype(np.float32)
    alpha = np.clip(color[:, 3], _ALPHA_MIN, _ALPHA_MAX)
    opacities = np.log(alpha / (1.0 - alpha)).astype(np.float32)

    quats = (buffer[:, 28:32].astype(np.float64) - 128.0) / 128.0
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    if np.any(norms == 0):
        row = int(np.flatnonzero(norms.ravel() == 0)[0])
        raise ArtifactError(f"quats decodes to a zero-length quaternion at row {row}")
    quats = (quats / norms).astype(np.float32)

    cloud = GaussianCloud(
        means=means,
        scales=scales,
        quats=quats,
        opacities=opacities,
        sh0=sh0,
        shN=np.zeros((n, 0, 3), dtype=np.float32),
    )
    cloud.validate()
    return cloud
