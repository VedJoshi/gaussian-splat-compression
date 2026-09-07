"""The .splatc container: typed, independently addressable blocks in one file."""

from __future__ import annotations

import io
import math
import zlib
from dataclasses import dataclass

import numpy as np

from splatpipe.compress.quantize import PackedField, quantize_affine
from splatpipe.errors import ArtifactError, ConfigError
from splatpipe.gaussians import GaussianCloud


def _encode_raw(data: bytes) -> bytes:
    return data


def _decode_raw(data: bytes) -> bytes:
    return data


def _encode_deflate(data: bytes) -> bytes:
    # zlib-wrapped, not raw: this is what DecompressionStream("deflate") reads.
    return zlib.compress(data, 9)


def _decode_deflate(data: bytes) -> bytes:
    try:
        return zlib.decompress(data)
    except zlib.error as error:
        raise ArtifactError(f"block is not a valid deflate stream: {error}") from error


def _png_size(length: int) -> tuple[int, int]:
    """A square-ish grid, so PNG's row filters have a neighbourhood to exploit."""
    width = max(1, math.ceil(math.sqrt(length)))
    return width, math.ceil(length / width)


def _pillow():
    try:
        from PIL import Image
    except ImportError as error:
        raise ConfigError(
            "the png block codec needs Pillow. Install the bench extra."
        ) from error
    return Image


def _encode_png(data: bytes) -> bytes:
    Image = _pillow()
    width, height = _png_size(len(data))
    padded = data + bytes(width * height - len(data))
    buffer = io.BytesIO()
    Image.frombytes("L", (width, height), padded).save(
        buffer, format="PNG", optimize=True
    )
    return buffer.getvalue()


def _decode_png(data: bytes) -> bytes:
    Image = _pillow()
    with Image.open(io.BytesIO(data)) as image:
        return image.convert("L").tobytes()


BLOCK_CODECS = {
    "raw": (_encode_raw, _decode_raw),
    "deflate": (_encode_deflate, _decode_deflate),
    "png": (_encode_png, _decode_png),
}


def _codec(name: str):
    if name not in BLOCK_CODECS:
        raise ConfigError(
            f"unknown block codec {name!r}. Known codecs are: "
            f"{', '.join(sorted(BLOCK_CODECS))}"
        )
    return BLOCK_CODECS[name]


def encode_block(data: bytes, codec: str) -> bytes:
    return _codec(codec)[0](data)


def decode_block(data: bytes, codec: str, raw_length: int) -> bytes:
    decoded = _codec(codec)[1](data)
    # png pads its final row and deflate can be truncated, so the descriptor's
    # length is the authority on where the block actually ends.
    if len(decoded) < raw_length:
        raise ArtifactError(
            f"block decoded to {len(decoded):,} bytes, expected {raw_length:,}"
        )
    return decoded[:raw_length]


CONTAINER_ORDERS = ("none", "morton", "plas")

# The baseline scheme, held fixed so ordering, layout and entropy coder are the
# only variables in the milestone 5 measurement.
FIELD_BITS = {
    "means": 16,
    "scales": 8,
    "quats": 8,
    "opacities": 8,
    "sh0": 8,
    "shN": 8,
}


@dataclass(frozen=True)
class ShVq:
    codebook: np.ndarray  # (clusters, K*3) uint8, already quantised
    labels: np.ndarray  # (N,) uint8 or uint16
    mins: np.ndarray  # (K*3,) float32
    maxs: np.ndarray  # (K*3,) float32
    bits: int


@dataclass(frozen=True)
class PackedScene:
    count: int
    sh_degree: int
    order: str
    fields: dict[str, PackedField]


def _permutation(cloud: GaussianCloud, order: str) -> np.ndarray | None:
    if order not in CONTAINER_ORDERS:
        raise ConfigError(
            f"unknown order {order!r}, expected one of {', '.join(CONTAINER_ORDERS)}"
        )
    if order == "none":
        return None
    if order == "morton":
        from splatpipe.formats.splat import morton_order

        return morton_order(cloud.means)

    # PLAS is a 2D optimal-transport sort and needs a square grid, so it drops
    # the remainder. It exists here only as the measurement's reference point.
    import torch
    from plas import sort_with_plas

    side = int(math.isqrt(len(cloud)))
    if side < 16:
        raise ArtifactError(
            f"plas ordering needs at least 256 Gaussians, got {len(cloud):,}"
        )
    keep = side * side
    grid = torch.from_numpy(cloud.means[:keep]).cuda().permute(1, 0).reshape(3, side, side)
    _, indices = sort_with_plas(grid, improvement_break=1e-4, verbose=False)
    return indices.cpu().numpy().reshape(-1)


def _packed(values: np.ndarray, bits: int) -> PackedField:
    """Store flattened (N, C) values; PackedField.shape restores the rank."""
    quantized, mins, maxs = quantize_affine(values.reshape(len(values), -1), bits)
    return PackedField(
        values=quantized,
        bits=bits,
        mins=mins,
        maxs=maxs,
        shape=tuple(values.shape),
    )


def pack_scene(
    cloud: GaussianCloud,
    sh_vq: ShVq | None = None,
    order: str = "morton",
) -> PackedScene:
    cloud.validate()
    if len(cloud) == 0:
        raise ArtifactError("cloud is empty, there is nothing to pack")

    labels = None
    if sh_vq is not None:
        if len(sh_vq.labels) != len(cloud):
            raise ArtifactError(
                f"sh labels have {len(sh_vq.labels):,} entries "
                f"for {len(cloud):,} Gaussians"
            )
        if int(sh_vq.labels.max(initial=0)) >= len(sh_vq.codebook):
            raise ArtifactError(
                f"sh label {int(sh_vq.labels.max())} exceeds the "
                f"{len(sh_vq.codebook):,}-entry codebook"
            )
        labels = sh_vq.labels

    permutation = _permutation(cloud, order)
    if permutation is not None:
        cloud = cloud.take(permutation)
        # The labels index a shared codebook per Gaussian. Permuting the cloud
        # without permuting these rebinds every Gaussian's colour, and nothing
        # downstream would notice.
        if labels is not None:
            labels = labels[permutation]

    fields = {
        name: _packed(getattr(cloud, name), FIELD_BITS[name])
        for name in ("means", "scales", "quats", "opacities", "sh0")
    }
    if sh_vq is None:
        fields["shN"] = _packed(cloud.shN, FIELD_BITS["shN"])
    else:
        fields["sh_codebook"] = PackedField(
            values=sh_vq.codebook,
            bits=sh_vq.bits,
            mins=np.asarray(sh_vq.mins, dtype=np.float32),
            maxs=np.asarray(sh_vq.maxs, dtype=np.float32),
            shape=tuple(sh_vq.codebook.shape),
        )
        # Labels are indices, not measurements: they are stored verbatim with a
        # zero range so nothing rescales them, and are never dequantised.
        fields["sh_labels"] = PackedField(
            values=labels,
            bits=8 if labels.dtype == np.uint8 else 16,
            mins=np.zeros(1, dtype=np.float32),
            maxs=np.zeros(1, dtype=np.float32),
            shape=(len(labels),),
        )

    return PackedScene(
        count=len(cloud),
        sh_degree=cloud.sh_degree,
        order=order,
        fields=fields,
    )


def unpack_scene(scene: PackedScene) -> GaussianCloud:
    fields = scene.fields
    missing = {"means", "scales", "quats", "opacities", "sh0"} - set(fields)
    if missing:
        raise ArtifactError(
            f"container is missing block(s): {', '.join(sorted(missing))}"
        )

    if "shN" in fields:
        shN = fields["shN"].dequantize()
    elif {"sh_codebook", "sh_labels"} <= set(fields):
        codebook = fields["sh_codebook"].dequantize()
        labels = fields["sh_labels"].values
        if int(labels.max(initial=0)) >= len(codebook):
            raise ArtifactError("a stored sh label exceeds the stored codebook")
        coefficients = codebook.shape[1] // 3
        shN = codebook[labels].reshape(len(labels), coefficients, 3)
    else:
        raise ArtifactError("container has neither an shN block nor a codebook pair")

    cloud = GaussianCloud(
        means=fields["means"].dequantize(),
        scales=fields["scales"].dequantize(),
        quats=fields["quats"].dequantize(),
        opacities=fields["opacities"].dequantize(),
        sh0=fields["sh0"].dequantize(),
        shN=np.ascontiguousarray(shN, dtype=np.float32),
    )
    cloud.validate()
    return cloud
