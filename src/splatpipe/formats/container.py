"""The .splatc container: typed, independently addressable blocks in one file."""

from __future__ import annotations

import io
import json
import math
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from splatpipe.compress.quantize import PackedField, quantize_affine
from splatpipe.errors import ArtifactError, ConfigError
from splatpipe.gaussians import GaussianCloud
from splatpipe.manifest import collect_versions


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


def log_transform(values: np.ndarray) -> np.ndarray:
    """gsplat's own means transform, matched exactly so the PNG baseline and the
    container quantise the same numbers.

    Truck's means span [-5720, 8781] because of a handful of stray Gaussians,
    while 99.9% of the scene lies within +/-24. Quantising that raw range at 16
    bits costs 0.04 units of position RMS, which is several splat widths, and
    renders at 13.6 dB. In log space the same 16 bits cost ~1e-4 units.
    """
    return np.sign(values) * np.log1p(np.abs(values))


def inverse_log_transform(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.expm1(np.abs(values))


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

    fields = {"means": _packed(log_transform(cloud.means), FIELD_BITS["means"])}
    fields.update(
        (name, _packed(getattr(cloud, name), FIELD_BITS[name]))
        for name in ("scales", "quats", "opacities", "sh0")
    )
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
        means=inverse_log_transform(fields["means"].dequantize()),
        scales=fields["scales"].dequantize(),
        quats=fields["quats"].dequantize(),
        opacities=fields["opacities"].dequantize(),
        sh0=fields["sh0"].dequantize(),
        shN=np.ascontiguousarray(shN, dtype=np.float32),
    )
    cloud.validate()
    return cloud


MAGIC = b"SPLATC"
VERSION_MAJOR = 1
VERSION_MINOR = 0
PREFIX_LEN = 24
# Eight bytes covers every typed-array view a viewer can take over the payload.
# Uint16Array at an odd offset throws in JavaScript, so this is not cosmetic.
ALIGNMENT = 8
_PREFIX = "<6sBBIIQ"
DEFAULT_CODEC = "deflate"


def _pad_to(length: int) -> int:
    return (-length) % ALIGNMENT


def write_container(
    scene: PackedScene,
    path: Path | str,
    codecs: dict[str, str] | None = None,
) -> None:
    codecs = dict(codecs or {})
    unknown = set(codecs) - set(scene.fields)
    if unknown:
        raise ConfigError(
            f"codec given for unknown block(s): {', '.join(sorted(unknown))}"
        )

    blocks = []
    payload = bytearray()
    for name, field in scene.fields.items():
        codec = codecs.get(name, DEFAULT_CODEC)
        values = np.ascontiguousarray(field.values)
        raw = values.tobytes()
        stored = encode_block(raw, codec)
        payload.extend(bytes(_pad_to(len(payload))))
        blocks.append(
            {
                "name": name,
                "codec": codec,
                "offset": len(payload),
                "length": len(stored),
                "raw_length": len(raw),
                "crc32": zlib.crc32(stored) & 0xFFFFFFFF,
                "dtype": values.dtype.name,
                "stored_shape": list(values.shape),
                "shape": list(field.shape),
                "bits": field.bits,
                "mins": [float(v) for v in np.ravel(field.mins)],
                "maxs": [float(v) for v in np.ravel(field.maxs)],
            }
        )
        payload.extend(stored)

    descriptor = {
        "count": int(scene.count),
        "sh_degree": int(scene.sh_degree),
        "order": scene.order,
        "versions": collect_versions(),
        "blocks": blocks,
    }
    encoded = json.dumps(descriptor, separators=(",", ":")).encode("utf-8")
    # Padded with spaces, not NULs: JSON.parse tolerates trailing whitespace and
    # rejects NUL, and the viewer parses this slice directly.
    encoded += b" " * _pad_to(PREFIX_LEN + len(encoded))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(
            struct.pack(
                _PREFIX,
                MAGIC,
                VERSION_MAJOR,
                VERSION_MINOR,
                len(encoded),
                zlib.crc32(encoded) & 0xFFFFFFFF,
                len(payload),
            )
        )
        handle.write(encoded)
        handle.write(payload)


def _read_descriptor(data: bytes) -> tuple[dict, memoryview]:
    if len(data) < PREFIX_LEN:
        raise ArtifactError(f"file is {len(data)} bytes, shorter than the header")
    magic, major, _minor, json_len, json_crc, payload_len = struct.unpack(
        _PREFIX, data[:PREFIX_LEN]
    )
    if magic != MAGIC:
        raise ArtifactError(f"not a splatc container: magic is {magic!r}")
    if major != VERSION_MAJOR:
        raise ArtifactError(
            f"container major version {major} is not readable by this build, "
            f"which reads version {VERSION_MAJOR}"
        )
    end = PREFIX_LEN + json_len
    if len(data) < end + payload_len:
        raise ArtifactError(
            f"container declares {json_len:,} descriptor and {payload_len:,} payload "
            f"bytes but holds {len(data):,} in total"
        )
    encoded = data[PREFIX_LEN:end]
    if zlib.crc32(encoded) & 0xFFFFFFFF != json_crc:
        raise ArtifactError("descriptor checksum does not match its contents")
    try:
        descriptor = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactError(f"descriptor is not valid JSON: {error}") from error
    return descriptor, memoryview(data)[end : end + payload_len]


def container_descriptor(path: Path | str) -> dict:
    return _read_descriptor(Path(path).read_bytes())[0]


def read_container(path: Path | str) -> PackedScene:
    descriptor, payload = _read_descriptor(Path(path).read_bytes())

    seen: list[tuple[int, int, str]] = []
    fields: dict[str, PackedField] = {}
    for block in descriptor["blocks"]:
        name = block["name"]
        offset, length = int(block["offset"]), int(block["length"])
        if offset < 0 or length < 0 or offset + length > len(payload):
            raise ArtifactError(
                f"block {name!r} spans {offset:,}..{offset + length:,} "
                f"outside a {len(payload):,}-byte payload"
            )
        seen.append((offset, offset + length, name))
        stored = bytes(payload[offset : offset + length])
        if zlib.crc32(stored) & 0xFFFFFFFF != int(block["crc32"]):
            raise ArtifactError(f"block {name!r} failed its checksum")

        raw = decode_block(stored, block["codec"], int(block["raw_length"]))
        values = np.frombuffer(raw, dtype=np.dtype(block["dtype"])).reshape(
            tuple(block["stored_shape"])
        )
        fields[name] = PackedField(
            values=values,
            bits=int(block["bits"]),
            mins=np.asarray(block["mins"], dtype=np.float32),
            maxs=np.asarray(block["maxs"], dtype=np.float32),
            shape=tuple(block["shape"]),
        )

    seen.sort()
    for (_, end, first), (start, _, second) in zip(seen, seen[1:]):
        if start < end:
            raise ArtifactError(f"blocks {first!r} and {second!r} overlap")

    return PackedScene(
        count=int(descriptor["count"]),
        sh_degree=int(descriptor["sh_degree"]),
        order=descriptor["order"],
        fields=fields,
    )
