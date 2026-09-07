"""The .splatc container: typed, independently addressable blocks in one file."""

from __future__ import annotations

import io
import math
import zlib

from splatpipe.errors import ArtifactError, ConfigError


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
