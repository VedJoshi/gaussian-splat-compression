from __future__ import annotations

import numpy as np
import pytest

from splatpipe.errors import ArtifactError, ConfigError
from splatpipe.formats.container import BLOCK_CODECS, decode_block, encode_block


@pytest.mark.parametrize("codec", sorted(BLOCK_CODECS))
def test_every_codec_round_trips_exactly(codec):
    rng = np.random.default_rng(0)
    payload = rng.integers(0, 256, 5000, dtype=np.uint8).tobytes()
    stored = encode_block(payload, codec)
    assert decode_block(stored, codec, len(payload)) == payload


@pytest.mark.parametrize("codec", sorted(BLOCK_CODECS))
def test_every_codec_round_trips_a_single_byte(codec):
    assert decode_block(encode_block(b"\x07", codec), codec, 1) == b"\x07"


def test_deflate_shrinks_repetitive_data():
    payload = b"\x00" * 20000
    assert len(encode_block(payload, "deflate")) < len(payload) // 10


def test_raw_is_the_identity():
    assert encode_block(b"abc", "raw") == b"abc"


def test_unknown_codec_is_rejected():
    with pytest.raises(ConfigError):
        encode_block(b"abc", "brotli")
    with pytest.raises(ConfigError):
        decode_block(b"abc", "brotli", 3)


def test_decode_rejects_a_length_the_stream_cannot_supply():
    stored = encode_block(b"abcd", "deflate")
    with pytest.raises(ArtifactError):
        decode_block(stored, "deflate", 99)


def test_decode_rejects_a_corrupt_deflate_stream():
    with pytest.raises(ArtifactError):
        decode_block(b"not a deflate stream", "deflate", 8)
