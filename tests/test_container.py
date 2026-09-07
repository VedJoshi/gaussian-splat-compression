from __future__ import annotations

import numpy as np
import pytest

from splatpipe.errors import ArtifactError, ConfigError
from splatpipe.formats.container import (
    BLOCK_CODECS,
    CONTAINER_ORDERS,
    ShVq,
    decode_block,
    encode_block,
    pack_scene,
    unpack_scene,
)
from splatpipe.formats.splat import morton_order
from tests.test_codecs import a_cloud


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


def test_pack_unpack_recovers_every_field_within_its_bound():
    cloud = a_cloud(256)
    scene = pack_scene(cloud, order="none")
    back = unpack_scene(scene)
    assert len(back) == 256
    assert back.sh_degree == 3
    for name in ("means", "scales", "quats", "opacities", "sh0", "shN"):
        original = getattr(cloud, name)
        recovered = getattr(back, name)
        assert recovered.shape == original.shape, name
        assert recovered.dtype == np.float32, name
        span = original.max() - original.min()
        assert np.abs(recovered - original).max() <= span, name


def test_means_are_stored_at_sixteen_bits_and_the_rest_at_eight():
    scene = pack_scene(a_cloud(64), order="none")
    assert scene.fields["means"].values.dtype == np.uint16
    for name in ("scales", "quats", "opacities", "sh0"):
        assert scene.fields[name].values.dtype == np.uint8, name


def test_morton_order_permutes_the_stored_gaussians():
    cloud = a_cloud(128)
    scene = pack_scene(cloud, order="morton")
    assert scene.order == "morton"
    expected = cloud.take(morton_order(cloud.means))
    back = unpack_scene(scene)
    # Morton is a permutation, so the recovered means match the permuted cloud
    # position for position, within quantisation error, and are a permutation of
    # the original set rather than the original ordering.
    span = cloud.means.max() - cloud.means.min()
    assert np.abs(back.means - expected.means).max() <= span / (2 * 65535) + 1e-6
    np.testing.assert_allclose(
        np.sort(back.means, axis=0), np.sort(cloud.means, axis=0), atol=span / 65535 + 1e-6
    )


def test_reordering_carries_the_sh_labels_with_the_cloud():
    cloud = a_cloud(64)
    rng = np.random.default_rng(3)
    sh_vq = ShVq(
        codebook=rng.integers(0, 64, (8, 45), dtype=np.uint8),
        labels=np.arange(64, dtype=np.uint8) % 8,
        mins=np.full(45, -0.2, dtype=np.float32),
        maxs=np.full(45, 0.2, dtype=np.float32),
        bits=6,
    )
    scene = pack_scene(cloud, sh_vq=sh_vq, order="morton")
    permutation = morton_order(cloud.means)
    np.testing.assert_array_equal(
        scene.fields["sh_labels"].values, sh_vq.labels[permutation]
    )


def test_sh_vq_scene_stores_a_codebook_and_labels_instead_of_shN():
    cloud = a_cloud(64)
    rng = np.random.default_rng(4)
    sh_vq = ShVq(
        codebook=rng.integers(0, 64, (8, 45), dtype=np.uint8),
        labels=np.zeros(64, dtype=np.uint8),
        mins=np.full(45, -0.2, dtype=np.float32),
        maxs=np.full(45, 0.2, dtype=np.float32),
        bits=6,
    )
    scene = pack_scene(cloud, sh_vq=sh_vq, order="none")
    assert "shN" not in scene.fields
    assert {"sh_codebook", "sh_labels"} <= set(scene.fields)
    back = unpack_scene(scene)
    assert back.shN.shape == (64, 15, 3)
    # Every label is 0, so every Gaussian decodes to the same SH vector.
    assert np.abs(back.shN - back.shN[0]).max() == 0


def test_labels_out_of_codebook_range_are_rejected():
    cloud = a_cloud(64)
    sh_vq = ShVq(
        codebook=np.zeros((4, 45), dtype=np.uint8),
        labels=np.full(64, 9, dtype=np.uint8),
        mins=np.zeros(45, dtype=np.float32),
        maxs=np.ones(45, dtype=np.float32),
        bits=6,
    )
    with pytest.raises(ArtifactError):
        pack_scene(cloud, sh_vq=sh_vq, order="none")


def test_label_count_must_match_the_cloud():
    cloud = a_cloud(64)
    sh_vq = ShVq(
        codebook=np.zeros((4, 45), dtype=np.uint8),
        labels=np.zeros(63, dtype=np.uint8),
        mins=np.zeros(45, dtype=np.float32),
        maxs=np.ones(45, dtype=np.float32),
        bits=6,
    )
    with pytest.raises(ArtifactError):
        pack_scene(cloud, sh_vq=sh_vq, order="none")


def test_unknown_order_is_rejected():
    with pytest.raises(ConfigError):
        pack_scene(a_cloud(16), order="spiral")


def test_every_declared_order_is_accepted_or_needs_a_gpu():
    assert CONTAINER_ORDERS == ("none", "morton", "plas")
