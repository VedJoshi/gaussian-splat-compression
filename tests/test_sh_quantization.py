from __future__ import annotations

import json

import numpy as np
import pytest

from splatpipe.bench.codecs import Codec, ShVqCodec, build_codecs
from splatpipe.compress.sh import (
    dequantize_codebook,
    label_dtype,
    quantize_codebook,
    seeded_compression,
)
from splatpipe.errors import ArtifactError, ConfigError
from tests.test_codecs import a_cloud


def test_codebook_round_trip_stays_within_half_a_quantization_step():
    rng = np.random.default_rng(4)
    centroids = rng.uniform(-2, 3, (64, 45)).astype(np.float32)
    quantized, mins, maxs = quantize_codebook(centroids, bits=6)
    decoded = dequantize_codebook(quantized, mins, maxs, bits=6)

    allowance = (maxs - mins) / (2 * 63) + np.finfo(np.float32).eps
    assert np.all(np.abs(decoded - centroids) <= allowance)
    assert quantized.dtype == np.uint8


def test_constant_codebook_components_round_trip_exactly():
    centroids = np.full((8, 3), 0.25, dtype=np.float32)
    quantized, mins, maxs = quantize_codebook(centroids, bits=4)
    decoded = dequantize_codebook(quantized, mins, maxs, bits=4)
    np.testing.assert_array_equal(decoded, centroids)


@pytest.mark.parametrize("bits", (0, 9, True, 4.0))
def test_invalid_codebook_bits_are_rejected(bits):
    centroids = np.zeros((4, 3), dtype=np.float32)
    with pytest.raises(ConfigError, match="bits"):
        quantize_codebook(centroids, bits)


def test_nonfinite_centroids_are_rejected():
    centroids = np.zeros((4, 3), dtype=np.float32)
    centroids[0, 0] = np.nan
    with pytest.raises(ArtifactError, match="non-finite"):
        quantize_codebook(centroids, bits=6)


@pytest.mark.parametrize(
    ("mins", "maxs", "message"),
    (
        (np.array([np.nan], dtype=np.float32), np.ones(1, dtype=np.float32), "non-finite"),
        (np.ones(1, dtype=np.float32), np.zeros(1, dtype=np.float32), "below"),
    ),
)
def test_invalid_codebook_bounds_are_rejected(mins, maxs, message):
    quantized = np.zeros((2, 1), dtype=np.uint8)
    with pytest.raises(ArtifactError, match=message):
        dequantize_codebook(quantized, mins, maxs, bits=6)


def test_cluster_indices_use_the_narrowest_supported_type():
    assert label_dtype(256) is np.uint8
    assert label_dtype(257) is np.uint16
    assert label_dtype(65_536) is np.uint16
    with pytest.raises(ConfigError, match="65,536"):
        label_dtype(65_537)


def test_sh_vq_codecs_are_available_to_the_benchmark():
    codecs = build_codecs("shvq256,shvq1024,shvq4096")
    assert [codec.name for codec in codecs] == ["shvq256", "shvq1024", "shvq4096"]
    assert all(isinstance(codec, Codec) for codec in codecs)


@pytest.mark.gpu
def test_sh_vq_codec_records_and_uses_its_codebook(tmp_path):
    codec = ShVqCodec(16, use_sort=False)
    cloud = a_cloud(256)
    codec.encode(cloud, tmp_path)
    decoded = codec.decode(tmp_path)

    meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))["shN"]
    assert meta["n_clusters"] == 16
    assert meta["quantization"] == 6
    with np.load(tmp_path / "shN.npz", allow_pickle=False) as archive:
        assert archive["labels"].dtype == np.uint8
        assert archive["centroids"].shape == (16, 45)
    assert np.unique(decoded.shN.reshape(len(decoded), -1), axis=0).shape[0] <= 16
    assert decoded.shN.shape == cloud.shN.shape


@pytest.mark.gpu
def test_seeded_compression_restores_random_state_after_an_error():
    import torch

    numpy_before = np.random.get_state()
    torch_before = torch.random.get_rng_state()
    cuda_before = torch.cuda.get_rng_state_all()

    with pytest.raises(RuntimeError, match="stop"):
        with seeded_compression(42):
            np.random.random()
            torch.rand(1)
            torch.rand(1, device="cuda")
            raise RuntimeError("stop")

    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    np.testing.assert_array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_before)
    assert all(
        torch.equal(after, before)
        for after, before in zip(torch.cuda.get_rng_state_all(), cuda_before)
    )
