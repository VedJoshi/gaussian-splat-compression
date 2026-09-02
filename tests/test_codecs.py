"""Codec round trips. Pure numpy, no GPU, so this runs in the fast tier."""

from __future__ import annotations

import numpy as np
import pytest

from splatpipe.bench.codecs import PlyCodec, SplatCodec, directory_size
from splatpipe.gaussians import GaussianCloud


def a_cloud(n: int = 32) -> GaussianCloud:
    rng = np.random.default_rng(0)
    return GaussianCloud(
        means=rng.uniform(-1, 1, (n, 3)).astype(np.float32),
        scales=rng.uniform(-3, -1, (n, 3)).astype(np.float32),
        quats=rng.normal(size=(n, 4)).astype(np.float32),
        opacities=rng.uniform(-1, 2, n).astype(np.float32),
        sh0=rng.uniform(-1, 1, (n, 3)).astype(np.float32),
        shN=rng.uniform(-0.2, 0.2, (n, 15, 3)).astype(np.float32),
    )


def test_ply_codec_is_lossless(tmp_path):
    """PlyCodec is the 1.00x anchor. If it loses anything the curve has no origin."""
    cloud = a_cloud(64)
    codec = PlyCodec()
    codec.encode(cloud, tmp_path)
    back = codec.decode(tmp_path)
    np.testing.assert_array_equal(back.means, cloud.means)
    np.testing.assert_array_equal(back.shN, cloud.shN)
    assert back.sh_degree == 3


def test_splat_codec_reports_thirty_two_bytes_per_gaussian(tmp_path):
    cloud = a_cloud(64)
    codec = SplatCodec()
    codec.encode(cloud, tmp_path)
    assert codec.size(tmp_path) == 64 * 32


def test_splat_codec_round_trips_through_the_decoder(tmp_path):
    cloud = a_cloud(64)
    codec = SplatCodec(order="none")
    codec.encode(cloud, tmp_path)
    back = codec.decode(tmp_path)
    assert len(back) == 64
    np.testing.assert_array_equal(back.means, cloud.means)


def test_codec_names_are_distinct():
    assert PlyCodec().name != SplatCodec().name


def test_directory_size_counts_every_file_recursively(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "a.bin").write_bytes(b"x" * 10)
    (tmp_path / "nested" / "b.bin").write_bytes(b"y" * 5)
    assert directory_size(tmp_path) == 15
