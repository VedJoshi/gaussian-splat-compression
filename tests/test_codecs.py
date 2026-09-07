"""Codec round trips. The ply and splat cases are pure numpy and run in the fast
tier; the png cases are GPU-marked and deselected there."""

from __future__ import annotations

import numpy as np
import pytest

from splatpipe.bench.codecs import (
    Codec,
    PlyCodec,
    PngCodec,
    ShVqCodec,
    SplatCodec,
    directory_size,
)
from splatpipe.errors import ConfigError
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
    cloud = a_cloud(64)
    codec = PlyCodec()
    codec.encode(cloud, tmp_path)
    back = codec.decode(tmp_path)
    for name in ("means", "scales", "quats", "opacities", "sh0", "shN"):
        np.testing.assert_array_equal(getattr(back, name), getattr(cloud, name), err_msg=name)
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
    assert cloud.sh_degree == 3
    assert back.sh_degree == 0
    np.testing.assert_array_equal(back.means, cloud.means)


def test_both_codecs_satisfy_the_codec_protocol():
    class NotACodec:
        name = "incomplete"

        def encode(self, cloud, directory): ...

        def decode(self, directory): ...

    assert isinstance(PlyCodec(), Codec)
    assert isinstance(SplatCodec(), Codec)
    assert not isinstance(NotACodec(), Codec)


def test_each_codec_measures_only_its_own_directory(tmp_path):
    cloud = a_cloud(64)
    codecs = [PlyCodec(), SplatCodec(order="none")]
    assert len({codec.name for codec in codecs}) == len(codecs)

    sizes = {}
    for codec in codecs:
        codec.encode(cloud, tmp_path / codec.name)
        sizes[codec.name] = codec.size(tmp_path / codec.name)

    assert sizes["splat"] == 64 * 32
    assert sizes["ply"] + sizes["splat"] == directory_size(tmp_path)


@pytest.mark.parametrize("codec", (PlyCodec(), SplatCodec()), ids=("ply", "splat"))
def test_encode_creates_a_target_directory_that_does_not_exist(tmp_path, codec):
    target = tmp_path / "made_by_encode"
    codec.encode(a_cloud(8), target)
    assert target.is_dir()
    assert codec.size(target) > 0


def test_directory_size_counts_every_file_recursively(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "a.bin").write_bytes(b"x" * 10)
    (tmp_path / "nested" / "b.bin").write_bytes(b"y" * 5)
    assert directory_size(tmp_path) == 15


def test_png_codec_satisfies_the_protocol_and_names_itself_distinctly():
    assert isinstance(PngCodec(), Codec)
    assert PngCodec().name == "png"
    assert len({codec.name for codec in (PlyCodec(), SplatCodec(), PngCodec())}) == 3


def test_sh_vq_codec_validates_its_configuration():
    with pytest.raises(ConfigError, match="seed"):
        PngCodec(seed=-1)
    with pytest.raises(ConfigError, match="at least 2"):
        ShVqCodec(1)
    with pytest.raises(ConfigError, match="seed"):
        ShVqCodec(256, seed=-1)


# PngCompression has no CPU path.
pngmark = pytest.mark.gpu


@pngmark
def test_png_codec_round_trips_a_square_cloud(tmp_path):
    # The stock SH codebook fixes the minimum cloud size at 65,536.
    cloud = a_cloud(65536)
    codec = PngCodec()
    directory = tmp_path / "png"
    codec.encode(cloud, directory)
    back = codec.decode(directory)
    assert directory.is_dir()
    assert len(back) == 65536
    assert back.sh_degree == 3
    assert codec.size(directory) > 0
    np.testing.assert_allclose(
        np.sort(back.means, axis=0), np.sort(cloud.means, axis=0), rtol=0, atol=1e-4
    )
    np.testing.assert_allclose(
        np.sort(back.scales, axis=0), np.sort(cloud.scales, axis=0), rtol=0, atol=1e-2
    )


@pngmark
def test_png_codec_does_not_mutate_the_cloud_it_is_given(tmp_path):
    names = ("means", "scales", "quats", "opacities", "sh0", "shN")
    cloud = a_cloud(65536)
    before = {name: getattr(cloud, name).copy() for name in names}
    PngCodec().encode(cloud, tmp_path)
    for name in names:
        np.testing.assert_array_equal(getattr(cloud, name), before[name], err_msg=name)


@pngmark
def test_png_codec_reports_the_count_it_actually_encoded(tmp_path):
    codec = PngCodec()
    codec.encode(a_cloud(65537), tmp_path)
    assert len(codec.decode(tmp_path)) == 65536
