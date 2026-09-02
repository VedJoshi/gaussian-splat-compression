"""Codec round trips. Pure numpy, no GPU, so this runs in the fast tier."""

from __future__ import annotations

import numpy as np
import pytest

from splatpipe.bench.codecs import Codec, PlyCodec, SplatCodec, directory_size
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
    """PlyCodec is the 1.00x anchor. If it loses anything the curve has no origin.

    All six arrays are checked because four of them are not implied by the other
    two. Measured against sabotaged decoders: zeroing scales, quats, opacities
    or sh0 on the way back passes a version of this test that asserts only means
    and shN, so the anchor could silently stop being lossless.
    """
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
    """The sh_degree assertion is what Task 6 reads to size its colour tensor.

    A cloud encoded at degree 3 comes back at degree 0, because the format
    carries no f_rest. Hardcoding 3 downstream would claim 16 coefficients where
    1 is present.
    """
    cloud = a_cloud(64)
    codec = SplatCodec(order="none")
    codec.encode(cloud, tmp_path)
    back = codec.decode(tmp_path)
    assert len(back) == 64
    assert cloud.sh_degree == 3
    assert back.sh_degree == 0
    np.testing.assert_array_equal(back.means, cloud.means)


def test_both_codecs_satisfy_the_codec_protocol():
    """The isinstance that @runtime_checkable enables has no other caller here.

    Measured on Python 3.11: dropping the decorator makes these calls raise
    TypeError, and removing a codec's `size` or its `name` makes them return
    False. What it does not check is signatures. A class whose `encode` takes no
    directory argument still passes, so this pins the shape of the interface
    rather than its calling convention, and the negative case below is what
    keeps the positive ones from being vacuous.
    """

    class NotACodec:
        name = "incomplete"

        def encode(self, cloud, directory): ...

        def decode(self, directory): ...

    assert isinstance(PlyCodec(), Codec)
    assert isinstance(SplatCodec(), Codec)
    assert not isinstance(NotACodec(), Codec)


def test_each_codec_measures_only_its_own_directory(tmp_path):
    """Task 9 gives each codec `scratch / codec.name` to encode into.

    Distinct names are what keeps the rates apart. Sharing one directory makes
    the second codec's size() count the first codec's files as well as its own,
    which lands on the curve as a wrong ratio rather than as an error.
    """
    cloud = a_cloud(64)
    codecs = [PlyCodec(), SplatCodec(order="none")]
    assert len({codec.name for codec in codecs}) == len(codecs)

    sizes = {}
    for codec in codecs:
        codec.encode(cloud, tmp_path / codec.name)
        sizes[codec.name] = codec.size(tmp_path / codec.name)

    assert sizes["splat"] == 64 * 32
    # The two directories partition the root, so neither counted the other.
    assert sizes["ply"] + sizes["splat"] == directory_size(tmp_path)


@pytest.mark.parametrize("codec", (PlyCodec(), SplatCodec()), ids=("ply", "splat"))
def test_encode_creates_a_target_directory_that_does_not_exist(tmp_path, codec):
    """The two encoders disagreed about this until it was pinned.

    write_ply creates the parent, so PlyCodec already worked into a missing
    directory while SplatCodec raised FileNotFoundError on the same call. Task
    9's run_bench calls mkdir before every encode, so neither convention breaks
    it and nothing downstream would have reported the disagreement.
    """
    target = tmp_path / "made_by_encode"
    codec.encode(a_cloud(8), target)
    assert target.is_dir()
    assert codec.size(target) > 0


def test_directory_size_counts_every_file_recursively(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "a.bin").write_bytes(b"x" * 10)
    (tmp_path / "nested" / "b.bin").write_bytes(b"y" * 5)
    assert directory_size(tmp_path) == 15
