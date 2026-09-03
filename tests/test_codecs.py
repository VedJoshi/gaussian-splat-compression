"""Codec round trips. The ply and splat cases are pure numpy and run in the fast
tier; the png cases are GPU-marked and deselected there."""

from __future__ import annotations

import numpy as np
import pytest

from splatpipe.bench.codecs import (
    Codec,
    PlyCodec,
    PngCodec,
    SplatCodec,
    directory_size,
)
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
    directory while SplatCodec raised FileNotFoundError on the same call.
    run_bench in Task 9 calls mkdir before each encode (plan line 1623) and so
    tolerates either convention, but Task 6 does not: its
    test_render_uses_the_clouds_own_sh_degree encodes into `tmp_path / "splat"`
    without creating it (plan line 848). Removing the mkdir this test guards
    breaks that caller.
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


def test_png_codec_satisfies_the_protocol_and_names_itself_distinctly():
    """PngCodec imports torch and gsplat inside its methods, so constructing one
    and checking its shape needs no CUDA, which is why this is fast tier while
    the three tests below are not.

    The distinct name matters for the reason
    test_each_codec_measures_only_its_own_directory gives: Task 9 hands each
    codec `scratch / codec.name`, so a collision would make one codec's size()
    count another codec's files.
    """
    assert isinstance(PngCodec(), Codec)
    assert PngCodec().name == "png"
    assert len({codec.name for codec in (PlyCodec(), SplatCodec(), PngCodec())}) == 3


# PngCompression sorts with PLAS and clusters with torchpq, both of which need
# CUDA. There is no CPU path, so these are GPU tier.
pngmark = pytest.mark.gpu


@pngmark
def test_png_codec_round_trips_a_square_cloud(tmp_path):
    """65536 is 256 squared, so nothing is cropped, and it is also the smallest
    cloud PngCompression can encode at all.

    Two upstream floors set that number, and which one a reader meets depends on
    how far they shrink it. Below 256 sorting fails first, because sort_splats
    runs at png_compression.py:97 before the compress loop at :100: reorder_plas
    raises block_size to at least min_block_size, which sort_with_plas defaults
    to 16 (plas/core.py:490) and gsplat never overrides (sort.py:39), so
    num_pixel_blocks = sidelen // block_size (plas/core.py:368) is 0 for a grid
    under 16 a side and params_to_blocky raises (plas/core.py:194). Anywhere
    from 256 to 65535 it is clustering that fails instead: _compress_kmeans asks
    torchpq for 65536 clusters (png_compression.py:326) and initialize_centroids
    draws them with np.random.choice(..., replace=False)
    (torchpq/clustering/KMeans.py:272), which needs at least as many Gaussians
    as clusters. compress builds its kwargs from n_sidelen and verbose alone
    (png_compression.py:102), so the cluster count cannot be lowered through the
    public API, and reaching past that API would measure something other than
    the baseline. Measured on this build: 65535 raises ValueError, 65536 encodes
    and decodes in about 15 seconds.

    Encoding into a subdirectory rather than into tmp_path is what covers the
    mkdir in PngCodec.encode. gsplat creates a directory only in _compress_npz
    (png_compression.py:304), which none of these six fields reach, so the png
    writers at :174, :247 and :250 write into whatever encode made. Measured:
    deleting that mkdir makes this test fail with FileNotFoundError.

    The two value assertions are what stop a decode returning zeros of the right
    shape from passing, since len, sh_degree and size are shapes and byte counts
    and GaussianCloud.validate (gaussians.py:51) checks dtype and shape and
    nothing about values. PLAS permutes rows, so each column is compared against
    itself sorted, and quantisation is monotonic, so that comparison is exactly
    the per-element error. Both tolerances are measured on this fixture rather
    than derived: means deviates by at most 2.12e-05 through the 16-bit path and
    scales by at most 3.92e-03 through the 8-bit path, which is half a step of
    8-bit quantisation over the observed span. An all-zero decode deviates by
    1.0 on means, four orders above the tolerance allowed here.

    What this does not pin: quats, which compress renormalises before quantising
    (png_compression.py:85) so they are not comparable to the raw input; sh0 and
    opacities, which go through the same 8-bit writer as scales; and shN, which
    deviates by only 3.17e-03 here because n_data equals n_clusters at this
    size, so every Gaussian gets its own centroid. K-means is far lossier on a
    real scene and this fixture cannot show that.
    """
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
    """run_bench hands one cloud to every codec in turn, so encode must not
    mutate it. This holds that contract rather than catching a live bug: the
    .cuda() in encode allocates a separate device tensor, so compress cannot
    reach the caller's numpy arrays whatever it does to the dict it is given.
    Measured by deleting the six .copy() calls the brief specified, which left
    all six arrays byte-identical. It fails only if encode grows a CPU path or
    starts writing back into the cloud.

    All six arrays are checked because encode hands all six to compress, and
    checking two of them is the defect the Task 3 review found in
    test_ply_codec_is_lossless."""
    names = ("means", "scales", "quats", "opacities", "sh0", "shN")
    cloud = a_cloud(65536)
    before = {name: getattr(cloud, name).copy() for name in names}
    PngCodec().encode(cloud, tmp_path)
    for name in names:
        np.testing.assert_array_equal(getattr(cloud, name), before[name], err_msg=name)


@pngmark
def test_png_codec_reports_the_count_it_actually_encoded(tmp_path):
    """65537 is not a square, so PngCompression drops the lowest-opacity splat.

    It is one above the floor rather than one above any square, because cropping
    to 65536 has to leave enough Gaussians to clear the k-means minimum that
    test_png_codec_round_trips_a_square_cloud documents. 65 exercises the same
    crop branch and then raises. Measured: this encode prints "Removed 1
    Gaussians" and decodes 65536.
    """
    codec = PngCodec()
    codec.encode(a_cloud(65537), tmp_path)
    assert len(codec.decode(tmp_path)) == 65536
