import numpy as np
import pytest
from plyfile import PlyData

from splatpipe.errors import ArtifactError
from splatpipe.gaussians import GaussianCloud, read_ply, write_ply

RNG = np.random.default_rng(0)


def make_cloud(n=7, k=15):
    return GaussianCloud(
        means=RNG.standard_normal((n, 3)).astype(np.float32),
        scales=RNG.standard_normal((n, 3)).astype(np.float32),
        quats=RNG.standard_normal((n, 4)).astype(np.float32),
        opacities=RNG.standard_normal(n).astype(np.float32),
        sh0=RNG.standard_normal((n, 3)).astype(np.float32),
        shN=RNG.standard_normal((n, k, 3)).astype(np.float32),
    )


def test_length_and_degree():
    assert len(make_cloud(n=7, k=15)) == 7
    assert make_cloud(k=15).sh_degree == 3
    assert make_cloud(k=8).sh_degree == 2
    assert make_cloud(k=3).sh_degree == 1
    assert make_cloud(k=0).sh_degree == 0


def test_validate_rejects_a_length_mismatch():
    cloud = make_cloud(n=7)
    broken = GaussianCloud(
        means=cloud.means[:5],
        scales=cloud.scales,
        quats=cloud.quats,
        opacities=cloud.opacities,
        sh0=cloud.sh0,
        shN=cloud.shN,
    )
    with pytest.raises(ArtifactError, match="scales has 7 rows"):
        broken.validate()


def test_take_reorders_every_array_together():
    cloud = make_cloud(n=7)
    reversed_cloud = cloud.take(np.arange(6, -1, -1))
    assert len(reversed_cloud) == 7
    np.testing.assert_array_equal(reversed_cloud.means[0], cloud.means[6])
    np.testing.assert_array_equal(reversed_cloud.shN[0], cloud.shN[6])


def test_ply_round_trip_is_exact(tmp_path):
    cloud = make_cloud(n=7)
    path = tmp_path / "scene.ply"
    write_ply(cloud, path)
    back = read_ply(path)
    for name in ("means", "scales", "quats", "opacities", "sh0", "shN"):
        np.testing.assert_array_equal(getattr(back, name), getattr(cloud, name), err_msg=name)


def test_f_rest_is_channel_major(tmp_path):
    """f_rest_0..14 is red, 15..29 is green, 30..44 is blue.

    Getting this wrong swaps colour channels in the view-dependent term, which
    looks like a subtle lighting bug rather than an error.
    """
    n, k = 4, 15
    shN = np.empty((n, k, 3), dtype=np.float32)
    shN[:, :, 0] = 1.0
    shN[:, :, 1] = 2.0
    shN[:, :, 2] = 3.0
    cloud = GaussianCloud(
        means=np.zeros((n, 3), np.float32),
        scales=np.zeros((n, 3), np.float32),
        quats=np.zeros((n, 4), np.float32),
        opacities=np.zeros(n, np.float32),
        sh0=np.zeros((n, 3), np.float32),
        shN=shN,
    )
    path = tmp_path / "channels.ply"
    write_ply(cloud, path)

    vertex = PlyData.read(path)["vertex"]
    assert vertex["f_rest_0"][0] == 1.0
    assert vertex["f_rest_14"][0] == 1.0
    assert vertex["f_rest_15"][0] == 2.0
    assert vertex["f_rest_29"][0] == 2.0
    assert vertex["f_rest_30"][0] == 3.0
    assert vertex["f_rest_44"][0] == 3.0

    np.testing.assert_array_equal(read_ply(path).shN, shN)


def test_bytes_per_gaussian_matches_the_spike(tmp_path):
    """236 bytes per Gaussian at degree 3: 59 float32 fields."""
    n = 100
    path = tmp_path / "sized.ply"
    write_ply(make_cloud(n=n, k=15), path)
    header_end = path.read_bytes().index(b"end_header\n") + len(b"end_header\n")
    assert (path.stat().st_size - header_end) / n == 236
