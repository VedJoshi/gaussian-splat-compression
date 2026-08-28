import warnings

import numpy as np
import pytest

from splatpipe.errors import ArtifactError, ConfigError
from splatpipe.formats.splat import (
    BYTES_PER_GAUSSIAN,
    encode_splat,
    morton_order,
    size_opacity_order,
)
from splatpipe.gaussians import GaussianCloud

RNG = np.random.default_rng(1)


def make_cloud(n=64, k=15):
    return GaussianCloud(
        means=RNG.random((n, 3)).astype(np.float32) * 10,
        scales=RNG.standard_normal((n, 3)).astype(np.float32),
        quats=RNG.standard_normal((n, 4)).astype(np.float32),
        opacities=RNG.standard_normal(n).astype(np.float32),
        sh0=RNG.standard_normal((n, 3)).astype(np.float32),
        shN=RNG.standard_normal((n, k, 3)).astype(np.float32),
    )


def test_output_is_32_bytes_per_gaussian():
    cloud = make_cloud(n=64)
    assert len(encode_splat(cloud)) == 64 * BYTES_PER_GAUSSIAN


def test_orderings_are_permutations():
    cloud = make_cloud(n=64)
    for order in (
        morton_order(cloud.means),
        size_opacity_order(cloud.scales, cloud.opacities),
    ):
        np.testing.assert_array_equal(np.sort(order), np.arange(64))


def test_size_opacity_puts_the_biggest_splat_first():
    cloud = make_cloud(n=64)
    order = size_opacity_order(cloud.scales, cloud.opacities)
    weight = np.exp(cloud.scales.sum(axis=1)) / (1 + np.exp(-cloud.opacities))
    assert order[0] == int(np.argmax(weight))


def test_order_none_preserves_input_order():
    cloud = make_cloud(n=4)
    raw = encode_splat(cloud, order="none")
    first_mean = np.frombuffer(raw[:12], dtype="<f4")
    np.testing.assert_allclose(first_mean, cloud.means[0], rtol=0, atol=0)


def test_unknown_order_is_rejected():
    with pytest.raises(ConfigError, match="sideways"):
        encode_splat(make_cloud(n=4), order="sideways")


def test_scales_are_exponentiated_and_opacity_goes_through_sigmoid():
    cloud = make_cloud(n=1)
    raw = encode_splat(cloud, order="none")
    np.testing.assert_allclose(
        np.frombuffer(raw[12:24], dtype="<f4"), np.exp(cloud.scales[0]), rtol=1e-6
    )
    expected_alpha = int(
        np.clip(1 / (1 + np.exp(-cloud.opacities[0])) * 255, 0, 255).astype(np.uint8)
    )
    assert raw[27] == expected_alpha


def test_valid_input_emits_no_runtime_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        encode_splat(make_cloud(n=64))


def test_extreme_finite_opacities_encode_without_warnings():
    cloud = make_cloud(n=2)
    limit = np.finfo(np.float32).max
    cloud.opacities[:] = (limit, -limit)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        raw = encode_splat(cloud, order="none")

    assert raw[27] == 255
    assert raw[32 + 27] == 0


def test_large_finite_scales_have_warning_free_size_opacity_ordering():
    scales = np.array(((80, 80, 80), (79, 79, 79), (78, 78, 78)), dtype=np.float32)
    opacities = np.zeros(3, dtype=np.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        order = size_opacity_order(scales, opacities)

    np.testing.assert_array_equal(order, np.array((0, 1, 2)))


def test_extreme_finite_sh0_saturates_without_warnings():
    cloud = make_cloud(n=2)
    limit = np.finfo(np.float32).max
    cloud.sh0[0] = limit
    cloud.sh0[1] = -limit

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        raw = encode_splat(cloud, order="none")

    assert raw[24:27] == b"\xff\xff\xff"
    assert raw[32 + 24 : 32 + 27] == b"\x00\x00\x00"


@pytest.mark.parametrize("field", ("means", "scales", "quats", "opacities", "sh0", "shN"))
@pytest.mark.parametrize("invalid", (np.nan, np.inf))
def test_nonfinite_fields_are_rejected(field, invalid):
    cloud = make_cloud(n=4)
    array = getattr(cloud, field)
    array.flat[0] = invalid

    with pytest.raises(ArtifactError, match=rf"{field} contains non-finite values"):
        encode_splat(cloud)


def test_zero_length_quaternion_is_rejected():
    cloud = make_cloud(n=4)
    cloud.quats[2] = 0

    with pytest.raises(ArtifactError, match="quats contains zero-length quaternion at row 2"):
        encode_splat(cloud)


def test_non_normalizable_quaternion_is_rejected():
    cloud = make_cloud(n=4)
    cloud.quats[1] = np.finfo(np.float32).max

    with pytest.raises(ArtifactError, match="quats contains non-normalizable quaternion at row 1"):
        encode_splat(cloud)


def test_log_scale_that_overflows_float32_is_rejected():
    cloud = make_cloud(n=4)
    cloud.scales[1, 0] = np.float32(100)

    with pytest.raises(ArtifactError, match="scales contains values that overflow float32"):
        encode_splat(cloud)


def export_gsplat_splat(cloud):
    torch = pytest.importorskip("torch")
    from gsplat.exporter import export_splats

    return export_splats(
        means=torch.from_numpy(cloud.means),
        scales=torch.from_numpy(cloud.scales),
        quats=torch.from_numpy(cloud.quats),
        opacities=torch.from_numpy(cloud.opacities),
        sh0=torch.from_numpy(cloud.sh0).unsqueeze(1),
        shN=torch.from_numpy(cloud.shN.transpose(0, 2, 1).copy()).permute(0, 2, 1),
        format="splat",
    )


def test_matches_gsplat_export_splats_byte_for_byte_with_zero_log_scales():
    """The authority on this format is the library the viewer was built for.

    gsplat.exporter imports without the CUDA backend, so this needs no GPU.
    Positions are drawn from a continuous distribution, so Morton ties, where
    argsort order would be implementation-defined, do not arise.
    """
    cloud = make_cloud(n=64)
    cloud.scales[:] = 0
    ours = encode_splat(cloud, order="morton")
    theirs = export_gsplat_splat(cloud)
    assert ours == theirs


def test_random_scale_fields_are_within_two_ulps_of_gsplat():
    """Different exp implementations are ULP-close, not byte-stable.

    Positions, colors, and rotations do not depend on exp and must remain byte
    exact. The scales are parsed as float32 and compared by positive-float ULP
    distance, which preserves an implementation-independent format check.
    """
    cloud = make_cloud(n=64)
    ours = encode_splat(cloud, order="morton")
    theirs = export_gsplat_splat(cloud)

    ours_records = np.frombuffer(ours, dtype=np.uint8).reshape(-1, BYTES_PER_GAUSSIAN)
    theirs_records = np.frombuffer(theirs, dtype=np.uint8).reshape(-1, BYTES_PER_GAUSSIAN)
    np.testing.assert_array_equal(ours_records[:, 0:12], theirs_records[:, 0:12])
    np.testing.assert_array_equal(ours_records[:, 24:32], theirs_records[:, 24:32])

    ours_scales = np.frombuffer(ours_records[:, 12:24].copy().tobytes(), dtype="<f4")
    theirs_scales = np.frombuffer(theirs_records[:, 12:24].copy().tobytes(), dtype="<f4")
    ulps = np.abs(ours_scales.view(np.uint32).astype(np.int64) - theirs_scales.view(np.uint32).astype(np.int64))
    assert ulps.max() <= 2
    assert np.any(ulps > 0)
