import warnings
from dataclasses import replace

import numpy as np
import pytest

from splatpipe.errors import ArtifactError, ConfigError
from splatpipe.formats.splat import (
    BYTES_PER_GAUSSIAN,
    decode_splat,
    encode_splat,
    morton_order,
    size_opacity_order,
)
from splatpipe.gaussians import GaussianCloud, SH_C0

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


def a_representable_cloud(n: int = 64) -> GaussianCloud:
    """A cloud the .splat format can actually carry.

    The module's existing make_cloud draws sh0 from a standard normal, but the
    format only represents sh0 in about [-1.77, 1.77]: rgb is sh0 * SH_C0 + 0.5
    and the encoder clips outside [0, 1]. Asserting round-trip fidelity on
    values the format cannot represent would test nothing, so the tolerance
    assertions below use this instead. Saturation itself is already covered by
    the encoder's own tests.
    """
    rng = np.random.default_rng(0)
    return GaussianCloud(
        means=rng.uniform(-10, 10, (n, 3)).astype(np.float32),
        scales=rng.uniform(-3, -1, (n, 3)).astype(np.float32),
        quats=rng.standard_normal((n, 4)).astype(np.float32),
        opacities=rng.uniform(-4, 4, n).astype(np.float32),
        sh0=rng.uniform(-1.5, 1.5, (n, 3)).astype(np.float32),
        shN=rng.standard_normal((n, 15, 3)).astype(np.float32),
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
    scales = np.array(((78, 78, 78), (79, 79, 79), (80, 80, 80)), dtype=np.float32)
    opacities = np.zeros(3, dtype=np.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        order = size_opacity_order(scales, opacities)

    np.testing.assert_array_equal(order, np.array((2, 1, 0)))


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
    # Guard against a vacuous comparison: these have to be real, varied scale
    # values rather than a buffer of zeros that any tolerance would accept.
    # Deliberately not `np.any(ulps > 0)`, which would turn the two
    # implementations agreeing exactly into a test failure.
    assert ours_scales.size == theirs_scales.size > 0
    assert np.ptp(ours_scales) > 0


def test_an_empty_cloud_is_refused_by_every_ordering():
    """A prune that removes every Gaussian is a real milestone 4 ablation.

    Left unguarded the three orderings disagree: "none" and "size_opacity"
    write a zero-byte artifact while "morton" raises NumPy's raw zero-size
    reduction ValueError from means.min(axis=0), which is not a SplatpipeError
    and so escapes the CLI's error contract.
    """
    empty = GaussianCloud(
        means=np.zeros((0, 3), dtype=np.float32),
        scales=np.zeros((0, 3), dtype=np.float32),
        quats=np.zeros((0, 4), dtype=np.float32),
        opacities=np.zeros(0, dtype=np.float32),
        sh0=np.zeros((0, 3), dtype=np.float32),
        shN=np.zeros((0, 15, 3), dtype=np.float32),
    )
    empty.validate()  # the shapes are consistent; emptiness is the only problem

    for order in ("morton", "size_opacity", "none"):
        with pytest.raises(ArtifactError, match="cloud is empty"):
            encode_splat(empty, order=order)


def test_decode_recovers_positions_exactly():
    """Positions are stored as float32 and must survive the round trip bit for bit."""
    cloud = make_cloud(n=16)
    back = decode_splat(encode_splat(cloud, order="none"))
    np.testing.assert_array_equal(back.means, cloud.means)


def test_decode_discards_higher_order_harmonics():
    cloud = make_cloud(n=16)
    back = decode_splat(encode_splat(cloud, order="none"))
    assert back.shN.shape == (16, 0, 3)
    assert back.sh_degree == 0


def test_decode_round_trips_scales_and_colours_within_quantisation_error():
    cloud = a_representable_cloud(64)
    back = decode_splat(encode_splat(cloud, order="none"))
    # Scales survive as exp then log through float32, so only rounding is lost.
    np.testing.assert_allclose(back.scales, cloud.scales, atol=1e-5)
    # Colours are quantised to 8 bits. One step of sh0 is 1/255 scaled by 1/SH_C0.
    np.testing.assert_allclose(back.sh0, cloud.sh0, atol=(1.0 / 255) / SH_C0)


def test_decode_clamps_saturated_opacity_instead_of_returning_infinity():
    """alpha == 255 inverts to logit(1) == inf without a clamp.

    The encoder reaches this on real data: it clips to [0, 255] and casts, so a
    high enough logit saturates. The clamp uses the midpoint of the quantisation
    bucket, which is the best estimate available rather than an arbitrary guard.
    """
    cloud = replace(make_cloud(n=4), opacities=np.full(4, 40.0, dtype=np.float32))
    back = decode_splat(encode_splat(cloud, order="none"))
    assert np.isfinite(back.opacities).all()
    np.testing.assert_allclose(back.opacities, 6.2324480, atol=1e-4)


def test_decode_clamps_underflowing_scale_instead_of_returning_negative_infinity():
    """A very negative log scale exponentiates to zero, and log(0) is -inf."""
    cloud = replace(make_cloud(n=4), scales=np.full((4, 3), -200.0, dtype=np.float32))
    back = decode_splat(encode_splat(cloud, order="none"))
    assert np.isfinite(back.scales).all()
    np.testing.assert_allclose(back.scales, -87.33655, atol=1e-3)


def test_decode_rejects_a_truncated_buffer():
    with pytest.raises(ArtifactError, match="not a multiple"):
        decode_splat(b"\x00" * 33)


def test_decode_rejects_an_empty_buffer():
    with pytest.raises(ArtifactError, match="no gaussians"):
        decode_splat(b"")


def test_decode_round_trips_quaternions_within_quantisation_error():
    """Nothing else in this file reads back.quats numerically.

    What this pins is the byte slice: reading the colour bytes or a window
    shifted by one moves every component by more than a whole unit. It pins the
    128 offset in one direction only. Measured on this fixture, an offset of 129
    gives 0.0202 and fails, while an offset of 127 gives 0.0078 and passes,
    because encode_splat truncates rather than rounds when it casts to uint8, so
    127 and 128 land in the same quantisation bucket. Tightening the tolerance
    cannot separate them. It also does not pin the divisor, because
    (stored - offset) / d followed by normalisation cancels d entirely.

    The correct decoder's worst component error is 0.0105, measured over 20,000
    gaussians, against the 0.0156 asserted here.
    """
    cloud = a_representable_cloud(64)
    back = decode_splat(encode_splat(cloud, order="none"))
    expected = cloud.quats / np.linalg.norm(cloud.quats, axis=1, keepdims=True)
    np.testing.assert_allclose(back.quats, expected, atol=2.0 / 128)


def test_decode_rejects_a_zero_length_quaternion():
    """Unreachable through encode_splat, which always normalises first.

    decode_splat is public and .splat is an interchange format, so it can be
    handed bytes this project did not write.
    """
    data = bytearray(encode_splat(make_cloud(n=4), order="none"))
    data[28:32] = b"\x80\x80\x80\x80"
    with pytest.raises(ArtifactError, match="zero-length quaternion"):
        decode_splat(bytes(data))
