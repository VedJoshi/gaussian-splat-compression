from __future__ import annotations

import numpy as np
import pytest

from splatpipe.compress.quantize import (
    PackedField,
    dequantize_affine,
    dtype_for_bits,
    quantization_step,
    quantize_affine,
    validate_bits,
)
from splatpipe.errors import ArtifactError, ConfigError


def test_round_trip_stays_within_the_analytic_bound():
    rng = np.random.default_rng(0)
    values = rng.uniform(-4.0, 3.0, (500, 3)).astype(np.float32)
    quantized, mins, maxs = quantize_affine(values, bits=8)
    recovered = dequantize_affine(quantized, mins, maxs, bits=8)
    bound = quantization_step(mins, maxs, bits=8)
    assert np.all(np.abs(recovered - values) <= bound + 1e-6)


def test_wider_bit_depth_strictly_reduces_the_bound():
    rng = np.random.default_rng(1)
    values = rng.uniform(-1.0, 1.0, (200, 3)).astype(np.float32)
    _, mins, maxs = quantize_affine(values, bits=8)
    assert np.all(quantization_step(mins, maxs, 16) < quantization_step(mins, maxs, 8))


def test_a_constant_component_recovers_exactly():
    values = np.array([[1.5, 0.0], [1.5, 1.0], [1.5, 2.0]], dtype=np.float32)
    quantized, mins, maxs = quantize_affine(values, bits=8)
    recovered = dequantize_affine(quantized, mins, maxs, bits=8)
    np.testing.assert_array_equal(recovered[:, 0], np.full(3, 1.5, dtype=np.float32))


def test_each_component_uses_its_own_range():
    values = np.array([[0.0, 100.0], [1.0, 200.0]], dtype=np.float32)
    quantized, mins, maxs = quantize_affine(values, bits=8)
    np.testing.assert_allclose(mins, [0.0, 100.0])
    np.testing.assert_allclose(maxs, [1.0, 200.0])
    np.testing.assert_array_equal(quantized, [[0, 0], [255, 255]])


def test_one_dimensional_input_keeps_its_shape():
    values = np.linspace(-2.0, 2.0, 9).astype(np.float32)
    quantized, mins, maxs = quantize_affine(values, bits=8)
    assert quantized.shape == (9,)
    assert mins.shape == (1,)
    assert dequantize_affine(quantized, mins, maxs, bits=8).shape == (9,)


@pytest.mark.parametrize(
    "bits,expected", ((1, np.uint8), (8, np.uint8), (9, np.uint16), (16, np.uint16))
)
def test_dtype_for_bits_uses_the_narrowest_type(bits, expected):
    assert dtype_for_bits(bits) is expected


@pytest.mark.parametrize("bits", (0, 17, -1, True, 8.5, "8"))
def test_invalid_bit_depths_are_rejected(bits):
    with pytest.raises(ConfigError):
        validate_bits(bits)


@pytest.mark.parametrize(
    "values",
    (
        np.array([1.0, np.nan], dtype=np.float32),
        np.array([1.0, np.inf], dtype=np.float32),
        np.zeros((0, 3), dtype=np.float32),
        np.zeros((2, 2, 2), dtype=np.float32),
    ),
)
def test_malformed_values_are_rejected(values):
    with pytest.raises(ArtifactError):
        quantize_affine(values, bits=8)


def test_dequantize_rejects_a_mismatched_dtype():
    with pytest.raises(ArtifactError):
        dequantize_affine(
            np.zeros((4, 2), dtype=np.uint16),
            np.zeros(2, dtype=np.float32),
            np.ones(2, dtype=np.float32),
            bits=8,
        )


def test_dequantize_rejects_bounds_that_do_not_match_the_components():
    with pytest.raises(ArtifactError):
        dequantize_affine(
            np.zeros((4, 3), dtype=np.uint8),
            np.zeros(2, dtype=np.float32),
            np.ones(2, dtype=np.float32),
            bits=8,
        )


def test_dequantize_rejects_an_inverted_range():
    with pytest.raises(ArtifactError):
        dequantize_affine(
            np.zeros((4, 1), dtype=np.uint8),
            np.ones(1, dtype=np.float32),
            np.zeros(1, dtype=np.float32),
            bits=8,
        )


def test_packed_field_restores_a_higher_rank_shape():
    rng = np.random.default_rng(2)
    values = rng.uniform(-0.2, 0.2, (10, 15, 3)).astype(np.float32)
    flat = values.reshape(10, 45)
    quantized, mins, maxs = quantize_affine(flat, bits=8)
    field = PackedField(values=quantized, bits=8, mins=mins, maxs=maxs, shape=(10, 15, 3))
    recovered = field.dequantize()
    assert recovered.shape == (10, 15, 3)
    assert np.abs(recovered - values).max() <= quantization_step(mins, maxs, 8).max() + 1e-6
