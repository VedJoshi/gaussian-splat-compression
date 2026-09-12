"""Affine per-component quantisation, and its inverse."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from splatpipe.errors import ArtifactError, ConfigError

MAX_BITS = 16


def validate_bits(bits: int) -> None:
    if isinstance(bits, bool) or not isinstance(bits, int) or not 1 <= bits <= MAX_BITS:
        raise ConfigError(
            f"quantisation bits must be an integer from 1 to {MAX_BITS}, got {bits!r}"
        )


def dtype_for_bits(bits: int) -> type[np.unsignedinteger]:
    validate_bits(bits)
    return np.uint8 if bits <= 8 else np.uint16


def _as_matrix(values: np.ndarray) -> np.ndarray:
    if values.ndim not in (1, 2):
        raise ArtifactError(
            f"values must be one- or two-dimensional, got shape {values.shape}"
        )
    if values.size == 0:
        raise ArtifactError("values must not be empty")
    return values.reshape(len(values), -1)


def quantize_affine(
    values: np.ndarray, bits: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantise each column over its own observed range."""
    validate_bits(bits)
    values = np.asarray(values)
    matrix = _as_matrix(values).astype(np.float64)
    if not np.isfinite(matrix).all():
        raise ArtifactError("values contain non-finite entries")

    mins = matrix.min(axis=0)
    maxs = matrix.max(axis=0)

    # A component with no spread would divide by zero. Substituting a span of 1
    # sends every entry to level 0, which dequantises back to the minimum, which
    # is the exact original value.
    spans = np.where(maxs - mins == 0, 1.0, maxs - mins)
    levels = (1 << bits) - 1
    quantized = np.rint((matrix - mins) / spans * levels).clip(0, levels)
    return (
        quantized.astype(dtype_for_bits(bits)).reshape(values.shape),
        mins.astype(np.float32),
        maxs.astype(np.float32),
    )


def dequantize_affine(
    quantized: np.ndarray, mins: np.ndarray, maxs: np.ndarray, bits: int
) -> np.ndarray:
    validate_bits(bits)
    quantized = np.asarray(quantized)
    expected = np.dtype(dtype_for_bits(bits))
    if quantized.dtype != expected:
        raise ArtifactError(
            f"quantised values have dtype {quantized.dtype}, expected {expected}"
        )
    matrix = _as_matrix(quantized)
    mins = np.asarray(mins, dtype=np.float64)
    maxs = np.asarray(maxs, dtype=np.float64)
    if mins.shape != maxs.shape or mins.shape != (matrix.shape[1],):
        raise ArtifactError(
            f"bounds of shape {mins.shape} do not match {matrix.shape[1]} components"
        )
    if not np.isfinite(mins).all() or not np.isfinite(maxs).all():
        raise ArtifactError("bounds contain non-finite values")
    if np.any(maxs < mins):
        raise ArtifactError("a component maximum is below its minimum")

    levels = (1 << bits) - 1
    values = matrix.astype(np.float64) / levels * (maxs - mins) + mins
    return values.astype(np.float32).reshape(quantized.shape)


def quantization_step(mins: np.ndarray, maxs: np.ndarray, bits: int) -> np.ndarray:
    """Half the level spacing, which is the exact round-trip error bound."""
    validate_bits(bits)
    spans = np.asarray(maxs, dtype=np.float64) - np.asarray(mins, dtype=np.float64)
    return spans / (2 * ((1 << bits) - 1))


@dataclass(frozen=True)
class PackedField:
    values: np.ndarray  # quantised, uint8 or uint16, 1- or 2-dimensional
    bits: int
    mins: np.ndarray  # (C,) float32
    maxs: np.ndarray  # (C,) float32
    shape: tuple[int, ...]  # the logical shape to restore, e.g. (N, 15, 3)

    def dequantize(self) -> np.ndarray:
        flat = dequantize_affine(self.values, self.mins, self.maxs, self.bits)
        return flat.reshape(self.shape)
