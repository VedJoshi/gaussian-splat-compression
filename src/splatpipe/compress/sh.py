"""Vector-quantize higher-order spherical harmonics."""

from __future__ import annotations

import json
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import numpy as np

from splatpipe.errors import ArtifactError, ConfigError


def validate_codebook_bits(bits: int) -> None:
    if isinstance(bits, bool) or not isinstance(bits, int) or not 1 <= bits <= 8:
        raise ConfigError(f"codebook bits must be an integer from 1 to 8, got {bits!r}")


def quantize_codebook(
    centroids: np.ndarray, bits: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantize each codebook component over its own observed range.

    This duplicates compress.quantize.quantize_affine's rule and is deliberately
    not collapsed into it. That one promotes to float64 and this one stays in
    float32, so a component landing within ~5e-6 of a rounding tie can go either
    way; test_codebook_quantiser_agrees_with_the_container_to_within_one_level
    pins the difference at one level. Milestone 3's measured bytes come from this
    path, so sharing the implementation would silently move a published number.
    """
    validate_codebook_bits(bits)
    if centroids.dtype != np.float32 or centroids.ndim != 2:
        raise ArtifactError(
            f"centroids must be a float32 matrix, got {centroids.dtype} {centroids.shape}"
        )
    if centroids.shape[0] == 0 or centroids.shape[1] == 0:
        raise ArtifactError("centroids must not be empty")
    if not np.isfinite(centroids).all():
        raise ArtifactError("centroids contain non-finite values")

    mins = centroids.min(axis=0)
    maxs = centroids.max(axis=0)
    spans = maxs - mins
    safe_spans = np.where(spans == 0, np.float32(1), spans)
    levels = (1 << bits) - 1
    normalized = (centroids - mins) / safe_spans
    quantized = np.rint(normalized * levels).clip(0, levels).astype(np.uint8)
    return quantized, mins, maxs


def dequantize_codebook(
    quantized: np.ndarray,
    mins: np.ndarray,
    maxs: np.ndarray,
    bits: int,
) -> np.ndarray:
    """Reconstruct a float32 codebook from its component ranges."""
    validate_codebook_bits(bits)
    if quantized.dtype != np.uint8 or quantized.ndim != 2:
        raise ArtifactError(
            f"quantized codebook must be a uint8 matrix, got "
            f"{quantized.dtype} {quantized.shape}"
        )
    if mins.shape != maxs.shape or mins.shape != (quantized.shape[1],):
        raise ArtifactError("codebook bounds do not match the quantized component count")
    if not np.isfinite(mins).all() or not np.isfinite(maxs).all():
        raise ArtifactError("codebook bounds contain non-finite values")
    if np.any(maxs < mins):
        raise ArtifactError("codebook maximum is below its minimum")

    levels = (1 << bits) - 1
    values = quantized.astype(np.float32) / levels
    return (values * (maxs - mins) + mins).astype(np.float32)


def label_dtype(n_clusters: int) -> type[np.unsignedinteger]:
    """Use the narrowest index type that can address the codebook."""
    if isinstance(n_clusters, bool) or not isinstance(n_clusters, int) or n_clusters < 2:
        raise ConfigError(f"SH cluster count must be an integer of at least 2, got {n_clusters!r}")
    if n_clusters <= 256:
        return np.uint8
    if n_clusters <= 65_536:
        return np.uint16
    raise ConfigError(f"SH cluster count must not exceed 65,536, got {n_clusters}")


def _compress_sh_vq(
    compress_dir: str,
    param_name: str,
    params,
    n_clusters: int,
    codebook_bits: int,
    verbose: bool = False,
    **_kwargs,
) -> dict:
    from torchpq.clustering import KMeans

    if params.numel() == 0:
        raise ArtifactError("higher-order spherical harmonics are empty")
    if params.shape[0] < n_clusters:
        raise ArtifactError(
            f"SH vector quantization needs at least {n_clusters:,} Gaussians, "
            f"got {params.shape[0]:,}"
        )

    vectors = params.reshape(params.shape[0], -1).permute(1, 0).contiguous()
    kmeans = KMeans(n_clusters=n_clusters, distance="manhattan", verbose=verbose)
    labels = kmeans.fit(vectors).detach().cpu().numpy().astype(label_dtype(n_clusters))
    centroids = kmeans.centroids.permute(1, 0).detach().cpu().numpy().astype(np.float32)
    quantized, mins, maxs = quantize_codebook(centroids, codebook_bits)

    np.savez_compressed(
        Path(compress_dir) / f"{param_name}.npz",
        centroids=quantized,
        labels=labels,
    )
    return {
        "shape": list(params.shape),
        "dtype": str(params.dtype).split(".")[1],
        "mins": mins.tolist(),
        "maxs": maxs.tolist(),
        "quantization": codebook_bits,
        "n_clusters": n_clusters,
        "distance": "manhattan",
    }


def _decompress_sh_vq(
    compress_dir: str,
    param_name: str,
    meta: dict,
    **_kwargs,
):
    import torch

    shape = tuple(meta["shape"])
    with np.load(Path(compress_dir) / f"{param_name}.npz", allow_pickle=False) as archive:
        quantized = archive["centroids"]
        labels = archive["labels"].astype(np.int64)

    mins = np.asarray(meta["mins"], dtype=np.float32)
    maxs = np.asarray(meta["maxs"], dtype=np.float32)
    centroids = dequantize_codebook(
        quantized,
        mins,
        maxs,
        bits=meta["quantization"],
    )
    if labels.shape != (shape[0],):
        raise ArtifactError(
            f"SH labels have shape {labels.shape}, expected {(shape[0],)}"
        )
    if labels.size and (labels.min() < 0 or labels.max() >= len(centroids)):
        raise ArtifactError("SH labels contain an out-of-range codebook index")

    values = centroids[labels].reshape(shape).astype(np.float32)
    return torch.from_numpy(values).to(dtype=getattr(torch, meta["dtype"]))


def make_sh_vq_compressor(
    n_clusters: int,
    codebook_bits: int,
    use_sort: bool,
):
    """Build a PngCompression variant with a configurable SH codebook."""
    label_dtype(n_clusters)
    validate_codebook_bits(codebook_bits)

    from gsplat.compression import PngCompression

    class ShVqCompression(PngCompression):
        def _get_compress_fn(self, param_name: str):
            if param_name == "shN":
                return partial(
                    _compress_sh_vq,
                    n_clusters=n_clusters,
                    codebook_bits=codebook_bits,
                )
            return super()._get_compress_fn(param_name)

        def _get_decompress_fn(self, param_name: str):
            if param_name == "shN":
                return _decompress_sh_vq
            return super()._get_decompress_fn(param_name)

    return ShVqCompression(use_sort=use_sort, verbose=False)


@contextmanager
def seeded_compression(seed: int):
    """Seed compression randomness without changing the caller's RNG state."""
    import torch

    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all()
    try:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        yield
    finally:
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        torch.cuda.set_rng_state_all(cuda_states)


def load_sh_codebook(directory: Path | str) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Read the quantised centroids and their bounds out of an shvq artifact
    directory, so the container can be measured against the same codebook the
    PNG baseline was measured with rather than a freshly clustered one."""
    directory = Path(directory)
    meta_path = directory / "meta.json"
    if not meta_path.is_file():
        raise ArtifactError(f"{directory} holds no meta.json")
    archive_path = directory / "shN.npz"
    if not archive_path.is_file():
        raise ArtifactError(f"{directory} holds no shN.npz codebook")

    meta = json.loads(meta_path.read_text(encoding="utf-8")).get("shN")
    if meta is None:
        raise ArtifactError(f"{meta_path} records no shN block, so it is not an shvq run")

    with np.load(archive_path, allow_pickle=False) as archive:
        centroids = archive["centroids"]

    bits = int(meta["quantization"])
    mins = np.asarray(meta["mins"], dtype=np.float32)
    maxs = np.asarray(meta["maxs"], dtype=np.float32)
    if mins.shape != (centroids.shape[1],):
        raise ArtifactError(
            f"codebook has {centroids.shape[1]} components but "
            f"{mins.shape[0]} bounds"
        )
    return centroids, mins, maxs, bits


def assign_sh_labels(shN: np.ndarray, centroids: np.ndarray, chunk: int = 16384) -> np.ndarray:
    """Nearest centroid per Gaussian under the manhattan distance the codebook
    was fitted with. This is K-means' own final assignment step, run against a
    fixed codebook, so every Gaussian keeps a label without reclustering."""
    import torch

    if shN.ndim != 3:
        raise ArtifactError(f"shN must be (N, K, 3), got shape {shN.shape}")
    if centroids.ndim != 2 or centroids.shape[1] != shN.shape[1] * shN.shape[2]:
        raise ArtifactError(
            f"codebook of shape {centroids.shape} does not match "
            f"shN of shape {shN.shape}"
        )

    flat = np.ascontiguousarray(shN, dtype=np.float32).reshape(len(shN), -1)
    book = torch.from_numpy(np.ascontiguousarray(centroids, dtype=np.float32)).cuda()
    labels = np.empty(len(flat), dtype=label_dtype(len(centroids)))
    for start in range(0, len(flat), chunk):
        block = torch.from_numpy(flat[start : start + chunk]).cuda()
        nearest = torch.cdist(block, book, p=1).argmin(dim=1)
        labels[start : start + chunk] = nearest.cpu().numpy()
    return labels


def sh_vq_from_baseline(directory: Path | str, shN: np.ndarray):
    """Build the container's ShVq from an existing shvq artifact directory.

    The baseline stores its labels in PLAS-sorted order and the sort is not
    recorded, so the labels cannot be reused directly. The codebook can, and
    reassigning against it is what keeps the container's SH identical to the
    baseline's while preserving every Gaussian.
    """
    from splatpipe.formats.container import ShVq

    centroids, mins, maxs, bits = load_sh_codebook(directory)
    dequantized = dequantize_codebook(centroids, mins, maxs, bits)
    return ShVq(
        codebook=centroids,
        labels=assign_sh_labels(shN, dequantized),
        mins=mins,
        maxs=maxs,
        bits=bits,
    )
