"""Post-training Gaussian ranking and pruning."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from splatpipe.errors import ArtifactError, ConfigError
from splatpipe.gaussians import GaussianCloud


def parse_retained_percentages(value: str) -> tuple[int, ...]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise ConfigError("at least one retained percentage is required")
    try:
        percentages = tuple(int(part) for part in parts)
    except ValueError as error:
        raise ConfigError(
            f"retained percentages must be whole numbers, got {value!r}"
        ) from error
    if any(str(number) != part for number, part in zip(percentages, parts)):
        raise ConfigError(f"retained percentages must be whole numbers, got {value!r}")
    if any(not 0 < number < 100 for number in percentages):
        raise ConfigError("retained percentages must be strictly between 0 and 100")
    if len(set(percentages)) != len(percentages):
        raise ConfigError("retained percentages must not contain duplicates")
    return percentages


def volume_weights(
    log_scales: np.ndarray,
    power: float = 0.1,
    reference_percentile: float = 10.0,
) -> tuple[np.ndarray, float]:
    """Return capped relative-volume weights without exponentiating raw volumes."""
    if log_scales.ndim != 2 or log_scales.shape[1] != 3:
        raise ArtifactError(f"log scales must have shape (N, 3), got {log_scales.shape}")
    if len(log_scales) == 0 or not np.isfinite(log_scales).all():
        raise ArtifactError("log scales must be non-empty and finite")
    if not np.isfinite(power) or power < 0:
        raise ConfigError(f"volume power must be finite and non-negative, got {power!r}")
    if not np.isfinite(reference_percentile) or not 0 <= reference_percentile <= 100:
        raise ConfigError(
            f"volume percentile must be between 0 and 100, got {reference_percentile!r}"
        )

    log_volumes = log_scales.astype(np.float64).sum(axis=1)
    reference = float(np.percentile(log_volumes, reference_percentile))
    log_ratios = np.minimum(log_volumes - reference, 0.0)
    return np.exp(power * log_ratios), reference


def _validated_scores(scores: np.ndarray, expected_count: int | None = None) -> np.ndarray:
    scores = np.asarray(scores)
    if scores.ndim != 1:
        raise ArtifactError(f"pruning scores must be one-dimensional, got {scores.shape}")
    if expected_count is not None and len(scores) != expected_count:
        raise ArtifactError(
            f"pruning scores have {len(scores):,} values for {expected_count:,} Gaussians"
        )
    if len(scores) == 0:
        raise ArtifactError("pruning scores must not be empty")
    if not np.isfinite(scores).all():
        raise ArtifactError("pruning scores contain non-finite values")
    if np.any(scores < 0):
        raise ArtifactError("pruning scores must be non-negative")
    return scores.astype(np.float64, copy=False)


def contribution_scores(
    cloud: GaussianCloud,
    contributions: np.ndarray,
    volume_power: float = 0.1,
    volume_percentile: float = 10.0,
) -> tuple[np.ndarray, float]:
    contributions = _validated_scores(contributions, len(cloud))
    weights, reference = volume_weights(
        cloud.scales,
        power=volume_power,
        reference_percentile=volume_percentile,
    )
    return contributions * weights, reference


def opacity_scores(cloud: GaussianCloud) -> np.ndarray:
    if not np.isfinite(cloud.opacities).all():
        raise ArtifactError("opacity logits contain non-finite values")
    return np.exp(-np.logaddexp(0.0, -cloud.opacities.astype(np.float64)))


def select_retained_indices(
    scores: np.ndarray,
    retained_fraction: float,
    expected_count: int | None = None,
) -> np.ndarray:
    scores = _validated_scores(scores, expected_count)
    if not np.isfinite(retained_fraction) or not 0 < retained_fraction <= 1:
        raise ConfigError(
            f"retained fraction must be greater than 0 and at most 1, got {retained_fraction!r}"
        )
    retained_count = min(len(scores), math.ceil(len(scores) * retained_fraction))
    ranked = np.argsort(-scores, kind="stable")[:retained_count]
    return np.sort(ranked)


def prune_by_score(
    cloud: GaussianCloud,
    scores: np.ndarray,
    retained_fraction: float,
) -> GaussianCloud:
    indices = select_retained_indices(scores, retained_fraction, expected_count=len(cloud))
    pruned = cloud.take(indices)
    pruned.validate()
    return pruned


def collect_contributions(
    cloud: GaussianCloud,
    views: Sequence,
    device: str = "cuda",
) -> np.ndarray:
    """Sum each Gaussian's alpha-blending weight over calibration pixels."""
    if not views:
        raise ArtifactError("no calibration views were provided")

    import torch
    from gsplat import rasterization
    from splatpipe.bench.render import CAMERA_MODEL, FAR_PLANE, NEAR_PLANE, RASTERIZE_MODE

    means = torch.from_numpy(cloud.means).to(device)
    quats = torch.from_numpy(cloud.quats).to(device)
    scales = torch.exp(torch.from_numpy(cloud.scales).to(device))
    opacities = torch.sigmoid(torch.from_numpy(cloud.opacities).to(device))
    features = torch.ones((len(cloud), 1), device=device, requires_grad=True)
    total = torch.zeros(len(cloud), dtype=torch.float64, device=device)

    for view in views:
        camtoworld = torch.from_numpy(view.camtoworld).to(device).unsqueeze(0)
        intrinsics = torch.from_numpy(view.K).to(device).unsqueeze(0)
        rendered, _, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=features,
            viewmats=torch.linalg.inv(camtoworld),
            Ks=intrinsics,
            width=view.width,
            height=view.height,
            sh_degree=None,
            near_plane=NEAR_PLANE,
            far_plane=FAR_PLANE,
            camera_model=CAMERA_MODEL,
            rasterize_mode=RASTERIZE_MODE,
            packed=False,
        )
        gradient = torch.autograd.grad(rendered.sum(), features)[0]
        total.add_(gradient[:, 0].to(torch.float64))

    result = total.detach().cpu().numpy()
    return _validated_scores(result, len(cloud))
