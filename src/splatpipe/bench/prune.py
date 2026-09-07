"""Collect pruning scores and measure fixed retained-count sweeps."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np

from splatpipe.bench.cameras import load_train_views
from splatpipe.bench.codecs import Codec, PrunedCodec
from splatpipe.bench.curve import Curve
from splatpipe.bench.run import run_bench
from splatpipe.compress.prune import (
    collect_contributions,
    contribution_scores,
    opacity_scores,
    parse_retained_percentages,
)
from splatpipe.errors import ArtifactError
from splatpipe.gaussians import read_ply
from splatpipe.paths import RunPaths

PSNR_MIN_DELTA = -0.10
SSIM_MIN_DELTA = -0.002
LPIPS_MAX_DELTA = 0.005


def run_prune_bench(
    run_dir: Path,
    scene_dir: Path,
    codecs: list[Codec],
    retained_percentages: tuple[int, ...],
    device: str = "cuda",
    data_factor: int = 1,
    test_every: int = 8,
    volume_power: float = 0.1,
    volume_percentile: float = 10.0,
) -> Curve:
    retained_percentages = parse_retained_percentages(
        ",".join(str(value) for value in retained_percentages)
    )
    paths = RunPaths(root=Path(run_dir))
    if not paths.ply.is_file():
        raise ArtifactError(f"{run_dir} holds no scene.ply at {paths.ply}")
    if not codecs or codecs[0].name != "ply":
        raise ArtifactError("the pruning sweep must start with ply as its size anchor")
    if len({codec.name for codec in codecs}) != len(codecs):
        raise ArtifactError("the pruning sweep contains duplicate codec names")

    cloud = read_ply(paths.ply)
    calibration_views = load_train_views(
        scene_dir,
        data_factor=data_factor,
        test_every=test_every,
    )
    if not calibration_views:
        raise ArtifactError("the scene yielded no training views for pruning calibration")

    started = time.time()
    contribution = collect_contributions(cloud, calibration_views, device=device)
    score, reference = contribution_scores(
        cloud,
        contribution,
        volume_power=volume_power,
        volume_percentile=volume_percentile,
    )
    opacity = opacity_scores(cloud)
    score_seconds = time.time() - started

    paths.pruning_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        paths.prune_scores,
        contribution=contribution,
        score=score,
        opacity=opacity,
    )
    metadata = {
        "method": "accumulated_alpha_weight_times_capped_volume",
        "calibration_split": "train",
        "calibration_views": len(calibration_views),
        "calibration_pixels": sum(view.height * view.width for view in calibration_views),
        "source_gaussians": len(cloud),
        "retained_percentages": list(retained_percentages),
        "volume_power": volume_power,
        "volume_percentile": volume_percentile,
        "reference_log_volume": reference,
        "score_seconds": score_seconds,
        "recovery": "none",
        "codecs": [codec.name for codec in codecs],
        "quality_limits": {
            "psnr_min_delta": PSNR_MIN_DELTA,
            "ssim_min_delta": SSIM_MIN_DELTA,
            "lpips_max_delta": LPIPS_MAX_DELTA,
        },
    }
    paths.prune_meta.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    measured: list[Codec] = list(codecs)
    for retained_percent in retained_percentages:
        for codec in codecs:
            if codec.name == "png":
                continue
            measured.append(
                PrunedCodec(codec, score, retained_percent, score_name="prune")
            )
            if codec.name == "ply":
                measured.append(
                    PrunedCodec(
                        codec,
                        opacity,
                        retained_percent,
                        score_name="opacity",
                    )
                )

    curve = run_bench(
        paths.root,
        scene_dir,
        measured,
        device=device,
        data_factor=data_factor,
        test_every=test_every,
        output_namespace="pruning",
    )
    metadata["selection"] = select_quality_point(curve, retained_percentages)
    paths.prune_meta.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    write_pruning_plot(curve, paths.prune_curve_png, metadata["selection"])
    return curve


def select_quality_point(
    curve: Curve,
    retained_percentages: tuple[int, ...],
) -> dict:
    points = {point.codec: point for point in curve.points}
    if "ply" not in points:
        raise ArtifactError("pruning curve has no unpruned ply reference")
    reference = points["ply"]
    candidates = []
    for retained_percent in retained_percentages:
        name = f"prune{retained_percent}-ply"
        if name not in points:
            raise ArtifactError(f"pruning curve has no {name} point")
        point = points[name]
        deltas = {
            "psnr": point.psnr - reference.psnr,
            "ssim": point.ssim - reference.ssim,
            "lpips": point.lpips - reference.lpips,
        }
        passes = (
            deltas["psnr"] >= PSNR_MIN_DELTA
            and deltas["ssim"] >= SSIM_MIN_DELTA
            and deltas["lpips"] <= LPIPS_MAX_DELTA
        )
        candidates.append(
            {
                "retained_percent": retained_percent,
                "codec": name,
                "passes": passes,
                "deltas": deltas,
            }
        )
    passing = [candidate for candidate in candidates if candidate["passes"]]
    if not passing:
        return {"passes": False, "retained_percent": None, "candidates": candidates}
    selected = min(passing, key=lambda candidate: candidate["retained_percent"])
    return {**selected, "candidates": candidates}


def write_pruning_plot(curve: Curve, path: Path, selection: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = {point.codec: point for point in curve.points}
    reference = points["ply"]
    retained = sorted(
        int(match.group(1))
        for name in points
        if (match := re.fullmatch(r"prune(\d+)-ply", name))
    )
    metric_specs = (
        ("psnr", "PSNR change (dB)", PSNR_MIN_DELTA),
        ("ssim", "SSIM change", SSIM_MIN_DELTA),
        ("lpips", "LPIPS change", LPIPS_MAX_DELTA),
    )
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=150)
    for axis, (metric, label, limit) in zip(axes.flat[:3], metric_specs):
        baseline = getattr(reference, metric)
        contribution = [
            getattr(points[f"prune{percent}-ply"], metric) - baseline
            for percent in retained
        ]
        opacity = [
            getattr(points[f"opacity{percent}-ply"], metric) - baseline
            for percent in retained
        ]
        axis.plot(retained, contribution, "o-", label="contribution")
        axis.plot(retained, opacity, "s--", label="opacity")
        axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
        axis.axhline(limit, color="red", linestyle=":", linewidth=1, label="quality bound")
        axis.set_xlabel("Gaussians retained (%)")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
        if retained:
            axis.set_xlim(100, max(min(retained) - 2, 0))
        selected = selection.get("retained_percent")
        if selected is not None:
            axis.axvline(selected, color="green", linewidth=0.8, alpha=0.6)
    axes[0, 0].legend(fontsize=8)

    rate_axis = axes[1, 1]
    sh_points = []
    if "shvq4096" in points:
        sh_points.append((100, points["shvq4096"]))
    sh_points.extend(
        (percent, points[f"prune{percent}-shvq4096"])
        for percent in sorted(retained, reverse=True)
        if f"prune{percent}-shvq4096" in points
    )
    if sh_points:
        rate_axis.plot(
            [point.bytes / 1e6 for _, point in sh_points],
            [point.psnr for _, point in sh_points],
            "o-",
            label="SH VQ family",
        )
        for percent, point in sh_points:
            rate_axis.annotate(
                f"{percent}%",
                (point.bytes / 1e6, point.psnr),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
            )
        if "png" in points:
            png = points["png"]
            rate_axis.scatter(png.bytes / 1e6, png.psnr, marker="s", label="stock PNG")
    else:
        raw_points = [(100, reference)] + [
            (percent, points[f"prune{percent}-ply"]) for percent in retained
        ]
        rate_axis.plot(
            [point.bytes / 1e6 for _, point in raw_points],
            [point.psnr for _, point in raw_points],
            "o-",
            label="PLY family",
        )
    rate_axis.set_xlabel("Encoded size (MB)")
    rate_axis.set_ylabel("Held-out PSNR (dB)")
    rate_axis.set_title("Compressed rate-distortion")
    rate_axis.grid(alpha=0.25)
    rate_axis.legend(fontsize=8)

    figure.suptitle(f"{curve.scene}: contribution-pruning sweep")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
