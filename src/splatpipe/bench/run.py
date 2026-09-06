"""Measure every codec against one trained run.

The first codec in the list is the anchor: its size is the denominator of every
ratio, and it is expected to be lossless. Each codec encodes into its own
subdirectory of a scratch area so that sizes never overlap.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from splatpipe.bench.cameras import load_val_views
from splatpipe.bench.curve import Curve, CurvePoint
from splatpipe.bench.metrics import score_views
from splatpipe.bench.render import render_view
from splatpipe.errors import ArtifactError
from splatpipe.gaussians import read_ply
from splatpipe.paths import RunPaths


def run_bench(
    run_dir: Path,
    scene_dir: Path,
    codecs: list,
    device: str = "cuda",
    data_factor: int = 1,
    test_every: int = 8,
) -> Curve:
    run_dir = Path(run_dir)
    paths = RunPaths(root=run_dir)
    if not paths.ply.is_file():
        raise ArtifactError(
            f"{run_dir} holds no scene.ply at {paths.ply}. "
            f"Run splatpipe run on this scene first."
        )
    if not codecs:
        raise ArtifactError("no codecs to measure")

    cloud = read_ply(paths.ply)
    views = load_val_views(scene_dir, data_factor=data_factor, test_every=test_every)
    if not views:
        raise ArtifactError(
            f"{scene_dir} yielded no held-out views at test_every={test_every}"
        )
    targets = [view.image.astype("float32") / 255.0 for view in views]

    curve = Curve.start(
        scene=_scene_name(paths, run_dir),
        config_digest=_config_digest(paths),
        held_out_views=len(views),
    )

    scratch = run_dir / "bench"
    if scratch.exists():
        shutil.rmtree(scratch)

    anchor_bytes = None
    for codec in codecs:
        directory = scratch / codec.name
        directory.mkdir(parents=True, exist_ok=True)

        started = time.time()
        codec.encode(cloud, directory)
        encode_seconds = time.time() - started

        started = time.time()
        decoded = codec.decode(directory)
        decode_seconds = time.time() - started

        size_bytes = codec.size(directory)
        if anchor_bytes is None:
            anchor_bytes = size_bytes

        rendered = [render_view(decoded, view, device=device) for view in views]
        scores = score_views(rendered, targets, device=device)

        curve.add(
            CurvePoint.of(
                codec=codec.name,
                size_bytes=size_bytes,
                gaussians=len(decoded),
                scores=scores,
                encode_seconds=encode_seconds,
                decode_seconds=decode_seconds,
                anchor_bytes=anchor_bytes,
            )
        )
        print(
            f"{codec.name:6s} {size_bytes:>12,} bytes  "
            f"PSNR {scores['psnr']:.3f}  SSIM {scores['ssim']:.4f}  "
            f"LPIPS {scores['lpips']:.4f}"
        )

    curve.write_json(paths.curve_json)
    curve.write_plot(paths.curve_png)
    return curve


def _scene_name(paths: RunPaths, run_dir: Path) -> str:
    try:
        return json.loads(paths.manifest.read_text(encoding="utf-8"))["name"]
    except (OSError, KeyError, json.JSONDecodeError):
        return run_dir.name


def _config_digest(paths: RunPaths) -> str:
    try:
        return json.loads(paths.manifest.read_text(encoding="utf-8"))["config_digest"]
    except (OSError, KeyError, json.JSONDecodeError):
        return "unknown"
