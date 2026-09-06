"""Milestone 2's falsification criterion.

PlyCodec is lossless, so rendering its output must reproduce what the trainer
measured when it rendered the same Gaussians. If the camera pipeline has
drifted, this fails. Without it, every number the project reports afterwards is
unfalsifiable.

The tolerance is fixed here, before the run, and is not to be adjusted
afterwards. It is tighter than milestone 1's 0.1 dB because nothing is being
retrained: the only remaining variation is float reduction order.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from splatpipe.bench.cameras import load_val_views
from splatpipe.bench.metrics import score_views
from splatpipe.bench.render import render_view
from splatpipe.gaussians import read_ply

pytestmark = pytest.mark.gpu

REPO = Path(__file__).resolve().parents[1]
TRUCK_PLY = REPO / "out" / "truck" / "artifacts" / "scene.ply"
TRUCK_SCENE = REPO / "data" / "tandt" / "truck"

# Recorded by the trainer on 2026-08-31, out/truck/train/stats/val_step6999.json.
TRAINER_PSNR = 24.394817
TRAINER_SSIM = 0.8579996
TRAINER_LPIPS = 0.1375573
PSNR_TOLERANCE = 0.05


@pytest.mark.skipif(not TRUCK_PLY.is_file(), reason="truck run not present")
def test_bench_reproduces_the_trainers_own_held_out_metrics():
    cloud = read_ply(TRUCK_PLY)
    views = load_val_views(TRUCK_SCENE, data_factor=1, test_every=8)
    targets = [view.image.astype("float32") / 255.0 for view in views]
    rendered = [render_view(cloud, view) for view in views]
    scores = score_views(rendered, targets)

    assert scores["psnr"] == pytest.approx(TRAINER_PSNR, abs=PSNR_TOLERANCE)
    assert scores["ssim"] == pytest.approx(TRAINER_SSIM, abs=0.002)
    assert scores["lpips"] == pytest.approx(TRAINER_LPIPS, abs=0.005)
