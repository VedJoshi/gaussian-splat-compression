"""Curve assembly and its JSON schema. Pure data, so this is fast tier."""

from __future__ import annotations

import json

import pytest

from splatpipe.bench.curve import Curve, CurvePoint


def a_curve() -> Curve:
    return Curve.start(scene="truck", config_digest="c221a1fc2180", held_out_views=32)


def test_ratio_is_computed_against_the_anchor():
    curve = a_curve()
    curve.add(
        CurvePoint.of(
            codec="ply", size_bytes=236_001_478, gaussians=1_000_000,
            scores={"psnr": 24.4, "ssim": 0.858, "lpips": 0.137},
            encode_seconds=1.0, decode_seconds=2.0, anchor_bytes=236_001_478,
        )
    )
    curve.add(
        CurvePoint.of(
            codec="splat", size_bytes=32_000_000, gaussians=1_000_000,
            scores={"psnr": 20.1, "ssim": 0.71, "lpips": 0.29},
            encode_seconds=3.0, decode_seconds=0.5, anchor_bytes=236_001_478,
        )
    )
    assert curve.points[0].ratio == pytest.approx(1.0)
    assert curve.points[1].ratio == pytest.approx(7.375046, abs=1e-5)


def test_round_trips_through_json(tmp_path):
    curve = a_curve()
    curve.add(
        CurvePoint.of(
            codec="ply", size_bytes=100, gaussians=10,
            scores={"psnr": 1.0, "ssim": 0.5, "lpips": 0.25},
            encode_seconds=0.1, decode_seconds=0.2, anchor_bytes=100,
        )
    )
    path = tmp_path / "curve.json"
    curve.write_json(path)
    back = Curve.read_json(path)
    assert back.scene == "truck"
    assert back.points[0].codec == "ply"
    assert back.points[0].psnr == 1.0


def test_json_records_the_context_that_makes_a_point_traceable(tmp_path):
    path = tmp_path / "curve.json"
    a_curve().write_json(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["scene"] == "truck"
    assert data["config_digest"] == "c221a1fc2180"
    assert data["held_out_views"] == 32
    assert "splatpipe_commit" in data["versions"]
    assert "gsplat" in data["versions"]


def test_gaussian_count_is_recorded_per_point():
    """PngCompression crops to a square count, so a point can hold fewer
    Gaussians than were handed to it. Assuming they match would misreport it."""
    curve = a_curve()
    curve.add(
        CurvePoint.of(
            codec="png", size_bytes=16_258_005, gaussians=999_999,
            scores={"psnr": 24.0, "ssim": 0.85, "lpips": 0.14},
            encode_seconds=120.0, decode_seconds=4.0, anchor_bytes=236_001_478,
        )
    )
    assert curve.points[0].gaussians == 999_999


def test_plot_writes_a_png(tmp_path):
    curve = a_curve()
    curve.add(
        CurvePoint.of(
            codec="ply", size_bytes=100, gaussians=10,
            scores={"psnr": 30.0, "ssim": 0.9, "lpips": 0.1},
            encode_seconds=0.1, decode_seconds=0.1, anchor_bytes=100,
        )
    )
    path = tmp_path / "curve.png"
    curve.write_plot(path)
    assert path.is_file()
    assert path.stat().st_size > 0
