from __future__ import annotations

import json

import numpy as np

from splatpipe.bench import prune as prune_bench
from splatpipe.bench.codecs import PlyCodec, SplatCodec
from splatpipe.bench.curve import Curve, CurvePoint
from splatpipe.bench.cameras import CalibrationView
from splatpipe.cli import main
from tests.test_bench_cpu import a_run, stub_views


def stub_contributions(monkeypatch, count):
    views = [
        CalibrationView(
            camtoworld=np.eye(4, dtype=np.float32),
            K=np.eye(3, dtype=np.float32),
            height=8,
            width=8,
        )
    ]
    monkeypatch.setattr(prune_bench, "load_train_views", lambda *a, **k: views)
    monkeypatch.setattr(
        prune_bench,
        "collect_contributions",
        lambda cloud, views, device="cuda": np.arange(count, dtype=np.float64),
    )


def test_prune_bench_writes_scores_metadata_and_a_separate_curve(tmp_path, monkeypatch):
    paths = a_run(tmp_path, n=10)
    stub_views(monkeypatch)
    stub_contributions(monkeypatch, count=10)

    curve = prune_bench.run_prune_bench(
        paths.root,
        tmp_path / "scene",
        codecs=[PlyCodec()],
        retained_percentages=(50,),
        device="cpu",
    )

    assert [point.codec for point in curve.points] == [
        "ply",
        "prune50-ply",
        "opacity50-ply",
    ]
    assert [point.gaussians for point in curve.points] == [10, 5, 5]
    assert paths.prune_curve_json.is_file()
    assert paths.prune_curve_png.is_file()
    assert paths.curve_json != paths.prune_curve_json

    with np.load(paths.prune_scores, allow_pickle=False) as archive:
        assert set(archive.files) == {"contribution", "score", "opacity"}
        assert archive["score"].shape == (10,)
    metadata = json.loads(paths.prune_meta.read_text(encoding="utf-8"))
    assert metadata["calibration_split"] == "train"
    assert metadata["calibration_views"] == 1
    assert metadata["calibration_pixels"] == 64
    assert metadata["retained_percentages"] == [50]
    assert metadata["recovery"] == "none"
    assert metadata["selection"]["retained_percent"] == 50
    assert metadata["selection"]["passes"] is True
    assert metadata["quality_limits"]["ssim_min_delta"] == -0.002


def test_prune_cli_accepts_an_explicit_cpu_codec_sweep(tmp_path, monkeypatch):
    paths = a_run(tmp_path, n=10)
    stub_views(monkeypatch)
    stub_contributions(monkeypatch, count=10)

    assert (
        main(
            [
                "prune",
                str(paths.root),
                "--scene",
                str(tmp_path / "scene"),
                "--retain",
                "50",
                "--codecs",
                "ply",
            ]
        )
        == 0
    )
    curve = Curve.read_json(paths.prune_curve_json)
    assert [point.codec for point in curve.points] == [
        "ply",
        "prune50-ply",
        "opacity50-ply",
    ]


def test_prune_bench_composes_each_non_png_codec_and_keeps_png_as_a_reference(
    tmp_path, monkeypatch
):
    class CpuPngReference(PlyCodec):
        name = "png"

    paths = a_run(tmp_path, n=10)
    stub_views(monkeypatch)
    stub_contributions(monkeypatch, count=10)
    curve = prune_bench.run_prune_bench(
        paths.root,
        tmp_path / "scene",
        codecs=[PlyCodec(), CpuPngReference(), SplatCodec(order="none")],
        retained_percentages=(50,),
        device="cpu",
    )
    assert [point.codec for point in curve.points] == [
        "ply",
        "png",
        "splat",
        "prune50-ply",
        "opacity50-ply",
        "prune50-splat",
    ]


def test_selection_uses_all_three_quality_bounds_and_picks_the_most_aggressive():
    curve = Curve.start("scene", "digest", 2)

    def add(name, psnr, ssim, lpips):
        curve.add(
            CurvePoint.of(
                codec=name,
                size_bytes=100,
                gaussians=10,
                scores={"psnr": psnr, "ssim": ssim, "lpips": lpips},
                encode_seconds=0,
                decode_seconds=0,
                anchor_bytes=100,
            )
        )

    add("ply", 24.4, 0.858, 0.137)
    add("prune80-ply", 24.39, 0.857, 0.138)
    add("prune70-ply", 24.36, 0.8559, 0.139)
    selection = prune_bench.select_quality_point(curve, (80, 70))
    assert selection["retained_percent"] == 80
    assert selection["candidates"][1]["passes"] is False
