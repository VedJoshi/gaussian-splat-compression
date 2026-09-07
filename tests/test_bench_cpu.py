"""CPU coverage of the bench orchestration and its error contract.

The real thing needs a GPU. Everything about ordering, the curve's contents and
every way the command can fail is exercised here with rendering and scoring
stubbed, so it survives a two-week gap without a compiler shell.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from splatpipe.bench import run as bench_run
from splatpipe.bench.cameras import ValView
from splatpipe.bench.codecs import PlyCodec, SplatCodec
from splatpipe.bench.curve import Curve
from splatpipe.cli import main
from splatpipe.errors import ArtifactError
from splatpipe.gaussians import write_ply
from splatpipe.paths import RunPaths
from tests.test_codecs import a_cloud

SCORES = {"psnr": 22.5, "ssim": 0.81, "lpips": 0.19}


def a_run(tmp_path, name="truck", n=64):
    """Build the on-disk shape splatpipe run leaves behind."""
    paths = RunPaths.for_run(tmp_path / "out", name)
    paths.ensure()
    write_ply(a_cloud(n), paths.ply)
    paths.manifest.write_text(
        json.dumps({"name": name, "config_digest": "abc123def456"}), encoding="utf-8"
    )
    return paths


def stub_views(monkeypatch, count=3):
    views = [
        ValView(
            camtoworld=np.eye(4, dtype=np.float32),
            K=np.eye(3, dtype=np.float32),
            image=np.zeros((8, 8, 3), dtype=np.uint8),
        )
        for _ in range(count)
    ]
    monkeypatch.setattr(bench_run, "load_val_views", lambda *a, **k: views)
    monkeypatch.setattr(
        bench_run, "render_view", lambda cloud, view, device="cuda": np.zeros(
            (8, 8, 3), dtype=np.float32
        )
    )
    monkeypatch.setattr(bench_run, "score_views", lambda *a, **k: dict(SCORES))
    return views


def test_writes_a_curve_with_one_point_per_codec(tmp_path, monkeypatch):
    paths = a_run(tmp_path)
    stub_views(monkeypatch)
    curve = bench_run.run_bench(
        paths.root, tmp_path / "scene", [PlyCodec(), SplatCodec()], device="cpu"
    )
    assert [point.codec for point in curve.points] == ["ply", "splat"]
    assert paths.curve_json.is_file()


def test_the_anchor_is_the_first_codec(tmp_path, monkeypatch):
    paths = a_run(tmp_path)
    stub_views(monkeypatch)
    curve = bench_run.run_bench(
        paths.root, tmp_path / "scene", [PlyCodec(), SplatCodec()], device="cpu"
    )
    assert curve.points[0].ratio == pytest.approx(1.0)
    assert curve.points[1].ratio > 1.0


def test_records_the_held_out_view_count(tmp_path, monkeypatch):
    paths = a_run(tmp_path)
    stub_views(monkeypatch, count=5)
    curve = bench_run.run_bench(paths.root, tmp_path / "scene", [PlyCodec()], device="cpu")
    assert curve.held_out_views == 5


def test_missing_trained_ply_raises_an_artifact_error(tmp_path, monkeypatch):
    paths = RunPaths.for_run(tmp_path / "out", "truck")
    paths.ensure()
    stub_views(monkeypatch)
    with pytest.raises(ArtifactError, match="no scene.ply"):
        bench_run.run_bench(paths.root, tmp_path / "scene", [PlyCodec()], device="cpu")


def test_cli_reports_a_missing_run_directory(tmp_path, capsys):
    assert main(["bench", str(tmp_path / "nope"), "--scene", str(tmp_path)]) == 1
    assert "error:" in capsys.readouterr().err


def test_cli_returns_zero_on_success(tmp_path, monkeypatch):
    """--codecs keeps this on the fast tier.

    The default list includes png, whose encode needs CUDA, cupy and plas.
    Selecting the two numpy-only codecs is what lets the command's success path
    be exercised without a GPU.
    """
    paths = a_run(tmp_path)
    stub_views(monkeypatch)
    assert (
        main(
            [
                "bench",
                str(paths.root),
                "--scene",
                str(tmp_path / "scene"),
                "--codecs",
                "ply,splat",
            ]
        )
        == 0
    )
    assert paths.curve_json.is_file()
    assert paths.curve_png.is_file()


def test_cli_rejects_an_unknown_codec_name(tmp_path, capsys):
    paths = a_run(tmp_path)
    assert (
        main(
            [
                "bench",
                str(paths.root),
                "--scene",
                str(tmp_path / "scene"),
                "--codecs",
                "ply,jpeg",
            ]
        )
        == 1
    )
    assert "unknown codec" in capsys.readouterr().err


def test_bench_rejects_an_unsafe_output_namespace(tmp_path, monkeypatch):
    paths = a_run(tmp_path)
    stub_views(monkeypatch)
    with pytest.raises(ArtifactError, match="namespace"):
        bench_run.run_bench(
            paths.root,
            tmp_path / "scene",
            [PlyCodec()],
            device="cpu",
            output_namespace="../elsewhere",
        )
