from __future__ import annotations

import json

import numpy as np
import pytest

from splatpipe.bench import export
from splatpipe.compress import sh
from splatpipe.errors import ArtifactError
from splatpipe.formats.container import ShVq, read_container
from tests.test_bench_cpu import a_run, stub_views


def fake_codebook(monkeypatch):
    def assign(directory, shN):
        return ShVq(
            codebook=np.arange(8 * 45, dtype=np.uint8).reshape(8, 45) % 64,
            labels=np.arange(len(shN), dtype=np.uint8) % 8,
            mins=np.full(45, -0.2, dtype=np.float32),
            maxs=np.full(45, 0.2, dtype=np.float32),
            bits=6,
        )

    monkeypatch.setattr(sh, "sh_vq_from_baseline", assign)


def a_pruned_run(tmp_path, n=64, scores=None):
    paths = a_run(tmp_path, n=n)
    paths.pruning_dir.mkdir()
    np.savez_compressed(paths.prune_scores, score=np.arange(n, dtype=np.float64) if scores is None else scores)
    (paths.pruning_dir / "bench" / "prune80-shvq4096").mkdir(parents=True)
    return paths


def test_export_writes_and_scores_the_same_container(tmp_path, monkeypatch):
    paths = a_pruned_run(tmp_path)
    stub_views(monkeypatch)
    fake_codebook(monkeypatch)

    result = export.run_export(paths.root, tmp_path / "scene", device="cpu")

    deployed = paths.root / "export" / "scene.splatc"
    scored = paths.root / "export" / "bench" / "prune80-container" / "scene.splatc"
    assert read_container(deployed).count == 52
    assert deployed.read_bytes() == scored.read_bytes()
    curve = json.loads((paths.root / "export" / "curve.json").read_text(encoding="utf-8"))
    assert [point["codec"] for point in curve["points"]] == ["ply", "prune80-container"]
    assert curve["points"][-1]["bytes"] == deployed.stat().st_size == result["bytes"]
    meta = json.loads((paths.root / "export" / "meta.json").read_text(encoding="utf-8"))
    assert meta["retained_gaussians"] == 52
    assert set(meta["codecs"]) == set(read_container(deployed).fields)


def test_export_refuses_an_existing_output(tmp_path, monkeypatch):
    paths = a_pruned_run(tmp_path)
    (paths.root / "export").mkdir()
    with pytest.raises(ArtifactError, match="exists"):
        export.run_export(paths.root, tmp_path / "scene", device="cpu")


def test_export_needs_scores_and_the_pruned_codebook(tmp_path):
    paths = a_pruned_run(tmp_path)
    (paths.pruning_dir / "bench" / "prune80-shvq4096").rmdir()
    with pytest.raises(ArtifactError, match="prune80-shvq4096"):
        export.run_export(paths.root, tmp_path / "scene", device="cpu")
    paths.prune_scores.unlink()
    with pytest.raises(ArtifactError, match="scores"):
        export.run_export(paths.root, tmp_path / "scene", device="cpu")
    assert not (paths.root / "export").exists()


def test_export_refuses_scores_for_another_cloud(tmp_path, monkeypatch):
    paths = a_pruned_run(tmp_path, scores=np.arange(10, dtype=np.float64))
    fake_codebook(monkeypatch)
    with pytest.raises(ArtifactError):
        export.run_export(paths.root, tmp_path / "scene", device="cpu")
    assert not (paths.root / "export").exists()


def test_cli_reports_a_missing_prune_as_an_error(tmp_path, capsys):
    from splatpipe.cli import main

    paths = a_run(tmp_path)
    assert main(["export", str(paths.root), "--scene", str(tmp_path / "scene")]) == 1
    assert "splatpipe prune" in capsys.readouterr().err
