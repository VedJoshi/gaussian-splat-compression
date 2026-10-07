from __future__ import annotations

import hashlib
import re

import pytest

from scripts.build_site import build


def a_scene_dir(tmp_path):
    scenes = tmp_path / "scenes"
    scenes.mkdir()
    (scenes / "truck.splatc").write_bytes(b"SPLATC-truck")
    sums = tmp_path / "SHA256SUMS"
    sums.write_text(f"{hashlib.sha256(b'SPLATC-truck').hexdigest()}  truck.splatc\n", encoding="utf-8")
    return scenes, sums


def test_builds_the_page_figures_viewer_and_pinned_scenes(tmp_path):
    scenes, sums = a_scene_dir(tmp_path)
    build(scenes, tmp_path / "site", sums)
    site = tmp_path / "site"
    assert (site / "index.html").is_file()
    assert (site / "figures" / "curve.svg").is_file()
    assert (site / "viewer" / "index.html").is_file()
    assert (site / "viewer" / "worker.mjs").is_file()
    assert (site / "scenes" / "truck.splatc").read_bytes() == b"SPLATC-truck"


def test_a_changed_byte_is_refused(tmp_path):
    scenes, sums = a_scene_dir(tmp_path)
    (scenes / "truck.splatc").write_bytes(b"SPLATC-truch")
    with pytest.raises(SystemExit, match="truck.splatc"):
        build(scenes, tmp_path / "site", sums)
    assert not (tmp_path / "site").exists()


def test_a_missing_scene_is_refused(tmp_path):
    scenes, sums = a_scene_dir(tmp_path)
    (scenes / "truck.splatc").unlink()
    with pytest.raises(SystemExit, match="truck.splatc"):
        build(scenes, tmp_path / "site", sums)


def test_an_existing_site_is_not_overwritten(tmp_path):
    scenes, sums = a_scene_dir(tmp_path)
    (tmp_path / "site").mkdir()
    (tmp_path / "site" / "old").write_text("x", encoding="utf-8")
    with pytest.raises(SystemExit, match="not empty"):
        build(scenes, tmp_path / "site", sums)


def test_every_figure_the_page_references_is_built(tmp_path):
    scenes, sums = a_scene_dir(tmp_path)
    build(scenes, tmp_path / "site", sums)
    site = tmp_path / "site"
    sources = re.findall(r'src="(figures/[^"]+)"', (site / "index.html").read_text(encoding="utf-8"))
    assert "figures/curve.svg" in sources
    assert len(set(sources)) == 10  # the curve and nine stills; cards reuse the compressed stills
    assert [s for s in sources if not (site / s).is_file()] == []
