"""The viewer loads a synthetic container end to end in Chromium. Marked gpu
only to join the existing slow tier; it needs a browser, not CUDA."""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from scripts.viewer_harness import REPO, capture, serve, viewer_url, wait_for_frame, watch_errors
from splatpipe.formats.container import PackedScene, write_container
from tests.test_viewer_js import a_vq_scene

pytestmark = pytest.mark.gpu

# Camera at z = -3 looking down +z at the synthetic cloud in [-1, 1]^3.
CAMERA = {"view": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 3, 1], "fx": 300.0, "fy": 300.0}


@pytest.fixture
def site(tmp_path):
    # A copy proves viewer/ is self-contained, as the milestone 7 static site needs.
    shutil.copytree(REPO / "viewer", tmp_path / "viewer")
    write_container(a_vq_scene(), tmp_path / "scene.splatc")
    with serve(tmp_path) as base:
        yield tmp_path, base


def _open(browser, base, scene, **options):
    page = browser.new_page(viewport={"width": 320, "height": 240}, device_scale_factor=1)
    errors = watch_errors(page)
    page.goto(viewer_url(base, scene, camera=CAMERA, **options))
    wait_for_frame(page)
    return page, errors


def test_the_viewer_renders_a_container(browser, site):
    _, base = site
    page, errors = _open(browser, base, "/scene.splatc")
    image = capture(page)
    stats = page.evaluate("() => window.viewerStats")
    page.close()

    assert errors == []
    assert stats["error"] is None
    assert image.shape == (240, 320, 3)
    assert image.max() > 0 and image.std() > 1.0, "the canvas is blank or uniform"
    assert set(stats["phases"]) == {"fetch", "decode", "reconstruct", "pack", "upload"}
    assert stats["firstFrameMs"] > 0


def test_a_corrupted_container_is_reported_not_hung(browser, site):
    root, base = site
    data = bytearray((root / "scene.splatc").read_bytes())
    data[-1] ^= 0xFF
    (root / "scene.splatc").write_bytes(bytes(data))

    page, errors = _open(browser, base, "/scene.splatc")
    message = page.inner_text("#message")
    page.close()
    assert "checksum" in message
    assert len(errors) == 1


def test_a_missing_scene_is_reported_not_hung(browser, site):
    _, base = site
    page, _ = _open(browser, base, "/absent.splatc")
    message = page.inner_text("#message")
    page.close()
    assert "404" in message


def test_sh_changes_the_image_and_sh_off_matches_dc(browser, site):
    _, base = site
    with_sh, errors_sh = _open(browser, base, "/scene.splatc")
    image_sh = capture(with_sh)
    with_sh.close()
    without, errors_dc = _open(browser, base, "/scene.splatc", sh=False)
    image_dc = capture(without)
    without.close()

    assert errors_sh == [] and errors_dc == []
    assert np.abs(image_sh.astype(int) - image_dc.astype(int)).mean() > 1.0


def test_a_container_without_sh_renders_dc_only_without_webgl_errors(browser, site):
    root, base = site
    scene = a_vq_scene()
    fields = {k: v for k, v in scene.fields.items() if k not in ("sh_codebook", "sh_labels")}
    write_container(PackedScene(count=scene.count, sh_degree=0, order=scene.order, fields=fields), root / "dc.splatc")

    page, errors = _open(browser, base, "/dc.splatc")
    image = capture(page)
    page.close()
    assert errors == []
    assert image.max() > 0
