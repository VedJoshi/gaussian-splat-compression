"""The viewer loads a synthetic container end to end in Chromium. Marked gpu
only to join the existing slow tier; it needs a browser, not CUDA."""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from scripts.viewer_harness import REPO, capture, serve, viewer_url, wait_for_frame, watch_errors
from splatpipe.formats.container import PackedScene, pack_scene, write_container
from splatpipe.gaussians import GaussianCloud
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


def test_the_panel_names_the_scene_and_credits_the_renderer(browser, site):
    _, base = site
    page, errors = _open(browser, base, "/scene.splatc")
    name = page.inner_text("#scene-name")
    stats = page.inner_text("#scene-stats")
    home = page.get_attribute("#home", "href")
    credit = page.get_attribute("#credit a", "href")
    body = page.inner_text("body")
    page.close()

    assert errors == []
    assert name == "Scene"
    assert stats.startswith("300 Gaussians · ")
    assert " MB · loaded in " in stats and stats.endswith(" s")
    assert home == "../index.html"
    assert credit == "https://github.com/antimatter15/splat"
    assert "Kevin Kwok" not in body


def test_a_corrupted_container_is_reported_not_hung(browser, site):
    root, base = site
    data = bytearray((root / "scene.splatc").read_bytes())
    data[-1] ^= 0xFF
    (root / "scene.splatc").write_bytes(bytes(data))

    page, errors = _open(browser, base, "/scene.splatc")
    message = page.inner_text("#message")
    stats = page.inner_text("#scene-stats")
    page.close()
    assert "checksum" in message
    assert len(errors) == 1
    assert stats == ""


def test_a_missing_scene_is_reported_not_hung(browser, site):
    _, base = site
    page, _ = _open(browser, base, "/absent.splatc")
    message = page.inner_text("#message")
    stats = page.inner_text("#scene-stats")
    page.close()
    assert "404" in message
    assert stats == ""


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


def test_a_gaussian_centred_off_screen_still_covers_the_image(browser, site):
    root, base = site
    # Centred 60 px below the bottom edge with sigma 50 px, so gsplat draws it over the bottom rows.
    cloud = GaussianCloud(
        means=np.array([[0.0, 1.8, 0.0], [0.5, -0.5, 0.5]], np.float32),
        scales=np.log(np.array([[0.5, 0.5, 0.5], [0.01, 0.01, 0.01]], np.float32)),
        quats=np.array([[1, 0, 0, 0], [1, 0, 0, 0]], np.float32),
        opacities=np.array([4.0, -10.0], np.float32),
        sh0=np.full((2, 3), 1.5, np.float32),
        shN=np.zeros((2, 15, 3), np.float32),
    )
    scene = pack_scene(cloud)
    fields = {k: v for k, v in scene.fields.items() if k != "shN"}
    write_container(PackedScene(count=scene.count, sh_degree=0, order=scene.order, fields=fields), root / "edge.splatc")

    page, errors = _open(browser, base, "/edge.splatc")
    image = capture(page)
    page.close()
    assert errors == []
    assert image[-10:].mean() > 20


def test_gaussians_a_millimetre_apart_draw_in_depth_order(browser, site):
    root, base = site
    # Outliers stretch the depth range, as stray Gaussians do in real scenes; the green disc
    # sits 1 mm behind the red one but comes first in storage order.
    cloud = GaussianCloud(
        means=np.array([[0, 0, 0.001], [0, 0, 0], [1, 1, 100], [-1, -1, -60]], np.float32),
        scales=np.log(np.array([[0.3] * 3, [0.3] * 3, [0.01] * 3, [0.01] * 3], np.float32)),
        quats=np.tile(np.array([1, 0, 0, 0], np.float32), (4, 1)),
        opacities=np.array([6.0, 6.0, -10.0, -10.0], np.float32),
        sh0=np.array([[-1.5, 1.5, -1.5], [1.5, -1.5, -1.5], [0, 0, 0], [0, 0, 0]], np.float32),
        shN=np.zeros((4, 15, 3), np.float32),
    )
    scene = pack_scene(cloud, order="none")
    fields = {k: v for k, v in scene.fields.items() if k != "shN"}
    write_container(PackedScene(count=scene.count, sh_degree=0, order=scene.order, fields=fields), root / "stack.splatc")

    page, errors = _open(browser, base, "/stack.splatc")
    image = capture(page)
    page.close()
    assert errors == []
    red, green = image[110:130, 150:170, :2].reshape(-1, 2).mean(0)
    assert red > 150 > green, (red, green)
