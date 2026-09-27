"""The browser must render what the benchmark measured: same decoded cloud,
same held-out cameras, gsplat as the reference. The floor was fixed before
the first run and does not move."""

from __future__ import annotations

import pytest

from scripts.viewer_harness import psnr, render_in_viewer, serve
from scripts.viewer_measure import CONTAINER, FLOOR_DB, SCENE_URL, TRUCK_SCENE, VIEWS

pytestmark = pytest.mark.gpu


@pytest.mark.skipif(not CONTAINER.is_file(), reason="milestone 5 truck container not present")
def test_viewer_matches_gsplat_on_held_out_views(browser):
    from splatpipe.bench.cameras import load_val_views
    from splatpipe.bench.render import render_view
    from splatpipe.formats.container import read_container, unpack_scene

    cloud = unpack_scene(read_container(CONTAINER))
    views = load_val_views(TRUCK_SCENE, data_factor=1, test_every=8)
    scores = {}
    with serve() as base:
        for index in VIEWS:
            view = views[index]
            reference = render_view(cloud, view)
            height, width = reference.shape[:2]
            image, errors = render_in_viewer(browser, base, SCENE_URL, view.camtoworld, view.K, width, height)
            assert errors == [], errors
            assert image.shape == reference.shape
            scores[index] = psnr(image, reference)
    assert all(score >= FLOOR_DB for score in scores.values()), scores
