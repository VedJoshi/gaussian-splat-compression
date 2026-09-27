"""Measure the viewer on the pruned truck container. Writes out/truck/viewer-m6/.

    python -m scripts.viewer_measure quality       # needs scripts\\env.bat for gsplat
    python -m scripts.viewer_measure performance   # opens a visible Chromium window
"""

from __future__ import annotations

import json
import sys

from scripts.viewer_harness import REPO, launch, psnr, render_in_viewer, serve

CONTAINER = REPO / "out" / "truck" / "container-prune80" / "scene.splatc"
TRUCK_SCENE = REPO / "data" / "tandt" / "truck"
SCENE_URL = "/out/truck/container-prune80/scene.splatc"
OUTPUT = REPO / "out" / "truck" / "viewer-m6"
VIEWS = (0, 10, 20)
FLOOR_DB = 30.0


def _write(name: str, result: dict) -> None:
    path = OUTPUT / name
    if path.exists():
        sys.exit(f"{path} exists; move it aside to re-measure")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def quality() -> dict:
    from playwright.sync_api import sync_playwright

    from splatpipe.bench.cameras import load_val_views
    from splatpipe.bench.render import render_view
    from splatpipe.formats.container import read_container, unpack_scene

    cloud = unpack_scene(read_container(CONTAINER))
    views = load_val_views(TRUCK_SCENE, data_factor=1, test_every=8)
    rows = []
    with sync_playwright() as playwright, serve() as base:
        browser = launch(playwright)
        page = browser.new_page()
        renderer = page.evaluate(
            "() => { const gl = document.createElement('canvas').getContext('webgl2');"
            " const info = gl.getExtension('WEBGL_debug_renderer_info');"
            " return gl.getParameter(info ? info.UNMASKED_RENDERER_WEBGL : gl.RENDERER); }"
        )
        page.close()
        for index in VIEWS:
            view = views[index]
            reference = render_view(cloud, view)
            height, width = reference.shape[:2]
            row = {"view": index}
            for key, sh in (("sh_db", True), ("dc_db", False)):
                image, errors = render_in_viewer(browser, base, SCENE_URL, view.camtoworld, view.K, width, height, sh=sh)
                if errors:
                    sys.exit(f"viewer console errors on view {index}: {errors}")
                row[key] = psnr(image, reference)
            rows.append(row)
        browser.close()
    return {"floor_db": FLOOR_DB, "container": SCENE_URL, "views": rows, "renderer": renderer}


if __name__ == "__main__":
    modes = {"quality": quality}
    if len(sys.argv) != 2 or sys.argv[1] not in modes:
        sys.exit(f"usage: python -m scripts.viewer_measure {{{'|'.join(modes)}}}")
    _write(f"{sys.argv[1]}.json", modes[sys.argv[1]]())
