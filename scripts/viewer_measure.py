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
ORBIT_SECONDS = 10
RENDER_SAMPLES = 200
# Uncapped rAF intervals time only command submission. Frame intervals stay vsync-bound; render
# time runs from rAF start, past the viewer's earlier-registered draw, to a readback that waits for the GPU.
RENDER_MS = """async (samples) => {
    const gl = document.getElementById("canvas").getContext("webgl2");
    const pixel = new Uint8Array(4);
    const times = [];
    for (let i = 0; i < samples; i++) {
        const start = await new Promise((resolve) => requestAnimationFrame(resolve));
        gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixel);
        times.push(performance.now() - start);
    }
    return times;
}"""


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


def performance() -> dict:
    import numpy as np
    from playwright.sync_api import sync_playwright

    from scripts.viewer_harness import viewer_camera, viewer_url, wait_for_frame, watch_errors
    from splatpipe.bench.cameras import load_val_views

    view = load_val_views(TRUCK_SCENE, data_factor=1, test_every=8)[0]
    camera = viewer_camera(view.camtoworld, view.K)
    del camera["cx"], camera["cy"]  # centred in the 1280x800 window
    runs = []
    with sync_playwright() as playwright, serve() as base:
        browser = launch(playwright, headless=False)
        for sh in (True, False):
            # A fresh context each run, so the second load is not served from cache.
            context = browser.new_context(viewport={"width": 1280, "height": 800}, device_scale_factor=1)
            page = context.new_page()
            errors = watch_errors(page)
            page.goto(viewer_url(base, SCENE_URL, camera=camera, sh=sh, orbit=True))
            wait_for_frame(page)
            load = page.evaluate("() => ({ ...window.viewerStats, frameMs: [], sortMs: [] })")
            page.evaluate("() => { viewerStats.frameMs.length = 0; viewerStats.sortMs.length = 0; }")
            page.wait_for_timeout(ORBIT_SECONDS * 1000)
            timing = page.evaluate("() => ({ frameMs: viewerStats.frameMs, sortMs: viewerStats.sortMs })")
            render_ms = np.asarray(page.evaluate(RENDER_MS, RENDER_SAMPLES), dtype=np.float64)
            context.close()

            frame_ms = np.asarray(timing["frameMs"], dtype=np.float64)
            runs.append({
                "sh": sh,
                "renderer": load["renderer"],
                "error": load["error"],
                "first_frame_ms": load["firstFrameMs"],
                "phases": load["phases"],
                "frames": int(frame_ms.size),
                "frame_ms_median": float(np.median(frame_ms)),
                "frame_ms_p95": float(np.percentile(frame_ms, 95)),
                "fps_median": float(1000.0 / np.median(frame_ms)),
                "render_ms_median": float(np.median(render_ms)),
                "render_ms_p95": float(np.percentile(render_ms, 95)),
                "sort_ms_median": float(np.median(timing["sortMs"])) if timing["sortMs"] else None,
                "console_errors": list(errors),
            })
        browser.close()
    return {"viewport": [1280, 800], "container_bytes": CONTAINER.stat().st_size, "runs": runs}


if __name__ == "__main__":
    modes = {"quality": quality, "performance": performance}
    if len(sys.argv) != 2 or sys.argv[1] not in modes:
        sys.exit(f"usage: python -m scripts.viewer_measure {{{'|'.join(modes)}}}")
    _write(f"{sys.argv[1]}.json", modes[sys.argv[1]]())
