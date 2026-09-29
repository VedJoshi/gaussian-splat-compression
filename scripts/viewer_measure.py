"""Measure the viewer. Without --scene: the M6 truck run, written to out/truck/viewer-m6/.
With --scene: the built site in out/site, written to out/<scene>/viewer-m7/.

    python -m scripts.viewer_measure quality [--scene S]       # needs scripts\\env.bat for gsplat
    python -m scripts.viewer_measure performance [--scene S]   # opens a visible Chromium window
    python -m scripts.viewer_measure network --scene S --base https://...   # the deployed site
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from scripts.viewer_harness import REPO, channel_bias, launch, psnr, render_in_viewer, serve

CONTAINER = REPO / "out" / "truck" / "container-prune80" / "scene.splatc"
TRUCK_SCENE = REPO / "data" / "tandt" / "truck"
SCENE_URL = "/out/truck/container-prune80/scene.splatc"
OUTPUT = REPO / "out" / "truck" / "viewer-m6"
SITE = REPO / "out" / "site"
SCENES = {"truck": "tandt/truck", "train": "tandt/train", "playroom": "db/playroom"}
VIEWS = (0, 10, 20)
FLOOR_DB = 30.0
ORBIT_SECONDS = 10
RENDER_SAMPLES = 200
NETWORK_LOADS = 5
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


@dataclass(frozen=True)
class Target:
    container: Path
    data: Path
    root: Path  # served directory holding viewer/
    url: str  # scene URL as the viewer page receives it
    output: Path


M6 = Target(CONTAINER, TRUCK_SCENE, REPO, SCENE_URL, OUTPUT)


def site_target(scene: str) -> Target:
    return Target(
        SITE / "scenes" / f"{scene}.splatc",
        REPO / "data" / SCENES[scene],
        SITE,
        f"../scenes/{scene}.splatc",
        REPO / "out" / scene / "viewer-m7",
    )


def _write(target: Target, name: str, result: dict) -> None:
    path = target.output / name
    if path.exists():
        sys.exit(f"{path} exists; move it aside to re-measure")
    target.output.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def quality(target: Target) -> dict:
    from playwright.sync_api import sync_playwright

    from splatpipe.bench.cameras import load_val_views
    from splatpipe.bench.render import render_view
    from splatpipe.formats.container import read_container, unpack_scene

    cloud = unpack_scene(read_container(target.container))
    views = load_val_views(target.data, data_factor=1, test_every=8)
    rows = []
    with sync_playwright() as playwright, serve(target.root) as base:
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
            for key, sh in (("sh", True), ("dc", False)):
                image, errors = render_in_viewer(browser, base, target.url, view.camtoworld, view.K, width, height, sh=sh)
                if errors:
                    sys.exit(f"viewer console errors on view {index}: {errors}")
                row[f"{key}_db"] = psnr(image, reference)
                row[f"{key}_bias_levels"] = channel_bias(image, reference)
            rows.append(row)
        browser.close()
    return {"floor_db": FLOOR_DB, "container": target.url, "views": rows, "renderer": renderer}


def performance(target: Target, base: str | None = None, modes=(True, False), loads: int = 1) -> dict:
    import numpy as np
    from playwright.sync_api import sync_playwright

    from scripts.viewer_harness import viewer_camera, viewer_url, wait_for_frame, watch_errors
    from splatpipe.bench.cameras import load_val_views

    view = load_val_views(target.data, data_factor=1, test_every=8)[0]
    camera = viewer_camera(view.camtoworld, view.K)
    del camera["cx"], camera["cy"]  # centred in the 1280x800 window
    size = target.container.stat().st_size
    site = base or "local"
    runs = []
    server = contextlib.nullcontext(base) if base else serve(target.root)
    with sync_playwright() as playwright, server as base:
        browser = launch(playwright, headless=False)
        for sh in modes:
            for _ in range(loads):
                # A fresh context each run, so no load is served from cache.
                context = browser.new_context(viewport={"width": 1280, "height": 800}, device_scale_factor=1)
                page = context.new_page()
                errors = watch_errors(page)
                page.goto(viewer_url(base, target.url, camera=camera, sh=sh, orbit=True))
                wait_for_frame(page)
                load = page.evaluate("() => ({ ...window.viewerStats, frameMs: [], sortMs: [] })")
                if load["error"]:
                    context.close()
                    sys.exit(f"viewer failed at {base}: {load['error']}")
                page.evaluate("() => { viewerStats.frameMs.length = 0; viewerStats.sortMs.length = 0; }")
                page.wait_for_timeout(ORBIT_SECONDS * 1000)
                timing = page.evaluate("() => ({ frameMs: viewerStats.frameMs, sortMs: viewerStats.sortMs })")
                render_ms = np.asarray(page.evaluate(RENDER_MS, RENDER_SAMPLES), dtype=np.float64)
                context.close()

                frame_ms = np.asarray(timing["frameMs"], dtype=np.float64)
                runs.append({
                    "sh": sh,
                    "renderer": load["renderer"],
                    "first_frame_ms": load["firstFrameMs"],
                    "phases": load["phases"],
                    "fetch_mbit_s": size * 8 / load["phases"]["fetch"] / 1000,
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
    return {
        "viewport": [1280, 800],
        "base": site,
        "measured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "container_bytes": size,
        "runs": runs,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="python -m scripts.viewer_measure")
    parser.add_argument("mode", choices=("quality", "performance", "network"))
    parser.add_argument("--scene", choices=sorted(SCENES))
    parser.add_argument("--base", help="deployed site URL, for network mode")
    args = parser.parse_args()
    if args.mode == "network" and not (args.scene and args.base):
        parser.error("network needs --scene and --base")
    target = site_target(args.scene) if args.scene else M6
    if args.mode == "quality":
        result = quality(target)
    elif args.mode == "performance":
        result = performance(target)
    else:
        result = performance(target, base=args.base.rstrip("/"), modes=(True,), loads=NETWORK_LOADS)
    _write(target, f"{args.mode}.json", result)
