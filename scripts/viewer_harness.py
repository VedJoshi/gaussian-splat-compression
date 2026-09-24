"""Serve the repository and drive the viewer in Chromium. Run as a module to
serve the repo for manual viewing: python -m scripts.viewer_harness"""

from __future__ import annotations

import base64
import contextlib
import functools
import json
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote

import numpy as np

REPO = Path(__file__).resolve().parents[1]
GPU_ARGS = ("--enable-gpu", "--ignore-gpu-blocklist", "--use-angle=default")


class _Handler(SimpleHTTPRequestHandler):
    # Python's mimetypes reads the Windows registry, which can map .js to text/plain.
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".js": "text/javascript",
        ".mjs": "text/javascript",
    }

    def log_message(self, *args):
        pass


@contextlib.contextmanager
def serve(root: Path = REPO):
    server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(_Handler, directory=str(root)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


class BrowserMissing(RuntimeError):
    pass


def launch(playwright, headless: bool = True, extra_args=()):
    try:
        return playwright.chromium.launch(headless=headless, args=[*GPU_ARGS, *extra_args])
    except Exception as error:  # Playwright raises its own Error type
        if "Executable doesn't exist" in str(error):
            raise BrowserMissing("Chromium is missing; run: python -m playwright install chromium") from error
        raise


def viewer_camera(camtoworld: np.ndarray, K: np.ndarray) -> dict:
    """The viewer's ?camera= payload: world-to-camera in OpenCV axes, column-major."""
    world_to_camera = np.linalg.inv(np.asarray(camtoworld, dtype=np.float64))
    return {
        "view": world_to_camera.T.ravel().tolist(),
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
    }


def viewer_url(base: str, scene: str, camera: dict | None = None, sh: bool = True, orbit: bool = False) -> str:
    url = f"{base}/viewer/index.html?url={quote(scene)}"
    if camera is not None:
        url += "&camera=" + quote(json.dumps(camera))
    if not sh:
        url += "&sh=0"
    if orbit:
        url += "&orbit=1"
    return url


def watch_errors(page) -> list[str]:
    errors: list[str] = []

    def on_console(message):
        # Chromium reports WebGL errors as warnings.
        if message.type == "error" or (message.type == "warning" and "WebGL" in message.text):
            errors.append(message.text)

    page.on("console", on_console)
    page.on("pageerror", lambda error: errors.append(f"pageerror: {error}"))
    return errors


def wait_for_frame(page, timeout_ms: int = 120_000) -> None:
    page.wait_for_function(
        "() => window.viewerStats && (window.viewerStats.firstFrameMs !== null || window.viewerStats.error)",
        timeout=timeout_ms,
    )


def capture(page) -> np.ndarray:
    frame = page.evaluate("() => window.captureFrame()")
    rgba = np.frombuffer(base64.b64decode(frame["pixels"]), dtype=np.uint8)
    # readPixels returns the bottom row first.
    return np.ascontiguousarray(rgba.reshape(frame["height"], frame["width"], 4)[::-1, :, :3])


def render_in_viewer(browser, base, scene, camtoworld, K, width, height, sh=True):
    page = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=1)
    errors = watch_errors(page)
    try:
        page.goto(viewer_url(base, scene, camera=viewer_camera(camtoworld, K), sh=sh))
        wait_for_frame(page)
        error = page.evaluate("() => window.viewerStats.error")
        if error:
            raise AssertionError(f"viewer failed: {error}")
        return capture(page), errors
    finally:
        page.close()


def psnr(image: np.ndarray, reference: np.ndarray) -> float:
    mse = float(np.mean((image.astype(np.float64) / 255.0 - reference.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(1.0 / mse)


if __name__ == "__main__":
    with serve() as base:
        print(f"{base}/viewer/index.html?url=/out/truck/container-prune80/scene.splatc")
        threading.Event().wait()
