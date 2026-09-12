"""Whether a PNG block survives a browser canvas round-trip. Marked gpu only to
inherit the existing slow tier; it needs a browser, not a GPU.

A browser decodes PNG through its image pipeline, where colour management and
alpha premultiplication can rewrite pixel values. Pillow's path does not. If a
PNG block cannot be read back byte-exactly here, png must not be offered as a
block codec, and it is far cheaper to learn that now than in milestone 6."""

from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
import pytest

from splatpipe.formats.container import encode_block, pack_scene, write_container
from tests.test_codecs import a_cloud

pytestmark = pytest.mark.gpu

CHROMIUM = (
    "C:/Users/vedti/AppData/Local/ms-playwright/chromium-1208/chrome-win64/chrome.exe"
)

READ_BACK = """
async (dataUrl) => {
  const blob = await (await fetch(dataUrl)).blob();
  const bitmap = await createImageBitmap(blob, { premultiplyAlpha: 'none',
                                                 colorSpaceConversion: 'none' });
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const context = canvas.getContext('2d', { willReadFrequently: true });
  context.drawImage(bitmap, 0, 0);
  const pixels = context.getImageData(0, 0, bitmap.width, bitmap.height).data;
  const out = [];
  for (let i = 0; i < pixels.length; i += 4) out.push(pixels[i]);
  return out;
}
"""


def test_a_png_block_survives_a_canvas_round_trip():
    playwright = pytest.importorskip("playwright.sync_api")

    payload = np.arange(256, dtype=np.uint8).repeat(4).tobytes()
    encoded = encode_block(payload, "png")
    data_url = "data:image/png;base64," + base64.b64encode(encoded).decode("ascii")

    with playwright.sync_playwright() as p:
        browser = p.chromium.launch(executable_path=CHROMIUM, headless=True)
        try:
            page = browser.new_page()
            recovered = page.evaluate(READ_BACK, data_url)
        finally:
            browser.close()

    assert bytes(recovered[: len(payload)]) == payload, (
        "chromium did not return the stored bytes. Colour management or alpha "
        "premultiplication rewrote them, so png must not be used as a block codec."
    )


def test_browser_imports_the_decoder_and_recovers_a_mixed_codec_container(tmp_path):
    playwright = pytest.importorskip("playwright.sync_api")
    scene = pack_scene(a_cloud(300))
    path = tmp_path / "scene.splatc"
    write_container(scene, path, codecs={"means": "raw", "sh0": "png"})
    decoder = Path(__file__).resolve().parents[1] / "scripts/decode_container.mjs"
    module_url = "data:text/javascript;base64," + base64.b64encode(decoder.read_bytes()).decode()
    payload = "data:application/octet-stream;base64," + base64.b64encode(path.read_bytes()).decode()
    with playwright.sync_playwright() as p:
        browser = p.chromium.launch(executable_path=CHROMIUM, headless=True)
        try:
            decoded = browser.new_page().evaluate(
                """async ([moduleUrl, payload]) => {
                    const { decodeContainer } = await import(moduleUrl);
                    return decodeContainer(await (await fetch(payload)).arrayBuffer());
                }""",
                [module_url, payload],
            )
        finally:
            browser.close()
    assert decoded["count"] == scene.count
    assert decoded["sh_degree"] == scene.sh_degree
    for name, field in scene.fields.items():
        np.testing.assert_array_equal(decoded["blocks"][name]["values"], field.values.ravel())
