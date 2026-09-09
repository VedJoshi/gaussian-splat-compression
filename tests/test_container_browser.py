"""Whether a PNG block survives a browser canvas round-trip. Marked gpu only to
inherit the existing slow tier; it needs a browser, not a GPU.

A browser decodes PNG through its image pipeline, where colour management and
alpha premultiplication can rewrite pixel values. Pillow's path does not. If a
PNG block cannot be read back byte-exactly here, png must not be offered as a
block codec, and it is far cheaper to learn that now than in milestone 6."""

from __future__ import annotations

import base64

import numpy as np
import pytest

from splatpipe.formats.container import encode_block

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
