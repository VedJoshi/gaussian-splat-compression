"""The built landing page in Chromium. Marked gpu to join the browser tier; it needs a browser, not CUDA."""

from __future__ import annotations

import hashlib
from urllib.parse import parse_qs, urlparse

import pytest

from scripts.build_site import build
from scripts.viewer_harness import serve, watch_errors

pytestmark = pytest.mark.gpu


@pytest.fixture
def built(tmp_path):
    scenes = tmp_path / "scenes"
    scenes.mkdir()
    lines = []
    for name in ("truck", "train", "playroom"):
        data = f"SPLATC-{name}".encode()
        (scenes / f"{name}.splatc").write_bytes(data)
        lines.append(f"{hashlib.sha256(data).hexdigest()}  {name}.splatc")
    sums = tmp_path / "SHA256SUMS"
    sums.write_text("\n".join(lines) + "\n", encoding="utf-8")
    build(scenes, tmp_path / "site", sums)
    with serve(tmp_path / "site") as base:
        yield tmp_path / "site", base


@pytest.mark.parametrize("width", [1280, 390])
def test_the_landing_page_loads_whole_at_desktop_and_phone_width(browser, built, width):
    root, base = built
    page = browser.new_page(viewport={"width": width, "height": 800})
    errors = watch_errors(page)
    page.goto(f"{base}/index.html")
    # Lazy images off-screen, including the sideways comparison row on phones, must still load.
    page.evaluate("() => document.querySelectorAll('img').forEach((img) => { img.loading = 'eager'; })")
    page.wait_for_function("() => [...document.images].every((img) => img.complete)")
    broken = page.evaluate("() => [...document.images].filter((img) => img.naturalWidth === 0).map((img) => img.src)")
    hrefs = page.evaluate("() => [...document.querySelectorAll('a.card')].map((a) => a.getAttribute('href'))")
    overflow = page.evaluate("() => document.documentElement.scrollWidth - window.innerWidth")
    page.close()

    assert errors == []
    assert broken == []
    assert len(hrefs) == 3
    for href in hrefs:
        assert not href.startswith("/"), "absolute links break under the Pages sub-path"
        query = parse_qs(urlparse(href).query)
        assert "camera" in query
        assert (root / "viewer" / query["url"][0]).resolve().is_file()
    assert overflow <= 0, "the page scrolls sideways"
