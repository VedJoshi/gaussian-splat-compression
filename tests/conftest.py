from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def browser():
    sync_api = pytest.importorskip("playwright.sync_api")
    from scripts.viewer_harness import BrowserMissing, launch

    with sync_api.sync_playwright() as playwright:
        try:
            instance = launch(playwright)
        except BrowserMissing as error:
            pytest.skip(str(error))
        yield instance
        instance.close()
