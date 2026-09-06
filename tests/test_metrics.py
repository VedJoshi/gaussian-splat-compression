"""Known-answer checks on the metric code.

The spec asks for these explicitly, so that a regression in the metrics is
visible without a GPU and without a full render. torchmetrics runs on CPU, so
only LPIPS needs its weights, which are cached after the first download.
"""

from __future__ import annotations

import numpy as np
import pytest

from splatpipe.bench.metrics import score_views


def a_pair(seed: int = 0, shape=(64, 64, 3)):
    rng = np.random.default_rng(seed)
    target = rng.uniform(0, 1, shape).astype(np.float32)
    return target


def test_identical_images_give_infinite_psnr_and_unit_ssim():
    target = a_pair()
    scores = score_views([target], [target], device="cpu")
    assert scores["ssim"] == pytest.approx(1.0, abs=1e-4)
    assert scores["psnr"] > 60


def test_psnr_matches_its_closed_form():
    """PSNR is 10 log10(1 / mse) at data_range 1. A fixed offset is checkable."""
    target = np.full((64, 64, 3), 0.5, dtype=np.float32)
    rendered = np.full((64, 64, 3), 0.6, dtype=np.float32)
    expected = 10 * np.log10(1.0 / (0.1**2))
    scores = score_views([rendered], [target], device="cpu")
    assert scores["psnr"] == pytest.approx(expected, abs=0.01)


def test_scores_average_over_every_view():
    good = np.full((64, 64, 3), 0.5, dtype=np.float32)
    bad = np.full((64, 64, 3), 0.9, dtype=np.float32)
    target = np.full((64, 64, 3), 0.5, dtype=np.float32)
    both = score_views([good, bad], [target, target], device="cpu")
    only_bad = score_views([bad], [target], device="cpu")
    assert both["psnr"] > only_bad["psnr"]


def test_mismatched_lengths_are_rejected():
    target = a_pair()
    with pytest.raises(ValueError, match="same number"):
        score_views([target], [target, target], device="cpu")
