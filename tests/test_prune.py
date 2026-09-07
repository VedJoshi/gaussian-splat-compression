from __future__ import annotations

import numpy as np
import pytest

from splatpipe.compress.prune import (
    contribution_scores,
    opacity_scores,
    parse_retained_percentages,
    prune_by_score,
    select_retained_indices,
    volume_weights,
)
from splatpipe.errors import ArtifactError, ConfigError
from tests.test_codecs import a_cloud


def test_parse_retained_percentages_preserves_the_declared_sweep():
    assert parse_retained_percentages("95,90,80,70") == (95, 90, 80, 70)


@pytest.mark.parametrize("value", ("", "0", "100", "90,90", "90.5", "nope"))
def test_parse_retained_percentages_rejects_invalid_input(value):
    with pytest.raises(ConfigError):
        parse_retained_percentages(value)


def test_volume_weight_is_capped_and_only_penalizes_small_gaussians():
    log_scales = np.array(
        [[-2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    weights, reference = volume_weights(
        log_scales, power=0.1, reference_percentile=50.0
    )
    assert reference == pytest.approx(0.0)
    np.testing.assert_allclose(weights, [np.exp(-0.2), 1.0, 1.0])


def test_contribution_score_applies_the_volume_weight():
    cloud = a_cloud(3)
    cloud = cloud.__class__(
        means=cloud.means,
        scales=np.array(
            [[-2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
        quats=cloud.quats,
        opacities=cloud.opacities,
        sh0=cloud.sh0,
        shN=cloud.shN,
    )
    scores, reference = contribution_scores(
        cloud,
        np.array([10.0, 10.0, 10.0]),
        volume_power=0.1,
        volume_percentile=50.0,
    )
    assert reference == pytest.approx(0.0)
    np.testing.assert_allclose(scores, [10 * np.exp(-0.2), 10.0, 10.0])


def test_select_retained_indices_uses_ceil_and_stable_ties():
    scores = np.array([3.0, 2.0, 2.0, 1.0], dtype=np.float64)
    selected = select_retained_indices(scores, retained_fraction=0.625)
    np.testing.assert_array_equal(selected, [0, 1, 2])


@pytest.mark.parametrize(
    "scores",
    (
        np.array([1.0, np.nan]),
        np.array([1.0, np.inf]),
        np.array([1.0, -0.1]),
        np.ones((2, 1)),
    ),
)
def test_select_retained_indices_rejects_malformed_scores(scores):
    with pytest.raises(ArtifactError):
        select_retained_indices(scores, retained_fraction=0.5)


def test_prune_by_score_subsets_every_field_without_reordering_survivors():
    cloud = a_cloud(5)
    scores = np.array([0.0, 5.0, 1.0, 4.0, 3.0])
    pruned = prune_by_score(cloud, scores, retained_fraction=0.6)
    assert len(pruned) == 3
    np.testing.assert_array_equal(pruned.means, cloud.means[[1, 3, 4]])
    np.testing.assert_array_equal(pruned.shN, cloud.shN[[1, 3, 4]])


def test_opacity_score_activates_logits_stably():
    cloud = a_cloud(3)
    cloud.opacities[:] = [-1000.0, 0.0, 1000.0]
    np.testing.assert_allclose(opacity_scores(cloud), [0.0, 0.5, 1.0], atol=1e-7)
