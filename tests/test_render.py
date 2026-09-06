"""Rendering a cloud through gsplat. GPU tier: there is no CPU rasteriser."""

from __future__ import annotations

import numpy as np
import pytest

from splatpipe.bench.cameras import load_val_views
from splatpipe.bench.render import render_view
from splatpipe.gaussians import read_ply
from tests.fixtures.tiny_scene import make_tiny_scene

pytestmark = pytest.mark.gpu


def test_render_matches_the_view_shape_and_range(tmp_path):
    from tests.test_codecs import a_cloud

    scene = make_tiny_scene(tmp_path / "scene", n_images=24, n_points=512)
    view = load_val_views(scene, data_factor=1, test_every=8)[0]
    image = render_view(a_cloud(256), view)

    assert image.shape == view.image.shape
    assert image.dtype == np.float32
    assert image.min() >= 0.0 and image.max() <= 1.0


def test_render_uses_the_clouds_own_sh_degree(tmp_path):
    """A decoded .splat has sh_degree 0. Passing a fixed 3 would raise inside
    rasterization, because colors would carry 1 coefficient where 16 is claimed."""
    from splatpipe.bench.codecs import SplatCodec
    from tests.test_codecs import a_cloud

    scene = make_tiny_scene(tmp_path / "scene", n_images=24, n_points=512)
    view = load_val_views(scene, data_factor=1, test_every=8)[0]

    codec = SplatCodec(order="none")
    codec.encode(a_cloud(256), tmp_path / "splat")
    decoded = codec.decode(tmp_path / "splat")
    assert decoded.sh_degree == 0

    image = render_view(decoded, view)
    assert image.shape == view.image.shape
