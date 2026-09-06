"""The held-out views bench scores against.

No GPU needed: loading COLMAP poses and images is CPU work. This runs in the
fast tier so a camera regression is caught without a compiler shell.
"""

from __future__ import annotations

import numpy as np

from splatpipe.bench.cameras import ValView, load_val_views
from tests.fixtures.tiny_scene import make_tiny_scene


def test_loads_every_eighth_view(tmp_path):
    """gsplat holds out indices where index % test_every == 0.

    24 images at test_every 8 gives indices 0, 8 and 16.
    """
    scene = make_tiny_scene(tmp_path / "scene", n_images=24, n_points=512)
    views = load_val_views(scene, data_factor=1, test_every=8)
    assert len(views) == 3


def test_views_carry_poses_intrinsics_and_pixels(tmp_path):
    scene = make_tiny_scene(tmp_path / "scene", n_images=24, n_points=512)
    view = load_val_views(scene, data_factor=1, test_every=8)[0]
    assert isinstance(view, ValView)
    assert view.camtoworld.shape == (4, 4)
    assert view.K.shape == (3, 3)
    assert view.image.shape == (72, 96, 3)
    assert view.image.dtype == np.uint8


def test_test_every_changes_the_held_out_count(tmp_path):
    scene = make_tiny_scene(tmp_path / "scene", n_images=24, n_points=512)
    assert len(load_val_views(scene, data_factor=1, test_every=4)) == 6
