"""The held-out views bench scores against.

No GPU needed: loading COLMAP poses and images is CPU work. This runs in the
fast tier so a camera regression is caught without a compiler shell.
"""

from __future__ import annotations

import numpy as np

from splatpipe.bench.cameras import (
    NORMALIZE_WORLD_SPACE,
    CalibrationView,
    ValView,
    _import_colmap_dataset,
    load_train_views,
    load_val_views,
)
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


def test_training_views_exclude_the_held_out_cameras(tmp_path):
    scene = make_tiny_scene(tmp_path / "scene", n_images=24, n_points=512)
    views = load_train_views(scene, data_factor=1, test_every=8)
    assert len(views) == 21
    assert isinstance(views[0], CalibrationView)
    assert (views[0].height, views[0].width) == (72, 96)


def test_training_camera_metadata_matches_the_trainers_dataset(tmp_path):
    scene = make_tiny_scene(tmp_path / "scene", n_images=24, n_points=512)
    view = load_train_views(scene, data_factor=1, test_every=8)[0]
    Parser, Dataset = _import_colmap_dataset()
    parser = Parser(
        data_dir=str(scene.resolve()),
        factor=1,
        normalize=NORMALIZE_WORLD_SPACE,
        test_every=8,
    )
    reference = Dataset(parser, split="train")[0]
    np.testing.assert_array_equal(view.camtoworld, reference["camtoworld"].numpy())
    np.testing.assert_array_equal(view.K, reference["K"].numpy())
    assert (view.height, view.width) == tuple(reference["image"].shape[:2])
