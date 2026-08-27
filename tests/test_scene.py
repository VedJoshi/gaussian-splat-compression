"""Scene directory validation: enough images, and the folders gsplat needs."""

import pytest

from splatpipe.errors import SceneError
from splatpipe.scene import SceneLayout

COLMAP_FILES = ("cameras.bin", "images.bin", "points3D.bin")


def build_scene(root, n_images=12, factor=1, plain_images=None):
    """Build a scene directory. `plain_images`, when given, sets a separate
    image count for the plain `images/` folder, distinct from `n_images` in
    the downscaled `images_<factor>/` folder, so the two counts can diverge.
    """
    images = root / ("images" if factor == 1 else f"images_{factor}")
    images.mkdir(parents=True)
    for i in range(n_images):
        (images / f"{i:04d}.png").write_bytes(b"not really a png")
    if factor != 1:
        plain = root / "images"
        plain.mkdir()
        count = n_images if plain_images is None else plain_images
        for i in range(count):
            (plain / f"{i:04d}.png").write_bytes(b"not really a png")
    sparse = root / "sparse" / "0"
    sparse.mkdir(parents=True)
    for name in COLMAP_FILES:
        (sparse / name).write_bytes(b"\x00" * 8)
    return root


def test_discovers_a_well_formed_scene(tmp_path):
    layout = SceneLayout.discover(build_scene(tmp_path))
    assert layout.image_count == 12
    assert layout.images_dir == tmp_path / "images"
    assert layout.sparse_dir == tmp_path / "sparse" / "0"


def test_reports_every_missing_piece_at_once(tmp_path):
    with pytest.raises(SceneError) as excinfo:
        SceneLayout.discover(tmp_path)
    message = str(excinfo.value)
    assert "images" in message
    for name in COLMAP_FILES:
        assert name in message


def test_rejects_a_scene_with_too_few_images(tmp_path):
    with pytest.raises(SceneError, match="at least"):
        SceneLayout.discover(build_scene(tmp_path, n_images=3))


def test_factor_two_requires_the_downscaled_folder(tmp_path):
    build_scene(tmp_path, factor=1)
    with pytest.raises(SceneError, match="images_2"):
        SceneLayout.discover(tmp_path, data_factor=2)


def test_factor_two_uses_the_downscaled_folder(tmp_path):
    layout = SceneLayout.discover(build_scene(tmp_path, factor=2), data_factor=2)
    assert layout.images_dir == tmp_path / "images_2"


def test_factor_two_rejects_a_thin_plain_folder(tmp_path):
    """COLMAP records its image filenames against `images/`, and gsplat's
    colmap loader zips sorted(images/) with sorted(images_<factor>/) to map
    between the two (_gsplat_repo/examples/datasets/colmap.py, around line
    197, colmap_to_image). If `images/` is thin while `images_2/` is well
    populated, that zip truncates silently and a later lookup by a COLMAP
    image name raises a bare KeyError deep inside gsplat. The minimum image
    count applies to both folders so this is caught here instead, naming
    `images/` specifically since that is the folder actually short.
    """
    build_scene(tmp_path, factor=2, plain_images=3)
    with pytest.raises(SceneError) as excinfo:
        SceneLayout.discover(tmp_path, data_factor=2)
    message = str(excinfo.value)
    assert str(tmp_path / "images") in message
    assert "images_2" not in message
