import pytest

from tests.fixtures.tiny_scene import make_tiny_scene


def test_scene_has_the_files_gsplat_needs(tmp_path):
    root = make_tiny_scene(tmp_path / "tiny", n_images=24, n_points=512)
    assert len(list((root / "images").glob("*.png"))) == 24
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        assert (root / "sparse" / "0" / name).is_file()


def test_scene_passes_our_own_validation(tmp_path):
    from splatpipe.scene import SceneLayout

    layout = SceneLayout.discover(make_tiny_scene(tmp_path / "tiny", n_images=24))
    assert layout.image_count == 24


def test_pycolmap_reads_back_what_we_wrote(tmp_path):
    """The real check: the library that consumes these files agrees with us."""
    pytest.importorskip("pycolmap")
    from pycolmap import SceneManager

    root = make_tiny_scene(tmp_path / "tiny", n_images=24, n_points=512)
    manager = SceneManager(str(root / "sparse" / "0") + "/")
    manager.load_cameras()
    manager.load_images()
    manager.load_points3D()

    assert len(manager.cameras) == 1
    assert len(manager.images) == 24
    assert manager.points3D.shape == (512, 3)

    camera = next(iter(manager.cameras.values()))
    assert (camera.width, camera.height) == (96, 72)
