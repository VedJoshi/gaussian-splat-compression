import numpy as np
import pytest

from tests.fixtures.colmap_bin import look_at_quaternion, write_images_bin
from tests.fixtures.tiny_scene import _quaternion_to_matrix, make_tiny_scene


def test_write_images_bin_refuses_mismatched_observation_lists(tmp_path):
    """count2D is written from len(xys) and the body then zips the two lists.

    A mismatched caller would otherwise produce a structurally valid COLMAP
    file whose header claims more observations than follow it. That is silent
    corruption in the fixture the end-to-end test depends on, and it surfaces
    far from its cause, so the fixture refuses instead.
    """
    record = (
        1,
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.zeros(3),
        1,
        "0000.png",
        np.zeros((3, 2)),
        np.array([1, 2]),
    )
    with pytest.raises(ValueError, match="3 xys but 2 point3D_ids"):
        write_images_bin(tmp_path / "images.bin", [record])


def test_look_at_quaternion_is_finite_and_exact_across_the_default_ring():
    """Guards the branch-by-largest-component extraction.

    The obvious w = sqrt(1 + trace) / 2 form divides by a w that goes to zero
    at a 180 degree rotation, and this ring reaches exactly that: at
    n_images=24 the camera at index 6 sits on the +y axis and drives the trace
    to -1.0 bit-exact, not merely close. Reverting to the naive form makes this
    test produce a non-finite quaternion, while every other test in this file
    still passes, which is why the guard has to be here.
    """
    n_images = 24
    for index in range(n_images):
        angle = 2 * np.pi * index / n_images
        position = np.array([4.0 * np.cos(angle), 4.0 * np.sin(angle), 1.5])
        qvec, tvec = look_at_quaternion(position, np.zeros(3))

        assert np.isfinite(qvec).all(), f"camera {index} produced {qvec}"
        assert np.isclose(np.linalg.norm(qvec), 1.0)

        rotation = _quaternion_to_matrix(qvec)
        assert np.isclose(np.linalg.det(rotation), 1.0)
        # COLMAP's convention: the transform carries the camera centre to the
        # camera-frame origin, which is what makes tvec = -R @ position right.
        assert np.allclose(rotation @ position + tvec, 0.0, atol=1e-9)


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
