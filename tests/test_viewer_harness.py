from __future__ import annotations

import urllib.request

import numpy as np

from scripts.viewer_harness import REPO, psnr, serve, viewer_camera


def test_modules_are_served_as_javascript():
    # Module scripts and module workers refuse any other MIME type.
    with serve(REPO) as base:
        with urllib.request.urlopen(f"{base}/viewer/decode_container.mjs") as response:
            assert response.headers["Content-Type"].startswith("text/javascript")


def test_viewer_camera_projects_like_gsplat():
    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    camtoworld = np.eye(4)
    camtoworld[:3, :3] = q * np.sign(np.linalg.det(q))
    camtoworld[:3, 3] = [0.3, -0.2, -4.0]
    K = np.array([[900.0, 0, 640.5], [0, 910.0, 400.25], [0, 0, 1]])

    camera = viewer_camera(camtoworld, K)
    view = np.asarray(camera["view"]).reshape(4, 4).T  # column-major to row-major
    point = np.array([0.1, 0.2, 0.3, 1.0])
    in_camera = view @ point
    pixel = [camera["fx"] * in_camera[0] / in_camera[2] + camera["cx"],
             camera["fy"] * in_camera[1] / in_camera[2] + camera["cy"]]

    expected = K @ (np.linalg.inv(camtoworld) @ point)[:3]
    np.testing.assert_allclose(pixel, expected[:2] / expected[2], rtol=1e-12)


def test_psnr_of_an_image_against_itself_is_infinite_and_one_level_off_is_48db():
    image = np.full((4, 4, 3), 128, dtype=np.uint8)
    assert psnr(image, image / 255.0) == float("inf")
    assert abs(psnr(image, (image + 1) / 255.0) - 48.1308) < 1e-3
