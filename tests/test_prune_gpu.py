from __future__ import annotations

import numpy as np
import pytest

from splatpipe.bench.cameras import CalibrationView, load_val_views
from splatpipe.bench.render import CAMERA_MODEL, FAR_PLANE, NEAR_PLANE, RASTERIZE_MODE
from splatpipe.compress.prune import collect_contributions
from tests.fixtures.tiny_scene import make_tiny_scene
from tests.test_codecs import a_cloud

pytestmark = pytest.mark.gpu


def test_contribution_gradient_matches_explicit_per_gaussian_features(tmp_path):
    import torch
    from gsplat import rasterization

    scene = make_tiny_scene(tmp_path / "scene", n_images=8, n_points=64)
    val_view = load_val_views(scene, data_factor=1, test_every=8)[0]
    view = CalibrationView(
        camtoworld=val_view.camtoworld,
        K=val_view.K,
        height=val_view.image.shape[0],
        width=val_view.image.shape[1],
    )
    cloud = a_cloud(8)
    actual = collect_contributions(cloud, [view])

    means = torch.from_numpy(cloud.means).cuda()
    quats = torch.from_numpy(cloud.quats).cuda()
    scales = torch.exp(torch.from_numpy(cloud.scales).cuda())
    opacities = torch.sigmoid(torch.from_numpy(cloud.opacities).cuda())
    features = torch.eye(len(cloud), device="cuda")
    camtoworld = torch.from_numpy(view.camtoworld).cuda().unsqueeze(0)
    intrinsics = torch.from_numpy(view.K).cuda().unsqueeze(0)

    with torch.no_grad():
        rendered, _, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=features,
            viewmats=torch.linalg.inv(camtoworld),
            Ks=intrinsics,
            width=view.width,
            height=view.height,
            sh_degree=None,
            near_plane=NEAR_PLANE,
            far_plane=FAR_PLANE,
            camera_model=CAMERA_MODEL,
            rasterize_mode=RASTERIZE_MODE,
            packed=False,
        )
    expected = rendered[0].sum(dim=(0, 1)).cpu().numpy()
    assert expected.max() > 0
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-4)
