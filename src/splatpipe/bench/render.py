"""Render a GaussianCloud through gsplat's public rasteriser.

Mirrors simple_trainer.rasterize_splats field for field. The activation
convention is where this matters: GaussianCloud stores log scales and logit
opacities, exactly as the .ply does, so exp and sigmoid are applied here rather
than on read. The quaternions are passed unnormalised because rasterization
normalises internally.

The defaults below are simple_trainer.py's own. They are written out rather than
left implicit because a silent disagreement with the trainer shows up as a
slightly wrong PSNR, not as an error.
"""

from __future__ import annotations

import numpy as np

from splatpipe.gaussians import GaussianCloud

NEAR_PLANE = 0.01  # simple_trainer.py:110
FAR_PLANE = 1e10  # simple_trainer.py:112
CAMERA_MODEL = "pinhole"  # simple_trainer.py:69
RASTERIZE_MODE = "classic"  # antialiased defaults to False, simple_trainer.py:125


def render_view(cloud: GaussianCloud, view, device: str = "cuda") -> np.ndarray:
    """Render one held-out view. Returns (H, W, 3) float32 clamped to [0, 1]."""
    import torch
    from gsplat import rasterization

    height, width = view.image.shape[:2]
    camtoworld = torch.from_numpy(view.camtoworld).to(device).unsqueeze(0)
    Ks = torch.from_numpy(view.K).to(device).unsqueeze(0)

    means = torch.from_numpy(cloud.means).to(device)
    quats = torch.from_numpy(cloud.quats).to(device)
    scales = torch.exp(torch.from_numpy(cloud.scales).to(device))
    opacities = torch.sigmoid(torch.from_numpy(cloud.opacities).to(device))
    sh0 = torch.from_numpy(cloud.sh0).to(device).unsqueeze(1)
    shN = torch.from_numpy(cloud.shN).to(device)
    colors = torch.cat([sh0, shN], dim=1)

    with torch.no_grad():
        rendered, _, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworld),
            Ks=Ks,
            width=width,
            height=height,
            # Derived from the cloud, never hardcoded: a decoded .splat carries
            # only the DC term and renders at degree 0.
            sh_degree=cloud.sh_degree,
            near_plane=NEAR_PLANE,
            far_plane=FAR_PLANE,
            camera_model=CAMERA_MODEL,
            rasterize_mode=RASTERIZE_MODE,
            packed=False,
        )
    image = torch.clamp(rendered, 0.0, 1.0).squeeze(0)
    return image.cpu().numpy().astype(np.float32)
