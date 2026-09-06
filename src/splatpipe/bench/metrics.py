"""PSNR, SSIM and LPIPS, configured exactly as gsplat configures them.

Matching the reference implementation is what makes this project's numbers
comparable to gsplat's published ones. Any deviation, a different LPIPS backbone
most of all, still produces a plot but makes the comparison meaningless. The
settings below are simple_trainer.py:457-467.

LPIPS downloads AlexNet weights on first use, which makes the first run
network-dependent in a project that is otherwise reproducible offline. They are
cached afterwards.
"""

from __future__ import annotations

import numpy as np

DATA_RANGE = 1.0
LPIPS_NET = "alex"


def _to_nchw(image: np.ndarray, device: str):
    import torch

    tensor = torch.from_numpy(np.ascontiguousarray(image)).to(device)
    return tensor.unsqueeze(0).permute(0, 3, 1, 2)


def score_views(
    rendered: list[np.ndarray],
    targets: list[np.ndarray],
    device: str = "cuda",
) -> dict[str, float]:
    """Mean PSNR, SSIM and LPIPS over a set of views.

    Both lists hold (H, W, 3) float32 images in [0, 1]. Averaging per view then
    over views is what gsplat does, and it is not the same as averaging over all
    pixels at once when views differ in size.
    """
    if len(rendered) != len(targets):
        raise ValueError(
            f"rendered and targets must hold the same number of views, "
            f"got {len(rendered)} and {len(targets)}"
        )
    if not rendered:
        raise ValueError("no views to score")

    import torch
    from torchmetrics.image import (
        PeakSignalNoiseRatio,
        StructuralSimilarityIndexMeasure,
    )
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    psnr = PeakSignalNoiseRatio(data_range=DATA_RANGE).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=DATA_RANGE).to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(
        net_type=LPIPS_NET, normalize=True
    ).to(device)

    totals = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0}
    with torch.no_grad():
        for render, target in zip(rendered, targets):
            r = _to_nchw(render, device)
            t = _to_nchw(target, device)
            totals["psnr"] += float(psnr(r, t))
            totals["ssim"] += float(ssim(r, t))
            totals["lpips"] += float(lpips(r, t))

    return {key: value / len(rendered) for key, value in totals.items()}
