# Gaussian Splat Compression

A reproducible research pipeline for measuring and improving the compression of
3D Gaussian Splatting scenes. The project trains scenes with gsplat, exports
portable artifacts, and evaluates compression methods by rate, image quality,
and rendering performance.

The central question is practical: how much of a trained Gaussian scene can be
removed or quantized before the visual loss becomes unacceptable?

## Project status

This project is in active development and has no stable release yet. Milestone
1 is building the reproducible `splatpipe` training and export pipeline. Seven
of its eleven implementation tasks have passed review on the
[`milestone-1-scripted-pipeline`](https://github.com/VedJoshi/gaussian-splat-compression/tree/milestone-1-scripted-pipeline)
branch.

Available on that branch:

- validated run configuration and deterministic parameter digests
- Windows CUDA and compiler preflight checks
- idempotent patches for the two required upstream Windows fixes
- COLMAP scene validation and deterministic output paths
- validated 3DGS `.ply` input and output
- numpy-only `.splat` encoding with spatial and visual ordering

The run manifest, synthetic scene fixture, training CLI, and final end-to-end
reproduction are still in progress. The repository should therefore be treated
as pre-release research code.

## Measured baseline

The feasibility baseline uses the Tanks and Temples `truck` scene with gsplat's
MCMC strategy for 7,000 steps on an RTX 4050 Laptop GPU with 6 GB VRAM.

| Measurement | Result |
|---|---:|
| Training time | 7.57 minutes |
| Peak GPU memory | 2.89 GiB |
| Held-out PSNR | 24.406 dB |
| Held-out SSIM | 0.8580 |
| Held-out LPIPS | 0.1372 |
| Gaussian count | 1,000,000 |
| Raw 3DGS `.ply` | 236,001,478 bytes |
| Conventional `.splat` | 32,000,000 bytes |
| Desktop browser rendering | 60 fps, vsync limited |

These measurements establish that the hardware can train the target scenes and
that artifact size is large enough to make compression meaningful. Phone
rendering and gsplat's `PngCompression` baseline have not yet been measured.

## Compression target

The raw model stores 59 float32 values per Gaussian, or 236 bytes:

| Field | Float32 values |
|---|---:|
| Position | 3 |
| Scale | 3 |
| Rotation quaternion | 4 |
| Opacity | 1 |
| Spherical harmonic DC term | 3 |
| Higher-order spherical harmonics | 45 |

Higher-order spherical harmonics account for about 76 percent of the raw
payload. The common 32-byte `.splat` format obtains a 7.38x size reduction by
discarding all 45 of those values, which also removes view-dependent
appearance. A useful compression method must therefore be compared at matched
visual quality, not by file size alone.

Planned experiments combine spherical harmonic quantization, contribution-based
pruning, compact attribute coding, and entropy coding. Results will be reported
as rate-distortion curves using PSNR, SSIM, LPIPS, artifact size, load time, and
rendering performance.

## Roadmap

| Milestone | Outcome | State |
|---|---|---|
| 1. Scripted pipeline | One command reproduces the training and export baseline | In progress |
| 2. Baseline benchmark | Raw, `.splat`, and `PngCompression` rate-distortion points | Planned |
| 3. SH quantization | First measured compression improvement | Planned |
| 4. Contribution pruning | Quality-aware Gaussian reduction | Planned |
| 5. Container format | Entropy-coded artifacts with a documented schema | Planned |
| 6. Viewer integration | Browser renderer for the project format | Planned |
| 7. Three-scene site | Public comparison across real scenes | Planned |
| 8. Evaluation writeup | Curves, ablations, and failure analysis | Planned |

## Development

The currently validated environment is Windows 11, Python 3.11, PyTorch
2.7.1 with CUDA 12.8, gsplat 1.5.3, and MSVC 14.29. The CUDA extension build is
sensitive to version changes, so the pinned environment should be reproduced
before dependencies are upgraded.

From a provisioned checkout of the milestone branch:

```bat
scripts\env.bat
.venv\Scripts\python.exe scripts\setup_env.py --check
.venv\Scripts\python.exe -m pytest -q
```

The fast suite currently contains 76 passing tests and deselects one GPU test.
To run the complete suite:

```bat
cmd /c "call scripts\env.bat >nul 2>&1 && .venv\Scripts\python.exe -m pytest -q -o addopts="
```

The complete suite currently contains 77 passing tests. `setup_env.py` also
checks the pinned gsplat checkout and reports whether both Windows patches are
applied.

## Documentation

- [Pipeline design](docs/superpowers/specs/2026-08-25-splat-compression-pipeline-design.md)
- [Milestone 1 implementation plan](docs/superpowers/plans/2026-08-26-milestone-1-scripted-pipeline.md)
- [Current implementation handoff](https://github.com/VedJoshi/gaussian-splat-compression/blob/milestone-1-scripted-pipeline/HANDOFF.md)
- [`SPIKE_LOG.txt`](SPIKE_LOG.txt), the full feasibility experiment record

Large datasets, trained artifacts, third-party checkouts, virtual environments,
and build logs are intentionally excluded from Git.
