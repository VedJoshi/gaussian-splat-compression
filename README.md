# Gaussian Splat Compression

Reproducible research pipeline for compressing 3D Gaussian Splatting scenes.
Trains scenes with gsplat, exports portable artifacts, measures compression
trade-offs (rate vs. distortion) across codec families.

## Compression target

Raw model: 236 bytes per Gaussian (59 float32 values).

| Field | Bytes | Share |
|---|---:|---:|
| Higher-order spherical harmonics | 180 | 76% |
| Rotation quaternion | 16 | 7% |
| Position | 12 | 5% |
| DC spherical harmonic | 12 | 5% |
| Scale | 12 | 5% |
| Opacity | 4 | 2% |

Compression targets: quantize view-dependent color, prune low-contribution
Gaussians, or reduce Gaussian count.

## Input requirements

COLMAP reconstruction directory with:
- `images/` — at least 8 images
- `sparse/0/cameras.bin`, `images.bin`, `points3D.bin` — camera poses and 3D points

Run COLMAP on your photographs separately; this pipeline reads the output.

## Status

**Milestone 1 (done):** Training pipeline with manifest, config validation, preflight checks, COLMAP scene loading, `.ply` and `.splat` export.

**Milestone 2 (done):** Benchmarking harness measuring rate-distortion on held-out views across PLY, SPLAT, and PngCompression codecs.

**Milestone 3 (done):** Configurable vector quantization for higher-order SH coefficients, measured at 256, 1,024, and 4,096 codebook entries.

**Milestone 4 (done):** Occlusion-aware contribution pruning, an opacity ablation, and a measured retained-count sweep.

**Milestone 5–8 (planned):** Container format, viewer, multi-scene evaluation, writeup.

### Known constraints

Stock `PngCompression` requires at least 65,536 Gaussians because its SH codebook has 65,536 entries. The Milestone 3 codecs lower that floor to their configured codebook size; PLAS sorting still requires at least 256 Gaussians.

All PNG-family codecs require a CUDA GPU for encoding. The `.splat` format is CPU-only and 32 MB (7.38×), but discards view-dependent color.

## Baseline (Tanks and Temples truck, RTX 4050)

| Metric | Value |
|---|---|
| Training time | 7.57 min |
| Peak GPU memory | 2.89 GiB |
| Held-out PSNR | 24.395 dB |
| Held-out SSIM | 0.8580 |
| Held-out LPIPS | 0.1376 |
| Gaussian count | 1,000,000 |
| `.ply` size | 236.0 MB |
| `.splat` size | 32.0 MB |
| PngCompression size | 16.21 MB |
| Rendering | 60 fps (vsync-limited) |

Pipeline reproduces the training metrics within measurement tolerance (PSNR ±0.05 dB, SSIM ±0.002, LPIPS ±0.005).

## Milestone 3 result

The truck benchmark used the same 1,000,000-Gaussian PLY and 32 held-out views for every codec. PNG-family encodes used seed 42. Sizes are decimal MB; ratios use the exact 236,001,478-byte PLY as the anchor.

| Codec | Bytes | Size | Ratio | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| PLY | 236,001,478 | 236.00 MB | 1.00× | 24.400 | 0.8581 | 0.1375 |
| SPLAT | 32,000,000 | 32.00 MB | 7.38× | 23.251 | 0.8362 | 0.1599 |
| PngCompression | 16,212,775 | 16.21 MB | 14.56× | 24.314 | 0.8555 | 0.1402 |
| SH VQ, 256 entries | 13,675,073 | 13.68 MB | 17.26× | 24.107 | 0.8513 | 0.1453 |
| SH VQ, 1,024 entries | 14,183,988 | 14.18 MB | 16.64× | 24.195 | 0.8528 | 0.1437 |
| SH VQ, 4,096 entries | 14,526,458 | 14.53 MB | 16.25× | 24.251 | 0.8539 | 0.1425 |

The 4,096-entry point is the conservative improvement: 10.40% smaller than stock `PngCompression` with a 0.063 dB PSNR reduction. The encoder uses 6-bit, per-component codebook quantization and the narrowest safe label type. It changes only higher-order SH storage; geometry, opacity, DC color, and spatial sorting remain on the baseline path.

## Milestone 4 result

The pruning score sums each Gaussian's alpha-blending weight across all 117,062,946 pixels in the 219 training views, then applies the capped volume factor from LightGaussian with exponent 0.1. The 32 held-out views are used only for evaluation. No recovery or fine-tuning is performed.

The retained fractions and quality limits were fixed before the truck sweep. A raw pruned PLY passes only when its changes from the unpruned PLY stay within -0.10 dB PSNR, -0.002 SSIM, and +0.005 LPIPS. The most aggressive passing point is selected.

| Retained | Raw Gaussians | Raw PSNR | Raw SSIM | Raw LPIPS | SH VQ Gaussians | SH VQ bytes | SH VQ PSNR | SH VQ SSIM | SH VQ LPIPS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100% | 1,000,000 | 24.4002 | 0.8581 | 0.1375 | 1,000,000 | 14,526,288 | 24.2488 | 0.8539 | 0.1425 |
| 95% | 950,000 | 24.4003 | 0.8581 | 0.1375 | 948,676 | 14,051,503 | 24.2695 | 0.8544 | 0.1422 |
| 90% | 900,000 | 24.3999 | 0.8580 | 0.1378 | 898,704 | 13,317,667 | 24.2639 | 0.8543 | 0.1423 |
| **80%** | **800,000** | **24.3915** | **0.8573** | **0.1386** | **799,236** | **11,920,255** | **24.2632** | **0.8537** | **0.1432** |
| 70% | 700,000 | 24.3578 | 0.8557 | 0.1402 | 698,896 | 10,449,576 | 24.2336 | 0.8520 | 0.1448 |

The selected 80% raw point removes 200,000 Gaussians for changes of -0.0087 dB PSNR, -0.00078 SSIM, and +0.00114 LPIPS. At the same retained count, opacity ranking has 0.0098 dB lower PSNR but slightly better SSIM and LPIPS. Contribution ranking therefore does not dominate the simpler baseline on every metric.

Combined with `shvq4096`, the selected point is 11.92 MB and 19.80x smaller than PLY. It is 26.48% smaller than seeded stock `PngCompression` for a -0.0506 dB PSNR change, and 17.94% smaller than the same-run unpruned `shvq4096`. PNG-family square cropping accounts for the decoded count of 799,236 rather than exactly 800,000.

## Setup (Windows)

Prerequisites (fixed paths):
- Visual Studio 2019 Build Tools (C++ workload)
- CUDA Toolkit 12.8

From repository root:

```bat
%LOCALAPPDATA%\Programs\Python\Python311\python.exe -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip wheel setuptools
.venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
scripts\env.bat
.venv\Scripts\python.exe -m pip install -r requirements.lock.txt --no-build-isolation --extra-index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python.exe -m pip install -e .[dev]
.venv\Scripts\python.exe scripts\setup_env.py
```

Tested end-to-end on 2026-08-31. `requirements.lock.txt` pinned; regenerate with `pip freeze --exclude-editable`.

## Usage

Train and export (truck scene, ~8 min on RTX 4050):
```bat
scripts\env.bat
.venv\Scripts\python.exe scripts\get_data.py
.venv\Scripts\splatpipe.exe run data\tandt\truck --config configs\truck.toml --out out
```

Outputs: `out/truck/artifacts/{scene.ply, scene.splat}`, `manifest.json`, config, logs.

Benchmark codecs against held-out views:
```bat
.venv\Scripts\splatpipe.exe bench out/truck --scene data\tandt\truck --codecs ply,splat,png,shvq256,shvq1024,shvq4096
```

Outputs: `out/truck/curve.json`, `curve.png`.

Measure contribution pruning and its opacity ablation:
```bat
.venv\Scripts\splatpipe.exe prune out\truck --scene data\tandt\truck --retain 95,90,80,70
```

Outputs: `out/truck/pruning/{scores.npz, meta.json, curve.json, curve.png}` and per-point codec directories under `out/truck/pruning/bench/`.

Add `--skip-train` to re-export without retraining.

### Tests

Fast tier (CPU, no MSVC):
```bat
.venv\Scripts\python.exe -m pytest -q
```

Full tier (GPU, trains twice):
```bat
cmd /c "call scripts\env.bat >nul 2>&1 && .venv\Scripts\python.exe -m pytest -q -o addopts="
```

Expected counts after Milestone 4: 191 fast tests with 13 GPU tests deselected; 204 tests in the complete tier.

## Repository layout

| Path | Purpose |
|---|---|
| `src/splatpipe/` | Pipeline package |
| `src/splatpipe/bench/` | Benchmarking harness |
| `configs/truck.toml` | Reference configuration |
| `scripts/env.bat` | MSVC environment loader |
| `scripts/setup_env.py` | gsplat checkout and Windows patches |
| `scripts/get_data.py` | Data download and preprocessing |
| `tests/` | CPU tier (fast) and GPU tier (full)|

## Roadmap

| Milestone | State |
|---|---|
| 1. Training pipeline | Done |
| 2. Benchmarking harness | Done |
| 3. SH quantization | Done |
| 4. Contribution pruning | Done |
| 5. Container format | Planned |
| 6. Viewer integration | Planned |
| 7. Multi-scene evaluation | Planned |
| 8. Evaluation writeup | Planned |

## Excluded from Git

Large datasets, trained artifacts, third-party checkouts, virtual environments,
and build logs are intentionally excluded from Git.
