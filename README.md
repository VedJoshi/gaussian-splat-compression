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

**Milestone 5 (done):** A single-file `.splatc` container with typed, checksummed, independently addressable blocks, a reference JavaScript decoder, and a staged measurement of block codec, ordering and layout.

**Milestone 6–8 (planned):** Viewer, multi-scene evaluation, writeup.

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

## Milestone 5 result

The container replaces the directory of PNGs that `PngCompression` emits with one `.splatc` file: a 24-byte prefix, a JSON descriptor, then 8-byte-aligned blocks that each declare their own codec and carry their own CRC-32. A dependency-free JavaScript decoder reads it under Node and in Chromium, and a canvas round-trip test confirms PNG blocks survive the browser image pipeline byte-exactly, so `png` is usable as a block codec.

To separate the container from the SH scheme, it reuses the codebook `shvq4096` already fitted rather than clustering its own. The baseline stores its labels in PLAS-sorted order and does not record the sort, so labels are reassigned to the nearest centroid under the same manhattan distance; the codebook, its 6-bit quantization and its per-component bounds are taken verbatim from the baseline artifact.

All three points below come from one run over the same 1,000,000-Gaussian PLY and the same 32 held-out views.

| Codec | Bytes | Ratio | PSNR | SSIM | LPIPS | Encode | Decode |
|---|---:|---:|---:|---:|---:|---:|---:|
| PLY | 236,001,478 | 1.00x | 24.400 | 0.8581 | 0.1375 | 1.8 s | 1.6 s |
| SH VQ, 4,096 entries | 14,526,917 | 16.25x | 24.256 | 0.8539 | 0.1425 | 119.8 s | 1.2 s |
| `.splatc` container | 16,594,830 | 14.22x | 24.246 | 0.8538 | 0.1427 | 30.8 s | 0.7 s |

**The container does not beat the baseline on size.** It is 14.22x smaller than PLY, but 14.24% larger than the `shvq4096` directory it replaces, for a PSNR difference of 0.010 dB. Both paths reuse the same stored codebook, but reassignment against quantized centroids can change labels, so exact reconstructed SH equality is not guaranteed. Encode is 3.9x faster because the codebook is reused instead of refitted, and decode is 1.8x faster.

The 16,594,830-byte figure is the benchmark's container, which applies `deflate` uniformly. Selecting the measured best codec per block gives 16,427,638 bytes, and `plas` ordering with per-block selection gives 15,368,018 bytes, 5.79% above the baseline. Changing block codecs at fixed ordering is lossless. The quality row measures Morton ordering; the PLAS result is a size-only observation.

### Where the bytes go

Block sizes under the selected codec, against the corresponding file in the baseline directory:

| Block | Container | Codec | Baseline | Difference |
|---|---:|---|---:|---:|
| means | 5,423,375 | deflate | 4,671,912 | +16.08% |
| scales | 2,445,791 | deflate | 1,756,541 | +39.24% |
| quats | 3,488,279 | deflate | 3,686,653 | -5.38% |
| opacities | 796,293 | deflate | 602,131 | +32.25% |
| sh0 | 2,506,679 | png | 2,056,608 | +21.88% |
| SH codebook + labels | 1,763,189 | png, deflate | 1,749,469 | +0.78% |

Most of the gap is in the five per-Gaussian attribute blocks. A likely contributor is PNG layout: the baseline writes attributes over a PLAS-sorted 1,000x1,000 grid, whereas the container lays bytes into a grid sized from byte count alone. Rows can cut across Gaussian boundaries. Ordering and layout both differ here; a controlled grid experiment is needed to quantify their individual effects. Similar SH block sizes are consistent with codebook reuse, but do not establish identical labels.

### Resolved design questions

- **Struct-of-arrays uses 9.03% fewer compressed payload bytes:** 16,423,606 versus 18,054,209, with the shared SH codebook included and headers excluded on both sides. This corrects the initial comparison, which counted headers and the codebook inconsistently. Per-attribute blocks also keep each field independently addressable.
- **PLAS wins on the unpruned cloud**: 16,640,542 bytes unsorted, 16,427,638 Morton, 15,392,546 PLAS. Morton remains the container's default because PLAS crops to a square grid, which is what makes the baseline decode 799,236 of 800,000 pruned Gaussians. Morton stores the exact count it is given.
- **The unpruned experiment selected the same block codecs under both orderings.** Selecting per-block codecs against a PLAS-packed scene rather than a Morton-packed one produced the same assignment. This observation is specific to that experiment.
- **Splitting 16-bit means into high and low byte planes** stores 4,779,319 bytes against 5,423,375 interleaved, 11.88% smaller. It is measured but not adopted; adopting it is a format change and belongs with the PNG grid fix.
- **Means must be quantized in log space.** Truck's means span [-5720, 8781] because of a handful of stray Gaussians while 99.9% of the scene lies within +/-24. Quantizing that raw range at 16 bits gave 0.043 units of position RMS, several splat widths, and rendered at 13.580 dB against PLY's 24.400. Applying `sign(x) * log1p(|x|)` first, which is what `PngCompression` already does, cut the RMS to 0.00059 and recovered the 24.246 dB above.

### Limitations

The container is not yet smaller than the PNG-family baseline. PNG grid layout remains an optimization candidate, with its independent effect unmeasured. Seeded CUDA K-means is not bit-deterministic, and unpruned `shvq4096` measured 14,526,917 bytes here against 14,526,288 in the Milestone 4 run.

### Composition with 80% contribution pruning

The follow-up uses the saved Milestone 4 ranking and the selected pruned point's own 4,096-entry codebook. The retained percentage, bit depths, candidate codecs, and 32 held-out views are unchanged. The baseline row is the preserved Milestone 4 measurement; the container row was measured on September 12, 2026.

| Codec | Decoded Gaussians | Bytes | Ratio vs raw PLY | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| 80% pruning + SH VQ | 799,236 | 11,920,255 | 19.80x | 24.2632 | 0.85373 | 0.14315 |
| 80% pruning + `.splatc` | 800,000 | 12,278,198 | 19.22x | 24.2575 | 0.85362 | 0.14327 |

The container preserves all 800,000 survivors and is **3.00% larger** than the baseline, with changes of -0.0057 dB PSNR, -0.00011 SSIM, and +0.00012 LPIPS. Encoding took 23.93 seconds with the codebook reused; decoding took 0.48 seconds. These timings do not include the earlier codebook fitting.

PNG wins for DC color and the SH codebook; DEFLATE wins for means, scales, quaternions, opacities, and labels. On this pruned cloud, Morton wins the ordering sweep:

| Ordering | Stored Gaussians | Container bytes | Bytes per Gaussian |
|---|---:|---:|---:|
| None | 800,000 | 13,544,418 | 16.931 |
| Morton | 800,000 | 12,278,198 | 15.348 |
| PLAS | 799,236 | 12,583,100 | 15.744 |

With consistent payload accounting, struct-of-arrays uses 12,274,156 bytes against 13,921,208 for array-of-structs, an 11.83% saving. Splitting the means into byte planes would reduce that block from 3,785,100 to 3,066,739 bytes (18.98%); it remains an unadopted experiment.

Artifacts are under `out/truck/container-prune80/`. The measured container matches the separately packed file block-for-block, and the JavaScript decoder matches Python on all seven blocks. SHA-256 checks confirmed that all 76 existing M2/M3, M4, and unpruned M5 artifact files remained unchanged.

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

Measure the container's block codecs, ordering and layout:
```bat
.venv\Scripts\splatpipe.exe pack out\truck --orders none,morton,plas --candidates raw,deflate,png --sh-codebook out\truck\pruning\bench\shvq4096
```

Outputs: `out/truck/container/{scene.splatc, meta.json}`.

Score the container against held-out views without overwriting the Milestone 2/3 curve:
```bat
.venv\Scripts\splatpipe.exe bench out\truck --scene data\tandt\truck --codecs ply,shvq4096,container --sh-codebook out\truck\pruning\bench\shvq4096 --output-namespace container
```

Outputs: `out/truck/container/{curve.json, curve.png}`.

Read a container from JavaScript:
```bat
node scripts\decode_container.mjs out\truck\container\scene.splatc out\truck\container\decoded.json
```

Add `--skip-train` to re-export without retraining.

### Tests

Fast tier (CPU, no MSVC):
```bat
.venv\Scripts\python.exe -m pytest -q
```

Full tier (GPU training, rendering, and Chromium):
```bat
cmd /c "call scripts\env.bat >nul 2>&1 && .venv\Scripts\python.exe -m pytest -q -o addopts="
```

Expected counts after Milestone 5: 278 fast tests with 15 slow tests deselected; 293 tests in the complete tier. Node parity tests require Node, and the browser tests require Playwright and Chromium.

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
| 5. Container format | Done |
| 6. Viewer integration | Planned |
| 7. Multi-scene evaluation | Planned |
| 8. Evaluation writeup | Planned |

## Excluded from Git

Large datasets, trained artifacts, third-party checkouts, virtual environments,
and build logs are intentionally excluded from Git.
