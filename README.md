# Gaussian Splat Compression

A reproducible research pipeline for measuring and improving the compression of
3D Gaussian Splatting scenes. The project trains scenes with gsplat, exports
portable artifacts, and evaluates compression methods by rate, image quality,
and rendering performance.

The central question is practical: how much of a trained Gaussian scene can be
removed or quantized before the visual loss becomes unacceptable?

## Project status

Milestone 1 is complete on the
[`milestone-1-scripted-pipeline`](https://github.com/VedJoshi/gaussian-splat-compression/tree/milestone-1-scripted-pipeline)
branch. One command now trains a COLMAP scene and packages it, and it
reproduces the feasibility baseline below.

All eleven implementation tasks have passed review. The branch provides:

- validated run configuration and deterministic parameter digests
- Windows CUDA and compiler preflight checks
- idempotent patches for the two required upstream Windows fixes
- COLMAP scene validation and deterministic output paths
- validated 3DGS `.ply` reading and writing
- numpy-only `.splat` encoding with spatial and visual ordering
- a run manifest recording the resolved config and its digest, library
  versions, the Git commit, timings, held-out metrics, and artifact checksums

Compression itself begins at milestone 3. Until then the repository should be
treated as pre-release research code.

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

The scripted pipeline reproduces that baseline. The same scene run through
`splatpipe` measures PSNR 24.395 dB, SSIM 0.8580 and LPIPS 0.1376 against the
spike's 24.406, 0.8580 and 0.1372, and writes a `.ply` of exactly 236,001,478
bytes. PSNR differs by 0.011 dB, SSIM by 1e-5 and LPIPS by 3e-4, because CUDA
reductions are not bit-reproducible even at gsplat's fixed seed of 42. The
artifact sizes are exact, because they depend only on the Gaussian count and
the field list.

## Provisioning a checkout

The validated environment is Windows 11, Python 3.11, PyTorch 2.7.1 with CUDA
12.8, gsplat 1.5.3, and MSVC 14.29. The CUDA extension build is sensitive to
all of them, so reproduce these versions before upgrading anything.

```bat
%LOCALAPPDATA%\Programs\Python\Python311\python.exe -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip wheel setuptools
.venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
scripts\env.bat
.venv\Scripts\python.exe -m pip install gsplat==1.5.3 --no-build-isolation
git clone https://github.com/nerfstudio-project/gsplat.git _gsplat_repo
.venv\Scripts\python.exe -m pip install -r _gsplat_repo\examples\requirements.txt
.venv\Scripts\python.exe -m pip install -e .[dev]
.venv\Scripts\python.exe scripts\setup_env.py
```

The order matters, and each constraint in it cost time to find:

- Python 3.11 specifically. `python` and `py` resolve to 3.13 on the machine
  this was built on, which the package rejects.
- torch 2.7.1, not the latest. 2.11 fails to build the CUDA extension against
  the Windows SDK. The `cu128` index matches the installed CUDA 12.8 toolkit.
- `env.bat` before anything that compiles. `--no-build-isolation` makes gsplat
  build against the installed torch, which is why `wheel` and `setuptools`
  have to be present first.
- `setup_env.py` last. It pins the gsplat checkout and patches two files in
  `site-packages`, so both gsplat and pycolmap must already be installed. It
  is idempotent, and `--check` reports status without changing anything.

This sequence is reconstructed from [`SPIKE_LOG.txt`](SPIKE_LOG.txt), which
records the environment being built the first time, including the failures.
It has not been re-run on a clean machine. `requirements.lock.txt` records the
versions it resolved to.

## Running the pipeline

```bat
scripts\env.bat
.venv\Scripts\python.exe scripts\get_data.py
.venv\Scripts\splatpipe.exe run data\tandt\truck --config configs\truck.toml --out out
```

Installing the package puts `splatpipe` in `.venv\Scripts`. `.venv\Scripts\python.exe -m splatpipe.cli`
is the same entry point.

`get_data.py` downloads Tanks and Temples and builds the downscaled image
folders, once; on later runs it reports what it skipped. `env.bat` loads the
MSVC environment and is needed in every new shell, because the pipeline checks
for a working compiler before spending GPU minutes. The run itself takes
roughly 8 minutes on an RTX 4050 and writes:

    out/truck/artifacts/scene.ply    the trained 3DGS model
    out/truck/artifacts/scene.splat  the 32-byte-per-Gaussian viewer format
    out/truck/config.toml            the config as given, copied verbatim
    out/truck/manifest.json          what produced them
    out/truck/train/                 gsplat's own output, left in its layout
    out/truck/logs/train.log         the trainer's full output

Add `--skip-train` to re-export from an existing `train/` directory without
training again, which is the loop to use when changing export settings.

The supported path is the command above. Underneath it, the pipeline runs
gsplat's trainer directly, which is worth knowing when debugging a training
failure. Point it at a scratch directory rather than at `out/truck/train`,
whose contents `out/truck/manifest.json` describes:

```bat
cd _gsplat_repo\examples
..\..\.venv\Scripts\python.exe simple_trainer.py mcmc --data-dir ..\..\data\tandt\truck ^
  --result-dir ..\..\out\scratch --data-factor 1 --max-steps 7000 --test-every 8 ^
  --eval-steps 7000 --save-ply --disable-viewer --strategy.cap-max 1000000
```

Tests run in two tiers. The fast tier needs no GPU and no MSVC shell, which is
what makes the project resumable after a gap:

```bat
.venv\Scripts\python.exe -m pytest -q
cmd /c "call scripts\env.bat >nul 2>&1 && .venv\Scripts\python.exe -m pytest -q -o addopts="
```

The fast tier is 95 tests and deselects the 3 GPU-marked ones. The complete
tier is 98 and trains the real gsplat trainer twice on a synthetic 24-image
scene, which has measured between 1.5 and 2.5 minutes on this machine.

## Repository layout

| Path | Purpose |
|---|---|
| `src/splatpipe/` | The pipeline package: config, scene validation, paths, training, export, manifest |
| `configs/truck.toml` | The baseline run above, expressed as configuration |
| `scripts/env.bat` | Loads the MSVC 14.29 environment |
| `scripts/setup_env.py` | Pins the gsplat checkout and owns both upstream patches |
| `scripts/get_data.py` | Downloads Tanks and Temples and builds `images_2` and `images_4` |
| `scripts/vram_sampler.py` | Samples whole-board GPU memory through `nvidia-smi` |
| `tests/` | Two tiers: a fast CPU suite and GPU-marked end-to-end runs |

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
| 1. Scripted pipeline | One command reproduces the training and export baseline | Done |
| 2. Baseline benchmark | Raw, `.splat`, and `PngCompression` rate-distortion points | Planned |
| 3. SH quantization | First measured compression improvement | Planned |
| 4. Contribution pruning | Quality-aware Gaussian reduction | Planned |
| 5. Container format | Entropy-coded artifacts with a documented schema | Planned |
| 6. Viewer integration | Browser renderer for the project format | Planned |
| 7. Three-scene site | Public comparison across real scenes | Planned |
| 8. Evaluation writeup | Curves, ablations, and failure analysis | Planned |

## Documentation

- [Pipeline design](docs/superpowers/specs/2026-08-25-splat-compression-pipeline-design.md)
- [Milestone 1 implementation plan](docs/superpowers/plans/2026-08-26-milestone-1-scripted-pipeline.md)
- [Current implementation handoff](https://github.com/VedJoshi/gaussian-splat-compression/blob/milestone-1-scripted-pipeline/HANDOFF.md)
- [`SPIKE_LOG.txt`](SPIKE_LOG.txt), the full feasibility experiment record

Large datasets, trained artifacts, third-party checkouts, virtual environments,
and build logs are intentionally excluded from Git.
