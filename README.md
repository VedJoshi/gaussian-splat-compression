# Gaussian Splat Compression

A pipeline for training, benchmarking, compressing and deploying 3D Gaussian
splat scenes, with the compression implemented from the literature and measured
against baselines on held-out views.

A trained splat scene is roughly 236 bytes per Gaussian, and 76 percent of that
is spherical harmonic coefficients. The work of this repository is reducing that
figure and reporting the rate-distortion cost of doing so.

Target platform is Windows 11 with an RTX 4050 Laptop, 6 GB VRAM.

## Status

Milestone 1 of 8. The Python package is `splatpipe`.

| # | Milestone | Done when | State |
|---|---|---|---|
| 1 | Scripted pipeline | `splatpipe run <scene>` reproduces truck end to end | in progress |
| 2 | `bench/`, baseline measurement | First RD points: raw `.ply`, `.splat`, `PngCompression` | not started |
| 3 | Spherical harmonic quantisation | First measured win on the curve | not started |
| 4 | Contribution-based pruning | Second win, enough evidence to design the format | not started |
| 5 | Container format and entropy coding | Files the viewer can read | not started |
| 6 | Fork the viewer to the format | A scene rendering from this pipeline | not started |
| 7 | Static site with 3 scenes | A public link | not started |
| 8 | Writeup: curve, ablations, failures | Published results | not started |

Implemented so far, on branch `milestone-1-scripted-pipeline`:

| Module | Purpose |
|---|---|
| `splatpipe.errors` | Exception hierarchy, so modules do not import each other for exception types |
| `splatpipe.env` | Build environment preflight: `cl.exe`, `CUDA_HOME`, arch list. Reports every problem at once |
| `splatpipe.config` | `RunConfig` / `TrainConfig` / `ExportConfig`, TOML loading, validation, parameter digest |
| `splatpipe.scene` | COLMAP scene validation before GPU time is spent |
| `splatpipe.paths` | The run output layout, derived from `(out_root, name)` |
| `splatpipe.gaussians` | Validated numpy Gaussian cloud plus channel-correct 3DGS `.ply` I/O |

Remaining in milestone 1: the `.splat` writer, the run manifest, a synthetic
COLMAP fixture, the training stage and CLI, and a reproduction of the truck
baseline through the finished CLI.

Test suite: 49 tests in the fast tier, 50 including the GPU-marked tier.

## Measured baseline

Tanks and Temples `truck`, 251 images at 979x546, gsplat MCMC strategy, 7,000
steps, capped at 1,000,000 Gaussians. Run 2026-08-25 on Windows 11, RTX 4050
Laptop 6 GB.

```
wall clock      7.57 min
peak GPU mem    2960 MiB of 6141 MiB (2.891 GiB)
PSNR 24.406     SSIM 0.8580     LPIPS 0.1372
gaussians       1,000,000 (the configured cap, not a hardware limit)
render          0.0246 s/image, about 41 fps
output .ply     236,001,478 bytes (225 MiB)
```

The 3DGS paper reports about 25.2 PSNR on `truck` at 30,000 steps. This run used
7,000 steps, so 24.406 is consistent with the shorter schedule.

Every figure above is the target that milestone 1 must reproduce through
`splatpipe run`. The `.ply` size must match exactly, since it is a function of
Gaussian count and field list. Metrics are expected to match to about two
decimal places: the seed is fixed upstream at 42, but CUDA reductions are not
bit-reproducible.

### Where the bytes go

236,001,478 / 1,000,000 = 236.0 bytes per Gaussian, as 59 float32 fields:

| Field | Floats |
|---|---|
| means | 3 |
| scales | 3 |
| quaternion | 4 |
| opacity | 1 |
| SH DC | 3 |
| SH rest (`f_rest_0..44`) | 45 |
| total | 59 x 4 bytes = 236 |

45 of 59 fields, 76 percent of the file, are spherical harmonic coefficients.
Compression work starts there.

Converting to `.splat`, the format most web viewers use, gives 30 MiB, a 7.38x
reduction. It achieves that by dropping all 45 `f_rest_*` coefficients and
keeping only the DC term, so it discards every view-dependent appearance term.
The common web format is already lossy in appearance, not only in precision.

`gsplat.compression.PngCompression` already exists in the library, exposed as
`--compression png`. The framing of this project is therefore improvement on a
published baseline, and measuring that baseline is milestone 2. It has not been
measured yet.

### Browser rendering

The 30 MiB `.splat` served locally and loaded in antimatter15's WebGL viewer,
driven through Playwright's cached chromium.

```
load             1071 ms for 30 MiB over localhost
frame rate       60 fps by the viewer's counter, 60.5 by rAF count over 3 s,
                 vsync capped
GL renderer      ANGLE (NVIDIA, RTX 4050 Laptop GPU, Direct3D11), hardware
                 rather than a SwiftShader software fallback
JS heap          39.5 MiB
console          one 404 on favicon.ico, confirmed benign against the server log
```

`_browser_test.png` is the screenshot. The blurred foreground is the viewer's
default camera pose, which is hardcoded for its own demo scene and starts the
camera partly inside the geometry.

Phone rendering has not been tested.

## Requirements

This combination is exact. Substituting versions breaks the CUDA extension
build.

```
python       3.11.0        3.13 is the default on PATH and will not work
torch        2.7.1+cu128   see upstream bug 1
torchvision  0.22.1+cu128
numpy        1.26.4        gsplat 1.5.3 examples require numpy<2
gsplat       1.5.3         plus the MSVC flag patch, pinned at
                           937e29912570c372bed6747a5c9bf85fed877bae
packaging                  undeclared gsplat dependency
wheel                      required for --no-build-isolation builds

CUDA_HOME             C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
                      v12.9 is first on PATH by default and must be overridden,
                      since it does not match the torch cu128 build
host compiler         vcvars64.bat from VS Build Tools 2019 16.11,
                      MSVC 14.29.30133
TORCH_CUDA_ARCH_LIST  8.9, Ada only. Without it the build takes roughly 8x
                      longer with no benefit on this hardware
MAX_JOBS              4
```

Anything that imports gsplat must run inside a `vcvars64.bat` shell with
`CUDA_HOME` set to v12.8, because gsplat's backend runs `where cl` on import
even when the extension is already compiled and cached. `scripts/env.bat` sets
this up.

The first gsplat compile takes about 4.7 minutes and is then cached in
`%LOCALAPPDATA%\torch_extensions`. Delete that directory to force a rebuild.

## Setup

```
scripts\env.bat
.venv\Scripts\python.exe scripts\setup_env.py --check
```

`--check` reports the status of the pinned gsplat checkout and both
site-packages patches without changing anything. Dropping `--check` clones
gsplat at the pin and applies the patches. Both patches must report `applied`.

A status of `stale` means an upgrade changed the upstream source, and the patch
needs re-deriving rather than forcing.

## Usage

Run the test suite:

```
.venv\Scripts\python.exe -m pytest -q
```

That is the fast tier, CPU only, a few seconds. For the complete tier including
GPU-marked tests:

```
cmd /c "call scripts\env.bat >nul 2>&1 && .venv\Scripts\python.exe -m pytest -q -o addopts="
```

Reproducing the baseline, until the CLI lands in milestone 1:

```
scripts\env.bat
cd _gsplat_repo\examples
..\..\.venv\Scripts\python.exe simple_trainer.py mcmc ^
  --data-dir ..\..\data\tandt\truck ^
  --data-factor 1 ^
  --max-steps 7000 ^
  --save-ply --disable-viewer ^
  --result-dir ..\..\results\truck_spike ^
  --strategy.cap-max 1000000
```

tyro uses hyphens, not underscores. MCMC was chosen over the default strategy
because `cap-max` makes VRAM usage predictable, which matters on a 6 GB card.
The same parameters are expressed as configuration in `configs/truck.toml`.

Two properties of the truck data that resemble bugs:

- COLMAP reports the camera as 1957x1091 while the image files are 979x546.
  Tanks and Temples ships images already downscaled by 2. gsplat's parser reads
  an actual image and rescales `K` by the measured ratio
  (`datasets/colmap.py`), so `--data-factor 1` is correct.
- `images_2/` and `images_4/` in the scene folder are unused. `images_4` is
  245x136.

## Upstream bugs

Three bugs in upstream packages, none of them configuration problems on this
machine.

**1. PyTorch 2.11 cannot build CUDA extensions on Windows.**
`torch/include/c10/cuda/CUDACachingAllocator.h:105` declares a parameter named
`small`. The Windows SDK's `rpcndr.h:190` contains `#define small char`, so the
parameter becomes `bool char` and nvcc reports `invalid combination of type
specifiers`. Upgrading Visual Studio does not help, because the macro comes from
the SDK. Worked around by pinning `torch==2.7.1+cu128`, whose headers do not
contain the struct.

**2. gsplat 1.5.3 passes GCC flags to MSVC.**
`gsplat/cuda/_backend.py:177` sets `extra_cflags = [opt_level, "-Wno-attributes"]`.
`cl.exe` reads that as `/W` followed by `no-attributes` and fails with
`D8021: invalid numeric argument`. `-O3` is invalid MSVC syntax as well. Patched
to select flags by platform.

**3. The pycolmap fork only works on Linux.**
`pycolmap/scene_manager.py:102` reads COLMAP binaries with
`struct.unpack('L', f.read(8))`. Without a byte-order prefix, `'L'` uses native
width: 8 bytes on Linux LP64, 4 on Windows LLP64. It reads 8 bytes and unpacks
4. Patched five read sites to `'<Q'` and `'<IiQQ'`.

The write path in that same file, lines 313 to 421, still uses native `'L'` and
remains broken on Windows. Nothing in this repository writes COLMAP binaries
through pycolmap, and the planned test fixture writes them directly with
explicit byte order and width for the same reason.

Two smaller issues: gsplat imports `packaging` without declaring it as a
dependency, and `fused-ssim` requires `wheel` when building with
`--no-build-isolation`.

Both patches for bugs 2 and 3 live inside `.venv/Lib/site-packages/`, so a
reinstall or upgrade destroys them and `pip freeze` does not record them. They
are defined as exact search and replace pairs in
`scripts/patches/definitions.py`, applied idempotently by `scripts/setup_env.py`.

## Repository layout

| Path | Contents |
|---|---|
| `src/splatpipe/` | The package |
| `tests/` | Test suite, GPU-dependent tests marked `gpu` and deselected by default |
| `scripts/env.bat` | vcvars64 and CUDA_HOME setup, required before any build or training |
| `scripts/setup_env.py` | Clones gsplat at the pin and applies both patches, idempotently |
| `scripts/patches/` | Patch definitions and the apply logic |
| `configs/truck.toml` | The baseline run as configuration |
| `docs/superpowers/specs/` | Design spec |
| `docs/superpowers/plans/` | Implementation plans |
| `LEARNING.md` | Notes on the 3D vision background |
| `SPIKE_LOG.txt` | Full trace of the original feasibility spike, every command and failure with root causes |
| `_get_data.py` | Downloads and prepares the truck scene |
| `_vram_sampler.py` | Samples nvidia-smi, measuring the whole board rather than PyTorch's allocator |
| `_browser_test.png` | Screenshot of the scene rendering in a browser |

Not in git: `.venv/`, `data/`, `results/`, `_gsplat_repo/`, `_webviewer/`, build
logs. About 15 GB in total, all regenerable.

## Known gaps

- **Phone rendering is untested.** Desktop rendering works. A phone has less
  bandwidth, less memory and a weaker GPU.
- **The `PngCompression` baseline is unmeasured.** Without that number there is
  no way to state how much headroom remains. Milestone 2.
- **Pose estimation for phone captures is unscheduled.** Milestones 1 to 8 do
  not include running COLMAP on a raw capture, and milestone 7 requires three
  real scenes.
- **Resolutions above 979x546 and counts above 1M Gaussians are untested.** The
  2.891 GiB peak against 6141 MiB available suggests headroom.
- `requirements.lock.txt` records this package as a `git+ssh` URL to a private
  remote, so the file is a record rather than an installer.

## References

- Design spec: `docs/superpowers/specs/2026-08-25-splat-compression-pipeline-design.md`
- Milestone 1 plan: `docs/superpowers/plans/2026-08-26-milestone-1-scripted-pipeline.md`
- Current state and next steps: `HANDOFF.md`
