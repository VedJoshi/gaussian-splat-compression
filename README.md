# splat-spike

A feasibility test, not a project. It answers one question:

> Can an RTX 4050 Laptop (6 GB) train a 3D Gaussian Splatting scene, and is the output large
> enough that compressing it is a problem worth working on?

Both yes. Ran 2026-08-25 on Windows 11, RTX 4050 Laptop 6 GB.

## Result

| Criterion | Bar | Measured | |
|---|---|---|---|
| gsplat compiles and imports | must work | works, after two patches | pass |
| Scene trains end to end | under 45 min | 7.57 min | pass |
| Peak VRAM | under ~5 GiB usable | 2.89 GiB | pass |
| Output .ply large enough to be worth compressing | over 100 MB | 225 MiB | pass |
| Renders in a browser or on a phone | must work | not tested | open |

The 6 GB card was the risk I expected to fail. It had about 2 GiB to spare. The real difficulty was
the Windows toolchain, which needed three upstream bugs worked around.

## What was run

Tanks and Temples `truck`, 251 images at 979x546, gsplat MCMC strategy, 7,000 steps, capped at
1,000,000 Gaussians.

```
wall clock      7.57 min
peak GPU mem    2960 MiB of 6141 MiB (2.891 GiB)
PSNR 24.406     SSIM 0.8580     LPIPS 0.1372
gaussians       1,000,000 (hit our cap, not a hardware limit)
render          0.0246 s/image, about 41 fps
output .ply     236,001,478 bytes = 225 MiB
```

The 3DGS paper reports about 25.2 PSNR on `truck` at 30k steps. This ran 7k steps, so 24.4 is in
the expected range. The pipeline is working, not silently broken.

## The part that matters for the project

236,001,478 / 1,000,000 = 236.0 bytes per Gaussian, which breaks down as:

| Field | Floats |
|---|---|
| means | 3 |
| scales | 3 |
| quaternion | 4 |
| opacity | 1 |
| SH DC | 3 |
| SH rest (`f_rest_0..44`) | 45 |
| total | 59 x 4 bytes = 236 |

About three quarters of the file is spherical harmonic coefficients. That is a measured target for
compression work rather than a guess.

Converting to `.splat`, the format most web viewers use, gives 30 MiB, a 7.38x reduction. It gets
there by dropping all 45 `f_rest_*` coefficients and keeping only the DC term, so it discards every
view-dependent appearance term. The common web format is already lossy in appearance, not just
precision.

One thing to weigh before designing anything: `gsplat.compression.PngCompression` already exists in
the library, exposed as `--compression png`. Splat compression is not untouched ground. That makes
the honest framing "analysed and improved on an existing baseline" rather than "built compression".
That baseline was not measured here.

## Three upstream bugs

None of these are configuration problems on this machine.

**1. PyTorch 2.11 cannot build CUDA extensions on Windows.**
`torch/include/c10/cuda/CUDACachingAllocator.h:105` declares a parameter named `small`. The Windows
SDK's `rpcndr.h:190` contains `#define small char`, so the parameter becomes `bool char` and nvcc
reports `invalid combination of type specifiers`. Upgrading Visual Studio does not help, because the
macro comes from the SDK. Fixed by pinning `torch==2.7.1+cu128`, whose headers do not contain the
struct.

**2. gsplat 1.5.3 passes GCC flags to MSVC.**
`gsplat/cuda/_backend.py:177` sets `extra_cflags = [opt_level, "-Wno-attributes"]`. `cl.exe` reads
that as `/W` followed by `no-attributes` and fails with `D8021: invalid numeric argument`. `-O3` is
not valid MSVC syntax either. Patched to select flags by platform. Original saved as
`_backend.py.ORIGINAL.bak`.

**3. The pycolmap fork only works on Linux.**
`pycolmap/scene_manager.py:102` reads COLMAP binaries with `struct.unpack('L', f.read(8))`. Without
a byte-order prefix, `'L'` uses native width: 8 bytes on Linux LP64, 4 on Windows LLP64. It reads 8
bytes and tries to unpack 4. Patched five read sites to `'<Q'` and `'<IiQQ'`. Original saved as
`_scene_manager.py.ORIGINAL.bak`.

The write path in that same file, lines 313 to 421, still uses native `'L'` and is still broken on
Windows. Nothing here writes COLMAP binaries, so it does not matter yet.

Two smaller ones: gsplat imports `packaging` without declaring it as a dependency, and `fused-ssim`
needs `wheel` installed when building with `--no-build-isolation`.

## The patches are fragile

Both fixes were applied inside `.venv/Lib/site-packages/`. A reinstall, an upgrade, or a fresh venv
destroys them, and `pip freeze` will not show them. If this turns into a real project, they need to
become a setup script, an upstream PR, or a vendored fork.

## Working toolchain

This combination is not obvious and is easy to lose.

```
python       3.11.0        not 3.13, which is the default on PATH
torch        2.7.1+cu128   not 2.11, see bug 1
torchvision  0.22.1+cu128
numpy        1.26.4        gsplat v1.5.3 examples require numpy<2
gsplat       1.5.3         plus the MSVC flag patch
packaging                  undeclared gsplat dependency
wheel                      needed for --no-build-isolation builds

CUDA_HOME             C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
                      v12.9 is on PATH by default and has to be overridden,
                      since it does not match the torch cu128 build
host compiler         vcvars64.bat from VS Build Tools 2019 16.11, MSVC 14.29.30133
TORCH_CUDA_ARCH_LIST  8.9, pinning to Ada only. Without it the build takes
                      roughly 8x longer for no benefit on this machine
MAX_JOBS              4
```

Everything has to run inside a `vcvars64.bat` shell with `CUDA_HOME` set to v12.8. `scripts/env.bat`
does this. First gsplat compile takes about 4.7 minutes and is then cached in
`%LOCALAPPDATA%\torch_extensions`; delete that to force a rebuild.

## Reproducing the run

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

tyro uses hyphens, not underscores. MCMC was chosen over the default strategy because `cap-max`
makes VRAM usage predictable, which matters on a small card.

Two things about the data that look like bugs and are not:

- COLMAP reports the camera as 1957x1091 while the image files are 979x546. Tanks and Temples ships
  images already downscaled by 2. gsplat's parser reads an actual image and rescales `K` by the
  measured ratio (`datasets/colmap.py:265-273`), so `--data-factor 1` is correct.
- `images_2/` and `images_4/` in the scene folder are unused. I generated them before working the
  above out. `images_4` is 245x136, far too small to be useful.

## Not tested

- The browser and phone path, which is half the point of the project it was meant to de-risk. A
  `.splat` was produced and a viewer cloned into `_webviewer/`, but it was never loaded in a browser
  and never opened on a phone. Rendering 1M Gaussians at 60 fps on a mid-range phone is an
  assumption at this point.
- gsplat's `PngCompression` baseline. Without that number there is no way to know how much headroom
  is left.
- Higher resolutions or more than 1M Gaussians, though the spare VRAM suggests there is room.

## Files

| Path | What |
|---|---|
| `SPIKE_LOG.txt` | Full trace, every command and failure with root causes. Start here if something breaks. |
| `scripts/env.bat` | Sets up vcvars64 and CUDA_HOME. Required before any build or training. |
| `_ply_to_splat.py` | Vectorised .ply to .splat. The upstream converter loops per vertex and is unusable at 1M. |
| `_get_data.py` | Downloads and prepares the truck scene. |
| `_vram_sampler.py` | Samples nvidia-smi, so it measures the whole board rather than just PyTorch's allocator. |
| `*.ORIGINAL.bak` | Unmodified copies of the two patched library files. |

Not in git: `.venv/`, `data/`, `results/`, `_gsplat_repo/`, `_webviewer/`, build logs. About 15 GB
in total. The whole folder is safe to delete; nothing outside it was changed except the two patched
files inside `.venv`, which go with it.

## Where this leaves things

The hardware objection is gone. What is still open is framing, not feasibility:

1. `PngCompression` already exists, so the pitch is about improving on a baseline, not inventing one.
2. The browser half is unverified, and it is the half that makes the result shareable.
3. Three quarters of the file is SH coefficients, so that is where compression work should start.

This was a spike during an ideation discussion. No design has been agreed and no project code has
been written.
