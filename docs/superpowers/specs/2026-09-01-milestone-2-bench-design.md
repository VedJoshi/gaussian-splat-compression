# Milestone 2: rate-distortion benchmark harness

Design for `bench/`, the component that measures compression quality against file
size. Written 2026-09-01, after milestone 1 was merged into `master`.

The project design at
`docs/superpowers/specs/2026-08-25-splat-compression-pipeline-design.md` is the
binding authority. This document expands its `bench/` section into something
implementable and records the decisions and measurements made while doing so.

## Why this milestone exists

The project spec calls milestone 2 "the one most likely to be skipped and must
not be", because measuring baselines before optimising is what makes any later
claim defensible. Without it the project ends with a fast compressor and no
evidence that it beats what already ships in gsplat.

Milestone 2 delivers the first points on the rate-distortion curve: raw `.ply`,
the 32-byte `.splat`, and gsplat's own `PngCompression`. It produces no
compression of its own.

## What the preflight probe established

These are measurements, not estimates. The probe ran on 2026-09-01 against
`out/truck/artifacts/scene.ply`, the 1,000,000-Gaussian truck cloud produced by
milestone 1.

`PngCompression` runs end to end on this machine and round-trips. It took about
two minutes and produced 16,258,005 bytes:

| File | Bytes |
|---|---:|
| `quats.png` | 3,688,741 |
| `shN.npz` | 3,464,570 |
| `means_l.png` | 2,984,135 |
| `sh0.png` | 2,060,225 |
| `scales.png` | 1,761,984 |
| `means_u.png` | 1,693,404 |
| `opacities.png` | 603,880 |
| `meta.json` | 1,066 |
| Total | 16,258,005 |

Against the raw `.ply` at 236,001,478 bytes that is 14.52x. Against the `.splat`
at 32,000,000 bytes it is 1.97x smaller, and unlike the `.splat` it retains the
view-dependent spherical harmonics.

This reframes the target. The compressor built in milestones 3 to 5 is not
competing with a 32 MB format that discarded its harmonics. It is competing with
a 16.3 MB format that kept them. The project spec's risk note, that compression
"may not beat `PngCompression` by much", is well founded and is now quantified
on the rate axis. The quality axis is unmeasured, which is what `bench/` is for.

## Dependencies

`PngCompression` requires three packages that milestone 1 did not install, plus
their transitive dependencies. The install order and pins that work:

```
cupy-cuda12x==13.6.0
torchpq
git+https://github.com/fraunhoferhhi/PLAS.git
```

Two non-obvious points, both of which cost time to find and would cost more to
rediscover:

- **`cupy-cuda12x` must be pinned below 14.** Version 14.2.0 requires
  `numpy>=2.0`, and this project pins `numpy<2.0.0` in `pyproject.toml`.
  Installing the current release silently upgrades numpy to 2.4.6 and breaks
  that pin. Version 13.6.0 accepts `numpy<2.6,>=1.22`, ships a prebuilt
  `win_amd64` wheel, and needs no compiler.
- **`torchpq` does not declare `cupy`.** Its metadata lists only numpy and
  torch, and it imports `cupy` at runtime inside
  `torchpq.clustering.KMeans`. Installing `torchpq` alone therefore appears to
  succeed and fails later, at the point `PngCompression` compresses `shN`.

`plas` builds a pure-Python wheel from source and pulls `click`, `kornia`,
`kornia-rs`, `lapjv`, `pandas` and `tzdata`. None of these move numpy.

`torchmetrics==1.9.0`, `ImageIO==2.37.4` and `matplotlib` were already in
`requirements.lock.txt` from milestone 1, so the metrics and the plot need
nothing new.

`requirements.lock.txt` must be reconciled with all of this. The working venv
currently has the packages installed and the lock file does not list them, which
is exactly the divergence milestone 1 existed to prevent.

## Architecture

`src/splatpipe/bench/`, a sibling of `formats/` and `stages/`, exposed as a
`splatpipe bench` subcommand.

The project spec writes the interface as `bench run <artifact-dir>`. This design
deviates on spelling only: a second console script would mean a second entry
point and a second argument parser over one shared config vocabulary. The
responsibility boundary the spec asks for is unchanged.

| Module | Purpose |
|---|---|
| `bench/codecs.py` | The codec protocol and its three implementations |
| `bench/cameras.py` | The single place that imports gsplat's COLMAP `Parser` |
| `bench/render.py` | Rendering a `GaussianCloud` through `gsplat.rasterization` |
| `bench/metrics.py` | PSNR, SSIM and LPIPS, matching gsplat's own configuration |
| `bench/curve.py` | Assembling `curve.json` and plotting `curve.png` |
| `bench/run.py` | One measurement pass over a list of codecs, and the anchor rule |

### The codec protocol

```
encode(cloud: GaussianCloud, directory: Path) -> None
decode(directory: Path) -> GaussianCloud
size(directory: Path) -> int
```

Directory-based rather than the project spec's `encode(gaussians, config) ->
bytes`. `PngCompression` genuinely produces eight files, and flattening them
into a single blob to satisfy a byte-oriented interface would misreport the
rate, which is the one number this component exists to measure. A future
`compress/` codec that produces a single buffer satisfies this protocol by
writing one file, so the spec's interface is not contradicted, only widened.

Three implementations:

- `PlyCodec`. Passthrough via the existing `read_ply` and `write_ply`. The 1.00x
  anchor point, and the only lossless one.
- `SplatCodec`. The existing `encode_splat`, plus a decoder that does not yet
  exist.
- `PngCodec`. Wraps `gsplat.compression.PngCompression`, converting between
  `GaussianCloud` and the splats dictionary it expects.

### The `.splat` decoder

New code, and the part of this milestone with real numerical content. The format
stores three float32 positions, three float32 exponentiated scales, four uint8
RGBA values and four uint8 quaternion components. Decoding inverts each:

- Positions are float32 and exact.
- Scales invert as `log(stored)`. The encoder guards overflow through
  `_FLOAT32_LOG_MAX` but not underflow: a sufficiently negative log scale
  exponentiates to zero and `log(0)` is negative infinity. The decoder clamps
  the stored scale up to the smallest positive normal float32, about
  `1.1754944e-38`, which bounds the recovered log scale at about -87.34.
- Colours invert as `sh0 = (rgb / 255 - 0.5) / SH_C0`.
- Opacity inverts as `logit(alpha / 255)`, which diverges at both ends.
  `alpha = 255` gives `logit(1) = inf`. The encoder produces that value, because
  it clips to `[0, 255]` and casts, so this is reachable on real data rather
  than hypothetical. The decoder clamps the normalised alpha to the half-step
  interval `[0.5 / 255, 254.5 / 255]`, which bounds the recovered logit at about
  plus or minus 6.23. The half step is the midpoint of the quantisation bucket
  the value came from, so the clamp is the best estimate available rather than
  an arbitrary guard.
- Quaternions invert as `(stored - 128) / 128` followed by normalisation.

The round trip is lossy by construction, and saturated opacities do not survive
it exactly. That is a property of the format, not a defect in the decoder, and
the rate-distortion point must reflect it honestly rather than special-case it
away.

Writing the decoder also supplies the round-trip test the encoder has never had.

### Cameras

`bench/cameras.py` is the only module that inserts `_gsplat_repo/examples` on
`sys.path`, and it imports `Parser` and `Dataset` from `datasets.colmap`.

This is a deliberate departure from `stages/train.py`, which avoided importing
from that directory by shelling out to a subprocess. The reason to depart is
that correctness here depends on identity, not similarity. gsplat's `Parser`
applies a scene normalisation, a transform and a scale, to the camera poses, and
selects held-out views as `indices % test_every == 0`
(`_gsplat_repo/examples/datasets/colmap.py:369`). A reimplementation that drifts
from any of that produces plausible numbers that are quietly wrong, and nothing
downstream would reveal it. Reusing the trainer's own code makes pose parity a
property of construction rather than of agreement.

The `sys.path` insertion is contained in one module and documented there, the
same way `stages/train.py` contains the subprocess boundary in one place.

### Rendering

`bench/render.py` mirrors `simple_trainer.rasterize_splats`
(`_gsplat_repo/examples/simple_trainer.py`), calling the public
`gsplat.rasterization` with:

- `means` as stored
- `quats` unnormalised, because `rasterization` normalises internally
- `scales` as `exp(cloud.scales)`
- `opacities` as `sigmoid(cloud.opacities)`
- `colors` as `cat([sh0, shN], dim=1)`

This maps onto `GaussianCloud` exactly, because milestone 1 chose to store log
scales and logit opacities rather than converting on read. That decision pays
for itself here.

### Metrics

`bench/metrics.py` matches `simple_trainer.py` field for field: torchmetrics
`PeakSignalNoiseRatio` and `StructuralSimilarityIndexMeasure` at
`data_range=1.0`, and `LearnedPerceptualImagePatchSimilarity` on AlexNet, with
predictions clamped to `[0, 1]` and permuted to `[1, 3, H, W]` before scoring.

Matching the reference implementation is what makes the project's numbers
comparable to gsplat's published ones. Any deviation, including a different
LPIPS backbone, makes the comparison meaningless while still producing a plot.

## Outputs

`curve.json` and `curve.png`, written into the run directory beside
`manifest.json`.

`curve.json` records, per codec: the codec name, bytes, compression ratio
against the raw `.ply`, PSNR, SSIM, LPIPS, and encode and decode seconds. It
also records the shared context that makes a point traceable: the scene, the
config digest, the held-out view count, and library versions and the Git commit
obtained by reusing `manifest.collect_versions()` rather than duplicating it.

`curve.png` plots quality against size, with size on a logarithmic axis, one
labelled point per codec.

## Success criterion

**`bench` run on the raw `.ply` must reproduce the trainer's own held-out
metrics.** The truck run recorded PSNR 24.394817, SSIM 0.8579996 and LPIPS
0.1375573. The tolerance is 0.05 dB PSNR, fixed before the run and not to be
adjusted afterwards.

The tolerance is tighter than milestone 1's 0.1 dB because nothing is being
retrained. The only sources of difference are rendering and metric evaluation,
both of which are deterministic given identical inputs, so the remaining
variation is float reduction order rather than stochastic training.

This is the milestone's falsification criterion and it is the reason the whole
harness can be trusted. `PlyCodec` is lossless, so rendering its output must
reproduce what the trainer measured when it rendered the same Gaussians. If the
camera pipeline has drifted, this fails. Without this check, every number the
project reports afterwards is unfalsifiable.

## Testing

Fast tier, no GPU and no compiler shell, consistent with the tier that has to
survive a two-week gap:

- Codec round-trips on synthetic clouds, numpy only.
- The `.splat` decoder's saturated-opacity and underflowing-scale edges, which
  are the cases that produce infinities.
- A known-answer metric check on fixed image pairs, so a regression in the
  metric code is visible without a GPU. The project spec asks for this
  explicitly.
- `curve.json` schema and the ratio arithmetic.

GPU tier:

- One tiny-scene end-to-end run producing a curve with at least two points,
  using the existing synthetic 24-image fixture.
- The truck success criterion above, as a marked test.

## Risks

- **`PngCompression` crops to a square number of Gaussians.** `n_crop` is
  `n_gs - floor(sqrt(n_gs))**2`, and the cropped splats are the lowest-opacity
  ones. Truck holds exactly 1,000,000 Gaussians, which is 1000 squared, so
  nothing is dropped and the measured baseline above is unaffected. Pruning in
  milestone 4 will almost never produce a square count, so from that point the
  `PngCompression` point is measured on slightly fewer Gaussians than its
  competitors. `curve.json` must record the count actually encoded, per codec,
  rather than assuming it equals the input.
- **`PngCompression.compress` mutates the dictionary it is given.** It applies
  `log_transform` to means and normalises quats in place. `PngCodec` must pass
  copies, or the cloud is corrupted for every codec measured afterwards.
- **The `sys.path` insertion is load-bearing and fragile.** If gsplat's example
  layout changes at a future pin, `bench/cameras.py` breaks. The pin is recorded
  in `scripts/setup_env.py` and the failure would be an obvious ImportError
  rather than a silent wrong number, which is the acceptable failure mode.
- **LPIPS downloads AlexNet weights on first use.** This makes the first bench
  run network-dependent in a project that is otherwise reproducible offline. The
  weights are cached afterwards. Worth knowing before a run fails on a train.

## Out of scope

No compression of the project's own, which is milestone 3. No viewer or
container format work. No phone testing. No additional scenes: milestone 2
measures truck, and the harness is what generalises later, not the dataset.

## Open questions

- Whether `curve.json` should be merged into `manifest.json` rather than sitting
  beside it. Kept separate for now, because a bench run is repeatable against an
  unchanged training run and overwriting the manifest would lose that
  distinction.
- Whether decode time belongs on the curve at all. Recorded because it is nearly
  free to measure and because load time is a stated success criterion for the
  deployed viewer in milestone 7.
