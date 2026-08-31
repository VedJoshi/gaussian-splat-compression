# Design: compressed Gaussian splat pipeline and web viewer

Date: 2026-08-25
Status: agreed, not started
Author: Ved, with Claude

## Goal

A portfolio project demonstrating applied computer vision engineering: a working, deployed system
where one named component is a research technique implemented from papers and measured honestly.

The audience is hiring teams for research engineer and applied CV roles. The claim being made is
"I built and deployed this, and the hard part was the compression, which I implemented from the
literature and benchmarked." Not "I trained a model."

## Constraints

These are fixed and the design is shaped around them.

- Roughly 4 to 6 hours per week, weekends, with gaps of up to two weeks. Every unit of work must be
  independently verifiable so that returning after a gap costs momentum but not progress.
- RTX 4050 Laptop, 6 GB VRAM, of which about 4.95 GiB is usable.
- Deliberately different subject matter from the FYP, which covers adversarial robustness of vision
  transformers. This project shares no topic with it.
- The 3D vision background is new. `LEARNING.md` runs in parallel, which stretches the calendar.

## Evidence base

Everything below rests on measurements from the spike in this repository, not on estimates. See
`README.md` and `SPIKE_LOG.txt`.

| Fact | Value | Why it matters |
|---|---|---|
| Training time, truck scene, 7k steps | 7.57 min | Short enough to iterate inside one weekend |
| Peak VRAM at 1M Gaussians | 2.89 GiB of 4.95 usable | Hardware is not the binding constraint |
| Output `.ply` | 225 MiB | There is real compression headroom |
| Bytes per Gaussian | 236, being 59 float32 | The compression target is concrete |
| Of which spherical harmonics | 45 of 59 floats, about 76% | Where compression work must start |
| `.splat` baseline | 30 MiB, 7.38x smaller | The number to beat, and it cheats |
| Browser load, 30 MiB | 1.07 s, 60 fps, 39.5 MiB heap | Web path works, with headroom to spare |

The `.splat` figure needs its caveat carried everywhere: it reaches 32 bytes per Gaussian by
discarding all 45 `f_rest_*` coefficients, so it throws away every view-dependent appearance term.
Beating it on size alone is trivial and meaningless. Beating it on size at equal or better quality
is the actual problem.

## Scope

### In

- 3 to 6 real scenes, captured on a phone by Ved. Three is the bar; more only if time allows.
- A local CLI pipeline: poses, training, compression, packaging
- A compression library, which is the research core
- A rate-distortion benchmark harness
- A web viewer that reads the project's own format
- A static site, hosted on Vercel or GitHub Pages
- A writeup with the curve, ablations and honest failure cases

### Out

- User uploads, job queues, training as a service
- Authentication, user accounts, databases
- Multi-tenancy

The scope boundary was chosen deliberately. Upload and queue infrastructure is roughly a month of
work that demonstrates nothing about computer vision, and it would consume the time the compression
work needs. The narrow version is the more impressive artifact, not the lesser one.

### Uploads later

The decision is "narrow now, uploads later". The only concession the design makes to that is a
constraint, not a feature:

> The pipeline must be a pure function of `(input directory, config) -> artifacts`, with no
> interactive state and no hardcoded paths.

That is good design regardless of whether uploads ever happen, so it costs nothing. Nothing further
should be built speculatively for a frontend that may never exist.

## Architecture

Five units. Each has one purpose, a defined interface, and can be tested without the others.

### `pipeline/`

Orchestration. Takes a scene directory and a config, produces artifacts.

Stages: COLMAP poses, gsplat training, compression, packaging. This is the spike's manual sequence
turned into something reproducible.

- Depends on: gsplat, `compress/`
- Interface: `pipeline run <scene-dir> --config <cfg> --out <dir>`
- Tested by: a tiny synthetic scene that runs in seconds, asserting artifacts appear and are valid

### `compress/`

The research core, and the part that is genuinely the author's work.

Not one function but composable stages, each independently switchable:

1. Prune Gaussians by contribution to rendered pixels
2. Quantise spherical harmonics, by vector quantisation or a codebook
3. Quantise geometry, meaning positions, scales and rotations
4. Entropy-code the result
5. Pack into the container format

Composability is load-bearing, not cosmetic. Being able to switch each stage on and off is what
makes the ablation table possible, and the ablation table is the evidence that the work was measured
rather than guessed.

- Depends on: numpy only. Deliberately no torch, so tests are fast and the code is easy to reason
  about.
- Interface: `encode(gaussians, config) -> bytes` and `decode(bytes) -> gaussians`
- Tested by: round-trip tests asserting `decode(encode(x))` stays within a stated PSNR tolerance,
  per stage and in combination

### `bench/`

Rate-distortion measurement.

Renders held-out views, computes PSNR, SSIM and LPIPS against file size, emits the curve.

Held-out views, not training views. This is the single methodological point that separates the
project from a blog post, and it is not negotiable. Measuring reconstruction quality on the images
the scene was fitted to would overstate every result.

- Depends on: gsplat for rendering, torchmetrics for the metrics
- Interface: `bench run <artifact-dir> -> curve.json, curve.png`
- Tested by: known-answer checks on a fixed scene, so a regression in the metric code is visible

### `viewer/`

Web rendering. Starts as a fork of antimatter15's viewer, which the spike proved works, and diverges
as the format does.

Responsibilities: fetch the payload, unpack it on the CPU into GPU buffers, render. The unpacker is
JavaScript first; WebAssembly only if profiling shows JavaScript is too slow. The spike's 1.07 s
load and 39.5 MiB heap suggest a generous budget, so WASM should not be assumed necessary.

- Depends on: nothing at runtime. Static assets only.
- Tested by: `scripts/browser_test.js`, extended. It already measures load time, frame rate, GL
  renderer and console errors.

### `site/`

Static hosting for the viewer and a handful of scenes. Deliberately dumb. Vercel or GitHub Pages.

Page structure, in order: the interactive viewer first, then visual side-by-side comparisons at
different compression levels, then memory and load benchmarks, then the rate-distortion plots. The
demo leads; the evidence follows.

## The container format

### Decision: defer it

The format is the contract between the Python encoder and the JavaScript viewer, and it is the most
expensive interface to change. It is therefore deliberately not designed yet.

A format is a commitment to a particular set of compression stages. Which stages actually earn their
place is unknown until measured. Designing the container first means designing around guesses, then
discovering in week 5 that one stage gives 4x and another gives nothing, leaving a format with a
block type nothing uses and no room for what mattered.

**Weekends 1 to 4 therefore emit `.splat`**, which already works end to end and provides the 30 MiB
baseline. The project's own format is introduced around weekend 5, designed against measurements.

### The principle it will follow

When designed, the format should use a struct-of-arrays layout: all x-coordinates together, then all
y, then the spherical harmonics, rather than all of Gaussian 1's fields followed by all of
Gaussian 2's.

The reason is that compression exploits similarity between neighbouring bytes. Grouped
x-coordinates are similar numbers and entropy-code well. Interleaved attributes put unrelated values
next to each other and leave nothing to exploit. Since compression is the point of the project, the
layout should serve compression.

The cost is a repacking step in the viewer at load time, which the spike's headroom suggests is
affordable.

This claim is checkable and should be checked rather than assumed: encode one scene both ways and
compare compressed sizes. If the difference is negligible, prefer array-of-structs for the simpler
viewer.

Likely shape: a small header describing which stages were applied and with what parameters, followed
by typed binary blocks. This allows progressive decoding and lets stages be added without breaking
existing files.

## Milestones

Ordered so each weekend ends with something measurably better, and so stopping early still leaves a
real artifact.

| # | Work | Ends with |
|---|---|---|
| 1 | Turn the spike into a scripted pipeline | Done. `splatpipe run` reproduces truck: PSNR 24.395, 225 MiB ply |
| 2 | Build `bench/`, measure baselines | First RD points: raw `.ply`, `.splat`, `PngCompression` |
| 3 | Spherical harmonic quantisation | First real win on the curve |
| 4 | Contribution-based pruning | Second win; enough evidence to design the format |
| 5 | Container format and entropy coding | Files the viewer can read |
| 6 | Fork the viewer to the format | A scene rendering from the project's own pipeline |
| 7 | Deploy the static site with 3 scenes | A link that can be sent to someone |
| 8 | Writeup: curve, ablations, failures | The artifact recruiters actually read |

Milestone 2 is the one most likely to be skipped and must not be. Measuring baselines before
optimising is what makes any end claim defensible. Without it the project ends with a fast
compressor and no evidence it beats what already ships in the library.

Calendar: 8 working weekends, realistically about three months alongside the learning track and
coursework.

## Testing

- `compress/` is developed test-first. Round-trip correctness is expressible as a test before the
  encoder exists, which makes it a natural fit.
- Every compression stage gets a test asserting its round-trip error stays within tolerance, both
  alone and combined with others.
- `bench/` gets known-answer tests so metric regressions are caught.
- `pipeline/` gets one end-to-end test on a synthetic scene small enough to run in seconds.
- The browser is tested by the existing Playwright script, extended to assert at least 30 fps at
  1280x800 on the development machine, and zero console errors other than the known favicon 404.

The test suite is what makes two-week gaps survivable: run it, and the state of the project is
immediately legible.

## Risks

**The site-packages patches are fragile.** Both the gsplat MSVC fix and the pycolmap struct fix live
inside `.venv` and die on any reinstall. Milestone 1 must convert them into a setup script or a
vendored fork. Leaving them as tribal knowledge would make the project unreproducible, including for
its author after a break.

**Compression may not beat `PngCompression` by much.** gsplat ships a baseline and it has not been
measured. If the honest result is a modest improvement, that is still a legitimate outcome and gets
reported as one. The project's value is the measurement discipline, not a specific ratio. This
should be settled early, in milestone 2, not discovered at the end.

**Phones are untested.** Desktop rendering works. A phone has less bandwidth, memory and GPU. The
claim "opens on a phone" must not be made until someone opens it on a phone. Test this in
milestone 6, not milestone 8.

**Scope creep toward uploads.** The boundary is set. Revisit only after milestone 8.

**Learning curve.** The 3D vision material is new, which is the largest source of schedule
uncertainty. The milestone order is deliberately arranged so the earliest weekends need the least
new theory.

## Success criteria

The project is done when all of the following hold:

1. A public link loads a scene in under 3 seconds and holds at least 30 fps at 1280x800 in a
   desktop browser.
2. A rate-distortion curve exists, measured on held-out views, comparing the project's compressor
   against raw `.ply`, `.splat`, and gsplat's `PngCompression`.
3. An ablation table shows the contribution of at least three compression stages individually.
4. The pipeline reproduces a scene from a phone capture with a single command on a clean machine.
5. A writeup states what worked, what did not, and what the numbers were, including negative
   results.

Criterion 5 is not decoration. The FYP work already shows a willingness to record results that
contradict expectations, and that is a rarer signal than any compression ratio.

## Open questions

None blocking. Two to resolve during the work:

- Whether struct-of-arrays measurably beats array-of-structs for this data. Resolve in milestone 5
  by measurement.
- Whether the viewer's unpacker needs WebAssembly. Resolve in milestone 6 by profiling. Assume not.
