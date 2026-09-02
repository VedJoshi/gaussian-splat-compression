# Handoff: Milestone 2 bench harness: 2026-09-01 (in progress, 2 of 11 tasks)

## Goal

Build `src/splatpipe/bench/` and measure the rate-distortion baselines this
project has to beat: the raw `.ply`, the 32-byte `.splat`, and gsplat's own
`PngCompression`. Milestone 2 produces measurement infrastructure and numbers.
It adds no compression of the project's own, which is milestone 3.

The falsification criterion is the truck scene: three codecs measured on
held-out views, each producing a size and a PSNR, SSIM and LPIPS triple, with
the `.ply` point matching the milestone 1 manifest to the byte.

**Milestone 1 is complete, reviewed as a whole, and merged into `master`.** Its
detail is condensed below and its ledger holds the rest.

## Where the work is

Branch `milestone-2-bench`, forked from `master` at `4528129`, HEAD `bb7eb28`.
**Nothing on this branch has been pushed.** `master` is at `4528129` and is
pushed.

Tasks 1 and 2 of 11 are complete. Task 3 is next and has not started.

## The number to beat

`PngCompression` was probed end to end on this machine before the plan was
written. It runs, it round-trips, it takes about two minutes, and it produces
16,258,005 bytes.

| Reference | Bytes | Ratio |
|---|---:|---:|
| Raw `.ply` | 236,001,478 | 1.00x |
| `.splat` | 32,000,000 | 7.38x |
| `PngCompression` | 16,258,005 | 14.52x |

`PngCompression` is therefore half the size of the `.splat` **and** it retains
spherical harmonics, which the `.splat` discards. It, not the `.splat`, is the
baseline milestone 3 has to improve on. The spec's original worry, that this
project "may not beat `PngCompression` by much", is well founded and is now
quantified rather than guessed.

## Completed on this branch

- **Task 1, `0bf9f0e..862241f`: dependencies and the lock file.** Added
  `cupy-cuda12x==13.6.0`, `torchpq==0.3.0.6`, and `plas` pinned at
  `4f1109c94f29a5a6de62bdf8f20cb3ce1ff0e680`, plus their transitives, all 83
  lines verified verbatim against `pip freeze --exclude-editable`. Added
  `test_png_compression_dependencies_are_importable`. Reviewed with one
  Important and one Minor, both inherited from plan text rather than introduced
  by the implementer, and both fixed in one round.
- **Task 2, `61ecc76..a2ecfee`: the `.splat` decoder.** `decode_splat` in
  `src/splatpipe/formats/splat.py` inverts the 32-byte record back to a
  `GaussianCloud`, with explicit clamps at both quantisation edges. Reviewed
  with two Important and one Minor. Clean after three fix rounds.
- Controller documentation fixes committed separately as `61ecc76` and
  `bb7eb28`.

**Verified today at `bb7eb28`:** fast tier `126 passed, 3 deselected` in 21.0s;
complete tier `129 passed` in 121.4s; `setup_env.py --check` reports
`gsplat checkout: pinned` and `applied` for both patches.

## What the three Task 2 fix rounds were actually about

All three were claim accuracy, not code correctness. The code was right on the
first commit. Recording this because the pattern is the useful part:

- **A wrong divisor in the quaternion decode is undetectable and also
  harmless.** `(stored - 128) / d` followed by normalisation cancels `d`
  entirely. Dividing by 255 instead of 128 produces error identical to the
  correct decoder.
- **Dropping the normalisation gives error 0.0078, smaller than the correct
  decoder's 0.0105.** No tolerance catches it.
- **An offset of 127 passes while 129 fails.** `encode_splat` truncates rather
  than rounds when it casts to uint8, so 127 and 128 land in the same
  quantisation bucket. Tightening the tolerance cannot separate them.
- The test therefore pins the byte slice hard and the 128 offset in one
  direction only. Its docstring says exactly that.

Two of the three rounds corrected wording I had supplied myself, once after the
reviewer's stated rationale turned out to be false in both halves, and once
after my own corrected docstring still overclaimed. Measure before writing a
tolerance or a rationale into a test. Reasoning about this format on paper
produced a false claim three times.

## In Progress

- Nothing is under implementation. Task 2 finished and was reported.
- **Task 3 is next**: the codec protocol plus `PlyCodec` and `SplatCodec`.
  Creates `src/splatpipe/bench/__init__.py` and `src/splatpipe/bench/codecs.py`,
  tests in `tests/test_codecs.py`. BASE is `bb7eb28`. Expect
  `131 passed, 3 deselected` afterwards.
- If this section names a task as under way and the ledger has no matching
  `Task N: complete` line, that task did not finish. Read the ledger at
  `.superpowers/sdd/2026-09-01-milestone-2-bench/progress.md` before changing
  anything.

## Traps waiting in the remaining tasks

These were found while writing the spec and are recorded in it. They are listed
here because each is a silent wrong number rather than a crash.

- **`Parser` defaults `normalize=False` while the trainer passes
  `cfg.normalize_world_space`, which is `True`** (`simple_trainer.py:67`).
  Loading cameras with the default silently evaluates against a differently
  scaled world and every metric is quietly wrong. `bench/cameras.py` encodes
  this as the named constant `NORMALIZE_WORLD_SPACE = True`.
- **`render.py` must derive `sh_degree` from the cloud**, because a decoded
  `.splat` has degree 0 while a `.ply` has degree 3. Hardcoding it renders the
  wrong thing for one codec and produces a plausible number.
- **`PngCompression.compress` mutates the dictionary it is given.** It applies
  `log_transform` to means and normalises quats in place. `PngCodec` must pass
  copies, or the cloud is corrupted for every codec measured afterwards.
- **`PngCompression` crops to a square number of Gaussians.** Truck holds
  exactly 1,000,000, which is 1000 squared, so nothing is dropped now. From
  milestone 4 pruning will rarely produce a square count, so `curve.json`
  records the count actually encoded per codec rather than assuming it equals
  the input.
- **LPIPS downloads AlexNet weights on first use.** The first bench run is
  network-dependent in a project that is otherwise reproducible offline.

## Not Working / Blockers

- Nothing is broken. The repository, both venvs, both test tiers, the gsplat
  checkout, and both installed patches are healthy.
- **A `cmd /c` complete-tier run launched from Git Bash is a false green.** The
  MSYS layer rewrites `/c` into a filesystem path, so `cmd` opens an
  interactive shell, prints its banner and exits 0 having run no tests. It is
  indistinguishable from success by exit code alone. Run those lines through
  PowerShell and check that pytest output is actually present.
- **Four milestone 1 Minors are consciously accepted**, ruled on by the
  whole-branch review. They do not need rediscovering:
  - Task 7: an invalid cloud combined with an unknown order raises
    `ArtifactError` before `ConfigError`. Validating data before dispatching on
    an order string is the right sequence and both outcomes are errors.
  - Task 8: `collect_versions` catches bare `Exception` around the torch and
    gsplat imports, so a broken install records `not-imported` identically to
    an absent one. A manifest must never fail a run over a version string.
  - Task 8: no test covers the `_git_commit` fallback branch or
    `ArtifactRecord.of` with a path outside `relative_to`. The fallback is a
    plain try/except with nothing to get wrong, and no caller can reach the
    second case: `run_pipeline` always passes paths under `paths.root`.
  - Task 9: `write_images_bin` and `write_points3D_bin` annotate their inputs
    as `Sequence[tuple]` with an untyped inner tuple. Cosmetic, test-only.
- **Open, and worth doing before the container format in milestone 5**: the
  `.splat` byte-parity tests against gsplat use tie-free inputs, and both
  implementations sort Morton codes with an unstable argsort. A million
  Gaussians in a 1024 cubed grid will produce ties, so parity in that regime is
  untested. This does not affect this project's own reproducibility, which
  re-encodes a fixed `.ply` deterministically.
- `_anchor_positions` deliberately treats indentation-free pycolmap anchors as
  substrings. Its uniqueness guard handles the current patches, but a future
  patch could theoretically alias inside a larger expression. The failure mode
  is a loud refusal, not corruption.
- `scripts/browser_test.js` hardcodes two absolute paths under the author's
  home directory. It is spike-era tooling that nothing in the pipeline calls.
- Phone captures still have no scheduled pose-estimation stage. This blocks the
  later three-scene deployment milestone.

## Key Decisions

### Process

- **One task per user turn**: implement one task, run its review gate, record
  the outcome, then stop. Do not dispatch or begin the next task while review
  is open. Ved has asked for this directly. It overrides the
  `subagent-driven-development` skill's continuous-execution rule, and the
  reason is that this project runs on weekends with gaps of up to two weeks.
- **Use the milestone branch in the existing checkout**: `.venv/`,
  `_gsplat_repo/`, `data/`, `results/` and `out/` are untracked and bound to
  this repo root. A separate worktree lacks the 15 GB working environment.
- **Routine pushes are authorized**: Ved authorized remote pushes and
  default-branch documentation updates on 2026-08-28. Merges remain his call.
  The milestone 1 merge into `master` was made on 2026-09-01 by his explicit
  instruction, after both tiers were re-verified on the exact tree being merged.
- **The default branch is named `master`**: there is no `main` branch.
- **Style is binding**: no emoji, em dashes, litotes, irony, or exclamation
  marks. Use plain declarative prose and comment only non-obvious reasoning.

### Milestone 2

- **`cupy-cuda12x` is pinned to `13.6.0`**: version 14 and later require
  `numpy>=2.0`, which would silently break this project's `numpy<2.0.0` pin.
  Pin with `==`, because `<` is a redirect operator in `cmd`.
- **`plas` is pinned to its resolved commit.** An unpinned git URL in a lock
  file is not a lock: a later install would fetch whatever the default branch
  is that day.
- **`torchpq` does not declare `cupy`.** Its metadata lists only numpy and
  torch, so `pip install torchpq` succeeds without cupy present. The gap shows
  at the first import: `torchpq/__init__.py` imports cupy at module top and
  raises `ModuleNotFoundError` if it is absent.
- **`kornia_rs` stays spelled with an underscore in the lock**, because
  `pip freeze` reports the distribution name that way and the lock is verified
  against that output.
- **The codec protocol is directory-based**: `encode(cloud, dir)`,
  `decode(dir) -> GaussianCloud`, `size(dir) -> int`. `PngCompression`
  genuinely produces eight files, so a single-file protocol would have to
  invent a container, which is milestone 5's job rather than milestone 2's.
- **`GaussianCloud.scales` are logs and `opacities` are logits**, exactly as
  stored in the `.ply`. Nothing converts on read. Every codec inverts back to
  this convention.
- **`decode_splat` returns `shN` of shape `(N, 0, 3)`, so `sh_degree == 0`.**
  The `.splat` format discards `f_rest`. This is a real fidelity difference
  between the codecs, not a defect to paper over.
- **The decoder clamps at both quantisation edges.** Scales invert as
  `log(stored)`, clamped up to the smallest positive normal float32, about
  `1.1754944e-38`, bounding the recovered log scale at about -87.34. Opacity
  inverts as `logit(alpha / 255)`, clamped to `[0.5 / 255, 254.5 / 255]`,
  bounding the recovered logit at about plus or minus 6.23. Both clamps are
  verified load-bearing: deleting either makes its test fail.
- **`splatpipe bench` takes `--codecs`**, a comma-separated selection over a
  name-to-factory registry defaulting to all three. Without it the CPU test
  would construct `PngCodec`, which needs CUDA, cupy and plas. Re-running one
  codec without paying for the other two is wanted anyway from milestone 3.
- **`curve.json` stays separate from `manifest.json`**: a bench run is
  repeatable against an unchanged training run, and overwriting the manifest
  would lose that distinction.

### Milestone 1, still binding

- **The venv is mandatory**: use `.venv\Scripts\python.exe`; bare `python` and
  `py` resolve to Python 3.13, while the project requires Python 3.11.
- **GPU commands must chain `env.bat`**: tool calls do not retain shell state.
- **Do not upgrade the working stack**: PyTorch 2.7.1+cu128, CUDA 12.8, numpy
  1.26.4, gsplat 1.5.3, and MSVC 14.29 are the verified combination.
- **Do not hand-edit site-packages**: `scripts/setup_env.py` owns both patches.
- **Keep the accepted anchor rule**: anchors starting with whitespace must
  begin at a line boundary; indentation-free anchors match as substrings.
- **Invalid Gaussian rows fail export**: do not mirror gsplat's silent
  filtering. `.splat` size must equal `32 * len(cloud)`.
- **Separate layout authority from `exp` implementation**: NumPy and Torch CPU
  `exp` differ by up to 2 ULP on random float32 inputs.
- **The manifest never records `source_path`**: an absolute local path cannot
  reach `manifest.json` and break portability or the config digest.
- **`ArtifactRecord` paths are POSIX relative**: `as_posix()` keeps recorded
  paths stable across platforms.
- **Test-only imports are declared**: `pillow` is in the `dev` extras because
  the fixtures import `PIL`.
- **`look_at_quaternion` branches on the largest component**: the plan's
  `w = sqrt(1 + trace) / 2` extraction is degenerate at the default
  `n_images=24`, where ring index 6 gives `trace == -1.0` bit-exact.
- **Always pass `--eval-steps <max_steps>`**: gsplat's eval branch carries no
  `or step == max_steps - 1` fallback, unlike its checkpoint and ply branches,
  so any run shorter than 7000 steps produces no metrics at all.
- **gsplat pads the stats filename but not the ply filename**: the two
  spellings coincide at step 6999, which hides the difference on the truck run.
- **Errors are translated at the boundary, not caught broadly in `main`**.
- **Configs may carry a UTF-8 BOM**: `from_toml` decodes with `utf-8-sig`.
- **No configurable seed**: gsplat's trainer hardcodes `42 + local_rank`.
- **Validate both image folders**: COLMAP names refer to `images/`, while
  training may consume `images_<factor>/`.
- **`out/` is ignored, anchored as `/out/`**: a run writes roughly 700 MB there
  including two 236 MB plys.
- **`requirements.lock.txt` carries a hand-maintained `-e .` line**: pip
  resolves an editable install inside a Git checkout to that checkout's remote,
  which here is a private SSH URL pinned to the last pushed commit. Regenerate
  the dependency list with `pip freeze --exclude-editable` and leave the
  editable line alone.
- **`plyfile` stays `>=1.0` in `pyproject.toml` and `==1.1.3` in the lock**:
  the abstract floor and the concrete pin are different jobs. The version
  cannot affect the byte target, because gsplat writes the `.ply` through its
  own `splat2ply_bytes` and never imports plyfile.
- **`env.bat` sets `DISTUTILS_USE_SDK=1`**: torch's `cpp_extension._check_abi`
  raises when it sees an activated VC environment without it, so no CUDA
  extension builds at all.
- **`resolve_target` locates a patch target without importing it**: importing
  `gsplat.cuda._backend` runs gsplat's `__init__`, which JIT compiles, which
  fails with the exact error the patch prevents. It uses
  `importlib.util.find_spec` on the top-level package, never on the dotted
  name, because a dotted name imports the parent and restores the deadlock.
- **Provision from `requirements.lock.txt`, not gsplat's example
  requirements**: the example file pulls `fused-bilagrid`, which does not
  compile under MSVC and which this project has never had installed.

## Milestone 1, closed

Milestone 1 turned the manual feasibility spike into one reproducible command:
`splatpipe run <scene-dir> --config <cfg> --out <dir>`. All 11 tasks passed
their review gates, the branch was reviewed again as a whole, and it merged
into `master` on 2026-09-01 as a fast-forward, so history is linear and every
task commit is preserved. Full detail is in
`.superpowers/sdd/2026-08-26-milestone-1-scripted-pipeline/progress.md`.

**The pipeline reproduces the spike on the real truck scene.** This was the
milestone's falsification criterion. It passed on the tolerance set before the
run, with no expected value adjusted afterwards, and an independent reviewer
re-derived every number from disk.

| Measurement | Pipeline | Spike | Delta |
|---|---:|---:|---:|
| PSNR | 24.394817 | 24.406155 | -0.0113 dB |
| SSIM | 0.8579996 | 0.8580129 | -0.0000133 |
| LPIPS | 0.1375573 | 0.1372071 | +0.00035 |
| Gaussians | 1,000,000 | 1,000,000 | 0 |
| `.ply` bytes | 236,001,478 | 236,001,478 | exact |
| `.splat` bytes | 32,000,000 | n/a | 32 x N |

Training took 8.03 minutes, export 3.1 seconds. The tolerance was 0.1 dB PSNR.

**The encoder is proven unchanged across the milestone.** Re-encoding the
spike's own `point_cloud_6999.ply` through today's `read_ply` and
`encode_splat` reproduces
`3fd1a045aed8498eae1b11b900d12c9941c64791f05b617e0306c2c4e09926e3`, the hash
recorded weeks earlier. The new run's `.splat` differs only because CUDA
reductions are not bit-reproducible at a fixed seed. The reviewer reproduced
this control independently.

**The provisioning sequence in `README.md` is verified, not reconstructed.** It
was run against an empty directory on 2026-08-31. All seven steps succeed, both
patches apply to the fresh venv's own site-packages, and that venv passed the
complete tier in 489 seconds including the first-time CUDA compile. Running it
found three defects that reading had not.

## Next Steps

1. **(P0) Task 3 of the milestone 2 plan**: the codec protocol, `PlyCodec` and
   `SplatCodec`. BASE `bb7eb28`. Then tasks 4 through 11, one per turn.
2. **(P1) After Task 11**: whole-branch review on the most capable model, then
   `superpowers:finishing-a-development-branch`. Ved makes the merge call.
3. **(P1) Decide whether phone captures gain a COLMAP stage around milestone
   6.5, or whether the three deployment scenes come from public datasets.**
   This is an owner decision for Ved and is the one spec success criterion
   nothing in milestones 1 to 8 currently schedules.
4. **(P2) Revisit Morton tie-breaking before milestone 5**, if the container
   format is going to claim byte parity with gsplat.
5. **(P2) Consider the four accepted milestone 1 Minors closed** unless
   something changes.
6. **(P3) `milestone-1-scripted-pipeline` still exists locally and on the
   remote**, pointing at the same commit as `master`. It is kept as the record
   and is safe to delete whenever Ved wants. It has not been deleted because
   nobody asked.

## Context

- **Branches**: `master` at `4528129`, pushed, carrying all of milestone 1.
  `milestone-2-bench` at `bb7eb28`, unpushed, carrying the milestone 2 spec
  `0c04165`, the plan `d11e425`, and tasks 1 and 2. The spec and the plan exist
  only on this branch, so a reader on `master` cannot see them yet.
  `milestone-1-scripted-pipeline` at `9f7f3a1`, kept as a record. There has
  never been a pull request. GitHub SSH and `gh` access work as user
  `VedJoshi`.
- **Authority for milestone 2**, in order: this file, then
  `docs/superpowers/specs/2026-09-01-milestone-2-bench-design.md`, then
  `docs/superpowers/plans/2026-09-01-milestone-2-bench.md` and its **Global
  Constraints** section, then the git-ignored ledger at
  `.superpowers/sdd/2026-09-01-milestone-2-bench/progress.md`. The ledger holds
  the pre-flight scan table and Rulings 1 through 12, each with its cost if
  wrong. Trust the ledger and `git log` over anything remembered.
- **The project spec** at
  `docs/superpowers/specs/2026-08-25-splat-compression-pipeline-design.md`
  remains the binding authority above every milestone document.
- **Independent audit**: `REVIEWER_AGENT.md` holds a self-contained prompt for
  a fresh review agent. It is read-only by contract. Its first step is checking
  that the ledger matches `git log`, which is the cheapest way to catch an
  implementing agent that lost its place. On milestone 1 Task 11 that step
  caught a commit the ledger had not yet recorded.
- **The truck run lives at `out/truck/`** and is git-ignored. `manifest.json`
  there records config digest `c221a1fc2180`, `splatpipe_commit 01dfeaa`, and
  both artifacts with sizes and SHA-256 hashes. Reproducing it costs about 8
  minutes of GPU time.
- **Files changed on this branch so far**: `requirements.lock.txt`,
  `src/splatpipe/formats/splat.py`, `tests/test_package.py`,
  `tests/test_splat_format.py`, and the two milestone 2 documents.
- **Commands to resume**:

  ```powershell
  cd "C:\Users\vedti\NUS_CS(noOnedrive)\gaussian-splat-compression"
  git fetch --prune origin
  git status --short --branch
  git log --oneline -8
  .venv\Scripts\python.exe -m pytest -q
  .venv\Scripts\python.exe scripts\setup_env.py --check
  cmd /c "call scripts\env.bat >nul 2>&1 && .venv\Scripts\python.exe -m pytest -q -o addopts="
  ```

  Expected at `bb7eb28`: clean tree, `126 passed, 3 deselected` fast,
  `129 passed` complete, `pinned` plus two `applied`.

- **Command caveats**: run `cmd /c` lines through PowerShell, never through Git
  Bash. Use `-o addopts=` for the complete suite; `-m gpu` runs only the
  GPU-marked tests. PowerShell strips quotes from arguments passed to
  `python -c`, so run a scratch `.py` file instead of an inline snippet. Bash
  heredocs mangle backslashes in Windows paths and fail outright on large
  Markdown documents; use the file-writing tools for both.
- **Open questions**: how phone captures get COLMAP poses; whether decode time
  belongs on the rate-distortion curve at all, recorded for now because it is
  nearly free to measure and load time is a stated success criterion for the
  milestone 7 viewer.
