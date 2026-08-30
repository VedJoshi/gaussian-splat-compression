# Handoff: Milestone 1 scripted splat pipeline: 2026-08-30 (Task 10 complete)

## Goal

Turn the manual Gaussian-splatting feasibility spike into one reproducible
command: `splatpipe run <scene-dir> --config <cfg> --out <dir>`. Milestone 1
must train a COLMAP scene and emit a `.ply`, a `.splat`, and a manifest that
records exactly what produced them, providing the foundation for later
rate-distortion compression research.

## Completed

- Tasks 1 through 10 of the 11-task milestone plan are implemented, committed,
  and through their review gates on `milestone-1-scripted-pipeline`.
- **The pipeline runs end to end.** `splatpipe run <scene-dir> --config <cfg>
  --out <dir>` takes a COLMAP directory and a config and produces a trained
  `.ply`, a `.splat`, a copied config and a manifest. Only the real truck
  reproduction remains.
- Task 1, `dfef248`: package skeleton, error hierarchy, editable install, and
  dependency record.
- Task 2, `4d77440`: aggregate Windows compiler and CUDA environment preflight.
- Task 3, `69cbba9`, `446701e`, `78817e4`: idempotent gsplat and pycolmap
  patching, including fixes for false applied-state detection and substring
  matching that had corrupted an already-patched live file.
- Task 4, `9859804`, `1d2f7e6`: frozen run configuration, TOML loading, strict
  validation, and stable parameter digests.
- Task 5, `4827973`, `0468db0`: aggregate COLMAP scene validation and the
  deterministic run output layout.
- Task 6, `e6b7c60`: validated numpy `GaussianCloud` plus channel-correct 3DGS
  `.ply` reading and writing. The real truck PLY reads as 1,000,000 Gaussians
  at SH degree 3 with `shN.shape == (1000000, 15, 3)`.
- Task 7, `8a5e8ed`, `5ce9388`, `05f64c0`: numpy-only 32-byte `.splat`
  encoding with Morton, size-opacity, and input ordering. It was reviewed clean
  after two fix rounds, with two Minor edge cases deferred.
- Task 8, `b5e4328`: `RunManifest` and `ArtifactRecord`, recording the resolved
  configuration, its digest, versions, the Git commit, timings, metrics, and
  content-addressed artifact entries. Reviewed clean on the first pass with no
  Critical or Important findings and two Minor edge cases deferred.
- Task 10, `bbbe8fe`, `14b65fa`: the training stage and the `splatpipe run`
  CLI. Training shells out to gsplat's `simple_trainer.py`; the CLI validates
  the scene, trains or reuses `train/`, copies the config, exports both
  artifacts, and writes the manifest. Reviewed after one fix round, with three
  Minor edge cases deferred. `--skip-train` re-exports without retraining: 2.7
  seconds against 71.3 seconds of training on the tiny scene.
- Task 9, `94b80f6`: the synthetic COLMAP scene fixture. `tests/fixtures/`
  writes `cameras.bin`, `images.bin`, and `points3D.bin` by hand with explicit
  little-endian struct widths, and `make_tiny_scene` builds a 24-image ring
  scene that generates in about a second. The real installed pycolmap reads
  every file back. Reviewed clean on the first pass with two Minor edge cases
  deferred.
- The real truck PLY encodes to exactly 32,000,000 bytes. Its Morton-ordered
  SHA-256 is
  `3fd1a045aed8498eae1b11b900d12c9941c64791f05b617e0306c2c4e09926e3`.
- `ArtifactRecord.of` on the same real truck PLY reports 236,001,478 bytes,
  which is the exact size Task 11 must reproduce, with chunked SHA-256
  `afeeb5b192475f208e8ad301fd3181f9d0229979ea43708f0cdc92bfbd1ae3c1` equal to
  its whole-file hash.
- The repository and GitHub remote are named `gaussian-splat-compression`; the
  package and planned CLI remain `splatpipe`.
- `da1c671` replaces the default branch's spike-style README with a concise
  project overview. `458686e` merges that documentation forward into the
  milestone branch. `35daec6` records the remote and merge policy.
- The milestone branch was assessed for merging and deliberately kept separate
  from `master`: its stated end-to-end success criterion is still unfinished.
- Verification after Task 10: `95 passed, 3 deselected` in the fast tier;
  `98 passed, 1 warning` in the complete tier, in 91 seconds. The warning is the
  known pycolmap `np.uint64(-1)` deprecation warning. The complete tier trains
  the real gsplat trainer twice, once per end-to-end test.
- Live environment verification reports the gsplat checkout pinned at
  `937e29912570c372bed6747a5c9bf85fed877bae` and both required patches
  `applied`.

## In Progress

- Task 11 is the first unfinished task, and it is the last one. It runs the
  real truck scene through the new CLI and compares the result against the
  spike. Its complete brief is at
  `.superpowers/sdd/2026-08-26-milestone-1-scripted-pipeline/task-11-brief.md`.
- No Task 11 implementation has started.
- Task 11 is falsification, not a formality. Expected values are PSNR 24.406,
  SSIM 0.8580, LPIPS 0.1372, and a `.ply` of exactly 236,001,478 bytes. The
  `.ply` size must match exactly, since it is a function of Gaussian count and
  field list. Metrics should match to about two decimal places: the seed is
  fixed upstream at 42, but CUDA reductions are not bit-reproducible. A PSNR
  differing by more than about 0.1 means something is genuinely different and
  must be investigated before the milestone is called done.
- Task 11 also carries the milestone cleanup: the `requirements.lock.txt` SSH
  remote, the `plyfile` version pin, the nine deferred Minor findings, and the
  spike files still sitting in the repository root.
- If this section names a task as under way and the ledger has no matching
  `Task N: complete` line, that task did not finish. Read the ledger at
  `.superpowers/sdd/2026-08-26-milestone-1-scripted-pipeline/progress.md`
  before changing anything.

## Not Working / Blockers

- Nothing is currently broken. The repository, venv, GPU test tier, gsplat
  checkout, and both installed patches are healthy.
- `requirements.lock.txt` is a record, not a replayable installer. Its editable
  package line uses the private SSH remote. Resolve this in Task 11.
- `plyfile` is declared as `>=1.0` and resolved to 1.1.3, while the original
  spike used 1.1.5. Decide and pin the intended version in Task 11.
- `_anchor_positions` deliberately treats indentation-free pycolmap anchors as
  substrings. Its uniqueness guard handles the current patches, but a future
  patch could theoretically alias inside a larger expression.
- Task 7 deferred Minor: an invalid cloud combined with an unknown order raises
  `ArtifactError` before `ConfigError` because cloud validation runs first.
- Task 7 deferred Minor: an empty cloud with Morton ordering raises NumPy's raw
  zero-size reduction `ValueError`. Training cannot produce an empty final
  cloud.
- Task 8 deferred Minor: `collect_versions` catches bare `Exception` around the
  torch and gsplat imports, so a broken install records `not-imported`
  identically to an absent one. The field is informational.
- Task 8 deferred Minor: no test covers the `_git_commit` fallback branch or
  `ArtifactRecord.of` called with a path outside `relative_to`, which raises
  `ValueError` from `Path.relative_to`.
- Task 9 deferred Minor: `write_images_bin` and `write_points3D_bin` annotate
  their inputs as `Sequence[tuple]` with an untyped inner tuple, unlike
  `write_cameras_bin`. Inherited from the brief, test-only fixture code.
- Task 9 deferred Minor: `write_images_bin` trusts `len(xys) == len(point3D_ids)`
  without asserting it. A mismatched caller would make `zip` truncate silently
  while the already-written `count2D` header still claimed the longer length,
  producing a corrupt file with no write-time error. The invariant holds for the
  only caller.
- Task 10 deferred Minor: `tests/test_pipeline_e2e.py` imports `ExportConfig`
  and `TrainConfig` without using either. Inherited from the brief.
- Task 10 deferred Minor: no CPU test exercises `run_pipeline`'s orchestration,
  so stage ordering and the `--skip-train` `ArtifactError` path are covered only
  by the two GPU-marked tests. Widening this needs a `run_training` stub.
- Task 10 deferred Minor: the non-skip-train branch does not check
  `trained_ply.is_file()` before copying it, so a training that somehow omitted
  the final `.ply` would raise a bare `FileNotFoundError`. Unreachable in
  practice, since `--save-ply` is always passed and the trainer saves at
  `max_steps - 1` unconditionally.
- Phone captures still have no scheduled pose-estimation stage. This blocks the
  later three-scene deployment milestone, not milestone 1.

## Key Decisions

- **One task per user turn**: implement one task, run its review gate, record the
  outcome, then stop. Do not dispatch or begin the next task while review is
  open.
- **Use the milestone branch in the existing checkout**: `.venv/`,
  `_gsplat_repo/`, `data/`, and `results/` are untracked and bound to this repo
  root. A separate worktree lacks the 15 GB working environment.
- **Routine pushes are authorized**: Ved authorized remote pushes and
  default-branch documentation updates on 2026-08-28. The incomplete milestone
  branch must not merge into `master` until all 11 tasks and the real truck
  reproduction pass.
- **The default branch is named `master`**: there is no `main` branch. Do not
  create or rename branches merely to normalize the name.
- **The venv is mandatory**: use `.venv\Scripts\python.exe`; bare `python` and
  `py` resolve to Python 3.13, while the project requires Python 3.11.
- **GPU commands must chain `env.bat`**: tool calls do not retain shell state.
  Run setup and GPU tests inside the same `cmd /c` invocation.
- **Do not upgrade the working stack**: PyTorch 2.7.1+cu128, CUDA 12.8, numpy
  1.26.4, gsplat 1.5.3, and MSVC 14.29 are the verified combination.
- **Do not hand-edit site-packages**: `scripts/setup_env.py` owns both patches.
  A `stale` status means the patch must be re-derived, never forced.
- **Keep the accepted anchor rule**: anchors starting with whitespace must
  begin at a line boundary; indentation-free anchors match as substrings. Strict
  whole-line matching would fail all five pycolmap anchors at fresh-install
  nesting depths.
- **Invalid Gaussian rows fail export**: do not mirror gsplat's silent filtering.
  Task 10 requires `.splat` size to equal `32 * len(cloud)`.
- **Separate layout authority from `exp` implementation**: NumPy and Torch CPU
  `exp` differ by up to 2 ULP on random float32 inputs. Exact full-record tests
  use zero log-scales; random scales compare within 2 ULP while every other byte
  remains exact.
- **The manifest never records `source_path`**: `RunConfig.to_dict()` pops it,
  so an absolute local path cannot reach `manifest.json` and break portability
  or the config digest.
- **`ArtifactRecord` paths are POSIX relative**: `as_posix()` keeps recorded
  paths stable across platforms, which is what Task 10 asserts.
- **Test-only imports are declared**: `pillow` is in the `dev` optional
  dependencies because the fixtures import `PIL`. The CPU fast tier must not
  depend on a package that arrived transitively through torchvision.
- **`look_at_quaternion` branches on the largest component**: the plan's
  `w = sqrt(1 + trace) / 2` extraction is degenerate at the default
  `n_images=24`, where ring index 6 gives `trace == -1.0` bit-exact and the
  plan's own guard raises. The standard branch-by-largest-component
  construction has no degenerate branch for a proper rotation. Reverting it
  fails all three Task 9 tests, so the fix is pinned by construction.
- **Always pass `--eval-steps <max_steps>`**: gsplat evaluates only at
  `step in [i - 1 for i in cfg.eval_steps]`, defaulting to `[7000, 30000]`.
  Unlike the checkpoint and ply branches it carries no `or step == max_steps - 1`
  fallback, so any run shorter than 7000 steps produces no metrics at all.
- **gsplat pads the stats filename but not the ply filename**: stats are
  `{stage}_step{step:04d}.json`, the ply is `point_cloud_{step}.ply`. The two
  spellings coincide at step 6999, which hides the difference on the truck run
  and only breaks below 1000 steps.
- **Errors are translated at the boundary, not caught broadly in `main`**:
  `from_toml` raises `ConfigError` naming the file, `run_training` raises
  `ArtifactError` naming the exit code and the log path. Each message says what
  to do next, which a widened `except` clause could not.
- **Configs may carry a UTF-8 BOM**: `from_toml` decodes with `utf-8-sig`.
  Windows is this project's only platform, and PowerShell and several Windows
  editors emit a BOM by default.
- **No configurable seed**: gsplat's trainer hardcodes `42 + local_rank`, so a
  config field would claim control it does not have.
- **Validate both image folders**: COLMAP names refer to `images/`, while
  training may consume `images_<factor>/`; both need the minimum image count.
- **Style is binding**: no emoji, em dashes, litotes, irony, or exclamation
  marks. Use plain declarative prose and comment only non-obvious reasoning.

## Next Steps

1. **(P0) Implement and review Task 11, the last task.** Read the design, plan,
   ledger, and `task-11-brief.md`; perform its preflight, run the real truck
   scene through the CLI, compare against the spike, commit with the prescribed
   message and trailers, run a dedicated review, resolve all Critical and
   Important findings, update the ledger, then stop. Expect the training run
   itself to take several minutes on the 6 GB card.
2. **(P1) Treat Task 11 as falsification.** A PSNR difference greater than
   about 0.1 requires investigation before the milestone is called done. Do not
   adjust the expected values to match what comes out.
3. **(P2) Resolve the lock-file, `plyfile`, the nine deferred Minor findings,
   and root spike-file cleanup during the final branch review and Task 11
   documentation pass.
5. **(P2) Decide whether phone captures gain a COLMAP stage around milestone
   6.5 or whether the three deployment scenes come from public datasets. This
   is an owner decision for Ved.

## Context

- **Branch(es)**: active
  `milestone-1-scripted-pipeline`; default `master`. The milestone branch forked
  from `master` at `a70668b`, contains the default README merge, and remains
  intentionally unmerged. There is no pull request. GitHub SSH and `gh` access
  work as user `VedJoshi`.
- **Authority**: read `HANDOFF.md`, then
  `docs/superpowers/specs/2026-08-25-splat-compression-pipeline-design.md`, then
  `docs/superpowers/plans/2026-08-26-milestone-1-scripted-pipeline.md`, then the
  ignored ledger at
  `.superpowers/sdd/2026-08-26-milestone-1-scripted-pipeline/progress.md`.
- **Independent audit**: `REVIEWER_AGENT.md` holds a self-contained prompt for a
  fresh review agent. Paste it into a new session, optionally with a scope such
  as "review the last commit" or "review the whole branch against master". It is
  read-only by contract: the reviewer reports findings and never edits, commits,
  pushes, or mutates GitHub state. Its first step is checking that the ledger
  matches `git log`, which is the cheapest way to catch an implementing agent
  that lost its place.
- **Key files changed**: `README.md`, `HANDOFF.md`, `RESUME_PROMPT.md`,
  `pyproject.toml`, `configs/truck.toml`, `scripts/setup_env.py`,
  `scripts/patches/`, `src/splatpipe/`, and `tests/`.
- **Commands to resume**:

  ```powershell
  cd "C:\Users\vedti\NUS_CS(noOnedrive)\gaussian-splat-compression"
  git fetch --prune origin
  git status --short --branch
  git log --oneline -8
  .venv\Scripts\python.exe -m pytest -q
  cmd /c "call scripts\env.bat >nul 2>&1 && .venv\Scripts\python.exe scripts\setup_env.py --check"
  cmd /c "call scripts\env.bat >nul 2>&1 && .venv\Scripts\python.exe -m pytest -q -o addopts="
  ```

- **Command caveat**: run `cmd /c` lines through PowerShell. Use
  `-o addopts=` for the complete suite; `-m gpu` runs only the GPU-marked test.
  PowerShell strips quotes from arguments passed to `python -c`, so run a
  scratch `.py` file instead of an inline snippet.
- **Open questions**: how phone captures get COLMAP poses; whether gsplat
  `PngCompression` leaves enough rate-distortion headroom to beat; final
  disposition of the nine deferred Minor edge cases from Tasks 7 through 10.
