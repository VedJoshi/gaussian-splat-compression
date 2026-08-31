# Handoff: Milestone 1 scripted splat pipeline: 2026-08-31 (all 11 tasks complete)

## Goal

Turn the manual Gaussian-splatting feasibility spike into one reproducible
command: `splatpipe run <scene-dir> --config <cfg> --out <dir>`. Milestone 1
must train a COLMAP scene and emit a `.ply`, a `.splat`, and a manifest that
records exactly what produced them, providing the foundation for later
rate-distortion compression research.

**This goal is met.** The remaining work on this branch is the broad
whole-branch review and the merge decision, not implementation.

## Completed

- All 11 tasks of the milestone plan are implemented, committed, and through
  their review gates on `milestone-1-scripted-pipeline`.
- **The pipeline reproduces the spike on the real truck scene.** This was the
  milestone's falsification criterion and it passed on the tolerance set before
  the run, with no expected value adjusted afterwards.

  | Measurement | Pipeline | Spike | Delta |
  |---|---:|---:|---:|
  | PSNR | 24.394817 | 24.406155 | -0.0113 dB |
  | SSIM | 0.8579996 | 0.8580129 | -0.0000133 |
  | LPIPS | 0.1375573 | 0.1372071 | +0.00035 |
  | Gaussians | 1,000,000 | 1,000,000 | 0 |
  | `.ply` bytes | 236,001,478 | 236,001,478 | exact |
  | `.splat` bytes | 32,000,000 | n/a | 32 x N |

  Training took 8.03 minutes, export 3.1 seconds. The tolerance was 0.1 dB PSNR.
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
  `.ply` reading and writing.
- Task 7, `8a5e8ed`, `5ce9388`, `05f64c0`: numpy-only 32-byte `.splat` encoding
  with Morton, size-opacity, and input ordering. Clean after two fix rounds.
- Task 8, `b5e4328`: `RunManifest` and `ArtifactRecord`. Clean on the first pass.
- Task 9, `94b80f6`: the synthetic COLMAP scene fixture, written by hand with
  explicit little-endian struct widths. Clean on the first pass.
- Task 10, `bbbe8fe`, `14b65fa`: the training stage and the `splatpipe run` CLI.
  Clean after one fix round. `--skip-train` re-exports without retraining.
- Task 11, `c68e0c3` and its fix commit: the real truck reproduction, the spike
  script cleanup, the lock-file fix, and the documentation pass. Reviewed with
  2 Critical, 3 Important and 7 Minor findings, all resolved.
- **The encoder is proven unchanged across the milestone.** Re-encoding the
  spike's own `point_cloud_6999.ply` through today's `read_ply` and
  `encode_splat` reproduces
  `3fd1a045aed8498eae1b11b900d12c9941c64791f05b617e0306c2c4e09926e3`, the hash
  recorded weeks earlier. The new run's `.splat` differs only because CUDA
  reductions are not bit-reproducible at a fixed seed, not because any code
  moved. The reviewer reproduced this control independently.
- Verification at the close of Task 11: `95 passed, 3 deselected` in the fast
  tier; `98 passed, 1 warning` in the complete tier. The warning is the known
  pycolmap `np.uint64(-1)` deprecation. The gsplat checkout reports `pinned`
  and both patches `applied`.
- `da1c671` replaced the default branch's spike-style README with a project
  overview. `458686e` merged that forward. `35daec6` records the remote and
  merge policy.

## In Progress

- Nothing is under implementation. Task 11 was the last task.
- If this section names a task as under way and the ledger has no matching
  `Task N: complete` line, that task did not finish. Read the ledger at
  `.superpowers/sdd/2026-08-26-milestone-1-scripted-pipeline/progress.md`
  before changing anything.

## Not Working / Blockers

- Nothing is broken. The repository, venv, GPU test tier, gsplat checkout, and
  both installed patches are healthy.
- **The nine deferred Minor findings from Tasks 7 through 10 are still open.**
  They were deliberately left to the whole-branch review rather than folded
  into Task 11, whose job was falsification:
  - Task 7: an invalid cloud combined with an unknown order raises
    `ArtifactError` before `ConfigError`, because cloud validation runs first.
  - Task 7: an empty cloud with Morton ordering raises NumPy's raw zero-size
    reduction `ValueError`. Training cannot produce an empty final cloud.
  - Task 8: `collect_versions` catches bare `Exception` around the torch and
    gsplat imports, so a broken install records `not-imported` identically to
    an absent one. The field is informational.
  - Task 8: no test covers the `_git_commit` fallback branch, or
    `ArtifactRecord.of` called with a path outside `relative_to`.
  - Task 9: `write_images_bin` and `write_points3D_bin` annotate their inputs
    as `Sequence[tuple]` with an untyped inner tuple, unlike `write_cameras_bin`.
  - Task 9: `write_images_bin` trusts `len(xys) == len(point3D_ids)` without
    asserting it. A mismatched caller would make `zip` truncate silently while
    the already-written `count2D` header still claimed the longer length.
  - Task 10: `tests/test_pipeline_e2e.py` imports `ExportConfig` and
    `TrainConfig` without using either.
  - Task 10: no CPU test exercises `run_pipeline`'s orchestration, so stage
    ordering and the `--skip-train` `ArtifactError` path are covered only by the
    two GPU-marked tests. Widening this needs a `run_training` stub.
  - Task 10: the non-skip-train branch does not check `trained_ply.is_file()`
    before copying, so a training that somehow omitted the final `.ply` would
    raise a bare `FileNotFoundError`. Unreachable in practice.
- `_anchor_positions` deliberately treats indentation-free pycolmap anchors as
  substrings. Its uniqueness guard handles the current patches, but a future
  patch could theoretically alias inside a larger expression.
- The provisioning sequence now in `README.md` is reconstructed from
  `SPIKE_LOG.txt` and **has not been re-run on a clean machine**. It is the best
  available account, not a verified one. Verifying it needs a second machine or
  a throwaway venv, and is the strongest remaining threat to the reproducibility
  claim.
- Phone captures still have no scheduled pose-estimation stage. This blocks the
  later three-scene deployment milestone, not milestone 1.

## Key Decisions

- **One task per user turn**: implement one task, run its review gate, record
  the outcome, then stop. Do not dispatch or begin the next task while review
  is open.
- **Use the milestone branch in the existing checkout**: `.venv/`,
  `_gsplat_repo/`, `data/`, `results/` and `out/` are untracked and bound to
  this repo root. A separate worktree lacks the 15 GB working environment.
- **Routine pushes are authorized**: Ved authorized remote pushes and
  default-branch documentation updates on 2026-08-28. **The merge into `master`
  is a separate decision and has not been made.** The branch's stated success
  criterion is now met, so the merge is unblocked on the merits but still
  Ved's call.
- **The default branch is named `master`**: there is no `main` branch.
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
  including two 236 MB plys, and Task 11's own brief ended in `git add -A`,
  which would have produced a commit GitHub rejects at push.
- **`requirements.lock.txt` carries a hand-maintained `-e .` line**: pip
  resolves an editable install inside a Git checkout to that checkout's remote,
  which here is a private SSH URL pinned to the last pushed commit. Regenerate
  the dependency list with `pip freeze --exclude-editable` and leave the
  editable line alone. Reinstalling does not fix this; `direct_url.json`
  already records a correct local path.
- **`plyfile` stays `>=1.0` in `pyproject.toml` and `==1.1.3` in the lock**:
  the abstract floor and the concrete pin are different jobs. The version
  cannot affect the byte target, because gsplat writes the `.ply` through its
  own `splat2ply_bytes` and never imports plyfile. Closed, not deferred.
- **Style is binding**: no emoji, em dashes, litotes, irony, or exclamation
  marks. Use plain declarative prose and comment only non-obvious reasoning.

## Next Steps

1. **(P0) Run the broad whole-branch review**, `master..milestone-1-scripted-pipeline`.
   Paste `REVIEWER_AGENT.md` into a fresh session with that scope. The
   per-task reviews each saw one commit; nothing has yet read the branch as a
   whole.
2. **(P1) Clear the nine deferred Minors** listed above, or consciously accept
   each one. They are individually small and collectively the last code debt in
   the milestone.
3. **(P1) Decide the merge into `master`.** The success criterion is met, so
   the branch is mergeable on the merits. Use
   `superpowers:finishing-a-development-branch` once the whole-branch review is
   clean.
4. **(P2) Verify the provisioning sequence on a clean venv.** It is
   reconstructed, not tested, and it is the one documented path a second
   machine would follow.
5. **(P2) Decide whether phone captures gain a COLMAP stage around milestone
   6.5, or whether the three deployment scenes come from public datasets.**
   This is an owner decision for Ved and is the one spec success criterion
   nothing in milestones 1 to 8 currently schedules.

## Context

- **Branch(es)**: active `milestone-1-scripted-pipeline`; default `master`. The
  milestone branch forked from `master` at `a70668b`, contains the default
  README merge, and remains unmerged. There is no pull request. GitHub SSH and
  `gh` access work as user `VedJoshi`.
- **Authority**: read `HANDOFF.md`, then
  `docs/superpowers/specs/2026-08-25-splat-compression-pipeline-design.md`, then
  `docs/superpowers/plans/2026-08-26-milestone-1-scripted-pipeline.md`, then the
  ignored ledger at
  `.superpowers/sdd/2026-08-26-milestone-1-scripted-pipeline/progress.md`.
- **Independent audit**: `REVIEWER_AGENT.md` holds a self-contained prompt for a
  fresh review agent. It is read-only by contract. Its first step is checking
  that the ledger matches `git log`, which is the cheapest way to catch an
  implementing agent that lost its place. On Task 11 that step worked: it caught
  a commit the ledger had not yet recorded.
- **The run output lives at `out/truck/`** and is git-ignored. `manifest.json`
  there records config digest `c221a1fc2180`, `splatpipe_commit 01dfeaa`, and
  both artifacts with sizes and SHA-256 hashes.
- **Key files changed**: `README.md`, `HANDOFF.md`, `RESUME_PROMPT.md`,
  `REVIEWER_AGENT.md`, `pyproject.toml`, `requirements.lock.txt`,
  `configs/truck.toml`, `scripts/`, `src/splatpipe/`, and `tests/`.
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

- **Command caveats**: run `cmd /c` lines through PowerShell. Use
  `-o addopts=` for the complete suite; `-m gpu` runs only the GPU-marked tests.
  PowerShell strips quotes from arguments passed to `python -c`, so run a
  scratch `.py` file instead of an inline snippet. Bash heredocs mangle
  backslashes in Windows paths and fail outright on large Markdown documents;
  use the file-writing tools for both.
- **Open questions**: how phone captures get COLMAP poses; whether gsplat
  `PngCompression` leaves enough rate-distortion headroom to beat; whether the
  reconstructed provisioning sequence actually works on a clean machine.
