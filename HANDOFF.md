# Handoff: Milestone 1 scripted splat pipeline: 2026-08-28

## Goal

Turn the manual Gaussian-splatting feasibility spike into one reproducible
command: `splatpipe run <scene-dir> --config <cfg> --out <dir>`. Milestone 1
must train a COLMAP scene and emit a `.ply`, a `.splat`, and a manifest that
records exactly what produced them, providing the foundation for later
rate-distortion compression research.

## Completed

- Tasks 1 through 7 of the 11-task milestone plan are implemented, committed,
  and through their review gates on `milestone-1-scripted-pipeline`.
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
- The real truck PLY encodes to exactly 32,000,000 bytes. Its Morton-ordered
  SHA-256 is
  `3fd1a045aed8498eae1b11b900d12c9941c64791f05b617e0306c2c4e09926e3`.
- The repository and GitHub remote are named `gaussian-splat-compression`; the
  package and planned CLI remain `splatpipe`.
- `da1c671` replaces the default branch's spike-style README with a concise
  project overview. `458686e` merges that documentation forward into the
  milestone branch. `35daec6` records the remote and merge policy.
- The milestone branch was assessed for merging and deliberately kept separate
  from `master`: its stated end-to-end success criterion is still unfinished.
- Both branches were pushed atomically. Before this handoff-only refresh,
  `origin/master == da1c671` and
  `origin/milestone-1-scripted-pipeline == 35daec6`.
- Verification after the README merge: `76 passed, 1 deselected` in the fast
  tier; `77 passed, 1 warning` in the complete tier. The warning is the known
  pycolmap `np.uint64(-1)` deprecation warning.
- Live environment verification reports the gsplat checkout pinned at
  `937e29912570c372bed6747a5c9bf85fed877bae` and both required patches
  `applied`.

## In Progress

- Task 8 is the first unfinished task. It adds `manifest.json` containing the
  resolved configuration, environment, versions, timings, metrics, Git commit,
  and content-addressed artifact metadata.
- No Task 8 implementation has started. Its complete brief is already at
  `.superpowers/sdd/2026-08-26-milestone-1-scripted-pipeline/task-8-brief.md`.
- Tasks 9 through 11 are also unstarted. Their briefs are already extracted in
  the same directory: synthetic COLMAP fixture, training stage and CLI, then
  real truck reproduction and milestone cleanup.

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
- **No configurable seed**: gsplat's trainer hardcodes `42 + local_rank`, so a
  config field would claim control it does not have.
- **Validate both image folders**: COLMAP names refer to `images/`, while
  training may consume `images_<factor>/`; both need the minimum image count.
- **Style is binding**: no emoji, em dashes, litotes, irony, or exclamation
  marks. Use plain declarative prose and comment only non-obvious reasoning.

## Next Steps

1. **(P0) Implement and review Task 8 only.** Read the design, plan, ledger, and
   `task-8-brief.md`; perform its preflight, work test-first, commit with the
   prescribed message and trailers, run a dedicated review, resolve all
   Critical and Important findings, update the ledger, then stop.
2. **(P1) Implement Tasks 9 through 11 in order, one reviewed task per user
   turn.** Task 9 writes explicit-width little-endian COLMAP binaries; Task 10
   introduces the first full CLI path; Task 11 runs the real truck scene.
3. **(P1) Treat Task 11 as falsification.** Expected truck values are PSNR
   24.406, SSIM 0.8580, LPIPS 0.1372, and `.ply` size 236,001,478 bytes. A PSNR
   difference greater than about 0.1 requires investigation.
4. **(P2) Resolve the lock-file, `plyfile`, two Task 7 Minor, and root spike-file
   cleanup items during the final branch review and Task 11 documentation pass.
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
- **Open questions**: how phone captures get COLMAP poses; whether gsplat
  `PngCompression` leaves enough rate-distortion headroom to beat; final
  disposition of the two Task 7 Minor edge cases.
