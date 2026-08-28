# Handoff: Milestone 1, scripted splat pipeline: 2026-08-28

## Working agreement

**One task at a time.** Implement a single task, take it through its review, record the outcome, then stop and report. Do not chain into the next task, and do not dispatch the next task's implementer while the current one is still under review. Ved has asked for this directly. The point is that he sees each task land and can redirect before the next one starts, which matters more here than throughput: this project runs on weekends with gaps of up to two weeks between sessions.

Push only when asked. Nothing touches `master` without asking.

## Goal

Turn the manual Gaussian-splatting spike in this repository into `splatpipe run <scene-dir> --config <cfg> --out <dir>`: one reproducible command that takes a COLMAP scene, trains it, and emits a `.ply`, a `.splat`, and a manifest recording exactly what produced them. This is milestone 1 of an eight-milestone portfolio project whose real subject is splat compression. The design doc is `docs/superpowers/specs/2026-08-25-splat-compression-pipeline-design.md` and the implementation plan is `docs/superpowers/plans/2026-08-26-milestone-1-scripted-pipeline.md`.

Work is being executed task by task by subagents, with a review after each, one task per session turn. Progress is tracked in a ledger, described under Context.

## Completed

Seven of eleven tasks are done. All commits are on branch `milestone-1-scripted-pipeline`, which forked from `master` at `a70668b`. The working tree is clean.

- **`a70668b` (on `master`)**: the implementation plan, 11 tasks, 76 steps.
- **`d9ca995`**: fixes for three defects found by scanning the plan against itself before any code was written. See Key Decisions.
- **`c2e6e2b`**: records the `env.bat` chaining recipe in the plan's constraints. See Key Decisions.
- **Task 1, `dfef248`**: package skeleton. `pyproject.toml` with an `src` layout, `src/splatpipe/__init__.py`, `src/splatpipe/errors.py` (`SplatpipeError` plus `BuildEnvError`, `ConfigError`, `SceneError`, `ArtifactError`), `tests/__init__.py`, `tests/test_package.py`, `requirements.lock.txt`. Installed editable. Reviewed clean.
- **Task 2, `4d77440`**: `src/splatpipe/env.py`, `check_build_env()`, which reports every build-environment problem at once instead of failing several layers inside PyTorch minutes into a run. `tests/test_env.py` injects a fake `which`, so it needs no compiler and no CUDA. Reviewed clean.
- **Task 3, `69cbba9`, `446701e`, then `78817e4`**: the two hand-edits inside `.venv/Lib/site-packages/` are now scripted and idempotent: `scripts/patches/`, `scripts/setup_env.py`. Verified against the real environment and reviewed clean after two fix rounds.
- **Task 4, `9859804`, then `1d2f7e6`**: frozen run configuration, strict validation, TOML loading, a stable 12-character parameter digest, and `configs/truck.toml`. Review found that integer fields accepted positive floats and booleans. The fix rejects both at the configuration boundary and passed a scoped re-review.
- **Task 5, `4827973`, then `0468db0`**: `src/splatpipe/scene.py` validates a COLMAP scene before any GPU time is spent, reporting every problem at once, and `src/splatpipe/paths.py` fixes the run output layout that every later task and milestone consumes. Review found that the folder deduplication was not pinned by any test, proved by mutation: removing it made the problem line appear twice and the suite stayed green, because `pytest.raises(match=...)` is a search. The fix asserts the number of problems reported. Reviewed clean after one fix round.
- **Task 6, `e6b7c60`**: `GaussianCloud`, a frozen numpy struct of arrays, plus validated 3DGS `.ply` reading and writing. Log-scales and logit-opacities remain untransformed, higher-order spherical harmonics preserve gsplat's channel-major disk layout, and the real million-Gaussian truck PLY reads as degree 3 with shape `(1000000, 15, 3)`. Reviewed clean with no findings.
- **Task 7, `8a5e8ed`, `5ce9388`, then `05f64c0`**: the numpy-only `.splat` writer with Morton, size-opacity and input ordering. Exact records are checked against gsplat with representable scales, randomized scale fields are bounded to 2 ULP, and numeric edge cases fail deliberately or saturate without warnings. The superseded `_ply_to_splat.py` is deleted. Reviewed clean after two fix rounds, with two Minors deferred.
- **numpy drift corrected.** The venv had drifted to 2.4.6 against gsplat's `numpy<2.0.0` requirement, so the environment did not match what `README.md` documented. Now 1.26.4, enforced by a test.
- **Repository renamed.** The local directory and private GitHub repository are now `gaussian-splat-compression`. The Python package and CLI remain `splatpipe`.
- **Suite state**: `76 passed, 1 deselected` in the fast tier; `77 passed, 1 warning` in the complete tier.

## What happened in the last session

Task 7 was implemented test-first, then completed two review fix rounds. The original all-random byte-equality test exposed a real cross-library limit: NumPy and Torch `exp` can differ in the final float32 bits. Production remains numpy-only. Full records compare exactly when log-scales are zero, while randomized scale fields must stay within 2 ULP and every other byte remains exact. Review then found overflow warnings in finite extreme opacity and size ordering; controller probing found the same issue in saturated SH colors. The fix uses log-domain size-opacity ordering and scoped saturation handling. A mutation check proves the large-scale regression test catches the old `Inf`-collapsed ordering.

The real truck cloud encodes as exactly 32,000,000 bytes for 1,000,000 Gaussians. Its Morton-ordered SHA-256 is `3fd1a045aed8498eae1b11b900d12c9941c64791f05b617e0306c2c4e09926e3`.

## In Progress

- **Task 8 is next.** It records the resolved configuration, environment, timings, metrics and content-addressed artifacts in `manifest.json`.
- **Tasks 8 to 11 are not started.** Briefs for all of them are already extracted into the workspace directory, so each dispatch is immediate. In order: 8 run manifest, 9 synthetic COLMAP scene fixture, 10 training stage and CLI, 11 reproduce the truck scene and update documentation.

## Not Working / Blockers

Nothing is broken. The environment and the working tree are both healthy. Five things are known and deliberately deferred.

- **`requirements.lock.txt` is not replayable.** Line 53 records the package as `-e git+ssh://git@github.com/VedJoshi/gaussian-splat-compression.git@d9ca995#egg=splatpipe`. That is pip's normal behaviour for an editable install inside a repo with a remote, but it means anyone without SSH keys for the private remote cannot install from the file, including you on a fresh machine. The file is a record rather than an installer, so nothing downstream breaks. Fix candidate is Task 11, which already touches documentation.
- **`plyfile` is pinned `>=1.0` and resolved to 1.1.3**, not the 1.1.5 the spike ran on. "Worked when frozen" and "resolves the same next install" are different guarantees. Deferred because tightening pins mid-milestone would invalidate the lock file just captured.
- **A residual weakness in anchor matching.** `_anchor_positions` in `scripts/patches/__init__.py` requires a line boundary only for anchors that *start with whitespace*. An anchor that does not (all five pycolmap ones) still matches as a plain substring, so it could in principle alias inside a longer expression. The `len(positions) == 1` ambiguity guard catches the realistic cases and the current anchors are unique, so this is theoretical. Worth a look if a third patch is ever added.
- **Task 7 order-error precedence.** `encode_splat` validates cloud contents before the order string, so an invalid cloud combined with an unknown order raises `ArtifactError` before `ConfigError`. Configuration normally arrives through validated `ExportConfig`; final branch review will decide whether precedence needs changing.
- **Task 7 empty-cloud behavior.** An empty cloud with Morton ordering reaches NumPy's minimum reduction and raises a raw `ValueError`. Training cannot produce an empty final cloud; final branch review will decide whether empty export returns `b""` or raises a deliberate `ArtifactError`.

Two things were broken during this session and are now fixed. Both are recorded because the reasoning matters more than the outcome.

- **`apply_patch` corrupted a live library file.** Matching used `text.count()` and `text.replace()`, which are substring operations. The gsplat anchor is a complete line at indent 8; the already-patched file holds that same code at indent 12 inside an `else:` branch, and a 12-space line is a 4-space prefix plus the 8-space line. The anchor matched inside it, the "appears exactly once" guard passed, and the replace spliced a nested `if/else` into the branch, producing an `IndentationError` in `gsplat/cuda/_backend.py`. The implementer had taken a backup first, caught it immediately, and restored. Fixed in `446701e`. Verified against the real file: the old check counts 1 and would corrupt again, the new one returns `[]` and refuses.
- **`patch_status` reported a patched file as unpatched.** It compared the full replacement text including its comment banner. The venv carries the spike's wording and the definition carries different wording, so a genuinely applied patch read as `appliable`. That false negative is what let the corruption bug fire. Applied-state evidence now lives per replacement, so comment wording cannot affect gsplat and all five pycolmap replacements must be present.

## Key Decisions

- **Branch, not a git worktree**: the toolchain is bound to untracked directories in the repo root (`.venv/` with two patched `site-packages` files, `_gsplat_repo/`, `data/`, `results/`, roughly 15 GB). A worktree would have none of them, so no task past Task 1 could run its tests there.
- **`tests/__init__.py` moved from Task 9 to Task 1**: under pytest's `prepend` import mode the directory added to `sys.path` is the first one above the test file lacking `__init__.py`. Without it that is `tests/`, and Task 3's `from scripts.patches import ...` would have failed six tasks before the file that fixes it.
- **`RunConfig.source_path` defined in Task 4, not added by Task 10**: the plan originally had Task 10 write a helper against a field that did not exist, then repair it a step later. A deliberate broken intermediate state costs a fix round and buys nothing.
- **No venv surgery to normalise the gsplat comment banner**: the installed patch is functionally identical to the canonical one and works. Only the comment differs. With the marker check in place the status reads `applied` and the file is correctly left alone. Editing a working library file to make a comment match is risk for no gain.
- **The implementer's anchor rule was accepted over the controller's**: the controller specified strict whole-line equality. The implementer showed that would never match the five pycolmap anchors, which deliberately omit indentation so they match at any nesting depth, and would have broken detection on any fresh install. Its rule (an anchor starting with whitespace must begin at a line boundary; one that does not matches as a substring) handles both anchor styles with a single test. Verified before accepting.
- **Export ordering is a config field from day one**: gsplat's `.splat` exporter sorts by Morton code while the spike sorted by size times opacity. Morton gives spatial locality, which is directly a compression lever for milestone 5, so both need to be measurable.
- **No `seed` config field**: `simple_trainer.py` calls `set_random_seed(42 + local_rank)` and exposes no flag. A seed setting would appear in the manifest and control nothing.
- **The image-count minimum applies to `images/` as well as `images_<factor>/`**: the plan checked only the folder actually trained on. COLMAP records its filenames against plain `images/`, and gsplat zips the two sorted listings then indexes the result by every COLMAP image name (`_gsplat_repo/examples/datasets/colmap.py`, `colmap_to_image`). A scene with a full `images_2/` and a thin `images/` passed the plan's check and would then have died inside gsplat on a bare `KeyError`.
- **Invalid Gaussians fail export rather than disappearing.** gsplat silently filters rows containing NaN or Inf. `encode_splat` rejects them, non-normalizable quaternions and unrepresentable scales with `ArtifactError`, preserving the later invariant that `.splat` size is exactly `32 * len(cloud)`.
- **The gsplat authority check separates layout from `exp` implementation.** Exact full-record equality uses zero log-scales. Random log-scales compare within 2 ULP while every other byte remains exact. Adding Torch to production solely for final-bit equality would violate the numpy-only boundary used by later compression work.
- **Repository name is `gaussian-splat-compression`**: it states the technical subject and portfolio claim directly. `splatpipe` remains the shorter command and package name.

## Next Steps

1. **(P0) Implement and review Task 8.** The brief is already extracted as `task-8-brief.md` and carries complete code for the run manifest.
2. **(P1) Continue Tasks 9 through 11 in order.** Briefs are pre-extracted. Use a standard model for these tasks because they need judgment across binary fixtures, training integration and real-scene reproduction.
3. **(P1) Task 11 is the falsification step.** It reruns the truck scene through the new CLI and compares against the spike: PSNR 24.406, SSIM 0.8580, LPIPS 0.1372, `.ply` exactly 236,001,478 bytes. The `.ply` size must match exactly. Metrics should match to about two decimal places, since CUDA reductions are not bit-reproducible. A PSNR differing by more than about 0.1 means something is genuinely different.
4. **(P2) Fold the two deferred dependency-record issues into Task 11**: the non-replayable `git+ssh` lock file line and the loose `plyfile` pin.
5. **(P2) Give the two unowned spike leftovers an owner in Task 11.** `git ls-files` still tracks `SPIKE_LOG.txt` and `_browser_test.png` at the repository root. Task 7 deleted `_ply_to_splat.py`; Task 11 already handles `_get_data.py`, `_gsplat_smoke.py` and `_vram_sampler.py`. Recruiters read this repository root.
6. **(P2) Decide how phone captures get their poses.** Nothing in milestones 1 to 8 schedules running COLMAP on a raw capture, but success criterion 4 requires it and milestone 7 needs three real scenes. Either it becomes milestone 6.5 or the scenes come from public datasets. This is a decision for Ved, not for an agent.

## Context

- **Branch**: `milestone-1-scripted-pipeline`, forked from `master` at `a70668b`. Task 7 is complete at `05f64c0`, before this handoff refresh. The branch tracks `origin/milestone-1-scripted-pipeline`; Tasks 6 and 7 plus their handoff refreshes are local and unpushed. `master` is untouched and there is no pull request.
- **Ledger**: `.superpowers/sdd/2026-08-26-milestone-1-scripted-pipeline/progress.md`. This is the authoritative record of what is done, every ruling made, and why. Read it before dispatching anything. It is git-ignored, so `git clean -fdx` would destroy it; recover from `git log` if that happens.
- **Task briefs and reports**: same directory, `task-N-brief.md` and `task-N-report.md`. Briefs for all 11 tasks are already extracted.
- **Key files created so far**: `pyproject.toml`, `src/splatpipe/{__init__,errors,env,config,scene,paths,gaussians}.py`, `src/splatpipe/formats/{__init__,splat}.py`, `configs/truck.toml`, `scripts/patches/{__init__,definitions}.py`, `scripts/setup_env.py`, `tests/{test_package,test_env,test_patches,test_config,test_scene,test_paths,test_gaussians,test_splat_format}.py`, `requirements.lock.txt`.

**Commands to resume:**

```
cd "C:\Users\vedti\NUS_CS(noOnedrive)\gaussian-splat-compression"

# Fast tier, ordinary shell, a few seconds. Currently 76 passed, 1 deselected.
.venv\Scripts\python.exe -m pytest -q

# Complete tier. Clearing addopts runs both CPU and GPU tests.
cmd /c "call scripts\env.bat >nul 2>&1 && .venv\Scripts\python.exe -m pytest -q -o addopts="

# Environment health. Both patches must report "applied".
cmd /c "call scripts\env.bat >nul 2>&1 && .venv\Scripts\python.exe scripts\setup_env.py --check"
```

Run the two `cmd /c` lines through the PowerShell tool, not the Bash tool. Git Bash rewrites the quoted argument and you get an interactive `cmd` banner with the command never run. It exits cleanly and prints no error, so it reads as a command that did nothing rather than as a failure.

Use `-o addopts=` for the complete suite. `-m gpu` selects only GPU-marked tests and does not run the CPU tests. Do not verify the environment by adding `echo %CUDA_HOME%` to the same `cmd /c` line. `cmd` expands variables when it parses the line, before `env.bat` has run, so it prints the literal text and looks like a failure when it is not. Check from a child Python process instead.

**Open questions:**

- How phone captures get COLMAP poses. See Next Steps. This blocks milestone 7, not milestone 1.
- Whether `PngCompression`, gsplat's existing baseline, leaves enough headroom to be worth beating. Unmeasured. This is milestone 2's job and the design doc flags it as the risk most likely to be skipped.

**Style constraint that binds all work here**, stated by Ved and carried in the plan's Global Constraints, in every dispatch prompt, and in project memory: no emoji, no em-dashes, no litotes, no irony, no exclamation marks, plain declarative sentences, and comments only where the code cannot speak for itself. It binds prose, code comments, docstrings and commit messages alike.
