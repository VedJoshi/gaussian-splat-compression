# Handoff — Milestone 1, scripted splat pipeline — 2026-08-26

## Goal

Turn the manual Gaussian-splatting spike in this repository into `splatpipe run <scene-dir> --config <cfg> --out <dir>`: one reproducible command that takes a COLMAP scene, trains it, and emits a `.ply`, a `.splat`, and a manifest recording exactly what produced them. This is milestone 1 of an eight-milestone portfolio project whose real subject is splat compression. The design doc is `docs/superpowers/specs/2026-08-25-splat-compression-pipeline-design.md` and the implementation plan is `docs/superpowers/plans/2026-08-26-milestone-1-scripted-pipeline.md`.

Work is being executed task by task by subagents, with a review after each. Progress is tracked in a ledger, described under Context.

## Completed

Three of eleven tasks are done. All commits are on branch `milestone-1-scripted-pipeline`, which forked from `master` at `a70668b`. The working tree is clean.

- **`a70668b` (on `master`)** — the implementation plan, 11 tasks, 76 steps.
- **`d9ca995`** — fixes for three defects found by scanning the plan against itself before any code was written. See Key Decisions.
- **`c2e6e2b`** — records the `env.bat` chaining recipe in the plan's constraints. See Key Decisions.
- **Task 1, `dfef248`** — package skeleton. `pyproject.toml` with an `src` layout, `src/splatpipe/__init__.py`, `src/splatpipe/errors.py` (`SplatpipeError` plus `BuildEnvError`, `ConfigError`, `SceneError`, `ArtifactError`), `tests/__init__.py`, `tests/test_package.py`, `requirements.lock.txt`. Installed editable. Reviewed clean.
- **Task 2, `4d77440`** — `src/splatpipe/env.py`, `check_build_env()`, which reports every build-environment problem at once instead of failing several layers inside PyTorch minutes into a run. `tests/test_env.py` injects a fake `which`, so it needs no compiler and no CUDA. Reviewed clean.
- **Task 3, `69cbba9` then `446701e`** — the two hand-edits inside `.venv/Lib/site-packages/` are now scripted and idempotent: `scripts/patches/`, `scripts/setup_env.py`. Verified against the real environment, not just in tests. Review still pending, see In Progress.
- **numpy drift corrected.** The venv had drifted to 2.4.6 against gsplat's `numpy<2.0.0` requirement, so the environment did not match what `README.md` documented. Now 1.26.4, enforced by a test.
- **Suite state**: `15 passed, 1 deselected` in the fast tier; `9 passed` in the full tier.

## In Progress

- **Task 3's task review has not been dispatched.** The fix is verified working by the controller, but it has not been through the review gate that every other task gets. This is the single unfinished action. Diff range for the reviewer is `c2e6e2b..446701e`. Brief and report are in the workspace directory named under Context.
- **Tasks 4 to 11 are not started.** Briefs for all of them are already extracted into the workspace directory, so each dispatch is immediate. In order: 4 run configuration, 5 scene validation and output paths, 6 `GaussianCloud` and `.ply` I/O, 7 the `.splat` writer, 8 run manifest, 9 synthetic COLMAP scene fixture, 10 training stage and CLI, 11 reproduce the truck scene and update documentation.

## Not Working / Blockers

Nothing is broken. The environment and the working tree are both healthy. Three things are known and deliberately deferred.

- **`requirements.lock.txt` is not replayable.** Line 53 records the package as `-e git+ssh://git@github.com/VedJoshi/splat-spike.git@d9ca995#egg=splatpipe`. That is pip's normal behaviour for an editable install inside a repo with a remote, but it means anyone without SSH keys for the private remote cannot install from the file, including you on a fresh machine. The file is a record rather than an installer, so nothing downstream breaks. Fix candidate is Task 11, which already touches documentation.
- **`plyfile` is pinned `>=1.0` and resolved to 1.1.3**, not the 1.1.5 the spike ran on. "Worked when frozen" and "resolves the same next install" are different guarantees. Deferred because tightening pins mid-milestone would invalidate the lock file just captured.
- **A residual weakness in anchor matching.** `_anchor_positions` in `scripts/patches/__init__.py` requires a line boundary only for anchors that *start with whitespace*. An anchor that does not (all five pycolmap ones) still matches as a plain substring, so it could in principle alias inside a longer expression. The `len(positions) == 1` ambiguity guard catches the realistic cases and the current anchors are unique, so this is theoretical. Worth a look if a third patch is ever added.

Two things were broken during this session and are now fixed. Both are recorded because the reasoning matters more than the outcome.

- **`apply_patch` corrupted a live library file.** Matching used `text.count()` and `text.replace()`, which are substring operations. The gsplat anchor is a complete line at indent 8; the already-patched file holds that same code at indent 12 inside an `else:` branch, and a 12-space line is a 4-space prefix plus the 8-space line. The anchor matched inside it, the "appears exactly once" guard passed, and the replace spliced a nested `if/else` into the branch, producing an `IndentationError` in `gsplat/cuda/_backend.py`. The implementer had taken a backup first, caught it immediately, and restored. Fixed in `446701e`. Verified against the real file: the old check counts 1 and would corrupt again, the new one returns `[]` and refuses.
- **`patch_status` reported a patched file as unpatched.** It compared the full replacement text including its comment banner. The venv carries the spike's wording and the definition carries different wording, so a genuinely applied patch read as `appliable`. That false negative is what let the corruption bug fire. Now decided by `Patch.applied_marker`, a code fragment that comment wording cannot affect.

## Key Decisions

- **Branch, not a git worktree**: the toolchain is bound to untracked directories in the repo root (`.venv/` with two patched `site-packages` files, `_gsplat_repo/`, `data/`, `results/`, roughly 15 GB). A worktree would have none of them, so no task past Task 1 could run its tests there.
- **`tests/__init__.py` moved from Task 9 to Task 1**: under pytest's `prepend` import mode the directory added to `sys.path` is the first one above the test file lacking `__init__.py`. Without it that is `tests/`, and Task 3's `from scripts.patches import ...` would have failed six tasks before the file that fixes it.
- **`RunConfig.source_path` defined in Task 4, not added by Task 10**: the plan originally had Task 10 write a helper against a field that did not exist, then repair it a step later. A deliberate broken intermediate state costs a fix round and buys nothing.
- **No venv surgery to normalise the gsplat comment banner**: the installed patch is functionally identical to the canonical one and works. Only the comment differs. With the marker check in place the status reads `applied` and the file is correctly left alone. Editing a working library file to make a comment match is risk for no gain.
- **The implementer's anchor rule was accepted over the controller's**: the controller specified strict whole-line equality. The implementer showed that would never match the five pycolmap anchors, which deliberately omit indentation so they match at any nesting depth, and would have broken detection on any fresh install. Its rule (an anchor starting with whitespace must begin at a line boundary; one that does not matches as a substring) handles both anchor styles with a single test. Verified before accepting.
- **Export ordering is a config field from day one**: gsplat's `.splat` exporter sorts by Morton code while the spike sorted by size times opacity. Morton gives spatial locality, which is directly a compression lever for milestone 5, so both need to be measurable.
- **No `seed` config field**: `simple_trainer.py` calls `set_random_seed(42 + local_rank)` and exposes no flag. A seed setting would appear in the manifest and control nothing.

## Next Steps

1. **(P0) Dispatch the Task 3 task reviewer** over `c2e6e2b..446701e`. Generate the package with `scripts/review-package` from the subagent-driven-development skill directory, then hand the reviewer the task 3 brief, the task 3 report, and the printed diff path. Task 3 is the only task that has not passed its review gate.
2. **(P1) Continue Tasks 4 through 11 in order.** Briefs are pre-extracted. Use the cheap model tier for tasks whose brief carries complete code (4, 5, 6, 8) and a standard model for those needing judgment across files (9, 10, 11). Task 7's `.splat` writer has the most valuable test in the milestone: a byte-for-byte comparison against `gsplat.exporter.export_splats`, which imports without the CUDA backend and so runs in the fast tier.
3. **(P1) Task 11 is the falsification step.** It reruns the truck scene through the new CLI and compares against the spike: PSNR 24.406, SSIM 0.8580, LPIPS 0.1372, `.ply` exactly 236,001,478 bytes. The `.ply` size must match exactly. Metrics should match to about two decimal places, since CUDA reductions are not bit-reproducible. A PSNR differing by more than about 0.1 means something is genuinely different.
4. **(P2) Fold the two deferred minors into Task 11**: the `git+ssh` lock file line and the loose `plyfile` pin.
5. **(P2) Decide how phone captures get their poses.** Nothing in milestones 1 to 8 schedules running COLMAP on a raw capture, but success criterion 4 requires it and milestone 7 needs three real scenes. Either it becomes milestone 6.5 or the scenes come from public datasets. This is a decision for Ved, not for an agent.
6. **(P2) Consider renaming the repository.** It is still `splat-spike`; it stopped being a spike at Task 1. Renaming is free now and annoying once anything links to it.

## Context

- **Branch**: `milestone-1-scripted-pipeline`, forked from `master` at `a70668b`. HEAD is `446701e`. Working tree clean. Nothing has been pushed.
- **Ledger**: `.superpowers/sdd/2026-08-26-milestone-1-scripted-pipeline/progress.md`. This is the authoritative record of what is done, every ruling made, and why. Read it before dispatching anything. It is git-ignored, so `git clean -fdx` would destroy it; recover from `git log` if that happens.
- **Task briefs and reports**: same directory, `task-N-brief.md` and `task-N-report.md`. Briefs for all 11 tasks are already extracted.
- **Key files created so far**: `pyproject.toml`, `src/splatpipe/{__init__,errors,env}.py`, `scripts/patches/{__init__,definitions}.py`, `scripts/setup_env.py`, `tests/{test_package,test_env,test_patches}.py`, `requirements.lock.txt`.

**Commands to resume:**

```
cd "C:\Users\vedti\NUS_CS(noOnedrive)\splat-spike"

# Fast tier, ordinary shell, a few seconds. Currently 15 passed, 1 deselected.
.venv\Scripts\python.exe -m pytest -q

# Full tier. env.bat only affects the shell that runs it and tool calls do not
# share shell state, so both must be chained into one cmd invocation.
cmd /c "call scripts\env.bat >nul 2>&1 && .venv\Scripts\python.exe -m pytest -m gpu -q"

# Environment health. Both patches must report "applied".
cmd /c "call scripts\env.bat >nul 2>&1 && .venv\Scripts\python.exe scripts\setup_env.py --check"
```

Use `-m gpu` rather than `-m ""`: a command-line `-m` overrides the `-m 'not gpu'` in `addopts`, and it avoids nesting empty quotes inside an already-quoted `cmd /c` string. Do not verify the environment by adding `echo %CUDA_HOME%` to the same `cmd /c` line. `cmd` expands variables when it parses the line, before `env.bat` has run, so it prints the literal text and looks like a failure when it is not. Check from a child Python process instead.

**Open questions:**

- How phone captures get COLMAP poses. See next step 5. This blocks milestone 7, not milestone 1.
- Whether to keep the repository name `splat-spike`.
- Whether `PngCompression`, gsplat's existing baseline, leaves enough headroom to be worth beating. Unmeasured. This is milestone 2's job and the design doc flags it as the risk most likely to be skipped.

**Style constraint that binds all work here**, stated by Ved and carried in the plan's Global Constraints, in every dispatch prompt, and in project memory: no emoji, no em-dashes, no litotes, no irony, no exclamation marks, plain declarative sentences, and comments only where the code cannot speak for itself. It binds prose, code comments, docstrings and commit messages alike.
