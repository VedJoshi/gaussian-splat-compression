# Resume prompt

Paste everything below the line into a fresh coding agent session started in
`C:\Users\vedti\NUS_CS(noOnedrive)\gaussian-splat-compression`. It is written to be self-contained.

---

You are continuing work on a computer vision portfolio project. Read this whole prompt before touching anything.

## What this project is

A 3D Gaussian Splatting pipeline: capture a scene, train a splat model, compress it, and deploy it to a browser. The portfolio claim is "I built and deployed this, and the hard part was the compression, which I implemented from the literature and benchmarked." Not "I trained a model."

The work is split into eight milestones. **Milestone 1 is complete and merged into `master`.** It turned an earlier throwaway spike into a reproducible pipeline: `splatpipe run <scene-dir> --config <cfg> --out <dir>`. **Milestone 2 is under way on branch `milestone-2-bench`: 2 of its 11 tasks are done.** It builds `src/splatpipe/bench/` and measures the rate-distortion baselines this project has to beat.

Read these four files first, in this order. They are the authority and they disagree with nothing:

1. `HANDOFF.md`: current state, what is done, what is deferred, and why.
2. `docs/superpowers/specs/2026-08-25-splat-compression-pipeline-design.md`: the project design. Binding above every milestone document.
3. `docs/superpowers/specs/2026-09-01-milestone-2-bench-design.md`: the milestone 2 design the current plan argues from. It records the probe measurements, the dependency traps, the module table, and the codec protocol.
4. `docs/superpowers/plans/2026-09-01-milestone-2-bench.md`: 11 tasks with complete code for each. Read its **Global Constraints** section carefully; every task inherits it.

Then read the ledger at `.superpowers/sdd/2026-09-01-milestone-2-bench/progress.md`. It is the authoritative record of what is complete and every ruling made so far, and it holds the pre-flight scan table plus Rulings 1 through 12, each with its cost if wrong. **Trust the ledger and `git log` over anything you think you remember.**

The milestone 1 plan and ledger are still on disk at
`docs/superpowers/plans/2026-08-26-milestone-1-scripted-pipeline.md` and
`.superpowers/sdd/2026-08-26-milestone-1-scripted-pipeline/progress.md`. Read
them when you need the history behind a milestone 1 decision, not to find out
what to do next.

## Where things stand

**Work on `milestone-2-bench`, which is at `bb7eb28` and has not been pushed.** It forked from `master` at `4528129`, which is pushed and carries all of milestone 1. The milestone 2 spec and plan are on this branch only, so they are not visible from `master` or from the remote. There is no `main` branch and there has never been a pull request. `milestone-1-scripted-pipeline` still exists at `9f7f3a1`; it is kept as the record, not as a place to add work.

**Tasks 1 and 2 of milestone 2 are complete and reviewed. Task 3 is next and has not started.** Task 3 is the codec protocol plus `PlyCodec` and `SplatCodec`: it creates `src/splatpipe/bench/__init__.py` and `src/splatpipe/bench/codecs.py` with tests in `tests/test_codecs.py`. BASE is `bb7eb28` and the fast tier should read `131 passed, 3 deselected` afterwards.

Task 1 added the `PngCompression` dependencies and locked them: `cupy-cuda12x==13.6.0`, `torchpq==0.3.0.6`, and `plas` pinned at commit `4f1109c9`. Three traps are recorded there. cupy 14 and later require `numpy>=2.0` and would silently break this project's `numpy<2.0.0` pin, so the version is pinned with `==` rather than `<`, which is a redirect operator in `cmd`. An unpinned git URL is not a lock, so `plas` carries its resolved commit like the three git entries that preceded it. And `torchpq` does not declare `cupy` in its metadata at all: `pip install torchpq` succeeds without it, and `torchpq/__init__.py` raises `ModuleNotFoundError` at first import.

Task 2 added `decode_splat` to `src/splatpipe/formats/splat.py`, inverting the 32-byte record back to a `GaussianCloud` with explicit clamps at both quantisation edges. It returns `shN` of shape `(N, 0, 3)`, so a decoded `.splat` has `sh_degree == 0` while a `.ply` has 3. That is a real fidelity difference between the codecs and later code has to handle it rather than assume a degree.

**The number milestone 3 has to beat is 16,258,005 bytes.** `PngCompression` was probed end to end before the plan was written: it runs, it round-trips, it takes about two minutes. Against the raw `.ply` at 236,001,478 bytes that is 14.52x, and it is half the size of the 32,000,000-byte `.splat` while also retaining spherical harmonics, which the `.splat` discards. `PngCompression`, not the `.splat`, is the baseline. The project spec's original worry that this pipeline "may not beat `PngCompression` by much" is well founded and is now quantified.

**The pipeline reproduces the spike on the real truck scene.** That was the milestone's falsification criterion. Measured PSNR 24.394817 against the spike's 24.406155, a delta of 0.0113 dB against a 0.1 tolerance fixed before the run; SSIM 0.8579996 against 0.8580129; LPIPS 0.1375573 against 0.1372071; exactly 1,000,000 Gaussians; a `.ply` of exactly 236,001,478 bytes, matching the spike to the byte; and a `.splat` of exactly 32,000,000. Training took 8.03 minutes. No expected value was adjusted after the fact, and an independent reviewer re-derived every number from disk.

GitHub access works through both SSH and `gh` as `VedJoshi`. Ved authorized routine pushes and default-branch documentation updates on 2026-08-28, and instructed the merge into `master` on 2026-09-01 after the evaluation passed. Both tiers were re-run on the exact tree being merged before it was taken.

### What milestone 1 left behind, in short

The `.splat` writer is numpy-only, 32 bytes per Gaussian, with Morton, size-opacity and input ordering. Exact records check against gsplat with representable scales; randomized scale fields stay within 2 ULP while every other byte is exact. The real truck cloud encodes to exactly 32,000,000 bytes.

`RunManifest` and `ArtifactRecord` live in `src/splatpipe/manifest.py`. Two contracts are pinned: `RunConfig.to_dict()` pops `source_path`, so no absolute local path reaches `manifest.json`, and `ArtifactRecord.of` calls `as_posix()`, so a recorded path reads `artifacts/scene.ply` on Windows rather than a backslash form.

The synthetic COLMAP fixture is under `tests/fixtures/`: `colmap_bin.py` writes `cameras.bin`, `images.bin` and `points3D.bin` by hand with explicit little-endian struct widths, and `tiny_scene.py` builds a 24-image ring scene in about a second. The real installed pycolmap reads all three back, which is what proves the widths. `look_at_quaternion` branches on the largest quaternion component, because the `w = sqrt(1 + trace) / 2` extraction is degenerate at its own default `n_images=24`: ring index 6 gives `trace == -1.0` bit-exact. Milestone 2 reuses this fixture for its tiny-scene end-to-end test.

Task 10 added `src/splatpipe/stages/train.py` and `src/splatpipe/cli.py`. Training shells out to gsplat's `simple_trainer.py` as a subprocess, because that trainer resolves `datasets.colmap` and `utils` relative to its own directory. Two plan defects were found by reading the trainer's source and ruled in before implementation. First, gsplat evaluates only at `step in [i - 1 for i in cfg.eval_steps]`, defaulting to `[7000, 30000]`, and unlike the checkpoint and ply branches that condition has no `or step == max_steps - 1` fallback, so any run shorter than 7000 steps produced no metrics: `build_train_command` now always passes `--eval-steps <max_steps>`. Second, gsplat writes stats as `{stage}_step{step:04d}.json` but the ply as `point_cloud_{step}.ply`, and the two spellings coincide at step 6999, so the defect is invisible on the truck run and only breaks below 1000 steps. A later review round moved error translation to the boundary: `from_toml` raises `ConfigError` naming the file and accepts a UTF-8 BOM, and `run_training` raises `ArtifactError` naming the exit code and the log path.

`--skip-train` re-exports from an existing `train/` directory without retraining. On the tiny scene that is 2.7 seconds against 71.3 seconds, and it needs no vcvars shell, since nothing on that path imports gsplat. That is the loop to use when iterating on export, and from milestone 3 on compression.

A control worth knowing about from the truck run: the new `.splat` hashes differently from the one recorded for the spike, and re-encoding the spike's own ply through today's code reproduces the old hash exactly, which proves the difference is CUDA non-determinism in training rather than a change in the encoder. `out/` is git-ignored, anchored as `/out/`, because a run writes roughly 700 MB there including two 236 MB plys.

Current suite at `bb7eb28`, measured on 2026-09-01: `126 passed, 3 deselected` fast tier in 21 seconds, `129 passed` complete tier in 121 seconds on a settled GPU, longer when it is already warm. The complete tier trains the real gsplat trainer twice, once per end-to-end test. Milestone 2 ends at 160 in the complete tier if every task lands its planned tests; the plan carries a per-task count table you can check against.

**The provisioning sequence in `README.md` is verified, not guessed.** It was run against an empty directory on 2026-08-31: all seven steps succeed, both patches apply to the fresh venv's own site-packages, and that venv passes all 119 tests. Running it found three defects reading had not, all now fixed. `DISTUTILS_USE_SDK` was set nowhere, so torch refused to build any CUDA extension inside a vcvars shell. `setup_env.py` could not bootstrap at all: it imported gsplat to locate the file it patches, and that import triggers the JIT compile which fails with exactly the error the patch exists to prevent. And the documented dependency step pulled `fused-bilagrid`, which does not compile under MSVC and which this project has never had installed.

The nine deferred Minors are closed: five fixed after the whole-branch review, four consciously accepted with the reasoning recorded in `HANDOFF.md`. The review raised two of them above Minor and was right to: `write_images_bin` trusting its two list lengths was silent corruption in the fixture the end-to-end test depends on, and the absent CPU coverage of `run_pipeline` was the largest gap on the branch.

## How to work

**One task per session turn.** Execute a single task, take it through its review, record the outcome in the ledger, then stop and report to Ved. Do not chain into the next task and do not dispatch the next implementer while the current task is still under review. Ved has asked for this directly, and the reason is that this project runs on weekends with gaps of up to two weeks: he needs to see each task land and be able to redirect before the next one starts.

**Record as you go, not at the end.** The Task 11 review raised the ledger's missing closing entry to Critical, correctly: for a stretch, the commit existed and no record of it did. Append the ruling, the outcome and the commit hash as each happens, so that a reviewer arriving at any moment sees a record that matches `git log`.

For the task you are on:

1. Read `.superpowers/sdd/2026-09-01-milestone-2-bench/task-N-brief.md`. It carries the complete code, exact values, and exact test assertions. Transcribe them; do not improvise names, versions, or assertions. The briefs are generated from the plan by the `subagent-driven-development` skill's `scripts/task-brief` and are regenerable if the workspace was cleaned.
2. Work test-first. Write the failing test, run it, confirm it fails for the stated reason, implement, run again. Do not skip the confirm-it-fails step.
3. Commit with the message the brief gives.
4. Record the outcome in the ledger before moving on.

Append these trailers to every commit message, after a blank line:

```
Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: <your own session URL>
```

Use your own session URL, not one copied from here. Earlier commits carry the session that produced them, which is what makes the trailer worth having.

Push completed work after its review and documentation are finished. Nothing on `milestone-2-bench` has been pushed yet, so the first push creates the remote branch. Milestone 1 passed its whole-branch review and is merged, so that gate is closed; apply the same discipline to milestone 2 and let Ved make each merge call.

## Environment: the parts that will waste your time if you do not know them

**The interpreter is the repo-local venv.** Always `.venv\Scripts\python.exe -m <module>`. Never bare `python` or `py`; the system default is 3.13 and this project needs 3.11.0.

**`scripts\env.bat` only affects the shell that runs it, and tool calls do not share shell state.** Running `env.bat` in one call and a test in the next silently fails, because the second call sees none of it. Chain both into one `cmd` invocation:

```
cmd /c "call scripts\env.bat >nul 2>&1 && .venv\Scripts\python.exe -m pytest -q -o addopts="
```

Use `-o addopts=` for the complete suite. `-m gpu` selects only GPU-marked tests and leaves the CPU tests deselected.

**Run every `cmd /c` line through the PowerShell tool, not the Bash tool.** Git Bash rewrites the quoted argument and you get an interactive `cmd` banner with the command never run. It exits cleanly and prints no error, so it reads as a command that did nothing rather than as a failure.

**PowerShell strips quotes from arguments passed to `python -c`.** An inline snippet loses its string literals and dies with a `SyntaxError` on the first `print("label:", value)`, which reads as broken code rather than as a quoting problem. Write a scratch `.py` file and run that instead. A bash heredoc writing a large Markdown document has a matching failure: it aborts with `unexpected EOF while looking for matching quote` and writes nothing. Use the file-writing tool for those.

**Bash heredocs also mangle backslashes in Windows paths.** A Python literal containing `data\tandt\truck` arrives with `\t` as a tab, so a string comparison against file contents silently finds zero matches and the script aborts on an assertion that looks like a stale expectation rather than a quoting bug. Write the script to a scratch file with the file-writing tool and use raw strings, or use the editing tool directly.

**Do not verify that setup by adding `echo %CUDA_HOME%` to the same `cmd /c` line.** `cmd` expands variables when it parses the line, before `env.bat` has run, so it prints the literal text `%CUDA_HOME%` and looks like a failure when everything is fine. Check from a child Python process instead, which sees the real environment.

**Anything that imports gsplat needs that build shell.** gsplat's CUDA backend runs `where cl` on import even when the extension is already compiled and cached. `scripts/setup_env.py` is the exception and no longer needs it: it locates the files it patches on disk rather than importing them, which is what lets it provision a machine where gsplat cannot yet compile.

**`env.bat` sets `DISTUTILS_USE_SDK=1`.** Without it torch's `cpp_extension._check_abi` refuses to build any CUDA extension inside an activated VC environment, which blocks `fused-ssim`. This is invisible on a machine where everything is already built, and fatal on a fresh one.

**Do not upgrade anything.** In particular:
- `torch==2.7.1+cu128` and `torchvision==0.22.1+cu128`. torch 2.11 cannot build CUDA extensions on Windows: `CUDACachingAllocator.h:105` declares a parameter named `small`, and the Windows SDK's `rpcndr.h:190` has `#define small char`.
- `numpy<2.0.0`. gsplat's examples pin it. The venv drifted off this once already and a test now guards it.

**Two files inside `.venv/Lib/site-packages/` are patched** and the patches are load-bearing. `scripts/setup_env.py --check` reports their status. Both must read `applied`. If either reads `stale`, an upgrade changed the upstream source and the patch needs re-deriving, not forcing. Never hand-edit those files.

**`_gsplat_repo/` is a 57 MB checkout pinned at `937e29912570c372bed6747a5c9bf85fed877bae`.** Do not delete or re-clone it. `data/` and `results/` hold about 15 GB of regenerable input and output; leave them alone.

## Health check before you start

```
cd "C:\Users\vedti\NUS_CS(noOnedrive)\gaussian-splat-compression"
git status --short
git log --oneline -5
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe scripts\setup_env.py --check
```

Expected: a clean tree on `milestone-2-bench` at `bb7eb28` or later, with `bb7eb28` present in recent history; the branch has no upstream, so `git status` reporting no tracking ref is correct rather than a problem; `126 passed, 3 deselected`; and `setup_env.py --check` reporting `gsplat checkout: pinned` plus `applied` for both patches. If any of that differs, stop and read the ledger before changing anything.

`--check` needs no compiler shell and no GPU: `resolve_target` locates the installed files rather than importing them, which is what lets it run on a machine where gsplat cannot yet compile. Only the complete test tier needs `scripts\env.bat`, and that line must be run from PowerShell or `cmd`, never from Git Bash, which rewrites `/c` into a path and silently runs nothing while exiting 0.

The truck run itself lives at `out/truck/` and is git-ignored, so a fresh clone will not have it. Reproducing it costs about 8 minutes of GPU time.

## Writing style, non-negotiable

Ved has asked for this twice. It binds prose, code comments, docstrings, commit messages and any document you produce.

- No emoji.
- No em-dashes. Use a comma, a colon, or a full stop.
- No litotes. Write "this is slow", never "this is not exactly fast".
- No irony, no jokes, no asides about the code or about writing it.
- No exclamation marks.
- Plain declarative sentences.
- Comment only what the code cannot say for itself: a measured number, an upstream bug, a non-obvious format detail, a reason a slower approach was chosen. Never a comment that restates the line below it.
- Prefer a clear name over a comment explaining an unclear one.

`README.md`, `LEARNING.md` and the design spec are already written this way. Match them.

## What is left

1. **Milestone 2, tasks 3 through 11**, one per turn, each through `superpowers:subagent-driven-development`: implementer dispatch, task review, fix loop, scoped re-review, ledger completion line. Task 3 is the codec protocol with `PlyCodec` and `SplatCodec`, BASE `bb7eb28`.
2. **After task 11**: a whole-branch review on the most capable model, then `superpowers:finishing-a-development-branch`. Ved makes the merge call.
3. Everything from milestone 1 is closed. The whole-branch review has been run; the nine deferred Minors are five fixed and four consciously accepted, with the reasoning for each recorded in `HANDOFF.md`; the provisioning sequence has been verified against an empty directory rather than reconstructed; and the merge into `master` has been taken.

### The four traps in the remaining milestone 2 tasks

Each of these is a silent wrong number rather than a crash, which is why they are here rather than only in the spec.

- **`Parser` defaults `normalize=False` while gsplat's trainer passes `cfg.normalize_world_space`, which is `True`** (`simple_trainer.py:67`). Loading cameras with the default silently evaluates against a differently scaled world and every metric is quietly wrong. `bench/cameras.py` encodes this as the named constant `NORMALIZE_WORLD_SPACE = True`.
- **`render.py` must derive `sh_degree` from the cloud.** A decoded `.splat` has degree 0 and a `.ply` has degree 3. Hardcoding it renders the wrong thing for one codec and still produces a plausible number.
- **`PngCompression.compress` mutates the dictionary it is given.** It applies `log_transform` to means and normalises quats in place, so `PngCodec` must pass copies or the cloud is corrupted for every codec measured afterwards.
- **`PngCompression` crops to a square number of Gaussians.** Truck holds exactly 1,000,000, which is 1000 squared, so nothing is dropped now. From milestone 4 pruning will rarely produce a square count, so `curve.json` records the count actually encoded per codec rather than assuming it equals the input.

One more thing worth knowing before a run fails on a train: LPIPS downloads AlexNet weights on first use, which makes the first bench run network-dependent in a project that is otherwise reproducible offline.

The two packaging issues that were deferred to Task 11 are closed. `requirements.lock.txt` no longer records the package through a private `git+ssh` URL; note that a plain `pip freeze` will put it back, because pip resolves an editable install inside a Git checkout to that checkout's remote, so regenerate with `pip freeze --exclude-editable` and leave the hand-maintained `-e .` line alone. `plyfile` deliberately stays `>=1.0` in `pyproject.toml` and `==1.1.3` in the lock: the version cannot affect the byte target, since gsplat writes the `.ply` through its own `splat2ply_bytes` and never imports plyfile.

## Auditing the work

`REVIEWER_AGENT.md` in the repository root holds a self-contained prompt for an independent review agent. Paste it into a fresh session, optionally with a scope such as "review the last commit" or "review the whole branch against master". It is read-only by contract: it reports findings and never edits, commits, pushes, or mutates GitHub state. Its first step is reconciling the ledger against `git log`, which is the cheapest way to catch an agent that lost its place. Use it whenever you want a second opinion that did not inherit this session's assumptions.

## Judgment

Make routine calls yourself and record them. If you find something in the plan that is wrong, say so and fix it rather than implementing a known defect. Ten plan defects were found and fixed this way across milestone 1, almost all of them at the boundary with upstream gsplat or with Windows, and almost all found by reading the upstream source or measuring rather than by trusting the brief. The last of them, in Task 11, would have produced a commit that could not be pushed. Milestone 2's pre-flight scan found three more before any code was written. That history is in the two ledgers and is worth reading, because it shows the kind of mistake this codebase invites.

**Measure before you write a rationale into a test.** Milestone 2 Task 2 needed three fix rounds, all of them claim accuracy rather than code correctness, and two of them corrected wording the controller had supplied. Reasoning about the `.splat` quaternion encoding on paper produced a false claim three separate times: a wrong divisor is undetectable and also harmless because it cancels under normalisation; dropping the normalisation gives a smaller error than the correct decoder; and an offset of 127 passes while 129 fails, because `encode_splat` truncates rather than rounds. A docstring that states what a test pins is only worth having if the statement was measured.

Do not treat a brief as current merely because it is detailed. Milestone 1's Task 11 brief edited README sections that a later commit had already deleted. Check the file before transcribing an edit into it. Milestone 2's Task 2 brief called a test helper `a_cloud` that does not exist; the real one is `make_cloud`. Check the names too.

Stop and ask Ved for destructive or irreversible operations and decisions that are genuinely his. The remaining open owner decision is how phone captures get COLMAP poses. The milestone 1 merge was one of these and has been taken.
