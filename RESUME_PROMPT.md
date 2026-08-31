# Resume prompt

Paste everything below the line into a fresh coding agent session started in
`C:\Users\vedti\NUS_CS(noOnedrive)\gaussian-splat-compression`. It is written to be self-contained.

---

You are continuing work on a computer vision portfolio project. Read this whole prompt before touching anything.

## What this project is

A 3D Gaussian Splatting pipeline: capture a scene, train a splat model, compress it, and deploy it to a browser. The portfolio claim is "I built and deployed this, and the hard part was the compression, which I implemented from the literature and benchmarked." Not "I trained a model."

The work is split into eight milestones. You are executing **milestone 1**, which turns an earlier throwaway spike into a reproducible pipeline: `splatpipe run <scene-dir> --config <cfg> --out <dir>`.

Read these three files first, in this order. They are the authority and they disagree with nothing:

1. `HANDOFF.md`: current state, what is done, what is deferred, and why.
2. `docs/superpowers/specs/2026-08-25-splat-compression-pipeline-design.md`: the design the plan argues from. The spec is binding when anything conflicts.
3. `docs/superpowers/plans/2026-08-26-milestone-1-scripted-pipeline.md`: 11 tasks, 76 steps, with complete code for each. Read its **Global Constraints** section carefully; every task inherits it.

Then read the ledger at `.superpowers/sdd/2026-08-26-milestone-1-scripted-pipeline/progress.md`. It is the authoritative record of what is complete and every ruling made so far. **Trust the ledger and `git log` over anything you think you remember.**

## Where things stand

Use branch `milestone-1-scripted-pipeline`. `origin/master` is at `da1c671`. There is no `main` branch and no pull request.

**All 11 tasks are implemented, reviewed and complete. Milestone 1 is done.** No task is in flight. What remains on this branch is the broad whole-branch review and the merge decision, not implementation.

**The pipeline reproduces the spike on the real truck scene.** That was the milestone's falsification criterion. Measured PSNR 24.394817 against the spike's 24.406155, a delta of 0.0113 dB against a 0.1 tolerance fixed before the run; SSIM 0.8579996 against 0.8580129; LPIPS 0.1375573 against 0.1372071; exactly 1,000,000 Gaussians; a `.ply` of exactly 236,001,478 bytes, matching the spike to the byte; and a `.splat` of exactly 32,000,000. Training took 8.03 minutes. No expected value was adjusted after the fact, and an independent reviewer re-derived every number from disk.

GitHub access works through both SSH and `gh` as `VedJoshi`. Ved authorized routine pushes and default-branch documentation updates on 2026-08-28. The end-to-end criterion now passes, so the merge into `master` is unblocked on the merits, but it is still Ved's decision and has not been made.

Task 7 added the numpy-only 32-byte `.splat` writer with Morton, size-opacity and input ordering. Exact records are checked against gsplat with representable scales; randomized scale fields stay within 2 ULP while every other byte remains exact. Review fixed warning-producing numeric edges and strengthened the ordering regression by mutation. The real truck cloud encodes to exactly 32,000,000 bytes.

Task 8 added `RunManifest` and `ArtifactRecord` in `src/splatpipe/manifest.py`. The manifest records the resolved config, its digest, versions, the Git commit, timings, metrics and content-addressed artifact entries. It reviewed clean on the first pass. Two contracts Task 10 depends on are pinned: `RunConfig.to_dict()` pops `source_path`, so no absolute local path reaches `manifest.json`, and `ArtifactRecord.of` calls `as_posix()`, so a recorded path reads `artifacts/scene.ply` on Windows rather than a backslash form.

Task 9 added the synthetic COLMAP scene fixture under `tests/fixtures/`: `colmap_bin.py` writes `cameras.bin`, `images.bin` and `points3D.bin` by hand with explicit little-endian struct widths, and `tiny_scene.py` builds a 24-image ring scene in about a second. The real installed pycolmap reads all three files back, which is what proves the widths. Two deviations from the brief were ruled in: `pillow` joined the `dev` optional dependencies, because the fixture imports `PIL` and pillow was present only as a transitive dependency of torchvision; and `look_at_quaternion` now branches on the largest quaternion component, because the brief's `w = sqrt(1 + trace) / 2` extraction is degenerate at its own default `n_images=24`. Ring index 6 gives `trace == -1.0` bit-exact, so the brief's own guard raised and every Task 9 test would have errored. Reverting that fix fails all three tests, so it is pinned by construction.

Task 10 added `src/splatpipe/stages/train.py` and `src/splatpipe/cli.py`. Training shells out to gsplat's `simple_trainer.py` as a subprocess, because that trainer resolves `datasets.colmap` and `utils` relative to its own directory. Two plan defects were found by reading the trainer's source and ruled in before implementation. First, gsplat evaluates only at `step in [i - 1 for i in cfg.eval_steps]`, defaulting to `[7000, 30000]`, and unlike the checkpoint and ply branches that condition has no `or step == max_steps - 1` fallback, so any run shorter than 7000 steps produced no metrics: `build_train_command` now always passes `--eval-steps <max_steps>`. Second, gsplat writes stats as `{stage}_step{step:04d}.json` but the ply as `point_cloud_{step}.ply`, and the two spellings coincide at step 6999, so the defect is invisible on the truck run and only breaks below 1000 steps. A later review round moved error translation to the boundary: `from_toml` raises `ConfigError` naming the file and accepts a UTF-8 BOM, and `run_training` raises `ArtifactError` naming the exit code and the log path.

`--skip-train` re-exports from an existing `train/` directory without retraining. On the tiny scene that is 2.7 seconds against 71.3 seconds, and it needs no vcvars shell, since nothing on that path imports gsplat. That is the loop to use when iterating on export, and from milestone 3 on compression.

Task 11 ran the real truck scene through the CLI and compared it against the spike. Its preflight found that `out/` was not git-ignored while the brief's own final step was `git add -A`, which would have staged roughly 700 MB including two 236 MB plys and produced a commit GitHub rejects at push; `/out/` is now ignored. Its step 6 could not be followed literally, because commit `da1c671` had already deleted the README sections it names, so the intent was honoured against the current structure instead. A control worth knowing about: the new `.splat` hashes differently from the one recorded for the spike, and re-encoding the spike's own ply through today's code reproduces the old hash exactly, which proves the difference is CUDA non-determinism in training rather than a change in the encoder.

Current suite: `95 passed, 3 deselected` fast tier, `98 passed, 1 warning` complete tier, measured between 91 and 151 seconds on the same machine. The complete tier trains the real gsplat trainer twice, once per end-to-end test.

Nine Minors are deferred to final branch review. Task 7: error precedence when both the cloud and order are invalid, and the raw NumPy `ValueError` for empty Morton input. Task 8: `collect_versions` catching bare `Exception` so a broken torch or gsplat install is indistinguishable from an absent one, and no test covering the `_git_commit` fallback or `ArtifactRecord.of` with a path outside `relative_to`. Task 9: loose inner-tuple type hints on two writers, and `write_images_bin` trusting that `xys` and `point3D_ids` are the same length. None affects trained non-empty clouds, validated run configuration, or the single fixture caller.

## How to work

**One task per session turn.** Execute a single task, take it through its review, record the outcome in the ledger, then stop and report to Ved. Do not chain into the next task and do not dispatch the next implementer while the current task is still under review. Ved has asked for this directly, and the reason is that this project runs on weekends with gaps of up to two weeks: he needs to see each task land and be able to redirect before the next one starts.

**Record as you go, not at the end.** The Task 11 review raised the ledger's missing closing entry to Critical, correctly: for a stretch, the commit existed and no record of it did. Append the ruling, the outcome and the commit hash as each happens, so that a reviewer arriving at any moment sees a record that matches `git log`.

For the task you are on:

1. Read `.superpowers/sdd/2026-08-26-milestone-1-scripted-pipeline/task-N-brief.md`. It carries the complete code, exact values, and exact test assertions. Transcribe them; do not improvise names, versions, or assertions.
2. Work test-first. Write the failing test, run it, confirm it fails for the stated reason, implement, run again. Do not skip the confirm-it-fails step.
3. Commit with the message the brief gives.
4. Record the outcome in the ledger before moving on.

Append these trailers to every commit message, after a blank line:

```
Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: <your own session URL>
```

Use your own session URL, not one copied from here. Earlier commits carry the session that produced them, which is what makes the trailer worth having.

Push completed work after its review and documentation are finished. All eleven tasks and the end-to-end reproduction have now passed, so the merge is no longer blocked by the criterion; run the whole-branch review first and let Ved make the call.

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

**Anything that imports gsplat needs that build shell.** gsplat's CUDA backend runs `where cl` on import even when the extension is already compiled and cached.

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
cmd /c "call scripts\env.bat >nul 2>&1 && .venv\Scripts\python.exe scripts\setup_env.py --check"
```

Expected: clean tree and synchronized tracking ref; `c68e0c3` and the Task 11 documentation commit present in recent history; `master` at `da1c671`; `95 passed, 3 deselected`; and `setup_env.py --check` reporting `gsplat checkout: pinned` plus `applied` for both patches. If any of that differs, stop and read the ledger before changing anything.

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

1. **Run the broad whole-branch review**, `master..milestone-1-scripted-pipeline`. Each per-task review saw a single commit; nothing has read the branch as a whole. Use `REVIEWER_AGENT.md` with that scope.
2. **Clear the nine deferred Minors, or consciously accept each one.** They are listed individually in `HANDOFF.md`. None affects trained non-empty clouds, validated run configuration, or the single fixture caller, which is why they were deferred rather than fixed.
3. **Decide the merge into `master`.** Use `superpowers:finishing-a-development-branch` once the whole-branch review is clean.
4. **Verify the provisioning sequence on a clean venv.** The `## Provisioning a checkout` section of `README.md` is reconstructed from `SPIKE_LOG.txt` and has never been re-run from scratch. It is the one documented path a second machine would follow, and the project's claim is reproducibility, so this is the largest untested assertion in the repository.

The two packaging issues that were deferred to Task 11 are closed. `requirements.lock.txt` no longer records the package through a private `git+ssh` URL; note that a plain `pip freeze` will put it back, because pip resolves an editable install inside a Git checkout to that checkout's remote, so regenerate with `pip freeze --exclude-editable` and leave the hand-maintained `-e .` line alone. `plyfile` deliberately stays `>=1.0` in `pyproject.toml` and `==1.1.3` in the lock: the version cannot affect the byte target, since gsplat writes the `.ply` through its own `splat2ply_bytes` and never imports plyfile.

## Auditing the work

`REVIEWER_AGENT.md` in the repository root holds a self-contained prompt for an independent review agent. Paste it into a fresh session, optionally with a scope such as "review the last commit" or "review the whole branch against master". It is read-only by contract: it reports findings and never edits, commits, pushes, or mutates GitHub state. Its first step is reconciling the ledger against `git log`, which is the cheapest way to catch an agent that lost its place. Use it whenever you want a second opinion that did not inherit this session's assumptions.

## Judgment

Make routine calls yourself and record them. If you find something in the plan that is wrong, say so and fix it rather than implementing a known defect. Ten plan defects were found and fixed this way across the milestone, almost all of them at the boundary with upstream gsplat or with Windows, and almost all found by reading the upstream source or measuring rather than by trusting the brief. The last of them, in Task 11, would have produced a commit that could not be pushed. That history is in the ledger and is worth reading, because it shows the kind of mistake this codebase invites.

Do not treat a brief as current merely because it is detailed. Task 11's brief edited README sections that a later commit had already deleted. Check the file before transcribing an edit into it.

Stop and ask Ved for destructive or irreversible operations and decisions that are genuinely his. The remaining open owner decisions are how phone captures get COLMAP poses, and whether to merge this branch into `master`.
