# Resume prompt

Paste everything below the line into a fresh coding agent session started in
`C:\Users\vedti\NUS_CS(noOnedrive)\gaussian-splat-compression`. It is written to be self-contained.

---

You are continuing work on a computer vision portfolio project. Read this whole prompt before touching anything.

## What this project is

A 3D Gaussian Splatting pipeline: capture a scene, train a splat model, compress it, and deploy it to a browser. The portfolio claim is "I built and deployed this, and the hard part was the compression, which I implemented from the literature and benchmarked." Not "I trained a model."

The work is split into eight milestones. You are executing **milestone 1**, which turns an earlier throwaway spike into a reproducible pipeline: `splatpipe run <scene-dir> --config <cfg> --out <dir>`.

Read these three files first, in this order. They are the authority and they disagree with nothing:

1. `HANDOFF.md` — current state, what is done, what is deferred, and why.
2. `docs/superpowers/specs/2026-08-25-splat-compression-pipeline-design.md` — the design the plan argues from. The spec is binding when anything conflicts.
3. `docs/superpowers/plans/2026-08-26-milestone-1-scripted-pipeline.md` — 11 tasks, 76 steps, with complete code for each. Read its **Global Constraints** section carefully; every task inherits it.

Then read the ledger at `.superpowers/sdd/2026-08-26-milestone-1-scripted-pipeline/progress.md`. It is the authoritative record of what is complete and every ruling made so far. **Trust the ledger and `git log` over anything you think you remember.**

## Where things stand

Branch `milestone-1-scripted-pipeline`, with Task 3 complete at `78817e4` before the repository rename documentation commit. Nothing on this branch has been pushed.

Tasks 1, 2 and 3 are implemented and passed review. **Task 4 is the first unfinished task.** Tasks 4 through 11 have briefs already extracted into the workspace directory as `task-N-brief.md`.

Current suite: `17 passed, 1 deselected` fast tier, `18 passed, 1 warning` complete tier.

## How to work

Execute tasks one at a time, in order. For each task:

1. Read `.superpowers/sdd/2026-08-26-milestone-1-scripted-pipeline/task-N-brief.md`. It carries the complete code, exact values, and exact test assertions. Transcribe them; do not improvise names, versions, or assertions.
2. Work test-first. Write the failing test, run it, confirm it fails for the stated reason, implement, run again. Do not skip the confirm-it-fails step.
3. Commit with the message the brief gives.
4. Record the outcome in the ledger before moving on.

Append these trailers to every commit message, after a blank line:

```
Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YUgbMd3mogAV1hCkA2xkZr
```

Do not `git push` and do not touch `master` without asking. Both need explicit permission from Ved.

## Environment: the parts that will waste your time if you do not know them

**The interpreter is the repo-local venv.** Always `.venv\Scripts\python.exe -m <module>`. Never bare `python` or `py`; the system default is 3.13 and this project needs 3.11.0.

**`scripts\env.bat` only affects the shell that runs it, and tool calls do not share shell state.** Running `env.bat` in one call and a test in the next silently fails, because the second call sees none of it. Chain both into one `cmd` invocation:

```
cmd /c "call scripts\env.bat >nul 2>&1 && .venv\Scripts\python.exe -m pytest -q -o addopts="
```

Use `-o addopts=` for the complete suite. `-m gpu` selects only GPU-marked tests and leaves the CPU tests deselected.

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

Expected before Task 4: clean tree, Task 3 present at `78817e4`, `17 passed, 1 deselected`, and `setup_env.py --check` reporting `gsplat checkout: pinned` plus `applied` for both patches. If any of that differs, stop and read the ledger before changing anything.

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

## Task order and what matters in each

1. **Task 4, run configuration.** Complete code in the brief.
2. **Task 5, scene validation and output paths.** Complete code in the brief.
3. **Task 6, `GaussianCloud` and `.ply` I/O.** This is the load-bearing interface of the whole project: milestone 3's compressor consumes it and milestone 2's benchmark renders from it. Two traps it exists to contain, both verified against real output: `scales` are logarithms and `opacities` are logits exactly as stored in the `.ply`, converted by nobody on read; and `f_rest` is channel-major, so `f_rest_0..14` are red, `15..29` green, `30..44` blue. Reading those in the obvious order produces wrong colours rather than an error.
4. **Task 7, the `.splat` writer.** Contains the most valuable test in the milestone: encode a cloud and assert it matches `gsplat.exporter.export_splats` byte for byte. That verifies the format against the library the web viewer was built for rather than against anyone's reading of it. `gsplat.exporter` imports without the CUDA backend, so it runs in the fast tier with no GPU.
5. **Task 8, run manifest.** Complete code in the brief.
6. **Task 9, synthetic COLMAP scene fixture.** Writes `cameras.bin`, `images.bin`, `points3D.bin` by hand. Every struct format must carry explicit byte order and width (`'<Q'`, never `'L'`): native `'L'` is 4 bytes on Windows and 8 on Linux, which is the upstream bug the pycolmap patch fixes. pycolmap's own write path is still broken on Windows and cannot be used as a reference. The layouts are transcribed in the brief from pycolmap's reader, which is the consumer.
7. **Task 10, training stage and CLI.** First time the whole chain runs. Treat a failure here as the real work of the task.
8. **Task 11, reproduce the truck scene.** This is the falsification step and the point of the milestone. Run the real scene through the new CLI and compare against the spike: PSNR 24.406, SSIM 0.8580, LPIPS 0.1372, `.ply` exactly 236,001,478 bytes. The `.ply` size must match exactly, since it is a function of Gaussian count and field list. Metrics should match to about two decimal places; the seed is fixed upstream at 42 but CUDA reductions are not bit-reproducible. **A PSNR differing by more than about 0.1 means something is genuinely different and needs investigating before you call the milestone done.**

Also fold in the two deferred minor issues at Task 11: `requirements.lock.txt` records the package as a `git+ssh` URL to a private remote, which nobody without SSH keys can install from, and `plyfile` is pinned `>=1.0` so it resolved to 1.1.3 rather than the 1.1.5 the spike ran on. Both are documented in `HANDOFF.md`.

## Judgment

Make routine calls yourself and record them. If you find something in the plan that is wrong, say so and fix it rather than implementing a known defect: three plan defects were already found and fixed this way before any code was written, and a fourth was found during Task 3 when the patch machinery corrupted a live library file. That history is in the ledger and is worth reading, because it shows the kind of mistake this codebase invites.

Stop and ask Ved only for: pushing, anything touching `master`, anything destructive or irreversible, and decisions that are genuinely his. The remaining open owner decision is how phone captures get COLMAP poses.
