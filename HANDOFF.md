# Handoff: Milestone 2 bench harness: 2026-09-03 (in progress, 4 of 11 tasks)

## What this project is, in plain terms

Read this section if you are returning after a gap and the jargon has faded.

You photograph a real object from many angles. Software works out where each
photo was taken from, then fills the space with millions of tiny coloured
translucent blobs and nudges each blob's position, size, colour and
transparency until the cloud, viewed from where each photo was taken, matches
that photo. You can then view the object from angles you never photographed.
That is Gaussian splatting, and each blob is a Gaussian.

The blob cloud is enormous. The truck scene is 1,000,000 blobs and 236 MB.
Nobody downloads 236 MB to look at a truck in a browser. The whole project is
one question: how small can that get before it starts looking bad. Every
compression method yields a size and a quality score, and plotting size against
quality gives the rate-distortion curve this repository keeps referring to.
Rate is file size. Distortion is how degraded it looks.

Where the bytes actually are, per Gaussian, out of 236:

| Field | Bytes | Share |
|---|---:|---:|
| `shN`, view-dependent colour | 180 | 76% |
| `quats`, rotation | 16 | 7% |
| `means`, position | 12 | 5% |
| `sh0`, base colour | 12 | 5% |
| `scales`, size | 12 | 5% |
| `opacities`, transparency | 4 | 2% |

Three quarters of the file is the data describing how a blob's colour shifts as
you walk around it. Any real win comes from there, or from having fewer blobs.

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

Branch `milestone-2-bench`, forked from `master` at `4528129`. It is pushed and
tracks `origin/milestone-2-bench`. `master` is at `4528129` and is unchanged;
nothing has been merged and there has never been a pull request.

Tasks 1, 2, 3 and 4 of 11 are complete. Task 5 is next and has not started.

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

**The one structural advantage worth knowing**: `PngCompression` compresses
each Gaussian more cleverly but never asks whether a Gaussian should exist.
The truck holds 1,000,000 because `cap_max` says so, not because the scene
needs that many. Pruning composes with their approach rather than competing
with it, which is why milestone 4 is probably the strongest card in the plan.
This is reasoning from the method, not a measured result, and it stays a
hypothesis until milestone 4 measures it.

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
- **Task 3, `bb7eb28..add184c`: the codec protocol, `PlyCodec` and
  `SplatCodec`.** `src/splatpipe/bench/codecs.py` plus `tests/test_codecs.py`.
  Reviewed with one Important and two Minor. Clean after one fix round.
- **Task 4, `b5bd7a9..9d0ea66`: `PngCodec`.** Wraps gsplat's `PngCompression`,
  the baseline this project has to beat. Reviewed with two Important and six
  Minor. Clean after one fix round. Its fixtures sit at 65,536 Gaussians
  because that is the smallest cloud the upstream libraries can encode, which
  is explained below.
- Controller documentation commits: `61ecc76`, `bb7eb28`, `2fc4cfb`, `b5bd7a9`.

**Verified at `9d0ea66` on 2026-09-03:** fast tier `135 passed, 6 deselected`
in 6.5s; complete tier `141 passed` in 149.6s; GPU tier for the codec file
`3 passed, 9 deselected` in 48.9s; `setup_env.py --check` reports
`gsplat checkout: pinned` and `applied` for both patches.

The complete tier's 354s at `add184c` did not recur. It is 149.6s here on a
larger suite, so that was transient machine load rather than a regression, and
the note asking a future session to watch for it is retired.

## What Task 3 was actually about

Task 3 shipped twice. A cheap model transcribed the brief and committed
`46b440c`, faithfully. A capable implementer then took ownership and found that
**the brief itself was wrong**, which transcription structurally cannot catch.

- **`PlyCodec.encode` and `SplatCodec.encode` disagreed about a missing target
  directory.** `write_ply` calls `Path(path).parent.mkdir(parents=True,
  exist_ok=True)` at `src/splatpipe/gaussians.py:154`, so the ply side worked.
  `Path.write_bytes` does not, so the splat side raised `FileNotFoundError`.
  Task 6, at plan line 848, calls `codec.encode(a_cloud(256), tmp_path /
  "splat")` and nothing creates that directory. **Task 6 would have failed on
  its first codec line**, three tasks from now, looking like a Task 6 problem.
  `encode` now owns its directory and the protocol docstring says so.
- **`test_ply_codec_is_lossless` checked 2 of the 6 arrays.** Decoders that
  zeroed `scales`, `quats`, `opacities` or `sh0` all passed it while its
  docstring claimed the anchor loses nothing. It now checks all six.
- `@runtime_checkable` had no caller, which is the decorative-guard pattern
  this project was already burned by. It now has an isinstance test with a
  negative case.
- `test_codec_names_are_distinct` reduced to `"ply" != "splat"`, a restatement
  of two literals. It was replaced by a test of Task 9's `scratch / codec.name`
  layout, which catches the real failure: sharing one directory made the
  `.splat` report 18,625 bytes instead of 2,048.

The one review round was **entirely about a false docstring claim**, not code.
The implementer found the directory bug, fixed it, wrote the guarding test, and
then wrote in that test's docstring that "nothing downstream would have
reported the disagreement", which its own headline finding disproves. That
sentence sat on the test protecting the fix and read as an argument for
deleting both. The corrected wording draws the distinction that matters: Task 9
tolerates either convention, Task 6 forces one.

**The lesson, now four rounds old across two tasks: measure before writing a
rationale, and a false claim in a docstring is a defect at the severity of
wrong code.** Every seat this session verified rather than agreed, and each one
caught something the seat before it had not.

## What Task 4 found, and why it matters beyond Task 4

**`PngCompression` cannot encode fewer than 65,536 Gaussians. At all.**

This is the single most consequential thing learned this session and it is a
constraint on the roadmap, not on one task.

- `_compress_kmeans` defaults to `n_clusters=65536`
  (`png_compression.py:326`), and torchpq requires `n_data >= n_clusters`.
- `compress` builds its kwargs as only `n_sidelen` and `verbose`
  (`png_compression.py:102`), so `n_clusters` cannot be lowered through the
  public API. `PngCompression` is a dataclass whose only fields are `use_sort`
  and `verbose`.
- Measured, not inferred: 65,535 fails, 65,536 works in 15.9s producing
  3,439,163 bytes, 65,537 crops one and decodes 65,536.
- PLAS has a second, lower floor of 256, hit first by anything smaller. Its
  `min_block_size` defaults to 16 (`plas/core.py:490`) and gsplat never
  overrides it (`gsplat/compression/sort.py:39`).

**Why this bites at milestone 4.** Milestone 4 is contribution pruning. The
truck's 1,000,000 Gaussians leave roughly 15x of pruning headroom before the
baseline codec stops being able to run at all. Below that floor the
rate-distortion curve simply has no `PngCompression` point, at exactly the
aggressive operating points pruning exists to explore. Decide before milestone
4 how the comparison is framed there: either the sweep stops at 65,536, or the
curve carries a stated gap, or the baseline is reimplemented without the
torchpq dependency. Nothing currently schedules that decision.

**A second finding, about what the tests do not prove.** At the 65,536 fixture
size the spherical harmonics round-trip almost perfectly, deviating only
3.17e-03. That is not evidence of fidelity. It happens because `n_data` equals
`n_clusters` exactly, so every Gaussian gets its own centroid and the K-means
step does nothing. At the truck's 1,000,000 those same 65,536 clusters have to
do real quantisation. **`PngCompression`'s spherical-harmonic loss at real
scale is unmeasured by anything in this repository.** The implementer declined
to assert on `shN` and said so in the docstring rather than bank a flattering
number. Since `shN` is 76% of the file and milestone 3 attacks exactly that,
knowing this is unmeasured rather than assumed good matters.

**The brief was wrong about mutation, and the copies were removed.** The brief
said `PngCompression.compress` normalises quats and log-transforms means in
place, so the codec must pass copies. Measured false: with all six `.copy()`
calls deleted, every caller array came back byte-identical. `.cuda()` allocates
a separate device tensor, so `compress` cannot reach the caller's numpy arrays
whatever it does to the dictionary it is handed. That mechanism is what the
docstring now states, because it holds even if a future gsplat starts writing
in place.

## In Progress

- Nothing is under implementation. Task 4 finished, passed its review gate, and
  was recorded.
- **Task 5 is next**: cameras. Creates `src/splatpipe/bench/cameras.py` with
  `load_val_views` and the `ValView` type, tests in `tests/test_cameras.py`.
  BASE is the head after this documentation commit. Expect
  `138 passed, 6 deselected` afterwards.
- Task 5 carries the `NORMALIZE_WORLD_SPACE` trap below, which is a silent
  wrong number rather than a crash. Read it before starting.
- If this section names a task as under way and the ledger has no matching
  `Task N: complete` line, that task did not finish. Read the ledger at
  `.superpowers/sdd/2026-09-01-milestone-2-bench/progress.md` before changing
  anything.

## Can I photograph something and render it yet

**No, and the gap is bigger than one missing feature.** Recorded here because
it was asked directly and the answer was not written down anywhere.

- **One photograph is never enough.** `SceneLayout.discover` sets a floor of 8
  images at `src/splatpipe/scene.py:18`, and that is a sanity check rather than
  a quality bar. Realistically 50 to 200 photographs walking around the
  subject.
- **Nothing in this repository turns photographs into camera poses.**
  `SceneLayout.discover` at `src/splatpipe/scene.py:28` requires
  `sparse/0/cameras.bin`, `images.bin` and `points3D.bin`, which is a finished
  COLMAP reconstruction. The installed `pycolmap` is the reader package: it
  exposes `SceneManager` and `COLMAPDatabase` and nothing that reconstructs.
  There is no COLMAP binary on this machine. Verified 2026-09-02.
- **The path that works today, with no code change**: install COLMAP proper,
  run its automatic reconstructor over the photographs to produce `sparse/0/`,
  then point `splatpipe run` at that directory.
- The project spec makes this success criterion 4, "the pipeline reproduces a
  scene from a phone capture with a single command on a clean machine", and
  lists poses as a pipeline stage at spec line 88. **No milestone from 1 to 8
  schedules it.** That is the standing gap, not an oversight of this session.

## The open strategic decision

Raised by Ved on 2026-09-02 and not yet decided. Recorded so it is not
rediscovered from scratch.

The question was: the goal is to beat `PngCompression`, so what happens if that
fails, given this is a portfolio project that can change course.

- **The compression ratio is close to worthless as a portfolio artifact.**
  Nobody reviewing the work can calibrate 15.1x against 14.5x. What is rare and
  visible is that this project can reproduce its own results: checksummed
  manifests, two test tiers, a verified from-scratch provisioning path, and a
  documented habit of correcting its own false claims. That is already banked.
- **Four directions were sketched**: change the win condition to speed rather
  than size, since `PngCompression` takes about two minutes; go for the viewer
  and the shareable link, which demonstrates far better than a table; close the
  capture gap above, which is a week rather than a research programme; or lean
  into measurement and publish a replication study of the published claims.
- **The suggestion on the table**: finish milestone 2, then close the capture
  gap before milestones 3 to 5, then attempt 3 and 4 with a pre-declared
  stopping rule so the attempt does not become a sunk-cost march.
- **Ved has not ruled on any of this.** Do not act on it as though he has.

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
  wrong thing for one codec and produces a plausible number. Task 3 now asserts
  both degrees at the codec layer, so the trap is pinned one layer earlier.
- **RESOLVED in Task 4: the mutation trap was not real.** The spec warned that
  `PngCompression.compress` mutates its input in place and that `PngCodec` must
  pass copies. Measured false, see the Task 4 section above. The copies were
  removed. Left here because the spec still carries the original claim.
- **`PngCompression` crops to a square number of Gaussians.** Truck holds
  exactly 1,000,000, which is 1000 squared, so nothing is dropped now. From
  milestone 4 pruning will rarely produce a square count, so `curve.json`
  records the count actually encoded per codec rather than assuming it equals
  the input. Note the crop keeps the **highest**-opacity splats: it sorts
  descending and slices off the tail (`png_compression.py:135-140`).
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
- **`git commit -m "..."` silently commits nothing under PowerShell 5.1.** It
  word-splits double quotes inside an argument to a native executable, so the
  message is parsed as pathspecs. Use `git commit -F` with a message file. Same
  family as the two traps above and the `python -c` one below.
- **`PngCodec` has no test below 65,536 Gaussians and cannot have one.** Its
  three GPU tests each cost a real encode, so the codec file's GPU tier runs
  about 49 seconds. That is the floor, not slack to be optimised away.
- **`Codec` has no production caller yet.** The protocol is enforced only by
  `tests/test_codecs.py`. Task 9's `run_bench` is duck-typed over
  `codec.name/encode/decode/size` rather than annotated against the protocol.
  Raised by the Task 3 reviewer as a cannot-verify item, ruled a Task 4 and 9
  question rather than a Task 3 defect.
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
- Phone captures still have no scheduled pose-estimation stage. See the capture
  section above.

## Key Decisions

### Process

- **One task per user turn**: implement one task, run its review gate, record
  the outcome, then stop. Do not dispatch or begin the next task while review
  is open. Ved has asked for this directly. It overrides the
  `subagent-driven-development` skill's continuous-execution rule, and the
  reason is that this project runs on weekends with gaps of up to two weeks.
- **Spend the capable model on the task, not only on the review.** Task 3
  demonstrated the cost of not doing this: a faithful transcription of a
  defective brief passed its own tests and would have broken Task 6. Ved
  instructed the upgrade directly. Reserve the cheapest tier for work where the
  brief has already been validated against the code it touches.
- **Use the milestone branch in the existing checkout**: `.venv/`,
  `_gsplat_repo/`, `data/`, `results/` and `out/` are untracked and bound to
  this repo root. A separate worktree lacks the 15 GB working environment.
- **Routine pushes are authorized**: Ved authorized remote pushes and
  default-branch documentation updates on 2026-08-28. Merges remain his call.
  The milestone 1 merge into `master` was made on 2026-09-01 by his explicit
  instruction, after both tiers were re-verified on the exact tree being merged.
- **Task commits are made by implementers.** This is the established cadence
  across both milestones and sits alongside Ved's standing rule that `git
  commit` needs permission. An implementer flagging the tension is right to;
  the resolution is that task commits are pre-authorized and pushes are not.
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
- **`encode` owns its target directory and creates it when missing.** Settled
  in Task 3 after the two implementations disagreed. Task 9's `run_bench`
  tolerates either convention because it calls mkdir first; Task 6 does not,
  and that is the caller that makes the contract mandatory.
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
- **Citations into the plan carry a stable identifier beside the line number.**
  The plan has been renumbered three times this milestone, and three separate
  line numbers supplied from memory have been off by one. A docstring citing
  `plan line 848` alone rots; citing
  `test_render_uses_the_clouds_own_sh_degree (plan line 848)` survives. The
  same applies to citations into `.venv/Lib/site-packages/`, which rot on any
  upgrade and which no test catches.
- **`PngCodec` fixtures are 65,536 and 65,537, not 64 and 65.** Forced by the
  upstream floor above. 65,536 is 256 squared so it needs no crop, and 65,537
  crops exactly one, preserving the square and non-square pair the brief
  intended. Reaching past `PngCompression`'s public API to force a smaller
  cluster count was rejected: it would measure something that is not the
  baseline, which defeats the purpose of having the codec.
- **A tolerance is measured before it is written.** Task 4's round trip pins
  means at `atol=1e-4` against a measured 2.12e-05 through the 16-bit path, and
  scales at `atol=1e-2` against a measured 3.92e-03 through the 8-bit path. The
  check that makes it real is the discriminating one: an all-zero decode
  deviates 0.99999863, four orders above the allowance. A tolerance that passes
  both the real decoder and a stub pins nothing.
- **A commit message is the one artifact that cannot be corrected later.** Task
  4's brief specified "copying its input because compress mutates in place",
  false in both halves once measured. The implementer wrote an accurate subject
  instead. A brief's literal text does not outrank a measurement that disproves
  it, and unlike a docstring a pushed commit message cannot be amended.

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

1. **(P0) Task 5 of the milestone 2 plan**: cameras. Then tasks 6 through 11,
   one per turn. Task 5 carries the `NORMALIZE_WORLD_SPACE` trap, which
   silently evaluates against a differently scaled world and makes every metric
   quietly wrong.
2. **(P1) After Task 11**: whole-branch review on the most capable model, then
   `superpowers:finishing-a-development-branch`. Ved makes the merge call.
3. **(P1) Ved to rule on the open strategic decision above**, ideally before
   milestone 3 starts, because a pre-declared stopping rule is what keeps the
   attempt from becoming a sunk-cost march.
4. **(P1) Decide how the curve handles the 65,536 floor before milestone 4.**
   Pruning below it leaves the comparison with no `PngCompression` point.
   Nothing currently schedules this decision. See the Task 4 section above.
4. **(P1) Decide whether phone captures gain a COLMAP stage**, and if so
   whether it lands before milestone 3 rather than around 6.5. This is the one
   spec success criterion nothing in milestones 1 to 8 currently schedules.
5. **(P2) Revisit Morton tie-breaking before milestone 5**, if the container
   format is going to claim byte parity with gsplat.
6. **(P2) Consider the four accepted milestone 1 Minors closed** unless
   something changes.
7. **(P3) `milestone-1-scripted-pipeline` still exists locally and on the
   remote**, pointing at the same commit as `master`. It is kept as the record
   and is safe to delete whenever Ved wants. It has not been deleted because
   nobody asked.

## Context

- **Branches**: `master` at `4528129`, pushed, carrying all of milestone 1.
  `milestone-2-bench` pushed and tracking `origin/milestone-2-bench`, carrying
  the milestone 2 spec `0c04165`, the plan `d11e425`, and tasks 1 to 4. The
  spec and the plan exist only on this branch, so a reader on `master` cannot
  see them yet.
  `milestone-1-scripted-pipeline` at `9f7f3a1`, kept as a record. There has
  never been a pull request. GitHub SSH and `gh` access work as user
  `VedJoshi`.
- **Authority for milestone 2**, in order: this file, then
  `docs/superpowers/specs/2026-09-01-milestone-2-bench-design.md`, then
  `docs/superpowers/plans/2026-09-01-milestone-2-bench.md` and its **Global
  Constraints** section, then the git-ignored ledger at
  `.superpowers/sdd/2026-09-01-milestone-2-bench/progress.md`. The ledger holds
  the pre-flight scan table and Rulings 1 through 26, each with its cost if
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
  `src/splatpipe/formats/splat.py`, `src/splatpipe/bench/__init__.py`,
  `src/splatpipe/bench/codecs.py`, `tests/test_package.py`,
  `tests/test_splat_format.py`, `tests/test_codecs.py`, the two milestone 2
  documents, and the four root Markdown files.
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

  Expected after this commit: clean tree, `135 passed, 6 deselected` fast,
  `141 passed` complete, `pinned` plus two `applied`.

- **Command caveats**: run `cmd /c` lines through PowerShell, never through Git
  Bash. Use `-o addopts=` for the complete suite; `-m gpu` runs only the
  GPU-marked tests. PowerShell strips quotes from arguments passed to
  `python -c`, so run a scratch `.py` file instead of an inline snippet, and it
  does the same to `git commit -m`, so use `git commit -F` with a message file.
  Bash heredocs mangle backslashes in Windows paths and fail outright on large
  Markdown documents; use the file-writing tools for both.
- **Open questions**: how phone captures get COLMAP poses; whether the project
  pivots per the strategic decision above; how the curve handles the 65,536
  floor once milestone 4 prunes below it; whether `PngCompression`'s
  spherical-harmonic loss at real scale should be measured directly, since
  nothing does and it is the field milestone 3 attacks; whether decode time
  belongs on the rate-distortion curve at all, recorded for now because it is
  nearly free to measure and load time is a stated success criterion for the
  milestone 7 viewer.
