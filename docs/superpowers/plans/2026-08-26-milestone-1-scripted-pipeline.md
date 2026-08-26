# Milestone 1: Scripted Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the manual spike sequence into `splatpipe run <scene-dir> --config <cfg> --out <dir>`, a reproducible command that takes a COLMAP scene directory and produces a trained `.ply`, a `.splat`, and a manifest recording exactly what was run.

**Architecture:** A small Python package under `src/splatpipe/`. Training shells out to gsplat's `simple_trainer.py` at a pinned commit, because that trainer needs its own `sys.path` and its own vcvars environment. Everything downstream of training operates on `GaussianCloud`, a numpy struct-of-arrays container with no torch dependency. That container is the interface the rest of the project hangs off: `compress/` in milestone 3 consumes it, `bench/` in milestone 2 renders from it. The environment patches that made the spike work become a scripted, idempotent, verifiable setup step rather than tribal knowledge.

**Tech Stack:** Python 3.11, numpy <2, plyfile, pytest. gsplat 1.5.3 and torch 2.7.1+cu128 for training only.

**Spec:** `docs/superpowers/specs/2026-08-25-splat-compression-pipeline-design.md`

---

## Global Constraints

Copied from the spec and from the verified spike environment. Every task inherits these.

- **Python 3.11.0.** Not 3.13, which is what is first on PATH. The venv at `.venv/` already has it.
- **torch 2.7.1+cu128, torchvision 0.22.1+cu128.** Do not upgrade. torch 2.11 cannot build CUDA extensions on Windows: `CUDACachingAllocator.h:105` declares a parameter named `small` and the Windows SDK's `rpcndr.h:190` has `#define small char`.
- **numpy <2.0.0.** `_gsplat_repo/examples/requirements.txt` pins it. The venv has drifted to 2.4.6 and Task 1 fixes that.
- **gsplat 1.5.3**, plus the MSVC flag patch. Repo pinned at `937e29912570c372bed6747a5c9bf85fed877bae`.
- **pycolmap** at `cc7ea4b7301720ac29287dbe450952511b32125e`, plus the struct-width patch.
- **`CUDA_HOME` must be a v12.8 install.** v12.9 is first on PATH by default and does not match the torch cu128 build.
- **`TORCH_CUDA_ARCH_LIST=8.9`, `MAX_JOBS=4`.**
- **Anything that imports gsplat must run inside a vcvars64 shell** (VS Build Tools 2019 16.11). gsplat's backend runs `where cl` on import even when the extension is already compiled and cached.
- **`GaussianCloud` and everything that consumes it is numpy only, no torch.** The spec makes this a hard rule for `compress/`; it starts here, so the compression tests stay fast and CPU-only.
- **Every `struct` format string carries an explicit byte order and width** (`'<Q'`, never `'L'`). Native `'L'` is 4 bytes on Windows LLP64 and 8 on Linux LP64. That is upstream bug 3 and there is no reason to reintroduce it.
- **The pipeline is a pure function of `(input directory, config) -> artifacts`.** No interactive state, no hardcoded paths, nothing read from the current working directory.
- **Writing style, for both prose and code comments.** This applies to every file produced, and to commit messages:
  - No emoji anywhere.
  - No em-dashes. Use a comma, a colon, or a full stop.
  - No litotes. Write "this is slow", not "this is not exactly fast".
  - No irony, no jokes, no asides about the code or about the process of writing it.
  - No exclamation marks.
  - Plain declarative sentences. Say the thing directly.
  - Comment only what the code cannot say for itself: a measured number, an upstream bug, a non-obvious format detail, a reason a slower approach was chosen. Do not comment what the next line plainly does. If a comment restates the code, delete it and let the code stand.
  - Prefer a clear name over a comment explaining an unclear one.
  - Match the tone of the existing `README.md`, `LEARNING.md` and the spec, which are already written this way.

---

## File Structure

```
pyproject.toml                          package metadata, deps, pytest config and markers
requirements.lock.txt                   pip freeze of the verified working environment
configs/truck.toml                      the spike's run, expressed as config
src/splatpipe/
  __init__.py                           version only
  errors.py                             the exception types, so modules never import each other for them
  env.py                                build-environment preflight: cl.exe, CUDA_HOME, arch list
  config.py                             RunConfig / TrainConfig / ExportConfig, TOML load, digest
  scene.py                              SceneLayout: validate an input scene directory
  paths.py                              RunPaths: every output path derived from (out_root, name)
  gaussians.py                          GaussianCloud + read_ply / write_ply     <- load-bearing
  formats/
    __init__.py
    splat.py                            GaussianCloud -> .splat bytes, selectable ordering
  stages/
    __init__.py
    train.py                            build and run the gsplat trainer subprocess
  manifest.py                           RunManifest: config, versions, timings, artifact digests
  cli.py                                `splatpipe run`
scripts/
  env.bat                               exists, unchanged
  setup_env.py                          idempotent: clone gsplat at the pin, apply patches
  patches/
    __init__.py                         Patch / Replacement dataclasses and the apply logic
    definitions.py                      the two real patches, as exact search/replace pairs
tests/
  test_package.py  test_env.py  test_patches.py  test_config.py  test_scene.py
  test_paths.py  test_gaussians.py  test_splat_format.py  test_manifest.py
  test_train_command.py
  test_pipeline_e2e.py                  marked `gpu`
  fixtures/
    __init__.py
    colmap_bin.py                       minimal COLMAP binary writer, explicit struct widths
    tiny_scene.py                       synthetic trainable scene, about a second to generate
```

**Why patches are Python search/replace rather than `.patch` files:** `patch.exe` is not present on a stock Windows install. A pair of exact anchor strings is checkable in both directions, so you can ask both "is this applied?" and "does this still apply to the version currently installed?"

**Why the export ordering is a config field from day one:** gsplat's `.splat` exporter sorts by Morton code (`gsplat/exporter.py`, `sort_centers`), while the spike's converter sorted by size times opacity, descending. These are different orderings serving different purposes. Morton gives spatial locality, which puts similar bytes next to each other and is therefore directly a compression lever. Size-descending gives progressive loading. Milestone 3 onward will want to measure both, so it is a parameter now rather than a rewrite later.

---

## Task 1: Package skeleton, pinned dependencies, test harness

**Files:**
- Create: `pyproject.toml`
- Create: `src/splatpipe/__init__.py`
- Create: `src/splatpipe/errors.py`
- Create: `tests/test_package.py`
- Create: `requirements.lock.txt`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: nothing
- Produces: the `splatpipe` package importable from the venv; `SplatpipeError`, `BuildEnvError`, `ConfigError`, `SceneError`, `ArtifactError` in `splatpipe.errors`; `pytest` runnable with a `gpu` marker deselected by default

- [ ] **Step 1: Write the failing test**

Create `tests/test_package.py`:

```python
"""The package installs and its error types are importable from one place."""


def test_package_version():
    import splatpipe

    assert splatpipe.__version__ == "0.1.0"


def test_numpy_is_below_2():
    """gsplat 1.5.3's examples pin numpy<2. The venv has drifted off this before."""
    import numpy as np

    assert np.__version__.startswith("1."), f"numpy {np.__version__} breaks the gsplat examples"


def test_error_types_share_a_base():
    from splatpipe.errors import BuildEnvError, ConfigError, SceneError, SplatpipeError

    for exc in (BuildEnvError, ConfigError, SceneError):
        assert issubclass(exc, SplatpipeError)
```

- [ ] **Step 2: Run test to verify it fails**

pytest is not installed yet, so install it first:

```
.venv\Scripts\python.exe -m pip install pytest
.venv\Scripts\python.exe -m pytest tests/test_package.py -q
```

Expected: 3 failures. Two are `ModuleNotFoundError: No module named 'splatpipe'`. The third is `test_numpy_is_below_2` failing with `numpy 2.4.6 breaks the gsplat examples`, which confirms the drift is real and not a guess.

- [ ] **Step 3: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "splatpipe"
version = "0.1.0"
description = "Reproducible Gaussian splat training, compression and packaging"
requires-python = "==3.11.*"
dependencies = [
    "numpy<2.0.0",
    "plyfile>=1.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[project.scripts]
splatpipe = "splatpipe.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "gpu: needs a CUDA GPU and a vcvars64 shell; takes minutes; deselected by default",
]
addopts = "-m 'not gpu'"
```

- [ ] **Step 4: Write the package files**

`src/splatpipe/__init__.py`:

```python
"""Reproducible Gaussian splat training, compression and packaging."""

__version__ = "0.1.0"
```

`src/splatpipe/errors.py`:

```python
"""Every exception this package raises deliberately.

Kept in one module so that no module imports another only to catch its errors.
"""


class SplatpipeError(Exception):
    """Base class. Catch this to catch anything the pipeline raises deliberately."""


class BuildEnvError(SplatpipeError):
    """The vcvars64 / CUDA environment is not set up. See scripts/env.bat."""


class ConfigError(SplatpipeError):
    """A run config is malformed, has unknown keys, or has out-of-range values."""


class SceneError(SplatpipeError):
    """An input scene directory is missing something the trainer needs."""


class ArtifactError(SplatpipeError):
    """An expected artifact is missing or does not have the shape we expect."""
```

- [ ] **Step 5: Install, which also fixes the numpy drift**

```
.venv\Scripts\python.exe -m pip install -e ".[dev]"
.venv\Scripts\python.exe -c "import numpy; print(numpy.__version__)"
```

The `numpy<2.0.0` dependency downgrades numpy from 2.4.6 as a side effect. Expected: a `1.26.x` version string.

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest -q`
Expected: 3 passed.

- [ ] **Step 7: Freeze the verified environment**

```
.venv\Scripts\python.exe -m pip freeze > requirements.lock.txt
```

This is the file that answers "what exactly was installed when this worked". It is a record, not an installer: the git-URL dependencies in it still need the setup script from Task 3.

- [ ] **Step 8: Add build outputs to `.gitignore`**

Append to `.gitignore`:

```
# Packaging and test artifacts
*.egg-info/
.pytest_cache/
build/
dist/
```

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml requirements.lock.txt src tests .gitignore
git commit -m "Add splatpipe package skeleton and pin the working dependency set"
```

---

## Task 2: Build-environment preflight

**Files:**
- Create: `src/splatpipe/env.py`
- Create: `tests/test_env.py`

**Interfaces:**
- Consumes: `splatpipe.errors.BuildEnvError`
- Produces: `splatpipe.env.check_build_env(env: Mapping[str, str] | None = None, which: Callable[..., str | None] = shutil.which) -> None`, which raises `BuildEnvError` listing every problem found; module constants `REQUIRED_CUDA_VERSION = "v12.8"` and `REQUIRED_ARCH_LIST = "8.9"`

Without this, a misconfigured shell surfaces as `CalledProcessError: Command '['where', 'cl']' returned non-zero exit status 1` from four layers inside torch, eight minutes into a run. The entire value of this module is failing in the first second with a sentence that says what to do.

- [ ] **Step 1: Write the failing test**

Create `tests/test_env.py`:

```python
import pytest

from splatpipe.env import REQUIRED_ARCH_LIST, REQUIRED_CUDA_VERSION, check_build_env
from splatpipe.errors import BuildEnvError

GOOD_ENV = {
    "PATH": r"C:\fake\msvc\bin",
    "CUDA_HOME": rf"C:\CUDA\{REQUIRED_CUDA_VERSION}",
    "TORCH_CUDA_ARCH_LIST": REQUIRED_ARCH_LIST,
}


def found(cmd, path=None):
    return r"C:\fake\msvc\bin\cl.exe"


def missing(cmd, path=None):
    return None


def test_good_environment_passes():
    check_build_env(GOOD_ENV, which=found)


def test_missing_compiler_is_reported():
    with pytest.raises(BuildEnvError) as excinfo:
        check_build_env(GOOD_ENV, which=missing)
    assert "env.bat" in str(excinfo.value)


def test_wrong_cuda_version_is_reported():
    env = GOOD_ENV | {"CUDA_HOME": r"C:\CUDA\v12.9"}
    with pytest.raises(BuildEnvError) as excinfo:
        check_build_env(env, which=found)
    assert REQUIRED_CUDA_VERSION in str(excinfo.value)


def test_all_problems_are_reported_at_once():
    with pytest.raises(BuildEnvError) as excinfo:
        check_build_env({}, which=missing)
    assert str(excinfo.value).count("\n  - ") == 3, str(excinfo.value)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_env.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'splatpipe.env'`

- [ ] **Step 3: Write the implementation**

Create `src/splatpipe/env.py`:

```python
"""Preflight checks for the gsplat build environment.

gsplat's CUDA backend runs `where cl` on import even when the extension is
already compiled and cached, so an ordinary shell cannot import gsplat at all.
Everything that touches gsplat has to run inside scripts/env.bat.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable, Mapping

from splatpipe.errors import BuildEnvError

REQUIRED_CUDA_VERSION = "v12.8"
REQUIRED_ARCH_LIST = "8.9"

_HINT = "Run scripts\\env.bat in this shell first."


def check_build_env(
    env: Mapping[str, str] | None = None,
    which: Callable[..., str | None] = shutil.which,
) -> None:
    """Raise BuildEnvError if this shell cannot build or load gsplat.

    Reports every problem at once rather than stopping at the first one.
    """
    env = os.environ if env is None else env
    problems: list[str] = []

    if which("cl", path=env.get("PATH")) is None:
        problems.append(f"cl.exe is not on PATH. {_HINT}")

    cuda_home = env.get("CUDA_HOME", "")
    if REQUIRED_CUDA_VERSION not in cuda_home:
        problems.append(
            f"CUDA_HOME is {cuda_home or '(unset)'}, but torch is built against cu128, "
            f"so it must point at a {REQUIRED_CUDA_VERSION} install. {_HINT}"
        )

    arch_list = env.get("TORCH_CUDA_ARCH_LIST")
    if arch_list != REQUIRED_ARCH_LIST:
        problems.append(
            f"TORCH_CUDA_ARCH_LIST is {arch_list or '(unset)'}, expected "
            f"{REQUIRED_ARCH_LIST}. Without it a rebuild targets every architecture "
            f"and takes roughly 8x longer. {_HINT}"
        )

    if problems:
        raise BuildEnvError("Build environment is not set up:\n  - " + "\n  - ".join(problems))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_env.py -q`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/splatpipe/env.py tests/test_env.py
git commit -m "Add build environment preflight check"
```

---

## Task 3: Scripted environment setup

This is the risk the spec singles out. As things stand the project does not survive a `pip install`, and does not survive its own author after a two-week gap.

**Files:**
- Create: `scripts/__init__.py` (empty)
- Create: `scripts/patches/__init__.py`
- Create: `scripts/patches/definitions.py`
- Create: `scripts/setup_env.py`
- Create: `tests/test_patches.py`
- Modify: `README.md`
- Delete: `_backend.py.ORIGINAL.bak`, `_scene_manager.py.ORIGINAL.bak`

**Interfaces:**
- Consumes: nothing from earlier tasks
- Produces: `scripts.patches.Patch(name, module, reason, replacements)`, `scripts.patches.Replacement(old, new)`, `apply_patch(patch: Patch, path: Path) -> str` returning `"applied"` or `"already-applied"`, `patch_status(patch: Patch, path: Path) -> str` returning `"applied" | "appliable" | "stale"`; `scripts.patches.definitions.ALL_PATCHES` and `resolve_target(patch) -> Path`; the command `python scripts/setup_env.py [--check]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_patches.py`:

```python
import pytest

from scripts.patches import Patch, Replacement, apply_patch, patch_status

SIMPLE = Patch(
    name="example",
    module="does.not.matter",
    reason="a test patch",
    replacements=(Replacement(old="a = 1", new="a = 2"),),
)


def write(tmp_path, text):
    target = tmp_path / "module.py"
    target.write_text(text, encoding="utf-8")
    return target


def test_apply_rewrites_the_anchor(tmp_path):
    target = write(tmp_path, "x = 0\na = 1\ny = 3\n")
    assert apply_patch(SIMPLE, target) == "applied"
    assert target.read_text(encoding="utf-8") == "x = 0\na = 2\ny = 3\n"


def test_apply_is_idempotent(tmp_path):
    target = write(tmp_path, "a = 1\n")
    apply_patch(SIMPLE, target)
    assert apply_patch(SIMPLE, target) == "already-applied"
    assert target.read_text(encoding="utf-8") == "a = 2\n"


def test_status_reports_stale_when_the_anchor_is_gone(tmp_path):
    assert patch_status(SIMPLE, write(tmp_path, "b = 9\n")) == "stale"


def test_apply_refuses_a_stale_anchor(tmp_path):
    with pytest.raises(RuntimeError, match="does not match"):
        apply_patch(SIMPLE, write(tmp_path, "b = 9\n"))


def test_apply_refuses_an_ambiguous_anchor(tmp_path):
    with pytest.raises(RuntimeError, match="2 times"):
        apply_patch(SIMPLE, write(tmp_path, "a = 1\na = 1\n"))


@pytest.mark.gpu
def test_real_patches_apply_to_the_installed_packages():
    """The two upstream bugs from the spike. This is the test that matters.

    Marked `gpu` because resolving the gsplat target imports gsplat, which needs
    a vcvars shell. Run with: scripts\\env.bat then pytest -m ""
    """
    from scripts.patches.definitions import ALL_PATCHES, resolve_target

    assert len(ALL_PATCHES) == 2
    for patch in ALL_PATCHES:
        status = patch_status(patch, resolve_target(patch))
        assert status in {"applied", "appliable"}, f"{patch.name} is {status}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_patches.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts'`

- [ ] **Step 3: Write the patch machinery**

Create `scripts/__init__.py` as an empty file, then `scripts/patches/__init__.py`:

```python
"""Apply small, exact source patches to installed third-party packages.

These are search/replace pairs rather than unified diffs because patch.exe is
not present on a stock Windows install, and because an exact anchor can be
checked in both directions: "is this already applied" and "does this still
apply to the version currently installed".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Replacement:
    """One exact search/replace. `old` must appear exactly once in the file."""

    old: str
    new: str


@dataclass(frozen=True)
class Patch:
    name: str
    module: str
    reason: str
    replacements: tuple[Replacement, ...]


def patch_status(patch: Patch, path: Path) -> str:
    """Return "applied", "appliable", or "stale".

    "stale" means the installed file no longer contains the anchors, which is
    what an upgrade looks like. That is a signal to go and re-derive the patch,
    not to force it.
    """
    text = path.read_text(encoding="utf-8")
    if all(r.new in text for r in patch.replacements):
        return "applied"
    if all(text.count(r.old) == 1 for r in patch.replacements):
        return "appliable"
    return "stale"


def apply_patch(patch: Patch, path: Path) -> str:
    """Apply `patch` to `path` in place. Returns "applied" or "already-applied"."""
    text = path.read_text(encoding="utf-8")

    if all(r.new in text for r in patch.replacements):
        return "already-applied"

    for replacement in patch.replacements:
        count = text.count(replacement.old)
        if count == 0:
            raise RuntimeError(
                f"{patch.name}: anchor does not match {path}. The installed version "
                f"has changed and the patch needs re-deriving. Anchor was:\n{replacement.old}"
            )
        if count > 1:
            raise RuntimeError(
                f"{patch.name}: anchor appears {count} times in {path}, so the "
                f"replacement is ambiguous. Anchor was:\n{replacement.old}"
            )
        text = text.replace(replacement.old, replacement.new)

    path.write_text(text, encoding="utf-8")
    return "applied"
```

- [ ] **Step 4: Write the two real patches**

Create `scripts/patches/definitions.py`. The anchor strings are copied verbatim from the installed sources; do not retype them from memory.

```python
"""The two upstream bugs the spike hit. See README.md for the full diagnosis."""

from __future__ import annotations

import importlib
from pathlib import Path

from scripts.patches import Patch, Replacement

GSPLAT_MSVC_FLAGS = Patch(
    name="gsplat-msvc-flags",
    module="gsplat.cuda._backend",
    reason=(
        "gsplat passes GCC/Clang flags to the host compiler unconditionally. "
        "cl.exe reads '-Wno-attributes' as /W followed by 'no-attributes' and "
        "fails with D8021, and '-O3' is not MSVC syntax either."
    ),
    replacements=(
        Replacement(
            old='        extra_cflags = [opt_level, "-Wno-attributes"]\n',
            new=(
                "        # PATCHED by scripts/setup_env.py: MSVC flag portability.\n"
                '        if os.name == "nt":\n'
                '            extra_cflags = ["/Od" if FAST_COMPILE else "/O2"]\n'
                "        else:\n"
                '            extra_cflags = [opt_level, "-Wno-attributes"]\n'
            ),
        ),
    ),
)

PYCOLMAP_STRUCT_WIDTHS = Patch(
    name="pycolmap-struct-widths",
    module="pycolmap.scene_manager",
    reason=(
        "The reader uses struct formats with no byte-order prefix, so 'L' takes "
        "native width: 8 bytes on Linux LP64, 4 on Windows LLP64. It reads 8 "
        "bytes from the file and then tries to unpack 4 of them."
    ),
    replacements=(
        Replacement(
            old="num_cameras = struct.unpack('L', f.read(8))[0]",
            new="num_cameras = struct.unpack('<Q', f.read(8))[0]",
        ),
        Replacement(
            old="camera_id, camera_type, w, h = struct.unpack('IiLL', f.read(24))",
            new="camera_id, camera_type, w, h = struct.unpack('<IiQQ', f.read(24))",
        ),
        Replacement(
            old="num_images = struct.unpack('L', f.read(8))[0]",
            new="num_images = struct.unpack('<Q', f.read(8))[0]",
        ),
        # Not a bug on either platform ('Q' is 8 bytes natively), but made
        # explicit so that grepping this file for native formats turns up only
        # the still-broken write path at lines 313-421.
        Replacement(
            old="num_points2D = struct.unpack('Q', f.read(8))[0]",
            new="num_points2D = struct.unpack('<Q', f.read(8))[0]",
        ),
        Replacement(
            old="num_points3D = struct.unpack('L', f.read(8))[0]",
            new="num_points3D = struct.unpack('<Q', f.read(8))[0]",
        ),
    ),
)

ALL_PATCHES = (GSPLAT_MSVC_FLAGS, PYCOLMAP_STRUCT_WIDTHS)


def resolve_target(patch: Patch) -> Path:
    """Find the installed source file a patch targets.

    Imports the module rather than guessing at site-packages layout, so this
    works for a venv, a user install, or an editable checkout.
    """
    module = importlib.import_module(patch.module)
    if module.__file__ is None:
        raise RuntimeError(f"{patch.module} has no __file__ and cannot be patched")
    return Path(module.__file__)
```

- [ ] **Step 5: Run tests to verify they pass**

Fast tier, ordinary shell:

```
.venv\Scripts\python.exe -m pytest tests/test_patches.py -q
```

Expected: 5 passed, 1 deselected.

Full tier, inside the build shell:

```
scripts\env.bat
.venv\Scripts\python.exe -m pytest tests/test_patches.py -q -m ""
```

Expected: 6 passed. The real-patches test reports `applied` for both, since the spike applied them by hand already.

- [ ] **Step 6: Write the setup script**

Create `scripts/setup_env.py`:

```python
"""Make a fresh checkout runnable. Idempotent; safe to run repeatedly.

    python scripts/setup_env.py --check    report status, change nothing
    python scripts/setup_env.py            clone gsplat at the pin, apply patches

Must be run inside scripts/env.bat, because resolving the gsplat patch target
imports gsplat.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.patches import apply_patch, patch_status  # noqa: E402
from scripts.patches.definitions import ALL_PATCHES, resolve_target  # noqa: E402

GSPLAT_URL = "https://github.com/nerfstudio-project/gsplat.git"
GSPLAT_PIN = "937e29912570c372bed6747a5c9bf85fed877bae"
GSPLAT_DIR = REPO_ROOT / "_gsplat_repo"


def gsplat_checkout_status() -> str:
    if not (GSPLAT_DIR / ".git").is_dir():
        return "missing"
    head = subprocess.run(
        ["git", "-C", str(GSPLAT_DIR), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return "pinned" if head == GSPLAT_PIN else f"at {head[:12]}, expected {GSPLAT_PIN[:12]}"


def ensure_gsplat_checkout() -> None:
    if not (GSPLAT_DIR / ".git").is_dir():
        print(f"cloning gsplat into {GSPLAT_DIR}")
        subprocess.run(["git", "clone", GSPLAT_URL, str(GSPLAT_DIR)], check=True)
    subprocess.run(["git", "-C", str(GSPLAT_DIR), "checkout", "--detach", GSPLAT_PIN], check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="report status, change nothing")
    args = parser.parse_args()

    print(f"gsplat checkout: {gsplat_checkout_status()}")
    for patch in ALL_PATCHES:
        target = resolve_target(patch)
        print(f"{patch.name}: {patch_status(patch, target)}  ({target})")

    if args.check:
        return 0

    ensure_gsplat_checkout()
    for patch in ALL_PATCHES:
        print(f"{patch.name}: {apply_patch(patch, resolve_target(patch))}")

    print("Environment ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 7: Verify the script against the real environment**

```
scripts\env.bat
.venv\Scripts\python.exe scripts\setup_env.py --check
```

Expected, three lines:

```
gsplat checkout: pinned
gsplat-msvc-flags: applied  (...\.venv\Lib\site-packages\gsplat\cuda\_backend.py)
pycolmap-struct-widths: applied  (...\.venv\Lib\site-packages\pycolmap\scene_manager.py)
```

Then prove it is genuinely idempotent:

```
.venv\Scripts\python.exe scripts\setup_env.py
git -C _gsplat_repo rev-parse HEAD
```

Expected: both patches report `already-applied`, and HEAD is still `937e29912570c372bed6747a5c9bf85fed877bae`.

- [ ] **Step 8: Replace the fragile-patches section of the README**

In `README.md`, replace this section:

```
## The patches are fragile

Both fixes were applied inside `.venv/Lib/site-packages/`. A reinstall, an upgrade, or a fresh venv
destroys them, and `pip freeze` will not show them. If this turns into a real project, they need to
become a setup script, an upstream PR, or a vendored fork.
```

with:

```
## The patches

Both fixes live inside `.venv/Lib/site-packages/`, so a reinstall or an upgrade destroys them and
`pip freeze` will not show them. They are scripted rather than manual:

    scripts\env.bat
    .venv\Scripts\python.exe scripts\setup_env.py --check

reports whether each is applied, and dropping `--check` applies them. If an upgrade changes the
upstream source the status reads `stale`, which means the patch needs re-deriving rather than
forcing. The definitions are in `scripts/patches/definitions.py`, with the reason for each.
```

- [ ] **Step 9: Delete the superseded backups**

The `.bak` files were a manual undo mechanism. The patch definitions now record the exact change and `patch_status` reports whether it is applied, which is strictly better. A `.bak` of an old upstream version is actively misleading after an upgrade.

```bash
git rm _backend.py.ORIGINAL.bak _scene_manager.py.ORIGINAL.bak
```

Also delete the `| `*.ORIGINAL.bak` |` row from the `## Files` table in `README.md`.

- [ ] **Step 10: Commit**

```bash
git add scripts tests/test_patches.py README.md
git commit -m "Script the two site-packages patches instead of applying them by hand"
```

---

## Task 4: Run configuration

**Files:**
- Create: `src/splatpipe/config.py`
- Create: `configs/truck.toml`
- Create: `tests/test_config.py`

**Interfaces:**
- Consumes: `splatpipe.errors.ConfigError`
- Produces:
  - `TrainConfig(max_steps: int = 7000, cap_max: int = 1_000_000, data_factor: int = 1, test_every: int = 8, strategy: str = "mcmc")`

There is deliberately no `seed` field. `simple_trainer.py` calls `set_random_seed(42 + local_rank)` at `Runner.__init__` and exposes no config field for it, so a `seed` setting here would be a lie: it would appear in the manifest, appear to control something, and control nothing. The manifest records the fixed 42 as a version fact instead.
  - `ExportConfig(order: str = "morton")`
  - `RunConfig(name: str, train: TrainConfig, export: ExportConfig)` with `.to_dict() -> dict`, `.digest() -> str` (12 hex chars), `RunConfig.from_dict(d) -> RunConfig`, `RunConfig.from_toml(path) -> RunConfig`
  - All three are frozen dataclasses.

`digest()` deliberately excludes `name`, because the name is an identity label and not a parameter. Two runs with the same parameters under different names should share a digest, so that "have I already computed this?" has a useful answer.

- [ ] **Step 1: Write the failing test**

Create `tests/test_config.py`:

```python
import pytest

from splatpipe.config import ExportConfig, RunConfig, TrainConfig
from splatpipe.errors import ConfigError


def test_defaults_match_the_spike():
    cfg = RunConfig(name="truck")
    assert cfg.train.max_steps == 7000
    assert cfg.train.cap_max == 1_000_000
    assert cfg.train.data_factor == 1
    assert cfg.train.strategy == "mcmc"
    assert cfg.export.order == "morton"
    assert not hasattr(cfg.train, "seed"), "gsplat hardcodes the seed; do not pretend otherwise"


def test_round_trips_through_a_dict():
    cfg = RunConfig(name="truck", train=TrainConfig(max_steps=50))
    assert RunConfig.from_dict(cfg.to_dict()) == cfg


def test_digest_is_stable_and_short():
    cfg = RunConfig(name="truck")
    assert cfg.digest() == cfg.digest()
    assert len(cfg.digest()) == 12


def test_digest_tracks_parameters_but_not_the_name():
    base = RunConfig(name="truck")
    assert RunConfig(name="lorry").digest() == base.digest()
    assert RunConfig(name="truck", train=TrainConfig(max_steps=50)).digest() != base.digest()
    assert RunConfig(name="truck", export=ExportConfig(order="none")).digest() != base.digest()


def test_unknown_keys_are_rejected():
    with pytest.raises(ConfigError, match="max_step"):
        RunConfig.from_dict({"name": "truck", "train": {"max_step": 50}})


def test_out_of_range_values_are_rejected():
    with pytest.raises(ConfigError, match="max_steps"):
        RunConfig(name="truck", train=TrainConfig(max_steps=0))
    with pytest.raises(ConfigError, match="test_every"):
        RunConfig(name="truck", train=TrainConfig(test_every=1))
    with pytest.raises(ConfigError, match="strategy"):
        RunConfig(name="truck", train=TrainConfig(strategy="magic"))
    with pytest.raises(ConfigError, match="order"):
        RunConfig(name="truck", export=ExportConfig(order="sideways"))


def test_loads_from_toml(tmp_path):
    path = tmp_path / "run.toml"
    path.write_text('name = "tiny"\n\n[train]\nmax_steps = 50\ncap_max = 5000\n', encoding="utf-8")
    cfg = RunConfig.from_toml(path)
    assert cfg.name == "tiny"
    assert cfg.train.max_steps == 50
    assert cfg.train.cap_max == 5000
    assert cfg.train.data_factor == 1  # unspecified keys keep their default


def test_the_checked_in_truck_config_loads():
    cfg = RunConfig.from_toml("configs/truck.toml")
    assert cfg.name == "truck"
    assert cfg.train.max_steps == 7000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_config.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'splatpipe.config'`

- [ ] **Step 3: Write the implementation**

Create `src/splatpipe/config.py`:

```python
"""Run configuration.

Frozen dataclasses with validation in __post_init__, loaded from TOML. Unknown
keys are an error rather than a silent no-op: a typo in a config file that is
quietly ignored costs a whole training run to notice.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

STRATEGIES = ("mcmc", "default")
ORDERS = ("morton", "size_opacity", "none")


def _reject_unknown(cls: type, data: dict[str, Any]) -> None:
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ConfigError(
            f"{cls.__name__}: unknown key(s) {', '.join(unknown)}. Known keys are: "
            f"{', '.join(sorted(known))}"
        )


from splatpipe.errors import ConfigError  # noqa: E402  (kept below _reject_unknown for readability)


@dataclass(frozen=True)
class TrainConfig:
    max_steps: int = 7000
    cap_max: int = 1_000_000
    data_factor: int = 1
    test_every: int = 8
    strategy: str = "mcmc"

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ConfigError(f"max_steps must be positive, got {self.max_steps}")
        if self.cap_max <= 0:
            raise ConfigError(f"cap_max must be positive, got {self.cap_max}")
        if self.data_factor < 1:
            raise ConfigError(f"data_factor must be at least 1, got {self.data_factor}")
        if self.test_every < 2:
            raise ConfigError(
                f"test_every must be at least 2, got {self.test_every}. Below 2 there "
                f"are no training images left, and at 1 every image is held out."
            )
        if self.strategy not in STRATEGIES:
            raise ConfigError(f"strategy must be one of {STRATEGIES}, got {self.strategy!r}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainConfig:
        _reject_unknown(cls, data)
        return cls(**data)


@dataclass(frozen=True)
class ExportConfig:
    order: str = "morton"

    def __post_init__(self) -> None:
        if self.order not in ORDERS:
            raise ConfigError(f"order must be one of {ORDERS}, got {self.order!r}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExportConfig:
        _reject_unknown(cls, data)
        return cls(**data)


@dataclass(frozen=True)
class RunConfig:
    name: str
    train: TrainConfig = field(default_factory=TrainConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    def __post_init__(self) -> None:
        if not self.name or any(c in self.name for c in r'\/:*?"<>| '):
            raise ConfigError(
                f"name must be a non-empty string usable as a directory name, got {self.name!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def digest(self) -> str:
        """Twelve hex characters identifying the parameters, ignoring the name.

        The name is an identity label, not a parameter. Excluding it means two
        runs with identical settings share a digest, which is what makes
        "have I already computed this?" answerable.
        """
        payload = self.to_dict()
        payload.pop("name")
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunConfig:
        data = dict(data)
        train = TrainConfig.from_dict(data.pop("train", {}))
        export = ExportConfig.from_dict(data.pop("export", {}))
        _reject_unknown(cls, data)
        return cls(train=train, export=export, **data)

    @classmethod
    def from_toml(cls, path: str | Path) -> RunConfig:
        with open(path, "rb") as handle:
            return cls.from_dict(tomllib.load(handle))
```

Note on the import placement: move `from splatpipe.errors import ConfigError` to the top of the file with the other imports when you write it. It is shown mid-file above only so that `_reject_unknown` reads next to its docstring context; there is no cycle and no reason for it to sit below.

- [ ] **Step 4: Write the truck config**

Create `configs/truck.toml`:

```toml
# The spike's run, expressed as configuration. See README.md for the measured result.
name = "truck"

[train]
max_steps = 7000
cap_max = 1000000
data_factor = 1
test_every = 8
strategy = "mcmc"
# No seed key. simple_trainer.py hardcodes set_random_seed(42 + local_rank)
# and exposes no flag for it.

[export]
# Morton gives spatial locality, which groups similar bytes together and is a
# compression lever. "size_opacity" is what the spike used and what
# antimatter15's converter does; it front-loads the biggest splats instead.
order = "morton"
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_config.py -q`
Expected: 8 passed.

- [ ] **Step 6: Commit**

```bash
git add src/splatpipe/config.py configs/truck.toml tests/test_config.py
git commit -m "Add run configuration with validation and a parameter digest"
```

---

## Task 5: Scene validation and output paths

**Files:**
- Create: `src/splatpipe/scene.py`
- Create: `src/splatpipe/paths.py`
- Create: `tests/test_scene.py`
- Create: `tests/test_paths.py`

**Interfaces:**
- Consumes: `splatpipe.errors.SceneError`, `splatpipe.config.RunConfig`
- Produces:
  - `SceneLayout(root: Path, images_dir: Path, sparse_dir: Path, image_count: int)` with `SceneLayout.discover(root: Path, data_factor: int = 1) -> SceneLayout`
  - `RunPaths(root: Path)` with `.config_file`, `.manifest`, `.train_dir`, `.artifacts_dir`, `.logs_dir`, `.ply`, `.splat`, `.train_log`, `.trained_ply(max_steps: int) -> Path`, `.ensure() -> None`, and `RunPaths.for_run(out_root: Path, name: str) -> RunPaths`

The output layout, fixed here and depended on by every later milestone:

```
<out_root>/<name>/
  config.toml          the resolved config exactly as run
  manifest.json        the run record
  train/               gsplat's --result-dir, left in gsplat's own layout
  artifacts/
    scene.ply          the trained cloud
    scene.splat        the 32-byte-per-Gaussian baseline
  logs/
    train.log
```

- [ ] **Step 1: Write the failing tests**

Create `tests/test_scene.py`:

```python
import pytest

from splatpipe.errors import SceneError
from splatpipe.scene import SceneLayout

COLMAP_FILES = ("cameras.bin", "images.bin", "points3D.bin")


def build_scene(root, n_images=12, factor=1):
    images = root / ("images" if factor == 1 else f"images_{factor}")
    images.mkdir(parents=True)
    for i in range(n_images):
        (images / f"{i:04d}.png").write_bytes(b"not really a png")
    if factor != 1:
        plain = root / "images"
        plain.mkdir()
        for i in range(n_images):
            (plain / f"{i:04d}.png").write_bytes(b"not really a png")
    sparse = root / "sparse" / "0"
    sparse.mkdir(parents=True)
    for name in COLMAP_FILES:
        (sparse / name).write_bytes(b"\x00" * 8)
    return root


def test_discovers_a_well_formed_scene(tmp_path):
    layout = SceneLayout.discover(build_scene(tmp_path))
    assert layout.image_count == 12
    assert layout.images_dir == tmp_path / "images"
    assert layout.sparse_dir == tmp_path / "sparse" / "0"


def test_reports_every_missing_piece_at_once(tmp_path):
    with pytest.raises(SceneError) as excinfo:
        SceneLayout.discover(tmp_path)
    message = str(excinfo.value)
    assert "images" in message
    for name in COLMAP_FILES:
        assert name in message


def test_rejects_a_scene_with_too_few_images(tmp_path):
    with pytest.raises(SceneError, match="at least"):
        SceneLayout.discover(build_scene(tmp_path, n_images=3))


def test_factor_two_requires_the_downscaled_folder(tmp_path):
    build_scene(tmp_path, factor=1)
    with pytest.raises(SceneError, match="images_2"):
        SceneLayout.discover(tmp_path, data_factor=2)


def test_factor_two_uses_the_downscaled_folder(tmp_path):
    layout = SceneLayout.discover(build_scene(tmp_path, factor=2), data_factor=2)
    assert layout.images_dir == tmp_path / "images_2"
```

Create `tests/test_paths.py`:

```python
from pathlib import Path

from splatpipe.paths import RunPaths


def test_layout_is_derived_from_root_and_name():
    paths = RunPaths.for_run(Path("out"), "truck")
    assert paths.root == Path("out") / "truck"
    assert paths.config_file == Path("out") / "truck" / "config.toml"
    assert paths.manifest == Path("out") / "truck" / "manifest.json"
    assert paths.ply == Path("out") / "truck" / "artifacts" / "scene.ply"
    assert paths.splat == Path("out") / "truck" / "artifacts" / "scene.splat"
    assert paths.train_log == Path("out") / "truck" / "logs" / "train.log"


def test_trained_ply_follows_gsplats_off_by_one_naming():
    """gsplat writes point_cloud_<max_steps - 1>.ply, so 7000 steps gives 6999."""
    paths = RunPaths.for_run(Path("out"), "truck")
    assert paths.trained_ply(7000).name == "point_cloud_6999.ply"
    assert paths.trained_ply(7000).parent == paths.train_dir / "ply"


def test_ensure_creates_every_directory(tmp_path):
    paths = RunPaths.for_run(tmp_path, "truck")
    paths.ensure()
    for directory in (paths.root, paths.train_dir, paths.artifacts_dir, paths.logs_dir):
        assert directory.is_dir()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_scene.py tests/test_paths.py -q`
Expected: FAIL with `ModuleNotFoundError` for `splatpipe.scene` and `splatpipe.paths`

- [ ] **Step 3: Write `scene.py`**

```python
"""Validate an input scene directory before spending GPU minutes on it.

gsplat's colmap parser raises deep inside its own loading code when something
is missing. Checking up front costs milliseconds and turns a confused traceback
into a list of exactly what to fix.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from splatpipe.errors import SceneError

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
COLMAP_FILES = ("cameras.bin", "images.bin", "points3D.bin")
MIN_IMAGES = 8


@dataclass(frozen=True)
class SceneLayout:
    root: Path
    images_dir: Path
    sparse_dir: Path
    image_count: int

    @classmethod
    def discover(cls, root: Path | str, data_factor: int = 1) -> SceneLayout:
        root = Path(root)
        problems: list[str] = []

        # gsplat needs `images/` regardless, because COLMAP's filenames are
        # recorded against it, plus `images_<factor>/` when downscaling.
        # See _gsplat_repo/examples/datasets/colmap.py, the image_dir_suffix block.
        colmap_images = root / "images"
        images_dir = colmap_images if data_factor == 1 else root / f"images_{data_factor}"

        for directory in dict.fromkeys((colmap_images, images_dir)):
            if not directory.is_dir():
                problems.append(f"missing image folder {directory}")

        sparse_dir = root / "sparse" / "0"
        for name in COLMAP_FILES:
            if not (sparse_dir / name).is_file():
                problems.append(f"missing COLMAP file {sparse_dir / name}")

        image_count = 0
        if images_dir.is_dir():
            image_count = sum(
                1 for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
            )
            if image_count < MIN_IMAGES:
                problems.append(
                    f"{images_dir} holds {image_count} images, need at least {MIN_IMAGES}"
                )

        if problems:
            raise SceneError(
                f"{root} is not a usable scene:\n  - " + "\n  - ".join(problems)
            )

        return cls(
            root=root, images_dir=images_dir, sparse_dir=sparse_dir, image_count=image_count
        )
```

- [ ] **Step 4: Write `paths.py`**

```python
"""Every output path in one place, derived from (out_root, name).

Nothing else in the package builds a path by string concatenation. When the
layout changes it changes here, and the tests catch anything that assumed the
old shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunPaths:
    root: Path

    @classmethod
    def for_run(cls, out_root: Path | str, name: str) -> RunPaths:
        return cls(root=Path(out_root) / name)

    @property
    def config_file(self) -> Path:
        return self.root / "config.toml"

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def train_dir(self) -> Path:
        """gsplat's --result-dir. Left in gsplat's own layout, untouched."""
        return self.root / "train"

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def ply(self) -> Path:
        return self.artifacts_dir / "scene.ply"

    @property
    def splat(self) -> Path:
        return self.artifacts_dir / "scene.splat"

    @property
    def train_log(self) -> Path:
        return self.logs_dir / "train.log"

    def trained_ply(self, max_steps: int) -> Path:
        """Where gsplat writes the final ply. It names files by step index, so
        a 7000-step run produces point_cloud_6999.ply."""
        return self.train_dir / "ply" / f"point_cloud_{max_steps - 1}.ply"

    def ensure(self) -> None:
        for directory in (self.root, self.train_dir, self.artifacts_dir, self.logs_dir):
            directory.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_scene.py tests/test_paths.py -q`
Expected: 8 passed.

- [ ] **Step 6: Commit**

```bash
git add src/splatpipe/scene.py src/splatpipe/paths.py tests/test_scene.py tests/test_paths.py
git commit -m "Add scene validation and the run output layout"
```

---

## Task 6: GaussianCloud and the .ply reader

This is the load-bearing interface of the whole project. `compress/` in milestone 3 consumes it, `bench/` in milestone 2 renders from it, and the container format in milestone 5 serialises it.

**Files:**
- Create: `src/splatpipe/gaussians.py`
- Create: `tests/test_gaussians.py`

**Interfaces:**
- Consumes: `splatpipe.errors.ArtifactError`
- Produces:
  - `GaussianCloud(means, scales, quats, opacities, sh0, shN)` -- a frozen dataclass of numpy arrays with shapes `(N,3) (N,3) (N,4) (N,) (N,3) (N,K,3)`, all float32
  - `len(cloud) -> int`, `cloud.sh_degree -> int`, `cloud.validate() -> None`, `cloud.take(idx) -> GaussianCloud`
  - `read_ply(path) -> GaussianCloud`, `write_ply(cloud, path) -> None`
  - `SH_C0 = 0.28209479177387814`

Two traps this module exists to contain, both verified against the spike's own output:

1. **`scales` are logarithms and `opacities` are logits**, exactly as stored in the `.ply`. Nothing converts them on read. Every consumer applies `exp` and `sigmoid` itself, so there is one convention rather than two.
2. **`f_rest` is channel-major.** gsplat writes `shN.permute(0, 2, 1).reshape(N, -1)`, so `f_rest_0..14` are red's 15 coefficients, `f_rest_15..29` are green's, `f_rest_30..44` are blue's. Reading them in the obvious `(N, K, 3)` order silently produces wrong colours rather than an error.

- [ ] **Step 1: Write the failing test**

Create `tests/test_gaussians.py`:

```python
import numpy as np
import pytest
from plyfile import PlyData

from splatpipe.errors import ArtifactError
from splatpipe.gaussians import GaussianCloud, read_ply, write_ply

RNG = np.random.default_rng(0)


def make_cloud(n=7, k=15):
    return GaussianCloud(
        means=RNG.standard_normal((n, 3)).astype(np.float32),
        scales=RNG.standard_normal((n, 3)).astype(np.float32),
        quats=RNG.standard_normal((n, 4)).astype(np.float32),
        opacities=RNG.standard_normal(n).astype(np.float32),
        sh0=RNG.standard_normal((n, 3)).astype(np.float32),
        shN=RNG.standard_normal((n, k, 3)).astype(np.float32),
    )


def test_length_and_degree():
    assert len(make_cloud(n=7, k=15)) == 7
    assert make_cloud(k=15).sh_degree == 3
    assert make_cloud(k=8).sh_degree == 2
    assert make_cloud(k=3).sh_degree == 1
    assert make_cloud(k=0).sh_degree == 0


def test_validate_rejects_a_length_mismatch():
    cloud = make_cloud(n=7)
    broken = GaussianCloud(
        means=cloud.means[:5],
        scales=cloud.scales,
        quats=cloud.quats,
        opacities=cloud.opacities,
        sh0=cloud.sh0,
        shN=cloud.shN,
    )
    with pytest.raises(ArtifactError, match="means"):
        broken.validate()


def test_take_reorders_every_array_together():
    cloud = make_cloud(n=7)
    reversed_cloud = cloud.take(np.arange(6, -1, -1))
    assert len(reversed_cloud) == 7
    np.testing.assert_array_equal(reversed_cloud.means[0], cloud.means[6])
    np.testing.assert_array_equal(reversed_cloud.shN[0], cloud.shN[6])


def test_ply_round_trip_is_exact(tmp_path):
    cloud = make_cloud(n=7)
    path = tmp_path / "scene.ply"
    write_ply(cloud, path)
    back = read_ply(path)
    for name in ("means", "scales", "quats", "opacities", "sh0", "shN"):
        np.testing.assert_array_equal(getattr(back, name), getattr(cloud, name), err_msg=name)


def test_f_rest_is_channel_major(tmp_path):
    """f_rest_0..14 is red, 15..29 is green, 30..44 is blue.

    Getting this wrong swaps colour channels in the view-dependent term, which
    looks like a subtle lighting bug rather than an error.
    """
    n, k = 4, 15
    shN = np.empty((n, k, 3), dtype=np.float32)
    shN[:, :, 0] = 1.0
    shN[:, :, 1] = 2.0
    shN[:, :, 2] = 3.0
    cloud = GaussianCloud(
        means=np.zeros((n, 3), np.float32),
        scales=np.zeros((n, 3), np.float32),
        quats=np.zeros((n, 4), np.float32),
        opacities=np.zeros(n, np.float32),
        sh0=np.zeros((n, 3), np.float32),
        shN=shN,
    )
    path = tmp_path / "channels.ply"
    write_ply(cloud, path)

    vertex = PlyData.read(path)["vertex"]
    assert vertex["f_rest_0"][0] == 1.0
    assert vertex["f_rest_14"][0] == 1.0
    assert vertex["f_rest_15"][0] == 2.0
    assert vertex["f_rest_29"][0] == 2.0
    assert vertex["f_rest_30"][0] == 3.0
    assert vertex["f_rest_44"][0] == 3.0

    np.testing.assert_array_equal(read_ply(path).shN, shN)


def test_bytes_per_gaussian_matches_the_spike(tmp_path):
    """236 bytes per Gaussian at degree 3: 59 float32 fields."""
    n = 100
    path = tmp_path / "sized.ply"
    write_ply(make_cloud(n=n, k=15), path)
    header_end = path.read_bytes().index(b"end_header\n") + len(b"end_header\n")
    assert (path.stat().st_size - header_end) / n == 236
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_gaussians.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'splatpipe.gaussians'`

- [ ] **Step 3: Write the implementation**

Create `src/splatpipe/gaussians.py`:

```python
"""Struct-of-arrays container for a Gaussian splat scene, plus .ply I/O.

numpy only, deliberately no torch. This is what compress/ consumes, and keeping
it torch-free is what makes the compression tests run in milliseconds on CPU.

Units, which are the main trap here: `scales` are logarithms and `opacities`
are logits, exactly as stored in the .ply. Nothing is converted on read.
Consumers apply exp() and sigmoid() themselves, so there is one convention in
the codebase rather than two.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from splatpipe.errors import ArtifactError

SH_C0 = 0.28209479177387814

_ARRAY_SHAPES = {
    "means": ("N", 3),
    "scales": ("N", 3),
    "quats": ("N", 4),
    "opacities": ("N",),
    "sh0": ("N", 3),
    "shN": ("N", "K", 3),
}


@dataclass(frozen=True)
class GaussianCloud:
    means: np.ndarray  # (N, 3) float32, world space
    scales: np.ndarray  # (N, 3) float32, LOG scale
    quats: np.ndarray  # (N, 4) float32, (w, x, y, z), not normalised
    opacities: np.ndarray  # (N,) float32, LOGIT
    sh0: np.ndarray  # (N, 3) float32, the view-independent DC term
    shN: np.ndarray  # (N, K, 3) float32, K = 15 at degree 3

    def __len__(self) -> int:
        return int(self.means.shape[0])

    @property
    def sh_degree(self) -> int:
        """Degree d has (d+1)^2 coefficients, one of which is the DC term."""
        return int(round((self.shN.shape[1] + 1) ** 0.5)) - 1

    def validate(self) -> None:
        n = len(self)
        for name, expected in _ARRAY_SHAPES.items():
            array = getattr(self, name)
            if array.dtype != np.float32:
                raise ArtifactError(f"{name} has dtype {array.dtype}, expected float32")
            if array.ndim != len(expected):
                raise ArtifactError(
                    f"{name} has shape {array.shape}, expected {len(expected)} dimensions"
                )
            if array.shape[0] != n:
                raise ArtifactError(
                    f"{name} has {array.shape[0]} rows but means has {n}"
                )
            for axis, size in enumerate(expected):
                if isinstance(size, int) and array.shape[axis] != size:
                    raise ArtifactError(
                        f"{name} has shape {array.shape}, expected axis {axis} to be {size}"
                    )
        if (self.shN.shape[1] + 1) ** 0.5 % 1 != 0:
            raise ArtifactError(
                f"shN has {self.shN.shape[1]} coefficients, which is not (d+1)^2 - 1 "
                f"for any integer degree d"
            )

    def take(self, index: np.ndarray) -> GaussianCloud:
        """Reorder or subset every array with the same index."""
        return replace(
            self,
            means=self.means[index],
            scales=self.scales[index],
            quats=self.quats[index],
            opacities=self.opacities[index],
            sh0=self.sh0[index],
            shN=self.shN[index],
        )


def _f_rest_names(k: int) -> list[str]:
    return [f"f_rest_{i}" for i in range(k * 3)]


def read_ply(path: Path | str) -> GaussianCloud:
    """Read a 3DGS .ply into a GaussianCloud.

    Reads by property name, never by byte offset: the field order in the file
    is x, y, z, f_dc, f_rest, opacity, scale, rot, which is not the order the
    struct is usually written down in.
    """
    vertex = PlyData.read(str(path))["vertex"]
    names = set(vertex.data.dtype.names)

    def column(name: str) -> np.ndarray:
        if name not in names:
            raise ArtifactError(f"{path} has no property {name!r}")
        return np.asarray(vertex[name], dtype=np.float32)

    k = sum(1 for name in names if name.startswith("f_rest_")) // 3
    n = len(vertex)

    # Channel-major on disk: (N, 3, K) flattened. Transpose back to (N, K, 3).
    if k:
        flat = np.stack([column(name) for name in _f_rest_names(k)], axis=1)
        shN = flat.reshape(n, 3, k).transpose(0, 2, 1).copy()
    else:
        shN = np.zeros((n, 0, 3), dtype=np.float32)

    cloud = GaussianCloud(
        means=np.stack([column("x"), column("y"), column("z")], axis=1),
        scales=np.stack([column(f"scale_{i}") for i in range(3)], axis=1),
        quats=np.stack([column(f"rot_{i}") for i in range(4)], axis=1),
        opacities=column("opacity"),
        sh0=np.stack([column(f"f_dc_{i}") for i in range(3)], axis=1),
        shN=shN,
    )
    cloud.validate()
    return cloud


def write_ply(cloud: GaussianCloud, path: Path | str) -> None:
    """Write a GaussianCloud as a 3DGS .ply, in the field order gsplat uses."""
    cloud.validate()
    n, k = len(cloud), cloud.shN.shape[1]

    columns: list[tuple[str, np.ndarray]] = [
        ("x", cloud.means[:, 0]),
        ("y", cloud.means[:, 1]),
        ("z", cloud.means[:, 2]),
        ("f_dc_0", cloud.sh0[:, 0]),
        ("f_dc_1", cloud.sh0[:, 1]),
        ("f_dc_2", cloud.sh0[:, 2]),
    ]
    if k:
        flat = cloud.shN.transpose(0, 2, 1).reshape(n, k * 3)
        columns += list(zip(_f_rest_names(k), flat.T))
    columns.append(("opacity", cloud.opacities))
    columns += [(f"scale_{i}", cloud.scales[:, i]) for i in range(3)]
    columns += [(f"rot_{i}", cloud.quats[:, i]) for i in range(4)]

    data = np.empty(n, dtype=[(name, "<f4") for name, _ in columns])
    for name, values in columns:
        data[name] = values

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(data, "vertex")]).write(str(path))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_gaussians.py -q`
Expected: 7 passed.

- [ ] **Step 5: Check it against the spike's real output**

This is a one-off sanity check, not a test -- the 225 MiB `.ply` is gitignored, so it cannot live in the suite.

```
scripts\env.bat
.venv\Scripts\python.exe -c "from splatpipe.gaussians import read_ply; c = read_ply(r'results\truck_spike\ply\point_cloud_6999.ply'); print(len(c), c.sh_degree, c.shN.shape)"
```

Expected: `1000000 3 (1000000, 15, 3)`. If the shN shape is `(1000000, 45, 1)` or similar, the channel-major reshape is wrong.

- [ ] **Step 6: Commit**

```bash
git add src/splatpipe/gaussians.py tests/test_gaussians.py
git commit -m "Add GaussianCloud container and .ply read/write"
```

---

## Task 7: The .splat writer

**Files:**
- Create: `src/splatpipe/formats/__init__.py` (empty)
- Create: `src/splatpipe/formats/splat.py`
- Create: `tests/test_splat_format.py`

**Interfaces:**
- Consumes: `splatpipe.gaussians.GaussianCloud`, `splatpipe.gaussians.SH_C0`, `splatpipe.errors.ConfigError`
- Produces:
  - `BYTES_PER_GAUSSIAN = 32`
  - `morton_order(means: np.ndarray) -> np.ndarray`
  - `size_opacity_order(scales: np.ndarray, opacities: np.ndarray) -> np.ndarray`
  - `encode_splat(cloud: GaussianCloud, order: str = "morton") -> bytes`

The cross-check against `gsplat.exporter.export_splats` is the most valuable test in this task: it proves the byte layout is right against the library the viewer was built for, rather than against my own reading of the format. `gsplat.exporter` imports standalone without triggering the CUDA backend, so this test runs in the fast tier with no vcvars shell.

- [ ] **Step 1: Write the failing test**

Create `tests/test_splat_format.py`:

```python
import numpy as np
import pytest

from splatpipe.errors import ConfigError
from splatpipe.formats.splat import (
    BYTES_PER_GAUSSIAN,
    encode_splat,
    morton_order,
    size_opacity_order,
)
from splatpipe.gaussians import GaussianCloud

RNG = np.random.default_rng(1)


def make_cloud(n=64, k=15):
    return GaussianCloud(
        means=RNG.random((n, 3)).astype(np.float32) * 10,
        scales=RNG.standard_normal((n, 3)).astype(np.float32),
        quats=RNG.standard_normal((n, 4)).astype(np.float32),
        opacities=RNG.standard_normal(n).astype(np.float32),
        sh0=RNG.standard_normal((n, 3)).astype(np.float32),
        shN=RNG.standard_normal((n, k, 3)).astype(np.float32),
    )


def test_output_is_32_bytes_per_gaussian():
    cloud = make_cloud(n=64)
    assert len(encode_splat(cloud)) == 64 * BYTES_PER_GAUSSIAN


def test_orderings_are_permutations():
    cloud = make_cloud(n=64)
    for order in (morton_order(cloud.means), size_opacity_order(cloud.scales, cloud.opacities)):
        np.testing.assert_array_equal(np.sort(order), np.arange(64))


def test_size_opacity_puts_the_biggest_splat_first():
    cloud = make_cloud(n=64)
    order = size_opacity_order(cloud.scales, cloud.opacities)
    weight = np.exp(cloud.scales.sum(axis=1)) / (1 + np.exp(-cloud.opacities))
    assert order[0] == int(np.argmax(weight))


def test_order_none_preserves_input_order():
    cloud = make_cloud(n=4)
    raw = encode_splat(cloud, order="none")
    first_mean = np.frombuffer(raw[:12], dtype="<f4")
    np.testing.assert_allclose(first_mean, cloud.means[0], rtol=0, atol=0)


def test_unknown_order_is_rejected():
    with pytest.raises(ConfigError, match="sideways"):
        encode_splat(make_cloud(n=4), order="sideways")


def test_scales_are_exponentiated_and_opacity_goes_through_sigmoid():
    cloud = make_cloud(n=1)
    raw = encode_splat(cloud, order="none")
    np.testing.assert_allclose(
        np.frombuffer(raw[12:24], dtype="<f4"), np.exp(cloud.scales[0]), rtol=1e-6
    )
    expected_alpha = int(
        np.clip(1 / (1 + np.exp(-cloud.opacities[0])) * 255, 0, 255).astype(np.uint8)
    )
    assert raw[27] == expected_alpha


def test_matches_gsplat_export_splats_byte_for_byte():
    """The authority on this format is the library the viewer was built for.

    gsplat.exporter imports without the CUDA backend, so this needs no GPU.
    Positions are drawn from a continuous distribution, so Morton ties -- where
    argsort order would be implementation-defined -- do not arise.
    """
    torch = pytest.importorskip("torch")
    from gsplat.exporter import export_splats

    cloud = make_cloud(n=64)
    ours = encode_splat(cloud, order="morton")
    theirs = export_splats(
        means=torch.from_numpy(cloud.means),
        scales=torch.from_numpy(cloud.scales),
        quats=torch.from_numpy(cloud.quats),
        opacities=torch.from_numpy(cloud.opacities),
        sh0=torch.from_numpy(cloud.sh0).unsqueeze(1),
        shN=torch.from_numpy(cloud.shN.transpose(0, 2, 1).copy()).permute(0, 2, 1),
        format="splat",
    )
    assert ours == theirs
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_splat_format.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'splatpipe.formats'`

- [ ] **Step 3: Write the implementation**

Create `src/splatpipe/formats/__init__.py` as an empty file, then `src/splatpipe/formats/splat.py`:

```python
"""The .splat format: 32 bytes per Gaussian, as read by antimatter15's viewer.

    position  3 x float32  = 12
    scale     3 x float32  = 12   (exponentiated)
    color     4 x uint8    =  4   (RGB from the SH DC term, alpha from opacity)
    rotation  4 x uint8    =  4   (normalised quaternion, mapped to 0..255)

Note what this throws away: all 45 f_rest coefficients, which is every
view-dependent appearance term. Beating 32 bytes per Gaussian on size alone is
trivial and meaningless; beating it at equal or better quality is the project.

The byte layout here is verified byte-for-byte against gsplat.exporter in
tests/test_splat_format.py.
"""

from __future__ import annotations

import numpy as np

from splatpipe.errors import ConfigError
from splatpipe.gaussians import SH_C0, GaussianCloud

BYTES_PER_GAUSSIAN = 32


def _part1by2(x: np.ndarray) -> np.ndarray:
    """Spread the low 10 bits of x out with two zero bits between each."""
    x = x.astype(np.uint64) & np.uint64(0x000003FF)
    x = (x ^ (x << np.uint64(16))) & np.uint64(0xFF0000FF)
    x = (x ^ (x << np.uint64(8))) & np.uint64(0x0300F00F)
    x = (x ^ (x << np.uint64(4))) & np.uint64(0x030C30C3)
    x = (x ^ (x << np.uint64(2))) & np.uint64(0x09249249)
    return x


def morton_order(means: np.ndarray) -> np.ndarray:
    """Sort indices by Morton code, giving spatial locality.

    Mirrors gsplat.exporter.sort_centers exactly, including its quirk: a point
    at the maximum along an axis scales to exactly 1024, which is 11 bits, and
    the & 0x3FF in _part1by2 truncates it to 0. Reproducing the quirk is what
    keeps the byte-for-byte cross-check honest.
    """
    lo = means.min(axis=0)
    lengths = means.max(axis=0) - lo
    lengths[lengths == 0] = 1
    scaled = np.floor((means - lo) / lengths * 1024).astype(np.int64)
    x, y, z = scaled[:, 0], scaled[:, 1], scaled[:, 2]
    codes = (_part1by2(z) << np.uint64(2)) + (_part1by2(y) << np.uint64(1)) + _part1by2(x)
    return np.argsort(codes)


def size_opacity_order(scales: np.ndarray, opacities: np.ndarray) -> np.ndarray:
    """Sort indices by projected size times opacity, biggest first.

    This is what antimatter15's own converter does. It front-loads the splats
    that matter most visually, so a partially loaded file still looks roughly
    right. Morton is the better default for compression; this one is here to be
    measured against it.
    """
    weight = np.exp(scales.sum(axis=1)) / (1 + np.exp(-opacities))
    return np.argsort(-weight)


def encode_splat(cloud: GaussianCloud, order: str = "morton") -> bytes:
    """Pack a GaussianCloud into .splat bytes."""
    cloud.validate()

    if order == "morton":
        cloud = cloud.take(morton_order(cloud.means))
    elif order == "size_opacity":
        cloud = cloud.take(size_opacity_order(cloud.scales, cloud.opacities))
    elif order != "none":
        raise ConfigError(f"unknown order {order!r}, expected morton, size_opacity or none")

    n = len(cloud)

    positions = np.ascontiguousarray(cloud.means, dtype="<f4")
    scales = np.ascontiguousarray(np.exp(cloud.scales), dtype="<f4")

    rgb = cloud.sh0 * SH_C0 + 0.5
    alpha = 1.0 / (1.0 + np.exp(-cloud.opacities))
    color = np.clip(np.concatenate([rgb, alpha[:, None]], axis=1) * 255, 0, 255).astype(np.uint8)

    quats = cloud.quats / np.linalg.norm(cloud.quats, axis=1, keepdims=True)
    rotation = np.clip(quats * 128 + 128, 0, 255).astype(np.uint8)

    buffer = np.empty((n, BYTES_PER_GAUSSIAN), dtype=np.uint8)
    buffer[:, 0:12] = positions.view(np.uint8).reshape(n, 12)
    buffer[:, 12:24] = scales.view(np.uint8).reshape(n, 12)
    buffer[:, 24:28] = color
    buffer[:, 28:32] = rotation
    return buffer.tobytes()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_splat_format.py -q`
Expected: 7 passed. `test_matches_gsplat_export_splats_byte_for_byte` is the one that matters; if it fails, the discrepancy is in the ordering or in the uint8 rounding, not in the float fields.

- [ ] **Step 5: Delete the superseded spike script**

`_ply_to_splat.py` is now covered by `read_ply` plus `encode_splat`, with tests.

```bash
git rm _ply_to_splat.py
```

Delete its row from the `## Files` table in `README.md`.

- [ ] **Step 6: Commit**

```bash
git add src/splatpipe/formats tests/test_splat_format.py README.md
git commit -m "Add .splat writer with selectable ordering, verified against gsplat"
```

---

## Task 8: Run manifest

**Files:**
- Create: `src/splatpipe/manifest.py`
- Create: `tests/test_manifest.py`

**Interfaces:**
- Consumes: `splatpipe.config.RunConfig`
- Produces:
  - `ArtifactRecord(path: str, bytes: int, sha256: str)` with `ArtifactRecord.of(path: Path, relative_to: Path) -> ArtifactRecord`
  - `RunManifest(name, config_digest, config, created_utc, versions, timings, artifacts, metrics)` with `.write(path)`, `RunManifest.read(path) -> RunManifest`, `RunManifest.start(cfg: RunConfig) -> RunManifest`
  - `collect_versions() -> dict[str, str]`

Without this you cannot answer "which config produced this file", and milestone 2's rate-distortion curve is a set of unlabelled points.

- [ ] **Step 1: Write the failing test**

Create `tests/test_manifest.py`:

```python
import hashlib

from splatpipe.config import RunConfig, TrainConfig
from splatpipe.manifest import ArtifactRecord, RunManifest, collect_versions


def test_artifact_record_hashes_the_file(tmp_path):
    payload = b"forty two" * 100
    target = tmp_path / "artifacts" / "scene.splat"
    target.parent.mkdir()
    target.write_bytes(payload)

    record = ArtifactRecord.of(target, relative_to=tmp_path)
    assert record.path == "artifacts/scene.splat"
    assert record.bytes == len(payload)
    assert record.sha256 == hashlib.sha256(payload).hexdigest()


def test_start_captures_the_config_and_its_digest():
    cfg = RunConfig(name="tiny", train=TrainConfig(max_steps=50))
    manifest = RunManifest.start(cfg)
    assert manifest.name == "tiny"
    assert manifest.config_digest == cfg.digest()
    assert manifest.config["train"]["max_steps"] == 50
    assert manifest.created_utc.endswith("Z")


def test_round_trips_through_json(tmp_path):
    manifest = RunManifest.start(RunConfig(name="tiny"))
    manifest.timings["train"] = 12.5
    manifest.metrics["psnr"] = 24.4
    manifest.artifacts.append(ArtifactRecord(path="artifacts/scene.splat", bytes=32, sha256="ab"))

    path = tmp_path / "manifest.json"
    manifest.write(path)
    assert RunManifest.read(path) == manifest


def test_versions_include_what_would_change_a_result():
    versions = collect_versions()
    assert versions["python"].startswith("3.11")
    for key in ("numpy", "splatpipe", "platform"):
        assert versions[key]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_manifest.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'splatpipe.manifest'`

- [ ] **Step 3: Write the implementation**

Create `src/splatpipe/manifest.py`:

```python
"""The record of what a run actually did.

Every artifact directory carries one of these. It is what makes a point on the
rate-distortion curve traceable back to the settings that produced it, which is
the difference between a measurement and a number.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from splatpipe import __version__
from splatpipe.config import RunConfig


@dataclass(frozen=True)
class ArtifactRecord:
    path: str
    bytes: int
    sha256: str

    @classmethod
    def of(cls, path: Path, relative_to: Path) -> ArtifactRecord:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return cls(
            path=path.relative_to(relative_to).as_posix(),
            bytes=path.stat().st_size,
            sha256=digest.hexdigest(),
        )


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def collect_versions() -> dict[str, str]:
    """Everything whose change could move a number in this run."""
    import numpy

    versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "splatpipe": __version__,
        "splatpipe_commit": _git_commit(),
    }
    for name in ("torch", "gsplat"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:  # noqa: BLE001 - torch/gsplat are optional outside training
            versions[name] = "not-imported"
    return versions


@dataclass
class RunManifest:
    name: str
    config_digest: str
    config: dict[str, Any]
    created_utc: str
    versions: dict[str, str] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    artifacts: list[ArtifactRecord] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    @classmethod
    def start(cls, cfg: RunConfig) -> RunManifest:
        return cls(
            name=cfg.name,
            config_digest=cfg.digest(),
            config=cfg.to_dict(),
            created_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            versions=collect_versions(),
        )

    def write(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def read(cls, path: Path | str) -> RunManifest:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data["artifacts"] = [ArtifactRecord(**record) for record in data["artifacts"]]
        return cls(**data)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_manifest.py -q`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/splatpipe/manifest.py tests/test_manifest.py
git commit -m "Add run manifest recording config, versions, timings and artifact digests"
```

---

## Task 9: A synthetic COLMAP scene for testing

**Files:**
- Create: `tests/fixtures/__init__.py` (empty)
- Create: `tests/fixtures/colmap_bin.py`
- Create: `tests/fixtures/tiny_scene.py`
- Create: `tests/test_tiny_scene.py`

**Interfaces:**
- Consumes: nothing from earlier tasks
- Produces:
  - `write_cameras_bin(path, cameras)` where `cameras` is `list[tuple[int, int, int, int, Sequence[float]]]` of `(camera_id, model_id, width, height, params)`
  - `write_images_bin(path, images)` where `images` is `list[tuple[int, Sequence[float], Sequence[float], int, str, np.ndarray, np.ndarray]]` of `(image_id, qvec, tvec, camera_id, name, xys, point3D_ids)`
  - `write_points3D_bin(path, points)` where `points` is `list[tuple[int, Sequence[float], Sequence[int], float, np.ndarray]]` of `(point_id, xyz, rgb, error, track)`
  - `make_tiny_scene(root: Path, n_images: int = 24, width: int = 96, height: int = 72, n_points: int = 512) -> Path`

Two reasons this exists rather than shipping a small real scene. A real scene is megabytes in git for something regenerable in a second. And writing the format by hand is the only way to be sure the struct widths are right, since pycolmap's own write path is still broken on Windows and cannot be used as a reference.

Formats read straight off pycolmap's reader in `scene_manager.py`:

| File | Header | Per record |
|---|---|---|
| `cameras.bin` | `<Q` count | `<I` id, `<i` model, `<Q` width, `<Q` height, `<{n}d` params |
| `images.bin` | `<Q` count | `<I 4d 3d I` id, qvec (w,x,y,z), tvec, camera_id; NUL-terminated name; `<Q` count2D; then per point `<2d` xy and `<Q` point3D_id |
| `points3D.bin` | `<Q` count | `<Q 3d 3B d Q` id, xyz, rgb, error, track_len; then per element `<II` image_id, point2D_idx |

The `<Q` point3D_id in `images.bin` looks wrong next to two doubles, and is not: the reader slurps three doubles per point and then reinterprets the third one's bits as a uint64.

- [ ] **Step 1: Write the failing test**

Create `tests/test_tiny_scene.py`:

```python
import pytest

from tests.fixtures.tiny_scene import make_tiny_scene


def test_scene_has_the_files_gsplat_needs(tmp_path):
    root = make_tiny_scene(tmp_path / "tiny", n_images=24, n_points=512)
    assert len(list((root / "images").glob("*.png"))) == 24
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        assert (root / "sparse" / "0" / name).is_file()


def test_scene_passes_our_own_validation(tmp_path):
    from splatpipe.scene import SceneLayout

    layout = SceneLayout.discover(make_tiny_scene(tmp_path / "tiny", n_images=24))
    assert layout.image_count == 24


def test_pycolmap_reads_back_what_we_wrote(tmp_path):
    """The real check: the library that consumes these files agrees with us."""
    pytest.importorskip("pycolmap")
    from pycolmap import SceneManager

    root = make_tiny_scene(tmp_path / "tiny", n_images=24, n_points=512)
    manager = SceneManager(str(root / "sparse" / "0") + "/")
    manager.load_cameras()
    manager.load_images()
    manager.load_points3D()

    assert len(manager.cameras) == 1
    assert len(manager.images) == 24
    assert manager.points3D.shape == (512, 3)

    camera = next(iter(manager.cameras.values()))
    assert (camera.width, camera.height) == (96, 72)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_tiny_scene.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'tests.fixtures'`

- [ ] **Step 3: Write the COLMAP binary writer**

Create `tests/fixtures/__init__.py` as an empty file and `tests/__init__.py` as an empty file (so `tests.fixtures` is importable), then `tests/fixtures/colmap_bin.py`:

```python
"""Write COLMAP sparse model binaries.

Every format string is explicitly little-endian with explicit widths. pycolmap's
own write path (scene_manager.py lines 313-421) still uses native 'L' and is
broken on Windows, so it cannot be used as a reference implementation here.

Layouts taken from pycolmap's reader in scene_manager.py.
"""

from __future__ import annotations

import struct
from collections.abc import Sequence
from pathlib import Path

import numpy as np

PINHOLE = 1  # model id; 4 params: fx, fy, cx, cy


def write_cameras_bin(path: Path, cameras: Sequence[tuple[int, int, int, int, Sequence[float]]]) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(cameras)))
        for camera_id, model_id, width, height, params in cameras:
            f.write(struct.pack("<IiQQ", camera_id, model_id, width, height))
            f.write(struct.pack(f"<{len(params)}d", *params))


def write_images_bin(path: Path, images: Sequence[tuple]) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for image_id, qvec, tvec, camera_id, name, xys, point3D_ids in images:
            f.write(struct.pack("<I4d3dI", image_id, *qvec, *tvec, camera_id))
            f.write(name.encode("utf-8") + b"\x00")
            f.write(struct.pack("<Q", len(xys)))
            for (x, y), point_id in zip(xys, point3D_ids):
                # Two doubles then a raw uint64. The reader slurps three doubles
                # and reinterprets the third one's bits as the point id.
                f.write(struct.pack("<2d", float(x), float(y)))
                f.write(struct.pack("<Q", int(point_id)))


def write_points3D_bin(path: Path, points: Sequence[tuple]) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(points)))
        for point_id, xyz, rgb, error, track in points:
            f.write(
                struct.pack(
                    "<Q3d3BdQ",
                    point_id,
                    *(float(v) for v in xyz),
                    *(int(v) for v in rgb),
                    float(error),
                    len(track),
                )
            )
            for image_id, point2D_idx in track:
                f.write(struct.pack("<II", int(image_id), int(point2D_idx)))


def look_at_quaternion(camera_position: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (qvec, tvec) in COLMAP's world-to-camera convention.

    qvec is (w, x, y, z). COLMAP stores the rotation that takes world points
    into the camera frame, and tvec is that same transform's translation, so
    tvec = -R @ camera_position rather than the camera position itself.
    """
    forward = target - camera_position
    forward = forward / np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)

    # Rows are the camera axes expressed in world coordinates: x right, y down,
    # z forward, which is COLMAP's image convention.
    rotation = np.stack([right, down, forward], axis=0)
    translation = -rotation @ camera_position

    trace = np.trace(rotation)
    w = np.sqrt(max(0.0, 1.0 + trace)) / 2.0
    if w < 1e-8:  # 180-degree rotation; not reachable for the ring below
        raise ValueError("degenerate rotation")
    x = (rotation[2, 1] - rotation[1, 2]) / (4 * w)
    y = (rotation[0, 2] - rotation[2, 0]) / (4 * w)
    z = (rotation[1, 0] - rotation[0, 1]) / (4 * w)
    return np.array([w, x, y, z]), translation
```

- [ ] **Step 4: Write the tiny scene generator**

Create `tests/fixtures/tiny_scene.py`:

```python
"""A synthetic scene small enough to train in seconds.

Cameras sit on a ring looking at the origin, and the images are the 3D points
projected into each view as coloured dots. It is not a good reconstruction and
it does not need to be -- it exists so the end-to-end test exercises the real
trainer without a 7 minute wait.

The defaults are chosen to clear gsplat's minimums: at least 8 images so that
test_every=8 leaves a non-empty test split, and enough points for the k=4
nearest-neighbour initialisation in examples/utils.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from tests.fixtures.colmap_bin import (
    PINHOLE,
    look_at_quaternion,
    write_cameras_bin,
    write_images_bin,
    write_points3D_bin,
)


def make_tiny_scene(
    root: Path,
    n_images: int = 24,
    width: int = 96,
    height: int = 72,
    n_points: int = 512,
    seed: int = 0,
) -> Path:
    """Write a complete COLMAP scene under `root` and return `root`."""
    root = Path(root)
    images_dir = root / "images"
    sparse_dir = root / "sparse" / "0"
    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    points = rng.uniform(-1.0, 1.0, size=(n_points, 3))
    colors = rng.integers(0, 256, size=(n_points, 3), dtype=np.uint8)

    focal = float(width)
    cx, cy = width / 2.0, height / 2.0
    intrinsics = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]])

    camera_records = [(1, PINHOLE, width, height, (focal, focal, cx, cy))]
    image_records = []
    tracks: dict[int, list[tuple[int, int]]] = {i: [] for i in range(1, n_points + 1)}

    for index in range(n_images):
        angle = 2 * np.pi * index / n_images
        position = np.array([4.0 * np.cos(angle), 4.0 * np.sin(angle), 1.5])
        qvec, tvec = look_at_quaternion(position, np.zeros(3))

        rotation = _quaternion_to_matrix(qvec)
        camera_points = points @ rotation.T + tvec
        in_front = camera_points[:, 2] > 1e-3
        projected = (camera_points[in_front] @ intrinsics.T)
        pixels = projected[:, :2] / projected[:, 2:3]

        on_screen = (
            (pixels[:, 0] >= 0) & (pixels[:, 0] < width)
            & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
        )
        visible_ids = np.flatnonzero(in_front)[on_screen]
        pixels = pixels[on_screen]

        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        for (x, y), point_index in zip(pixels, visible_ids):
            canvas[int(y), int(x)] = colors[point_index]
        name = f"{index:04d}.png"
        Image.fromarray(canvas).save(images_dir / name)

        for slot, point_index in enumerate(visible_ids):
            tracks[int(point_index) + 1].append((index + 1, slot))

        image_records.append(
            (index + 1, qvec, tvec, 1, name, pixels, visible_ids + 1)
        )

    point_records = [
        (i + 1, points[i], colors[i], 0.5, tracks[i + 1]) for i in range(n_points)
    ]

    write_cameras_bin(sparse_dir / "cameras.bin", camera_records)
    write_images_bin(sparse_dir / "images.bin", image_records)
    write_points3D_bin(sparse_dir / "points3D.bin", point_records)
    return root


def _quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_tiny_scene.py -q`
Expected: 3 passed. `test_pycolmap_reads_back_what_we_wrote` is the one that proves the struct widths; if it raises `struct.error: unpack requires a buffer of N bytes`, a format string is wrong.

- [ ] **Step 6: Commit**

```bash
git add tests/__init__.py tests/fixtures tests/test_tiny_scene.py
git commit -m "Add a synthetic COLMAP scene fixture for fast end-to-end tests"
```

---

## Task 10: Training stage and the CLI

**Files:**
- Create: `src/splatpipe/stages/__init__.py` (empty)
- Create: `src/splatpipe/stages/train.py`
- Create: `src/splatpipe/cli.py`
- Create: `tests/test_train_command.py`
- Create: `tests/test_pipeline_e2e.py`

**Interfaces:**
- Consumes: `RunConfig`, `RunPaths`, `SceneLayout`, `RunManifest`, `ArtifactRecord`, `check_build_env`, `read_ply`, `encode_splat`
- Produces:
  - `build_train_command(python: Path, trainer_dir: Path, scene: Path, result_dir: Path, cfg: TrainConfig) -> list[str]`
  - `run_training(scene: Path, paths: RunPaths, cfg: RunConfig, python: Path | None = None) -> float` returning elapsed seconds
  - `read_val_metrics(train_dir: Path, max_steps: int) -> dict[str, float]`
  - `run_pipeline(scene_dir: Path, cfg: RunConfig, out_root: Path, skip_train: bool = False) -> RunPaths`
  - `main(argv: list[str] | None = None) -> int`

`build_train_command` is pure and gets a real test, because the flag names are the part that breaks. tyro takes hyphens, not underscores, and the failure mode is a confusing argparse error several minutes into a session.

`--skip-train` reuses an existing `train/` directory. That is how you iterate on export and, from milestone 3, on compression without paying for training every time. On a weekends-only schedule it is the difference between a ten second loop and an eight minute one.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_train_command.py`:

```python
from pathlib import Path

import pytest

from splatpipe.config import TrainConfig
from splatpipe.stages.train import build_train_command, read_val_metrics


def command_for(**overrides):
    return build_train_command(
        python=Path("py.exe"),
        trainer_dir=Path("gsplat/examples"),
        scene=Path("data/truck"),
        result_dir=Path("out/truck/train"),
        cfg=TrainConfig(**overrides),
    )


def test_strategy_is_a_positional_subcommand():
    assert command_for(strategy="mcmc")[:3] == ["py.exe", "simple_trainer.py", "mcmc"]


def test_every_flag_uses_hyphens_not_underscores():
    """tyro derives flags from field names with hyphens. Underscores fail."""
    for token in command_for():
        if token.startswith("--"):
            assert "_" not in token, token


def test_values_come_from_the_config():
    command = command_for(max_steps=50, cap_max=5000, data_factor=2, test_every=4)
    for flag, value in (
        ("--max-steps", "50"),
        ("--strategy.cap-max", "5000"),
        ("--data-factor", "2"),
        ("--test-every", "4"),
    ):
        assert command[command.index(flag) + 1] == value


def test_no_seed_flag_is_passed():
    """simple_trainer.py has no seed field; passing one would make tyro exit 2."""
    assert not [token for token in command_for() if "seed" in token]


def test_viewer_is_disabled_and_ply_is_saved():
    command = command_for()
    assert "--disable-viewer" in command
    assert "--save-ply" in command


def test_cap_max_is_only_passed_to_the_mcmc_strategy():
    """--strategy.cap-max is an MCMC field; the default strategy rejects it."""
    assert "--strategy.cap-max" not in command_for(strategy="default")


def test_read_val_metrics(tmp_path):
    stats = tmp_path / "stats"
    stats.mkdir()
    (stats / "val_step6999.json").write_text(
        '{"psnr": 24.4, "ssim": 0.858, "lpips": 0.137, "num_GS": 1000000}', encoding="utf-8"
    )
    metrics = read_val_metrics(tmp_path, max_steps=7000)
    assert metrics["psnr"] == pytest.approx(24.4)
    assert metrics["num_GS"] == 1000000
```

Create `tests/test_pipeline_e2e.py`:

```python
import pytest

from splatpipe.cli import main
from splatpipe.config import ExportConfig, RunConfig, TrainConfig
from splatpipe.formats.splat import BYTES_PER_GAUSSIAN
from splatpipe.gaussians import read_ply
from splatpipe.manifest import RunManifest
from splatpipe.paths import RunPaths
from tests.fixtures.tiny_scene import make_tiny_scene

pytestmark = pytest.mark.gpu


def test_pipeline_produces_every_artifact(tmp_path):
    """Run the real trainer on a synthetic scene. Needs a GPU and env.bat.

    Deliberately tiny: 24 images at 96x72, 200 steps, 5000 Gaussians. The point
    is that every stage runs and hands its output to the next one, not that the
    result looks like anything.
    """
    scene = make_tiny_scene(tmp_path / "scene", n_images=24, n_points=512)
    config_path = tmp_path / "tiny.toml"
    config_path.write_text(
        'name = "tiny"\n\n[train]\nmax_steps = 200\ncap_max = 5000\ntest_every = 8\n',
        encoding="utf-8",
    )
    out_root = tmp_path / "out"

    assert main(["run", str(scene), "--config", str(config_path), "--out", str(out_root)]) == 0

    paths = RunPaths.for_run(out_root, "tiny")
    assert paths.ply.is_file()
    assert paths.splat.is_file()
    assert paths.config_file.is_file()

    cloud = read_ply(paths.ply)
    assert len(cloud) > 0
    assert cloud.sh_degree == 3
    assert paths.splat.stat().st_size == len(cloud) * BYTES_PER_GAUSSIAN

    manifest = RunManifest.read(paths.manifest)
    assert manifest.name == "tiny"
    assert manifest.config_digest == RunConfig.from_toml(config_path).digest()
    assert manifest.timings["train"] > 0
    assert {"psnr", "ssim", "lpips"} <= set(manifest.metrics)
    assert {record.path for record in manifest.artifacts} == {
        "artifacts/scene.ply",
        "artifacts/scene.splat",
    }


def test_skip_train_reuses_the_existing_training_output(tmp_path):
    """The loop that makes weekends work: re-export without re-training."""
    scene = make_tiny_scene(tmp_path / "scene", n_images=24, n_points=512)
    config_path = tmp_path / "tiny.toml"
    config_path.write_text(
        'name = "tiny"\n\n[train]\nmax_steps = 200\ncap_max = 5000\n\n[export]\norder = "morton"\n',
        encoding="utf-8",
    )
    out_root = tmp_path / "out"
    main(["run", str(scene), "--config", str(config_path), "--out", str(out_root)])

    paths = RunPaths.for_run(out_root, "tiny")
    morton_bytes = paths.splat.read_bytes()

    config_path.write_text(
        'name = "tiny"\n\n[train]\nmax_steps = 200\ncap_max = 5000\n\n'
        '[export]\norder = "size_opacity"\n',
        encoding="utf-8",
    )
    assert (
        main(
            [
                "run",
                str(scene),
                "--config",
                str(config_path),
                "--out",
                str(out_root),
                "--skip-train",
            ]
        )
        == 0
    )

    reordered = paths.splat.read_bytes()
    assert len(reordered) == len(morton_bytes)
    assert reordered != morton_bytes
    assert RunManifest.read(paths.manifest).timings.get("train") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_train_command.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'splatpipe.stages'`

- [ ] **Step 3: Write the training stage**

Create `src/splatpipe/stages/__init__.py` as an empty file, then `src/splatpipe/stages/train.py`:

```python
"""Run gsplat's simple_trainer.py as a subprocess.

A subprocess rather than an import, because the trainer does `from datasets.colmap
import Parser` and `from utils import knn`, both of which resolve relative to its
own directory. Importing it would mean mutating sys.path from inside a library,
and the subprocess boundary also gives us a clean log file and a clean timing.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from splatpipe.config import RunConfig, TrainConfig
from splatpipe.errors import ArtifactError
from splatpipe.paths import RunPaths

TRAINER_DIR = Path(__file__).resolve().parents[3] / "_gsplat_repo" / "examples"


def build_train_command(
    python: Path,
    trainer_dir: Path,
    scene: Path,
    result_dir: Path,
    cfg: TrainConfig,
) -> list[str]:
    """Build the trainer invocation.

    tyro derives flag names from field names with hyphens, so --max-steps works
    and --max_steps does not.
    """
    command = [
        str(python),
        "simple_trainer.py",
        cfg.strategy,
        "--data-dir",
        str(scene),
        "--result-dir",
        str(result_dir),
        "--data-factor",
        str(cfg.data_factor),
        "--max-steps",
        str(cfg.max_steps),
        "--test-every",
        str(cfg.test_every),
        "--save-ply",
        "--disable-viewer",
    ]
    if cfg.strategy == "mcmc":
        # cap_max is a field of MCMCStrategy; the default strategy has no such flag.
        command += ["--strategy.cap-max", str(cfg.cap_max)]
    return command


def read_val_metrics(train_dir: Path, max_steps: int) -> dict[str, float]:
    """Read gsplat's held-out evaluation stats for the final step."""
    stats = train_dir / "stats" / f"val_step{max_steps - 1}.json"
    if not stats.is_file():
        raise ArtifactError(f"training produced no validation stats at {stats}")
    return json.loads(stats.read_text(encoding="utf-8"))


def run_training(
    scene: Path,
    paths: RunPaths,
    cfg: RunConfig,
    python: Path | None = None,
    trainer_dir: Path = TRAINER_DIR,
) -> float:
    """Train, streaming output to paths.train_log. Returns elapsed seconds."""
    command = build_train_command(
        python=Path(python or sys.executable),
        trainer_dir=trainer_dir,
        scene=scene.resolve(),
        result_dir=paths.train_dir.resolve(),
        cfg=cfg.train,
    )

    started = time.time()
    with open(paths.train_log, "w", encoding="utf-8") as log:
        log.write(" ".join(command) + "\n\n")
        log.flush()
        subprocess.run(
            command,
            cwd=str(trainer_dir),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return time.time() - started
```

- [ ] **Step 4: Write the CLI**

Create `src/splatpipe/cli.py`:

```python
"""`splatpipe run <scene-dir> --config <cfg> --out <dir>`.

A pure function of (input directory, config) to artifacts: nothing is read from
the current working directory and nothing is prompted for.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

from splatpipe.config import RunConfig
from splatpipe.env import check_build_env
from splatpipe.errors import ArtifactError, SplatpipeError
from splatpipe.formats.splat import encode_splat
from splatpipe.gaussians import read_ply
from splatpipe.manifest import ArtifactRecord, RunManifest
from splatpipe.paths import RunPaths
from splatpipe.scene import SceneLayout
from splatpipe.stages.train import read_val_metrics, run_training


def run_pipeline(
    scene_dir: Path,
    cfg: RunConfig,
    out_root: Path,
    skip_train: bool = False,
) -> RunPaths:
    layout = SceneLayout.discover(scene_dir, data_factor=cfg.train.data_factor)
    paths = RunPaths.for_run(out_root, cfg.name)
    paths.ensure()

    manifest = RunManifest.start(cfg)
    shutil.copyfile(cfg_source(cfg, paths), paths.config_file)

    trained_ply = paths.trained_ply(cfg.train.max_steps)
    if skip_train:
        if not trained_ply.is_file():
            raise ArtifactError(
                f"--skip-train was given but {trained_ply} does not exist. "
                f"Run once without it first."
            )
        print(f"reusing existing training output at {paths.train_dir}")
    else:
        check_build_env()
        print(f"training {layout.image_count} images for {cfg.train.max_steps} steps")
        manifest.timings["train"] = run_training(scene_dir, paths, cfg)

    manifest.metrics = read_val_metrics(paths.train_dir, cfg.train.max_steps)

    started = time.time()
    shutil.copyfile(trained_ply, paths.ply)
    cloud = read_ply(paths.ply)
    paths.splat.write_bytes(encode_splat(cloud, order=cfg.export.order))
    manifest.timings["export"] = time.time() - started

    manifest.artifacts = [
        ArtifactRecord.of(paths.ply, relative_to=paths.root),
        ArtifactRecord.of(paths.splat, relative_to=paths.root),
    ]
    manifest.write(paths.manifest)

    print(f"{len(cloud):,} gaussians")
    for record in manifest.artifacts:
        print(f"  {record.path}  {record.bytes / 2**20:.1f} MiB")
    return paths


def cfg_source(cfg: RunConfig, paths: RunPaths) -> Path:
    """The config file this run was loaded from, recorded alongside the output."""
    return cfg.source_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="splatpipe")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="train and package one scene")
    run.add_argument("scene", type=Path, help="a COLMAP scene directory")
    run.add_argument("--config", type=Path, required=True, help="a run config TOML")
    run.add_argument("--out", type=Path, required=True, help="output root directory")
    run.add_argument(
        "--skip-train",
        action="store_true",
        help="reuse an existing train/ directory instead of training again",
    )

    args = parser.parse_args(argv)
    try:
        run_pipeline(args.scene, RunConfig.from_toml(args.config), args.out, args.skip_train)
    except SplatpipeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Note the `cfg_source` helper above references `cfg.source_path`, which `RunConfig` does not have. Fix this properly in the next step rather than leaving it: `RunConfig.from_toml` should record where it came from.

- [ ] **Step 5: Record the config's source path**

In `src/splatpipe/config.py`, add a non-comparing field to `RunConfig` and set it in `from_toml`:

```python
@dataclass(frozen=True)
class RunConfig:
    name: str
    train: TrainConfig = field(default_factory=TrainConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    source_path: Path | None = field(default=None, compare=False, repr=False)
```

In `to_dict`, drop it, because it is provenance and not a parameter:

```python
    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("source_path")
        return data
```

In `from_toml`, set it:

```python
    @classmethod
    def from_toml(cls, path: str | Path) -> RunConfig:
        with open(path, "rb") as handle:
            data = tomllib.load(handle)
        return replace(cls.from_dict(data), source_path=Path(path))
```

Add `replace` to the `dataclasses` import. Then in `cli.py` delete the `cfg_source` function and replace its call site with:

```python
    if cfg.source_path is not None:
        shutil.copyfile(cfg.source_path, paths.config_file)
```

Add a test to `tests/test_config.py`:

```python
def test_source_path_is_recorded_but_not_a_parameter(tmp_path):
    path = tmp_path / "run.toml"
    path.write_text('name = "tiny"\n', encoding="utf-8")
    cfg = RunConfig.from_toml(path)
    assert cfg.source_path == path
    assert "source_path" not in cfg.to_dict()
    assert cfg.digest() == RunConfig(name="tiny").digest()
```

- [ ] **Step 6: Run the fast tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest -q`
Expected: all fast tests pass, `tests/test_pipeline_e2e.py` deselected.

- [ ] **Step 7: Run the end-to-end test**

```
scripts\env.bat
.venv\Scripts\python.exe -m pytest tests/test_pipeline_e2e.py -q -m ""
```

Expected: 2 passed, in roughly a minute. This is the first time the whole chain runs, so treat a failure here as the real work of the task rather than a surprise. If gsplat rejects the synthetic scene, the trainer's output is in the test's temporary `logs/train.log`; raise `n_points` or `n_images` in the fixture rather than guessing.

- [ ] **Step 8: Commit**

```bash
git add src/splatpipe/stages src/splatpipe/cli.py src/splatpipe/config.py tests/
git commit -m "Add the training stage and the splatpipe run CLI"
```

---

## Task 11: Reproduce the truck scene and update the documentation

The point of this task is falsifying the claim that the pipeline reproduces the spike. Until the numbers match, everything above is untested against reality.

**Files:**
- Modify: `README.md`
- Delete: `_get_data.py`, `_gsplat_smoke.py` (or keep with a note; see step 5)
- Create: `scripts/get_data.py`

- [ ] **Step 1: Move the data downloader into scripts**

`git mv _get_data.py scripts/get_data.py`, then change its `ROOT` line so it resolves the repository root rather than its own directory:

```python
ROOT = Path(__file__).resolve().parent.parent
```

Verify it still finds the already-downloaded data rather than re-downloading 650 MB:

```
.venv\Scripts\python.exe scripts\get_data.py
```

Expected: `zip already present`, `images_2 already built, skipping`, `images_4 already built, skipping`, then `DATA READY`.

- [ ] **Step 2: Run the pipeline on truck**

```
scripts\env.bat
.venv\Scripts\python.exe -m splatpipe.cli run data\tandt\truck --config configs\truck.toml --out out
```

Expected: roughly 8 minutes, then `1,000,000 gaussians`, `artifacts/scene.ply  225.1 MiB`, `artifacts/scene.splat  30.5 MiB`.

- [ ] **Step 3: Check the result against the spike**

```
.venv\Scripts\python.exe -c "import json; m = json.load(open('out/truck/manifest.json')); print(m['metrics']); print(m['timings']); print([(a['path'], a['bytes']) for a in m['artifacts']])"
```

Compare against the spike's recorded numbers: PSNR 24.406, SSIM 0.8580, LPIPS 0.1372, `.ply` 236,001,478 bytes.

The `.ply` size should match exactly, since it is a function of the Gaussian count and the field list. The metrics should match to about two decimal places: the seed is fixed upstream at 42, so the run is deterministic in principle, but CUDA reductions are not bit-reproducible and the digits will drift. **A PSNR differing by more than about 0.1 means something is genuinely different and needs investigating before moving on.**

- [ ] **Step 4: Check the .splat against the spike's**

The spike's converter used size-times-opacity ordering and `configs/truck.toml` uses Morton, so the bytes differ by construction. The size must not:

```
.venv\Scripts\python.exe -c "from pathlib import Path; print(Path('out/truck/artifacts/scene.splat').stat().st_size, 1000000*32)"
```

Expected: `32000000 32000000`.

- [ ] **Step 5: Retire the remaining spike scripts**

`_gsplat_smoke.py` was a one-off to force the first JIT compile. The e2e test now covers that ground.

```bash
git rm _gsplat_smoke.py
```

Keep `_vram_sampler.py` -- it measures whole-board usage via nvidia-smi, which nothing else does, and milestone 2 will want it. Move it: `git mv _vram_sampler.py scripts/vram_sampler.py`.

- [ ] **Step 6: Update the README**

Add a `## Running the pipeline` section immediately after `## Result`:

```
## Running the pipeline

    scripts\env.bat
    .venv\Scripts\python.exe scripts\setup_env.py
    .venv\Scripts\python.exe scripts\get_data.py
    .venv\Scripts\python.exe -m splatpipe.cli run data\tandt\truck --config configs\truck.toml --out out

Roughly 8 minutes on an RTX 4050. Produces `out/truck/artifacts/scene.ply`,
`out/truck/artifacts/scene.splat`, and `out/truck/manifest.json` recording the config,
library versions, timings, held-out metrics and artifact checksums.

Add `--skip-train` to re-export from an existing training run without training again.

Tests:

    .venv\Scripts\python.exe -m pytest              # fast, no GPU, a few seconds
    scripts\env.bat && .venv\Scripts\python.exe -m pytest -m ""   # adds the end-to-end run
```

Then update the `## Reproducing the run` section: keep the raw `simple_trainer.py` invocation, since it is what the pipeline actually runs and is useful when debugging, but add one line above it saying the pipeline is the supported path and this is the underlying command.

Update the `## Files` table to list `scripts/setup_env.py`, `scripts/get_data.py`, `scripts/vram_sampler.py`, `src/splatpipe/`, and `configs/truck.toml`, and drop the rows for the deleted `_*.py` scripts.

- [ ] **Step 7: Update the spec's milestone table**

In `docs/superpowers/specs/2026-08-25-splat-compression-pipeline-design.md`, mark milestone 1 done by changing its row's "Ends with" cell to note the measured result:

```
| 1 | Turn the spike into a scripted pipeline | Done. `splatpipe run` reproduces truck: PSNR <measured>, 225 MiB ply |
```

Fill in the PSNR actually measured in step 3, not the spike's.

- [ ] **Step 8: Full verification before claiming the milestone**

```
.venv\Scripts\python.exe -m pytest -q
scripts\env.bat
.venv\Scripts\python.exe -m pytest -q -m ""
.venv\Scripts\python.exe scripts\setup_env.py --check
```

All three must pass. Do not mark this milestone complete on the basis of the fast tier alone.

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "Reproduce the truck scene through the scripted pipeline"
```

---

## Self-Review

Run through this after the plan is written, before starting Task 1.

**Spec coverage.** Milestone 1's row in the spec reads "Turn the spike into a scripted pipeline / `pipeline run <scene>` reproduces truck end to end". Tasks 1 through 10 build it, Task 11 proves it. The spec's risk "the site-packages patches are fragile / Milestone 1 must convert them into a setup script or a vendored fork" is Task 3. The spec's testing requirements are covered: `pipeline/` gets one end-to-end test on a synthetic scene (Task 10), and the two-week-gap property is served by the fast tier running with no GPU and no vcvars shell. The spec's `compress/` "numpy only, deliberately no torch" rule is enforced early by `gaussians.py` and `formats/splat.py`, which is what milestone 3 will extend.

Three spec items are deliberately **not** in this plan, and are not gaps:

- **`bench/`** is milestone 2. `read_val_metrics` reads the metrics gsplat already computes on its own held-out split, which is enough to record in the manifest but is not the rate-distortion harness.
- **Running COLMAP on a phone capture** is not here. `SceneLayout` validates that poses exist and says what is missing when they do not. Success criterion 4 ("reproduces a scene from a phone capture with a single command") therefore stays open after this milestone. **This is worth raising with Ved: it is the one part of the spec that nothing in milestones 1 to 8 explicitly schedules, and it is a prerequisite for the three real scenes in milestone 7.**
- **The container format** is milestone 5, deferred by the spec's own decision. `.splat` is the milestone 1 output, as the spec says it should be for weekends 1 to 4.

**Placeholder scan.** No "TBD", no "add error handling", no "similar to Task N", no test described rather than written. Two places deliberately point forward instead of implementing: Task 10 step 4 writes `cfg_source` referencing a field that does not exist yet, and step 5 fixes it. That is sequenced on purpose so the CLI reads cleanly before the config change lands, and both steps are fully specified.

**Type consistency.** Checked across tasks:

- `RunConfig.digest()` returns 12 characters (Task 4) and `RunManifest.start` stores it as `config_digest` (Task 8); the e2e test compares them (Task 10).
- `RunPaths.trained_ply(max_steps)` (Task 5) is called by `run_pipeline` with `cfg.train.max_steps` (Task 10), and `read_val_metrics(train_dir, max_steps)` applies the same `max_steps - 1` convention.
- `encode_splat(cloud, order=...)` (Task 7) takes the same three order strings that `ExportConfig` validates (Task 4): `morton`, `size_opacity`, `none`.
- `GaussianCloud.take(index)` (Task 6) is used by `encode_splat` (Task 7) with the output of `morton_order` / `size_opacity_order`.
- `ArtifactRecord.of(path, relative_to=...)` (Task 8) is called with `relative_to=paths.root` (Task 10), which is what makes the recorded paths `artifacts/scene.ply` as the e2e test asserts.
- `check_build_env()` (Task 2) is called with no arguments from `run_pipeline` (Task 10), which is why both parameters have defaults.

**Flag names verified against the installed source, not from memory.** `--data-dir`, `--result-dir`, `--data-factor`, `--max-steps`, `--test-every`, `--save-ply` and `--disable-viewer` all correspond to fields on `simple_trainer.Config`. `--strategy.cap-max` corresponds to `MCMCStrategy.cap_max`, which `DefaultStrategy` does not have, which is why it is gated on the strategy. There is no seed flag, which is why `TrainConfig` has no seed field.

**One thing to watch during execution.** `TRAINER_DIR` in `stages/train.py` is `Path(__file__).resolve().parents[3] / "_gsplat_repo" / "examples"`. From `src/splatpipe/stages/train.py` that is `src/splatpipe/stages` -> `src/splatpipe` -> `src` -> repo root, so `parents[3]` is right for a source checkout. Under an editable install it still resolves to the source tree, which is what we want. Under a real (non-editable) install it would not, but nothing here installs that way. If Task 10 step 7 fails with a missing trainer, this line is the first thing to check.
