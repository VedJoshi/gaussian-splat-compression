# Milestone 2 Rate-Distortion Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure the first rate-distortion points for raw `.ply`, `.splat` and gsplat's `PngCompression`, so that every later compression claim is comparable to a measured baseline.

**Architecture:** A new `src/splatpipe/bench/` package with a directory-based codec protocol, cameras reused from gsplat's own COLMAP `Parser` so held-out pose parity holds by construction, rendering through the public `gsplat.rasterization`, metrics matching `simple_trainer.py` field for field, and `curve.json` plus `curve.png` as outputs. Exposed as a `splatpipe bench` subcommand.

**Tech Stack:** Python 3.11, numpy 1.26.4, torch 2.7.1+cu128, gsplat 1.5.3, torchmetrics 1.9.0, matplotlib 3.11.1, cupy-cuda12x 13.6.0, torchpq, plas.

**Spec:** `docs/superpowers/specs/2026-09-01-milestone-2-bench-design.md`

## Global Constraints

- **Interpreter**: always `.venv\Scripts\python.exe`. Bare `python` and `py` resolve to 3.13; this project requires Python 3.11.
- **numpy stays below 2.0.** `pyproject.toml` pins `numpy<2.0.0`. Any dependency that would upgrade it is rejected, not accommodated.
- **`cupy-cuda12x` is pinned to `13.6.0`.** Version 14 and above requires `numpy>=2.0`.
- **GPU work needs `scripts\env.bat` in the same shell.** gsplat runs `where cl` on import even when its extension is already compiled.
- **`cmd /c` lines must be run from PowerShell or `cmd`, never Git Bash.** MSYS rewrites `/c` into a filesystem path, so `cmd` opens an interactive shell and exits 0 having run nothing. A green exit code there means nothing.
- **Two test tiers.** Fast tier `.venv\Scripts\python.exe -m pytest -q` needs no GPU and no compiler shell. Complete tier adds `-o addopts=` inside `env.bat`. Anything needing a GPU is marked `@pytest.mark.gpu`.
- **Style is binding**: no emoji, em dashes, litotes, irony, or exclamation marks. Plain declarative prose. Comment only non-obvious reasoning, never a comment restating the line below it.
- **One task per session turn.** Execute a task, take it through review, record the outcome in the ledger, then stop and report.
- **Units convention**: `GaussianCloud.scales` are logs and `opacities` are logits, exactly as stored in the `.ply`. Nothing converts on read.

---

### Task 1: Dependencies and lock reconciliation

The working venv already has these installed from the preflight probe, but `requirements.lock.txt` does not list them. That divergence is exactly what milestone 1 existed to prevent, so it is closed first.

**Files:**
- Modify: `requirements.lock.txt`
- Modify: `pyproject.toml`
- Test: `tests/test_package.py`

**Interfaces:**
- Consumes: nothing.
- Produces: an environment in which `import cupy`, `import torchpq`, `import plas` and `from gsplat.compression import PngCompression` all succeed.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_package.py`:

```python
def test_png_compression_dependencies_are_importable():
    """PngCompression's three undeclared dependencies must be installed.

    torchpq declares only numpy and torch in its metadata but imports cupy at
    runtime inside torchpq.clustering.KMeans, so importing torchpq alone proves
    nothing. All three are checked explicitly.
    """
    import cupy
    import plas
    import torchpq

    assert cupy.__version__.startswith("13."), (
        f"cupy must stay on 13.x; 14 and above require numpy>=2.0 and this "
        f"project pins numpy<2.0.0. Found {cupy.__version__}"
    )


def test_numpy_stays_below_2():
    import numpy

    assert numpy.__version__.startswith("1."), (
        f"numpy must stay below 2.0.0, found {numpy.__version__}"
    )
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_package.py -q`

Expected: FAIL. On a fresh checkout it fails with `ModuleNotFoundError: No module named 'cupy'`. On this machine, where the probe already installed them, it passes, which is itself the signal that the lock file is the thing out of date rather than the venv.

- [ ] **Step 3: Add the dependencies to the lock file**

Add to `requirements.lock.txt`, keeping its existing alphabetical grouping:

```
cupy-cuda12x==13.6.0
fastrlock==0.8.3
kornia==0.8.3
kornia-rs==0.1.14
lapjv==1.3.27
plas @ git+https://github.com/fraunhoferhhi/PLAS.git
torchpq==0.3.0.6
```

`click`, `pandas` and `tzdata` also arrive transitively through `plas`; add them at the versions `pip freeze` reports so the file stays a true lock.

- [ ] **Step 4: Record the pin reason in `pyproject.toml`**

Add a `bench` optional dependency group. The comment is the point of this step: the pin is not obvious and rediscovering it costs a broken numpy.

```toml
[project.optional-dependencies]
dev = ["pytest>=8.0", "pillow"]
# cupy 14 and above requires numpy>=2.0, which conflicts with the numpy<2.0.0
# pin above. torchpq imports cupy at runtime without declaring it. plas has no
# PyPI release. See docs/superpowers/specs/2026-09-01-milestone-2-bench-design.md
bench = ["cupy-cuda12x==13.6.0", "torchpq", "torchmetrics>=1.9", "matplotlib"]
```

- [ ] **Step 5: Verify the fast tier and the lock agree**

Run: `.venv\Scripts\python.exe -m pytest -q`

Expected: PASS, 118 passed, 3 deselected.

Then confirm the lock file is honest by diffing it against the live venv:

Run: `.venv\Scripts\python.exe -m pip freeze --exclude-editable`

Expected: every line of `requirements.lock.txt` except the hand-maintained `-e .` entry appears in the output. Do not regenerate the file with a bare `pip freeze`; it resolves the editable install to a private git URL. Use `--exclude-editable` and leave the `-e .` line alone.

- [ ] **Step 6: Commit**

```bash
git add requirements.lock.txt pyproject.toml tests/test_package.py
git commit -m "Pin the PngCompression dependencies and record why cupy stays on 13.x"
```

---

### Task 2: The `.splat` decoder

**Files:**
- Modify: `src/splatpipe/formats/splat.py`
- Test: `tests/test_splat_format.py`

**Interfaces:**
- Consumes: `GaussianCloud`, `SH_C0` from `splatpipe.gaussians`; `BYTES_PER_GAUSSIAN`, `encode_splat` from `splatpipe.formats.splat`.
- Produces: `decode_splat(data: bytes) -> GaussianCloud`. The returned cloud has `shN` of shape `(N, 0, 3)` and therefore `sh_degree == 0`, because the format discards higher-order harmonics.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_splat_format.py`:

```python
from splatpipe.formats.splat import decode_splat


def a_representable_cloud(n: int = 64) -> GaussianCloud:
    """A cloud the .splat format can actually carry.

    The module's existing make_cloud draws sh0 from a standard normal, but the
    format only represents sh0 in about [-1.77, 1.77]: rgb is sh0 * SH_C0 + 0.5
    and the encoder clips outside [0, 1]. Asserting round-trip fidelity on
    values the format cannot represent would test nothing, so the tolerance
    assertions below use this instead. Saturation itself is already covered by
    the encoder's own tests.
    """
    rng = np.random.default_rng(0)
    return GaussianCloud(
        means=rng.uniform(-10, 10, (n, 3)).astype(np.float32),
        scales=rng.uniform(-3, -1, (n, 3)).astype(np.float32),
        quats=rng.standard_normal((n, 4)).astype(np.float32),
        opacities=rng.uniform(-4, 4, n).astype(np.float32),
        sh0=rng.uniform(-1.5, 1.5, (n, 3)).astype(np.float32),
        shN=rng.standard_normal((n, 15, 3)).astype(np.float32),
    )


def test_decode_recovers_positions_exactly():
    """Positions are stored as float32 and must survive the round trip bit for bit."""
    cloud = make_cloud(n=16)
    back = decode_splat(encode_splat(cloud, order="none"))
    np.testing.assert_array_equal(back.means, cloud.means)


def test_decode_discards_higher_order_harmonics():
    cloud = make_cloud(n=16)
    back = decode_splat(encode_splat(cloud, order="none"))
    assert back.shN.shape == (16, 0, 3)
    assert back.sh_degree == 0


def test_decode_round_trips_scales_and_colours_within_quantisation_error():
    cloud = a_representable_cloud(64)
    back = decode_splat(encode_splat(cloud, order="none"))
    # Scales survive as exp then log through float32, so only rounding is lost.
    np.testing.assert_allclose(back.scales, cloud.scales, atol=1e-5)
    # Colours are quantised to 8 bits. One step of sh0 is 1/255 scaled by 1/SH_C0.
    np.testing.assert_allclose(back.sh0, cloud.sh0, atol=(1.0 / 255) / SH_C0)


def test_decode_clamps_saturated_opacity_instead_of_returning_infinity():
    """alpha == 255 inverts to logit(1) == inf without a clamp.

    The encoder reaches this on real data: it clips to [0, 255] and casts, so a
    high enough logit saturates. The clamp uses the midpoint of the quantisation
    bucket, which is the best estimate available rather than an arbitrary guard.
    """
    cloud = replace(make_cloud(n=4), opacities=np.full(4, 40.0, dtype=np.float32))
    back = decode_splat(encode_splat(cloud, order="none"))
    assert np.isfinite(back.opacities).all()
    np.testing.assert_allclose(back.opacities, 6.2324480, atol=1e-4)


def test_decode_clamps_underflowing_scale_instead_of_returning_negative_infinity():
    """A very negative log scale exponentiates to zero, and log(0) is -inf."""
    cloud = replace(make_cloud(n=4), scales=np.full((4, 3), -200.0, dtype=np.float32))
    back = decode_splat(encode_splat(cloud, order="none"))
    assert np.isfinite(back.scales).all()
    np.testing.assert_allclose(back.scales, -87.33655, atol=1e-3)


def test_decode_rejects_a_truncated_buffer():
    with pytest.raises(ArtifactError, match="not a multiple"):
        decode_splat(b"\x00" * 33)


def test_decode_rejects_an_empty_buffer():
    with pytest.raises(ArtifactError, match="no gaussians"):
        decode_splat(b"")
```

Add `from dataclasses import replace` and `from splatpipe.gaussians import SH_C0` to the imports if the file does not already have them. The file's existing cloud helper is `make_cloud(n=64, k=15)`, defined at `tests/test_splat_format.py:18`. Use that name; there is no `a_cloud` in this file.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_splat_format.py -q -k decode`

Expected: FAIL with `ImportError: cannot import name 'decode_splat'`.

- [ ] **Step 3: Implement the decoder**

Add to `src/splatpipe/formats/splat.py`:

```python
# The midpoint of the quantisation bucket an 8-bit alpha came from. Inverting
# logit at the bucket edge diverges; the midpoint is the best estimate the
# stored byte supports. Bounds the recovered logit at about plus or minus 6.23.
_ALPHA_MIN = 0.5 / 255
_ALPHA_MAX = 254.5 / 255
# exp() of a sufficiently negative log scale underflows to zero, and log(0) is
# negative infinity. The smallest positive normal float32 bounds it at -87.34.
_FLOAT32_TINY = float(np.finfo(np.float32).tiny)


def decode_splat(data: bytes) -> GaussianCloud:
    """Decode 32-byte .splat records back into a GaussianCloud.

    Lossy by construction, and deliberately so: the format quantises colour and
    rotation to 8 bits and discards f_rest entirely. The returned cloud has
    sh_degree 0. Nothing here special-cases the loss away, because the
    rate-distortion point has to reflect what the format actually costs.
    """
    if len(data) % BYTES_PER_GAUSSIAN:
        raise ArtifactError(
            f"buffer of {len(data)} bytes is not a multiple of {BYTES_PER_GAUSSIAN}"
        )
    n = len(data) // BYTES_PER_GAUSSIAN
    if n == 0:
        raise ArtifactError("buffer holds no gaussians")

    buffer = np.frombuffer(data, dtype=np.uint8).reshape(n, BYTES_PER_GAUSSIAN)

    means = buffer[:, 0:12].tobytes()
    means = np.frombuffer(means, dtype="<f4").reshape(n, 3).astype(np.float32)

    stored = np.frombuffer(buffer[:, 12:24].tobytes(), dtype="<f4").reshape(n, 3)
    scales = np.log(np.maximum(stored, _FLOAT32_TINY)).astype(np.float32)

    color = buffer[:, 24:28].astype(np.float64) / 255.0
    sh0 = ((color[:, :3] - 0.5) / SH_C0).astype(np.float32)
    alpha = np.clip(color[:, 3], _ALPHA_MIN, _ALPHA_MAX)
    opacities = np.log(alpha / (1.0 - alpha)).astype(np.float32)

    quats = (buffer[:, 28:32].astype(np.float64) - 128.0) / 128.0
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    if np.any(norms == 0):
        row = int(np.flatnonzero(norms.ravel() == 0)[0])
        raise ArtifactError(f"quats decodes to a zero-length quaternion at row {row}")
    quats = (quats / norms).astype(np.float32)

    cloud = GaussianCloud(
        means=means,
        scales=scales,
        quats=quats,
        opacities=opacities,
        sh0=sh0,
        shN=np.zeros((n, 0, 3), dtype=np.float32),
    )
    cloud.validate()
    return cloud
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_splat_format.py -q`

Expected: PASS.

- [ ] **Step 5: Run the whole fast tier**

Run: `.venv\Scripts\python.exe -m pytest -q`

Expected: PASS, 125 passed, 3 deselected.

- [ ] **Step 6: Commit**

```bash
git add src/splatpipe/formats/splat.py tests/test_splat_format.py
git commit -m "Add the .splat decoder with explicit clamps at both quantisation edges"
```

---

### Task 3: The codec protocol, PlyCodec and SplatCodec

**Files:**
- Create: `src/splatpipe/bench/__init__.py`
- Create: `src/splatpipe/bench/codecs.py`
- Test: `tests/test_codecs.py`

**Interfaces:**
- Consumes: `read_ply`, `write_ply`, `GaussianCloud` from `splatpipe.gaussians`; `encode_splat`, `decode_splat` from `splatpipe.formats.splat`.
- Produces:
  - `Codec` protocol with `name: str`, `encode(cloud: GaussianCloud, directory: Path) -> None`, `decode(directory: Path) -> GaussianCloud`, `size(directory: Path) -> int`.
  - `PlyCodec()`, `SplatCodec(order: str = "morton")`.
  - `directory_size(directory: Path) -> int`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_codecs.py`:

```python
"""Codec round trips. Pure numpy, no GPU, so this runs in the fast tier."""

from __future__ import annotations

import numpy as np
import pytest

from splatpipe.bench.codecs import PlyCodec, SplatCodec, directory_size
from splatpipe.gaussians import GaussianCloud


def a_cloud(n: int = 32) -> GaussianCloud:
    rng = np.random.default_rng(0)
    return GaussianCloud(
        means=rng.uniform(-1, 1, (n, 3)).astype(np.float32),
        scales=rng.uniform(-3, -1, (n, 3)).astype(np.float32),
        quats=rng.normal(size=(n, 4)).astype(np.float32),
        opacities=rng.uniform(-1, 2, n).astype(np.float32),
        sh0=rng.uniform(-1, 1, (n, 3)).astype(np.float32),
        shN=rng.uniform(-0.2, 0.2, (n, 15, 3)).astype(np.float32),
    )


def test_ply_codec_is_lossless(tmp_path):
    """PlyCodec is the 1.00x anchor. If it loses anything the curve has no origin."""
    cloud = a_cloud(64)
    codec = PlyCodec()
    codec.encode(cloud, tmp_path)
    back = codec.decode(tmp_path)
    np.testing.assert_array_equal(back.means, cloud.means)
    np.testing.assert_array_equal(back.shN, cloud.shN)
    assert back.sh_degree == 3


def test_splat_codec_reports_thirty_two_bytes_per_gaussian(tmp_path):
    cloud = a_cloud(64)
    codec = SplatCodec()
    codec.encode(cloud, tmp_path)
    assert codec.size(tmp_path) == 64 * 32


def test_splat_codec_round_trips_through_the_decoder(tmp_path):
    cloud = a_cloud(64)
    codec = SplatCodec(order="none")
    codec.encode(cloud, tmp_path)
    back = codec.decode(tmp_path)
    assert len(back) == 64
    np.testing.assert_array_equal(back.means, cloud.means)


def test_codec_names_are_distinct():
    assert PlyCodec().name != SplatCodec().name


def test_directory_size_counts_every_file_recursively(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "a.bin").write_bytes(b"x" * 10)
    (tmp_path / "nested" / "b.bin").write_bytes(b"y" * 5)
    assert directory_size(tmp_path) == 15
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_codecs.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'splatpipe.bench'`.

- [ ] **Step 3: Create the package and the codecs**

Create `src/splatpipe/bench/__init__.py` as an empty file.

Create `src/splatpipe/bench/codecs.py`:

```python
"""What a rate-distortion point is made of.

A codec turns a GaussianCloud into files on disk and back. The rate is the
total size of those files; the distortion is whatever rendering the decoded
cloud costs in image quality.

Directory-based rather than bytes-based because PngCompression genuinely
produces eight files, and flattening them into one blob would misreport the
rate, which is the single number this component exists to measure. A codec that
produces one buffer satisfies this protocol by writing one file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from splatpipe.formats.splat import decode_splat, encode_splat
from splatpipe.gaussians import GaussianCloud, read_ply, write_ply


def directory_size(directory: Path) -> int:
    """Total bytes of every file under `directory`, recursively."""
    return sum(p.stat().st_size for p in Path(directory).rglob("*") if p.is_file())


@runtime_checkable
class Codec(Protocol):
    name: str

    def encode(self, cloud: GaussianCloud, directory: Path) -> None: ...

    def decode(self, directory: Path) -> GaussianCloud: ...

    def size(self, directory: Path) -> int: ...


class PlyCodec:
    """The lossless anchor. Its point is the 1.00x origin of every ratio."""

    name = "ply"

    def encode(self, cloud: GaussianCloud, directory: Path) -> None:
        write_ply(cloud, Path(directory) / "scene.ply")

    def decode(self, directory: Path) -> GaussianCloud:
        return read_ply(Path(directory) / "scene.ply")

    def size(self, directory: Path) -> int:
        return directory_size(directory)


class SplatCodec:
    """The 32-byte web format. Discards f_rest, so it decodes at sh_degree 0."""

    name = "splat"

    def __init__(self, order: str = "morton") -> None:
        self.order = order

    def encode(self, cloud: GaussianCloud, directory: Path) -> None:
        path = Path(directory) / "scene.splat"
        path.write_bytes(encode_splat(cloud, order=self.order))

    def decode(self, directory: Path) -> GaussianCloud:
        return decode_splat((Path(directory) / "scene.splat").read_bytes())

    def size(self, directory: Path) -> int:
        return directory_size(directory)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_codecs.py -q`

Expected: PASS, 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/splatpipe/bench tests/test_codecs.py
git commit -m "Add the bench codec protocol with the ply and splat implementations"
```

---

### Task 4: PngCodec

**Files:**
- Modify: `src/splatpipe/bench/codecs.py`
- Test: `tests/test_codecs.py`

**Interfaces:**
- Consumes: `Codec`, `directory_size` from Task 3.
- Produces: `PngCodec(use_sort: bool = True)`. Its `encode` may drop the lowest-opacity Gaussians to reach a square count, so `decode` can return fewer Gaussians than were encoded.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_codecs.py`:

```python
from splatpipe.bench.codecs import PngCodec

# PngCompression sorts with PLAS and clusters with torchpq, both of which need
# CUDA. There is no CPU path, so these are GPU tier.
pngmark = pytest.mark.gpu


@pngmark
def test_png_codec_round_trips_a_square_cloud(tmp_path):
    cloud = a_cloud(64)  # 64 is 8 squared, so nothing is cropped
    codec = PngCodec()
    codec.encode(cloud, tmp_path)
    back = codec.decode(tmp_path)
    assert len(back) == 64
    assert back.sh_degree == 3
    assert codec.size(tmp_path) > 0


@pngmark
def test_png_codec_does_not_mutate_the_cloud_it_is_given(tmp_path):
    """PngCompression.compress applies log_transform to means and normalises
    quats in place. Without a copy, every codec measured after this one would
    receive a corrupted cloud, and the failure would be a wrong number rather
    than an error."""
    cloud = a_cloud(64)
    before_means = cloud.means.copy()
    before_quats = cloud.quats.copy()
    PngCodec().encode(cloud, tmp_path)
    np.testing.assert_array_equal(cloud.means, before_means)
    np.testing.assert_array_equal(cloud.quats, before_quats)


@pngmark
def test_png_codec_reports_the_count_it_actually_encoded(tmp_path):
    """65 is not a square, so PngCompression drops the lowest-opacity splat."""
    codec = PngCodec()
    codec.encode(a_cloud(65), tmp_path)
    assert len(codec.decode(tmp_path)) == 64
```

- [ ] **Step 2: Run the tests to verify they fail**

Run from PowerShell:

```
cmd /c "call scripts\env.bat >nul 2>&1 && .venv\Scripts\python.exe -m pytest tests/test_codecs.py -q -m gpu"
```

Expected: FAIL with `ImportError: cannot import name 'PngCodec'`.

- [ ] **Step 3: Implement PngCodec**

Append to `src/splatpipe/bench/codecs.py`:

```python
class PngCodec:
    """gsplat's own PngCompression, the baseline this project has to beat.

    Quantises to PNG images with a PLAS spatial sort, and vector-quantises the
    higher-order harmonics with K-means. Needs cupy, torchpq and plas, and needs
    a GPU: there is no CPU path.
    """

    name = "png"

    def __init__(self, use_sort: bool = True) -> None:
        self.use_sort = use_sort

    def encode(self, cloud: GaussianCloud, directory: Path) -> None:
        import torch
        from gsplat.compression import PngCompression

        Path(directory).mkdir(parents=True, exist_ok=True)
        # compress() applies log_transform to means and normalises quats in
        # place, so every tensor here is a fresh copy of the caller's arrays.
        splats = {
            "means": torch.from_numpy(cloud.means.copy()).cuda(),
            "scales": torch.from_numpy(cloud.scales.copy()).cuda(),
            "quats": torch.from_numpy(cloud.quats.copy()).cuda(),
            "opacities": torch.from_numpy(cloud.opacities.copy()).cuda(),
            "sh0": torch.from_numpy(cloud.sh0.copy()).unsqueeze(1).cuda(),
            "shN": torch.from_numpy(cloud.shN.copy()).cuda(),
        }
        PngCompression(use_sort=self.use_sort, verbose=False).compress(
            str(directory), splats
        )

    def decode(self, directory: Path) -> GaussianCloud:
        from gsplat.compression import PngCompression

        splats = PngCompression(verbose=False).decompress(str(directory))
        cloud = GaussianCloud(
            means=splats["means"].cpu().numpy().astype("float32"),
            scales=splats["scales"].cpu().numpy().astype("float32"),
            quats=splats["quats"].cpu().numpy().astype("float32"),
            opacities=splats["opacities"].cpu().numpy().astype("float32"),
            sh0=splats["sh0"].squeeze(1).cpu().numpy().astype("float32"),
            shN=splats["shN"].cpu().numpy().astype("float32"),
        )
        cloud.validate()
        return cloud

    def size(self, directory: Path) -> int:
        return directory_size(directory)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run from PowerShell:

```
cmd /c "call scripts\env.bat >nul 2>&1 && .venv\Scripts\python.exe -m pytest tests/test_codecs.py -q -m gpu"
```

Expected: PASS, 3 passed.

- [ ] **Step 5: Confirm the fast tier still deselects the GPU tests**

Run: `.venv\Scripts\python.exe -m pytest -q`

Expected: PASS, 130 passed, 6 deselected.

- [ ] **Step 6: Commit**

```bash
git add src/splatpipe/bench/codecs.py tests/test_codecs.py
git commit -m "Add PngCodec, copying its input because compress mutates in place"
```

---

### Task 5: Cameras

**Files:**
- Create: `src/splatpipe/bench/cameras.py`
- Test: `tests/test_cameras.py`

**Interfaces:**
- Consumes: gsplat's `datasets.colmap.Parser` and `datasets.colmap.Dataset` from `_gsplat_repo/examples`.
- Produces:
  - `ValView` frozen dataclass with `camtoworld: np.ndarray` shape `(4, 4)`, `K: np.ndarray` shape `(3, 3)`, `image: np.ndarray` shape `(H, W, 3)` dtype uint8.
  - `load_val_views(scene_dir: Path, data_factor: int = 1, test_every: int = 8) -> list[ValView]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cameras.py`:

```python
"""The held-out views bench scores against.

No GPU needed: loading COLMAP poses and images is CPU work. This runs in the
fast tier so a camera regression is caught without a compiler shell.
"""

from __future__ import annotations

import numpy as np

from splatpipe.bench.cameras import ValView, load_val_views
from tests.fixtures.tiny_scene import make_tiny_scene


def test_loads_every_eighth_view(tmp_path):
    """gsplat holds out indices where index % test_every == 0.

    24 images at test_every 8 gives indices 0, 8 and 16.
    """
    scene = make_tiny_scene(tmp_path / "scene", n_images=24, n_points=512)
    views = load_val_views(scene, data_factor=1, test_every=8)
    assert len(views) == 3


def test_views_carry_poses_intrinsics_and_pixels(tmp_path):
    scene = make_tiny_scene(tmp_path / "scene", n_images=24, n_points=512)
    view = load_val_views(scene, data_factor=1, test_every=8)[0]
    assert isinstance(view, ValView)
    assert view.camtoworld.shape == (4, 4)
    assert view.K.shape == (3, 3)
    assert view.image.shape == (72, 96, 3)
    assert view.image.dtype == np.uint8


def test_test_every_changes_the_held_out_count(tmp_path):
    scene = make_tiny_scene(tmp_path / "scene", n_images=24, n_points=512)
    assert len(load_val_views(scene, data_factor=1, test_every=4)) == 6
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_cameras.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'splatpipe.bench.cameras'`.

- [ ] **Step 3: Implement the camera loader**

Create `src/splatpipe/bench/cameras.py`:

```python
"""The held-out views, loaded through gsplat's own COLMAP parser.

This module is the one place that imports from _gsplat_repo/examples, and it
does so deliberately. stages/train.py avoided that directory by shelling out,
because it only needed to run the trainer. Here correctness depends on identity
rather than similarity: gsplat's Parser applies a scene normalisation, a
transform and a scale, to the camera poses, and selects held-out views as
index % test_every == 0. A reimplementation that drifts from any of that
produces plausible numbers that are quietly wrong, and nothing downstream would
reveal it. Reusing the trainer's own code makes pose parity a property of
construction.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TRAINER_EXAMPLES = Path(__file__).resolve().parents[3] / "_gsplat_repo" / "examples"

# simple_trainer.py's own defaults, which the poses depend on. normalize is the
# dangerous one: Parser defaults it to False while the trainer passes
# cfg.normalize_world_space, which defaults to True (simple_trainer.py:67).
# Taking Parser's default here would un-normalise every pose and destroy the
# metric parity this whole component rests on.
NORMALIZE_WORLD_SPACE = True


@dataclass(frozen=True)
class ValView:
    camtoworld: np.ndarray  # (4, 4) float32
    K: np.ndarray  # (3, 3) float32
    image: np.ndarray  # (H, W, 3) uint8


def _import_colmap_dataset():
    """Put gsplat's examples directory on sys.path and import its loader.

    The trainer resolves `datasets.colmap` relative to its own directory, so the
    path insertion is unavoidable if the code is to be reused rather than
    reimplemented. Idempotent, so repeated calls are harmless.
    """
    path = str(TRAINER_EXAMPLES)
    if path not in sys.path:
        sys.path.insert(0, path)
    from datasets.colmap import Dataset, Parser

    return Parser, Dataset


def load_val_views(
    scene_dir: Path,
    data_factor: int = 1,
    test_every: int = 8,
) -> list[ValView]:
    """Load the held-out views for a COLMAP scene, exactly as the trainer sees them."""
    Parser, Dataset = _import_colmap_dataset()
    parser = Parser(
        data_dir=str(Path(scene_dir).resolve()),
        factor=data_factor,
        normalize=NORMALIZE_WORLD_SPACE,
        test_every=test_every,
    )
    dataset = Dataset(parser, split="val")

    views = []
    for index in range(len(dataset)):
        item = dataset[index]
        views.append(
            ValView(
                camtoworld=np.asarray(item["camtoworld"], dtype=np.float32),
                K=np.asarray(item["K"], dtype=np.float32),
                image=np.asarray(item["image"], dtype=np.uint8),
            )
        )
    return views
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_cameras.py -q`

Expected: PASS, 3 passed.

If `item["camtoworld"]` or `item["image"]` arrives as a torch tensor rather than a numpy array, convert with `.numpy()` before `np.asarray`. Check the actual type before assuming; `Dataset.__getitem__` returns torch tensors in gsplat 1.5.3.

- [ ] **Step 5: Commit**

```bash
git add src/splatpipe/bench/cameras.py tests/test_cameras.py
git commit -m "Load held-out views through gsplat's own parser so poses match by construction"
```

---

### Task 6: Rendering

**Files:**
- Create: `src/splatpipe/bench/render.py`
- Test: `tests/test_render.py`

**Interfaces:**
- Consumes: `ValView` from Task 5, `GaussianCloud`.
- Produces: `render_view(cloud: GaussianCloud, view: ValView, device: str = "cuda") -> np.ndarray` returning `(H, W, 3)` float32 clamped to `[0, 1]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_render.py`:

```python
"""Rendering a cloud through gsplat. GPU tier: there is no CPU rasteriser."""

from __future__ import annotations

import numpy as np
import pytest

from splatpipe.bench.cameras import load_val_views
from splatpipe.bench.render import render_view
from splatpipe.gaussians import read_ply
from tests.fixtures.tiny_scene import make_tiny_scene

pytestmark = pytest.mark.gpu


def test_render_matches_the_view_shape_and_range(tmp_path):
    from tests.test_codecs import a_cloud

    scene = make_tiny_scene(tmp_path / "scene", n_images=24, n_points=512)
    view = load_val_views(scene, data_factor=1, test_every=8)[0]
    image = render_view(a_cloud(256), view)

    assert image.shape == view.image.shape
    assert image.dtype == np.float32
    assert image.min() >= 0.0 and image.max() <= 1.0


def test_render_uses_the_clouds_own_sh_degree(tmp_path):
    """A decoded .splat has sh_degree 0. Passing a fixed 3 would raise inside
    rasterization, because colors would carry 1 coefficient where 16 is claimed."""
    from splatpipe.bench.codecs import SplatCodec
    from tests.test_codecs import a_cloud

    scene = make_tiny_scene(tmp_path / "scene", n_images=24, n_points=512)
    view = load_val_views(scene, data_factor=1, test_every=8)[0]

    codec = SplatCodec(order="none")
    codec.encode(a_cloud(256), tmp_path / "splat")
    decoded = codec.decode(tmp_path / "splat")
    assert decoded.sh_degree == 0

    image = render_view(decoded, view)
    assert image.shape == view.image.shape
```

- [ ] **Step 2: Run the test to verify it fails**

Run from PowerShell:

```
cmd /c "call scripts\env.bat >nul 2>&1 && .venv\Scripts\python.exe -m pytest tests/test_render.py -q -o addopts="
```

Expected: FAIL with `ModuleNotFoundError: No module named 'splatpipe.bench.render'`.

- [ ] **Step 3: Implement the renderer**

Create `src/splatpipe/bench/render.py`:

```python
"""Render a GaussianCloud through gsplat's public rasteriser.

Mirrors simple_trainer.rasterize_splats field for field. The activation
convention is where this matters: GaussianCloud stores log scales and logit
opacities, exactly as the .ply does, so exp and sigmoid are applied here rather
than on read. The quaternions are passed unnormalised because rasterization
normalises internally.

The defaults below are simple_trainer.py's own. They are written out rather than
left implicit because a silent disagreement with the trainer shows up as a
slightly wrong PSNR, not as an error.
"""

from __future__ import annotations

import numpy as np

from splatpipe.gaussians import GaussianCloud

NEAR_PLANE = 0.01  # simple_trainer.py:110
FAR_PLANE = 1e10  # simple_trainer.py:112
CAMERA_MODEL = "pinhole"  # simple_trainer.py:69
RASTERIZE_MODE = "classic"  # antialiased defaults to False, simple_trainer.py:125


def render_view(cloud: GaussianCloud, view, device: str = "cuda") -> np.ndarray:
    """Render one held-out view. Returns (H, W, 3) float32 clamped to [0, 1]."""
    import torch
    from gsplat import rasterization

    height, width = view.image.shape[:2]
    camtoworld = torch.from_numpy(view.camtoworld).to(device).unsqueeze(0)
    Ks = torch.from_numpy(view.K).to(device).unsqueeze(0)

    means = torch.from_numpy(cloud.means).to(device)
    quats = torch.from_numpy(cloud.quats).to(device)
    scales = torch.exp(torch.from_numpy(cloud.scales).to(device))
    opacities = torch.sigmoid(torch.from_numpy(cloud.opacities).to(device))
    sh0 = torch.from_numpy(cloud.sh0).to(device).unsqueeze(1)
    shN = torch.from_numpy(cloud.shN).to(device)
    colors = torch.cat([sh0, shN], dim=1)

    with torch.no_grad():
        rendered, _, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworld),
            Ks=Ks,
            width=width,
            height=height,
            # Derived from the cloud, never hardcoded: a decoded .splat carries
            # only the DC term and renders at degree 0.
            sh_degree=cloud.sh_degree,
            near_plane=NEAR_PLANE,
            far_plane=FAR_PLANE,
            camera_model=CAMERA_MODEL,
            rasterize_mode=RASTERIZE_MODE,
            packed=False,
        )
    image = torch.clamp(rendered, 0.0, 1.0).squeeze(0)
    return image.cpu().numpy().astype(np.float32)
```

- [ ] **Step 4: Run the test to verify it passes**

Run from PowerShell:

```
cmd /c "call scripts\env.bat >nul 2>&1 && .venv\Scripts\python.exe -m pytest tests/test_render.py -q -o addopts="
```

Expected: PASS, 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/splatpipe/bench/render.py tests/test_render.py
git commit -m "Render through gsplat's rasteriser with the trainer's own settings"
```

---

### Task 7: Metrics

**Files:**
- Create: `src/splatpipe/bench/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `score_views(rendered: list[np.ndarray], targets: list[np.ndarray], device: str = "cuda") -> dict[str, float]` returning keys `psnr`, `ssim`, `lpips`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_metrics.py`:

```python
"""Known-answer checks on the metric code.

The spec asks for these explicitly, so that a regression in the metrics is
visible without a GPU and without a full render. torchmetrics runs on CPU, so
only LPIPS needs its weights, which are cached after the first download.
"""

from __future__ import annotations

import numpy as np
import pytest

from splatpipe.bench.metrics import score_views


def a_pair(seed: int = 0, shape=(32, 48, 3)):
    rng = np.random.default_rng(seed)
    target = rng.uniform(0, 1, shape).astype(np.float32)
    return target


def test_identical_images_give_infinite_psnr_and_unit_ssim():
    target = a_pair()
    scores = score_views([target], [target], device="cpu")
    assert scores["ssim"] == pytest.approx(1.0, abs=1e-4)
    assert scores["psnr"] > 60


def test_psnr_matches_its_closed_form():
    """PSNR is 10 log10(1 / mse) at data_range 1. A fixed offset is checkable."""
    target = np.full((16, 16, 3), 0.5, dtype=np.float32)
    rendered = np.full((16, 16, 3), 0.6, dtype=np.float32)
    expected = 10 * np.log10(1.0 / (0.1**2))
    scores = score_views([rendered], [target], device="cpu")
    assert scores["psnr"] == pytest.approx(expected, abs=0.01)


def test_scores_average_over_every_view():
    good = np.full((16, 16, 3), 0.5, dtype=np.float32)
    bad = np.full((16, 16, 3), 0.9, dtype=np.float32)
    target = np.full((16, 16, 3), 0.5, dtype=np.float32)
    both = score_views([good, bad], [target, target], device="cpu")
    only_bad = score_views([bad], [target], device="cpu")
    assert both["psnr"] > only_bad["psnr"]


def test_mismatched_lengths_are_rejected():
    target = a_pair()
    with pytest.raises(ValueError, match="same number"):
        score_views([target], [target, target], device="cpu")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_metrics.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'splatpipe.bench.metrics'`.

- [ ] **Step 3: Implement the metrics**

Create `src/splatpipe/bench/metrics.py`:

```python
"""PSNR, SSIM and LPIPS, configured exactly as gsplat configures them.

Matching the reference implementation is what makes this project's numbers
comparable to gsplat's published ones. Any deviation, a different LPIPS backbone
most of all, still produces a plot but makes the comparison meaningless. The
settings below are simple_trainer.py:457-467.

LPIPS downloads AlexNet weights on first use, which makes the first run
network-dependent in a project that is otherwise reproducible offline. They are
cached afterwards.
"""

from __future__ import annotations

import numpy as np

DATA_RANGE = 1.0
LPIPS_NET = "alex"


def _to_nchw(image: np.ndarray, device: str):
    import torch

    tensor = torch.from_numpy(np.ascontiguousarray(image)).to(device)
    return tensor.unsqueeze(0).permute(0, 3, 1, 2)


def score_views(
    rendered: list[np.ndarray],
    targets: list[np.ndarray],
    device: str = "cuda",
) -> dict[str, float]:
    """Mean PSNR, SSIM and LPIPS over a set of views.

    Both lists hold (H, W, 3) float32 images in [0, 1]. Averaging per view then
    over views is what gsplat does, and it is not the same as averaging over all
    pixels at once when views differ in size.
    """
    if len(rendered) != len(targets):
        raise ValueError(
            f"rendered and targets must hold the same number of views, "
            f"got {len(rendered)} and {len(targets)}"
        )
    if not rendered:
        raise ValueError("no views to score")

    import torch
    from torchmetrics.image import (
        PeakSignalNoiseRatio,
        StructuralSimilarityIndexMeasure,
    )
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    psnr = PeakSignalNoiseRatio(data_range=DATA_RANGE).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=DATA_RANGE).to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(
        net_type=LPIPS_NET, normalize=True
    ).to(device)

    totals = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0}
    with torch.no_grad():
        for render, target in zip(rendered, targets):
            r = _to_nchw(render, device)
            t = _to_nchw(target, device)
            totals["psnr"] += float(psnr(r, t))
            totals["ssim"] += float(ssim(r, t))
            totals["lpips"] += float(lpips(r, t))

    return {key: value / len(rendered) for key, value in totals.items()}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_metrics.py -q`

Expected: PASS, 4 passed. The first run downloads AlexNet weights and needs a network.

- [ ] **Step 5: Commit**

```bash
git add src/splatpipe/bench/metrics.py tests/test_metrics.py
git commit -m "Add PSNR, SSIM and LPIPS matching gsplat's own configuration"
```

---

### Task 8: The curve

**Files:**
- Create: `src/splatpipe/bench/curve.py`
- Test: `tests/test_curve.py`

**Interfaces:**
- Consumes: `manifest.collect_versions` from `splatpipe.manifest`.
- Produces:
  - `CurvePoint` frozen dataclass: `codec: str`, `bytes: int`, `ratio: float`, `gaussians: int`, `psnr: float`, `ssim: float`, `lpips: float`, `encode_seconds: float`, `decode_seconds: float`.
  - `Curve` dataclass: `scene: str`, `config_digest: str`, `held_out_views: int`, `created_utc: str`, `versions: dict[str, str]`, `points: list[CurvePoint]`, with `write_json(path)`, `read_json(path)` and `write_plot(path)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_curve.py`:

```python
"""Curve assembly and its JSON schema. Pure data, so this is fast tier."""

from __future__ import annotations

import json

import pytest

from splatpipe.bench.curve import Curve, CurvePoint


def a_curve() -> Curve:
    return Curve.start(scene="truck", config_digest="c221a1fc2180", held_out_views=32)


def test_ratio_is_computed_against_the_anchor():
    curve = a_curve()
    curve.add(
        CurvePoint.of(
            codec="ply", size_bytes=236_001_478, gaussians=1_000_000,
            scores={"psnr": 24.4, "ssim": 0.858, "lpips": 0.137},
            encode_seconds=1.0, decode_seconds=2.0, anchor_bytes=236_001_478,
        )
    )
    curve.add(
        CurvePoint.of(
            codec="splat", size_bytes=32_000_000, gaussians=1_000_000,
            scores={"psnr": 20.1, "ssim": 0.71, "lpips": 0.29},
            encode_seconds=3.0, decode_seconds=0.5, anchor_bytes=236_001_478,
        )
    )
    assert curve.points[0].ratio == pytest.approx(1.0)
    assert curve.points[1].ratio == pytest.approx(7.375046, abs=1e-5)


def test_round_trips_through_json(tmp_path):
    curve = a_curve()
    curve.add(
        CurvePoint.of(
            codec="ply", size_bytes=100, gaussians=10,
            scores={"psnr": 1.0, "ssim": 0.5, "lpips": 0.25},
            encode_seconds=0.1, decode_seconds=0.2, anchor_bytes=100,
        )
    )
    path = tmp_path / "curve.json"
    curve.write_json(path)
    back = Curve.read_json(path)
    assert back.scene == "truck"
    assert back.points[0].codec == "ply"
    assert back.points[0].psnr == 1.0


def test_json_records_the_context_that_makes_a_point_traceable(tmp_path):
    path = tmp_path / "curve.json"
    a_curve().write_json(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["scene"] == "truck"
    assert data["config_digest"] == "c221a1fc2180"
    assert data["held_out_views"] == 32
    assert "splatpipe_commit" in data["versions"]
    assert "gsplat" in data["versions"]


def test_gaussian_count_is_recorded_per_point():
    """PngCompression crops to a square count, so a point can hold fewer
    Gaussians than were handed to it. Assuming they match would misreport it."""
    curve = a_curve()
    curve.add(
        CurvePoint.of(
            codec="png", size_bytes=16_258_005, gaussians=999_999,
            scores={"psnr": 24.0, "ssim": 0.85, "lpips": 0.14},
            encode_seconds=120.0, decode_seconds=4.0, anchor_bytes=236_001_478,
        )
    )
    assert curve.points[0].gaussians == 999_999


def test_plot_writes_a_png(tmp_path):
    curve = a_curve()
    curve.add(
        CurvePoint.of(
            codec="ply", size_bytes=100, gaussians=10,
            scores={"psnr": 30.0, "ssim": 0.9, "lpips": 0.1},
            encode_seconds=0.1, decode_seconds=0.1, anchor_bytes=100,
        )
    )
    path = tmp_path / "curve.png"
    curve.write_plot(path)
    assert path.is_file()
    assert path.stat().st_size > 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_curve.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'splatpipe.bench.curve'`.

- [ ] **Step 3: Implement the curve**

Create `src/splatpipe/bench/curve.py`:

```python
"""The measured rate-distortion curve, and what makes a point traceable.

A point without its context is a number rather than a measurement, so every
curve carries the scene, the config digest, the held-out view count, the library
versions and the Git commit. Versions come from manifest.collect_versions rather
than a second copy, so a manifest and a curve written by the same run cannot
disagree about what produced them.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from splatpipe.manifest import collect_versions


@dataclass(frozen=True)
class CurvePoint:
    codec: str
    bytes: int
    ratio: float
    gaussians: int
    psnr: float
    ssim: float
    lpips: float
    encode_seconds: float
    decode_seconds: float

    @classmethod
    def of(
        cls,
        codec: str,
        size_bytes: int,
        gaussians: int,
        scores: dict[str, float],
        encode_seconds: float,
        decode_seconds: float,
        anchor_bytes: int,
    ) -> CurvePoint:
        return cls(
            codec=codec,
            bytes=size_bytes,
            ratio=anchor_bytes / size_bytes,
            gaussians=gaussians,
            psnr=scores["psnr"],
            ssim=scores["ssim"],
            lpips=scores["lpips"],
            encode_seconds=encode_seconds,
            decode_seconds=decode_seconds,
        )


@dataclass
class Curve:
    scene: str
    config_digest: str
    held_out_views: int
    created_utc: str
    versions: dict[str, str] = field(default_factory=dict)
    points: list[CurvePoint] = field(default_factory=list)

    @classmethod
    def start(cls, scene: str, config_digest: str, held_out_views: int) -> Curve:
        return cls(
            scene=scene,
            config_digest=config_digest,
            held_out_views=held_out_views,
            created_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            versions=collect_versions(),
        )

    def add(self, point: CurvePoint) -> None:
        self.points.append(point)

    def write_json(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def read_json(cls, path: Path | str) -> Curve:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data["points"] = [CurvePoint(**point) for point in data["points"]]
        return cls(**data)

    def write_plot(self, path: Path | str) -> None:
        """Quality against size, size on a log axis because the range is 15x."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        figure, axes = plt.subplots(figsize=(6, 4), dpi=150)
        for point in self.points:
            megabytes = point.bytes / 2**20
            axes.scatter(megabytes, point.psnr, s=40)
            axes.annotate(
                f"{point.codec} ({point.ratio:.2f}x)",
                (megabytes, point.psnr),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=8,
            )
        axes.set_xscale("log")
        axes.set_xlabel("Size (MiB, log scale)")
        axes.set_ylabel("Held-out PSNR (dB)")
        axes.set_title(f"{self.scene}: rate against distortion")
        axes.grid(True, which="both", alpha=0.3)
        figure.tight_layout()
        figure.savefig(path)
        plt.close(figure)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_curve.py -q`

Expected: PASS, 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/splatpipe/bench/curve.py tests/test_curve.py
git commit -m "Add the curve, its JSON schema and the rate-distortion plot"
```

---

### Task 9: The bench command

**Files:**
- Create: `src/splatpipe/bench/run.py`
- Modify: `src/splatpipe/bench/codecs.py`
- Modify: `src/splatpipe/cli.py`
- Modify: `src/splatpipe/paths.py`
- Test: `tests/test_bench_cpu.py`

**Interfaces:**
- Consumes: every module from Tasks 3 to 8.
- Produces:
  - `RunPaths.curve_json` and `RunPaths.curve_png` properties.
  - `run_bench(run_dir: Path, scene_dir: Path, codecs: list, device: str = "cuda") -> Curve`.
  - `splatpipe bench <run-dir> --scene <scene-dir>` returning exit code 0, or 1 with `error: ...` on stderr.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_bench_cpu.py`:

```python
"""CPU coverage of the bench orchestration and its error contract.

The real thing needs a GPU. Everything about ordering, the curve's contents and
every way the command can fail is exercised here with rendering and scoring
stubbed, so it survives a two-week gap without a compiler shell.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from splatpipe.bench import run as bench_run
from splatpipe.bench.cameras import ValView
from splatpipe.bench.codecs import PlyCodec, SplatCodec
from splatpipe.bench.curve import Curve
from splatpipe.cli import main
from splatpipe.errors import ArtifactError
from splatpipe.gaussians import write_ply
from splatpipe.paths import RunPaths
from tests.test_codecs import a_cloud

SCORES = {"psnr": 22.5, "ssim": 0.81, "lpips": 0.19}


def a_run(tmp_path, name="truck", n=64):
    """Build the on-disk shape splatpipe run leaves behind."""
    paths = RunPaths.for_run(tmp_path / "out", name)
    paths.ensure()
    write_ply(a_cloud(n), paths.ply)
    paths.manifest.write_text(
        json.dumps({"name": name, "config_digest": "abc123def456"}), encoding="utf-8"
    )
    return paths


def stub_views(monkeypatch, count=3):
    views = [
        ValView(
            camtoworld=np.eye(4, dtype=np.float32),
            K=np.eye(3, dtype=np.float32),
            image=np.zeros((8, 8, 3), dtype=np.uint8),
        )
        for _ in range(count)
    ]
    monkeypatch.setattr(bench_run, "load_val_views", lambda *a, **k: views)
    monkeypatch.setattr(
        bench_run, "render_view", lambda cloud, view, device="cuda": np.zeros(
            (8, 8, 3), dtype=np.float32
        )
    )
    monkeypatch.setattr(bench_run, "score_views", lambda *a, **k: dict(SCORES))
    return views


def test_writes_a_curve_with_one_point_per_codec(tmp_path, monkeypatch):
    paths = a_run(tmp_path)
    stub_views(monkeypatch)
    curve = bench_run.run_bench(
        paths.root, tmp_path / "scene", [PlyCodec(), SplatCodec()], device="cpu"
    )
    assert [point.codec for point in curve.points] == ["ply", "splat"]
    assert paths.curve_json.is_file()


def test_the_anchor_is_the_first_codec(tmp_path, monkeypatch):
    paths = a_run(tmp_path)
    stub_views(monkeypatch)
    curve = bench_run.run_bench(
        paths.root, tmp_path / "scene", [PlyCodec(), SplatCodec()], device="cpu"
    )
    assert curve.points[0].ratio == pytest.approx(1.0)
    assert curve.points[1].ratio > 1.0


def test_records_the_held_out_view_count(tmp_path, monkeypatch):
    paths = a_run(tmp_path)
    stub_views(monkeypatch, count=5)
    curve = bench_run.run_bench(paths.root, tmp_path / "scene", [PlyCodec()], device="cpu")
    assert curve.held_out_views == 5


def test_missing_trained_ply_raises_an_artifact_error(tmp_path, monkeypatch):
    paths = RunPaths.for_run(tmp_path / "out", "truck")
    paths.ensure()
    stub_views(monkeypatch)
    with pytest.raises(ArtifactError, match="no scene.ply"):
        bench_run.run_bench(paths.root, tmp_path / "scene", [PlyCodec()], device="cpu")


def test_cli_reports_a_missing_run_directory(tmp_path, capsys):
    assert main(["bench", str(tmp_path / "nope"), "--scene", str(tmp_path)]) == 1
    assert "error:" in capsys.readouterr().err


def test_cli_returns_zero_on_success(tmp_path, monkeypatch):
    """--codecs keeps this on the fast tier.

    The default list includes png, whose encode needs CUDA, cupy and plas.
    Selecting the two numpy-only codecs is what lets the command's success path
    be exercised without a GPU.
    """
    paths = a_run(tmp_path)
    stub_views(monkeypatch)
    assert (
        main(
            [
                "bench",
                str(paths.root),
                "--scene",
                str(tmp_path / "scene"),
                "--codecs",
                "ply,splat",
            ]
        )
        == 0
    )
    assert paths.curve_json.is_file()
    assert paths.curve_png.is_file()


def test_cli_rejects_an_unknown_codec_name(tmp_path, capsys):
    paths = a_run(tmp_path)
    assert (
        main(
            [
                "bench",
                str(paths.root),
                "--scene",
                str(tmp_path / "scene"),
                "--codecs",
                "ply,jpeg",
            ]
        )
        == 1
    )
    assert "unknown codec" in capsys.readouterr().err
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_bench_cpu.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'splatpipe.bench.run'`.

- [ ] **Step 3: Add the two output paths**

Add to `RunPaths` in `src/splatpipe/paths.py`, beside the existing properties:

```python
    @property
    def curve_json(self) -> Path:
        return self.root / "curve.json"

    @property
    def curve_png(self) -> Path:
        return self.root / "curve.png"
```

- [ ] **Step 4: Implement the orchestration**

Create `src/splatpipe/bench/run.py`:

```python
"""Measure every codec against one trained run.

The first codec in the list is the anchor: its size is the denominator of every
ratio, and it is expected to be lossless. Each codec encodes into its own
subdirectory of a scratch area so that sizes never overlap.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from splatpipe.bench.cameras import load_val_views
from splatpipe.bench.curve import Curve, CurvePoint
from splatpipe.bench.metrics import score_views
from splatpipe.bench.render import render_view
from splatpipe.errors import ArtifactError
from splatpipe.gaussians import read_ply
from splatpipe.paths import RunPaths


def run_bench(
    run_dir: Path,
    scene_dir: Path,
    codecs: list,
    device: str = "cuda",
    data_factor: int = 1,
    test_every: int = 8,
) -> Curve:
    run_dir = Path(run_dir)
    paths = RunPaths(root=run_dir)
    if not paths.ply.is_file():
        raise ArtifactError(
            f"{run_dir} holds no scene.ply at {paths.ply}. "
            f"Run splatpipe run on this scene first."
        )
    if not codecs:
        raise ArtifactError("no codecs to measure")

    cloud = read_ply(paths.ply)
    views = load_val_views(scene_dir, data_factor=data_factor, test_every=test_every)
    if not views:
        raise ArtifactError(
            f"{scene_dir} yielded no held-out views at test_every={test_every}"
        )
    targets = [view.image.astype("float32") / 255.0 for view in views]

    curve = Curve.start(
        scene=_scene_name(paths, run_dir),
        config_digest=_config_digest(paths),
        held_out_views=len(views),
    )

    scratch = run_dir / "bench"
    if scratch.exists():
        shutil.rmtree(scratch)

    anchor_bytes = None
    for codec in codecs:
        directory = scratch / codec.name
        directory.mkdir(parents=True, exist_ok=True)

        started = time.time()
        codec.encode(cloud, directory)
        encode_seconds = time.time() - started

        started = time.time()
        decoded = codec.decode(directory)
        decode_seconds = time.time() - started

        size_bytes = codec.size(directory)
        if anchor_bytes is None:
            anchor_bytes = size_bytes

        rendered = [render_view(decoded, view, device=device) for view in views]
        scores = score_views(rendered, targets, device=device)

        curve.add(
            CurvePoint.of(
                codec=codec.name,
                size_bytes=size_bytes,
                gaussians=len(decoded),
                scores=scores,
                encode_seconds=encode_seconds,
                decode_seconds=decode_seconds,
                anchor_bytes=anchor_bytes,
            )
        )
        print(
            f"{codec.name:6s} {size_bytes:>12,} bytes  "
            f"PSNR {scores['psnr']:.3f}  SSIM {scores['ssim']:.4f}  "
            f"LPIPS {scores['lpips']:.4f}"
        )

    curve.write_json(paths.curve_json)
    curve.write_plot(paths.curve_png)
    return curve


def _scene_name(paths: RunPaths, run_dir: Path) -> str:
    try:
        return json.loads(paths.manifest.read_text(encoding="utf-8"))["name"]
    # The manifest is informational here. A curve for a run whose manifest was
    # hand-edited or removed is still worth producing, named after its directory.
    except (OSError, KeyError, json.JSONDecodeError):
        return run_dir.name


def _config_digest(paths: RunPaths) -> str:
    try:
        return json.loads(paths.manifest.read_text(encoding="utf-8"))["config_digest"]
    except (OSError, KeyError, json.JSONDecodeError):
        return "unknown"
```

- [ ] **Step 5: Add the codec registry**

Append to `src/splatpipe/bench/codecs.py`. The registry exists so a codec can be
selected by name from the command line: running one codec without paying for the
other two matters from milestone 3 on, when the sweep gets long, and it is what
lets the CPU tests exercise the command without constructing a CUDA codec.

```python
CODECS = {
    "ply": PlyCodec,
    "splat": SplatCodec,
    "png": PngCodec,
}


def build_codecs(names: str) -> list:
    """Build codecs from a comma-separated name list, preserving order.

    Order is significant: the first codec is the anchor whose size is the
    denominator of every ratio, so it should be a lossless one.
    """
    selected = [name.strip() for name in names.split(",") if name.strip()]
    if not selected:
        raise ConfigError("no codecs selected")
    unknown = [name for name in selected if name not in CODECS]
    if unknown:
        raise ConfigError(
            f"unknown codec(s) {', '.join(unknown)}. "
            f"Known codecs are: {', '.join(sorted(CODECS))}"
        )
    return [CODECS[name]() for name in selected]
```

Add `from splatpipe.errors import ConfigError` to that module's imports.

- [ ] **Step 6: Wire the subcommand into the CLI**

In `src/splatpipe/cli.py`, add the parser beside the existing `run` parser inside `main`:

```python
    bench = subparsers.add_parser("bench", help="measure codecs against a finished run")
    bench.add_argument("run_dir", type=Path, help="an existing splatpipe run directory")
    bench.add_argument("--scene", type=Path, required=True, help="the COLMAP scene it was trained on")
    bench.add_argument("--data-factor", type=int, default=1)
    bench.add_argument("--test-every", type=int, default=8)
    bench.add_argument(
        "--codecs",
        default="ply,splat,png",
        help="comma-separated codec names in measurement order. The first is the anchor.",
    )
```

Replace the single-command dispatch in `main` with:

```python
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            run_pipeline(args.scene, RunConfig.from_toml(args.config), args.out, args.skip_train)
        else:
            from splatpipe.bench.codecs import build_codecs
            from splatpipe.bench.run import run_bench

            run_bench(
                args.run_dir,
                args.scene,
                build_codecs(args.codecs),
                data_factor=args.data_factor,
                test_every=args.test_every,
            )
    except SplatpipeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_bench_cpu.py -q`

Expected: PASS, 7 passed.

The CLI tests stub `bench_run`'s module-level names, so the default codec list including `PngCodec` is never constructed on the CPU path. If a test fails importing cupy, the stub is being applied to the wrong module object; patch `splatpipe.bench.run`, not `splatpipe.cli`.

- [ ] **Step 8: Run the whole fast tier**

Run: `.venv\Scripts\python.exe -m pytest -q`

Expected: PASS, 149 passed, 8 deselected.

- [ ] **Step 9: Commit**

```bash
git add src/splatpipe/bench/run.py src/splatpipe/bench/codecs.py src/splatpipe/cli.py src/splatpipe/paths.py tests/test_bench_cpu.py
git commit -m "Add the splatpipe bench subcommand and its CPU coverage"
```

---

### Task 10: End-to-end on the tiny scene

**Files:**
- Create: `tests/test_bench_e2e.py`

**Interfaces:**
- Consumes: everything above.
- Produces: nothing new. This is the first proof the whole chain runs on real rendered pixels.

- [ ] **Step 1: Write the test**

Create `tests/test_bench_e2e.py`:

```python
"""The real thing on a synthetic scene. Needs a GPU and env.bat.

Trains for 200 steps and benchmarks all three codecs. The reconstruction is
poor and that is fine: the point is that every stage hands its output to the
next one on real pixels, not that the numbers are good.
"""

from __future__ import annotations

import pytest

from splatpipe.bench.curve import Curve
from splatpipe.cli import main
from splatpipe.paths import RunPaths
from tests.fixtures.tiny_scene import make_tiny_scene

pytestmark = pytest.mark.gpu


def test_bench_measures_every_codec_on_a_trained_scene(tmp_path):
    scene = make_tiny_scene(tmp_path / "scene", n_images=24, n_points=512)
    config = tmp_path / "tiny.toml"
    # cap_max is an MCMC cap rather than an exact count, so the final Gaussian
    # count is not guaranteed square and PngCompression may crop a few. The
    # assertions below do not depend on it.
    config.write_text(
        'name = "tiny"\n\n[train]\nmax_steps = 200\ncap_max = 4096\ntest_every = 8\n',
        encoding="utf-8",
    )
    out_root = tmp_path / "out"
    assert main(["run", str(scene), "--config", str(config), "--out", str(out_root)]) == 0

    paths = RunPaths.for_run(out_root, "tiny")
    assert main(["bench", str(paths.root), "--scene", str(scene)]) == 0

    curve = Curve.read_json(paths.curve_json)
    assert [point.codec for point in curve.points] == ["ply", "splat", "png"]
    assert curve.held_out_views == 3
    assert paths.curve_png.is_file()

    ply, splat, png = curve.points
    assert ply.ratio == pytest.approx(1.0)
    # The .splat discards f_rest, so it is smaller than the ply and worse than it.
    assert splat.bytes < ply.bytes
    assert splat.psnr < ply.psnr
    # Every point must carry finite metrics; a nan here means a decoder clamp
    # was missed and the whole curve is meaningless.
    for point in curve.points:
        assert point.psnr == point.psnr
        assert point.gaussians > 0
```

- [ ] **Step 2: Run it**

Run from PowerShell:

```
cmd /c "call scripts\env.bat >nul 2>&1 && .venv\Scripts\python.exe -m pytest tests/test_bench_e2e.py -q -o addopts="
```

Expected: PASS, 1 passed. Takes a few minutes: it trains, then encodes and renders three times.

- [ ] **Step 3: Commit**

```bash
git add tests/test_bench_e2e.py
git commit -m "Benchmark all three codecs end to end on the synthetic scene"
```

---

### Task 11: The truck criterion and the documentation

This is the milestone's falsification criterion. It either reproduces the trainer's own numbers or the camera pipeline has drifted, and no later number can be trusted until it does.

**Files:**
- Create: `tests/test_bench_truck.py`
- Modify: `README.md`
- Modify: `HANDOFF.md`

**Interfaces:**
- Consumes: everything above.
- Produces: the measured baseline curve, recorded in the README.

- [ ] **Step 1: Write the criterion test**

Create `tests/test_bench_truck.py`:

```python
"""Milestone 2's falsification criterion.

PlyCodec is lossless, so rendering its output must reproduce what the trainer
measured when it rendered the same Gaussians. If the camera pipeline has
drifted, this fails. Without it, every number the project reports afterwards is
unfalsifiable.

The tolerance is fixed here, before the run, and is not to be adjusted
afterwards. It is tighter than milestone 1's 0.1 dB because nothing is being
retrained: the only remaining variation is float reduction order.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from splatpipe.bench.cameras import load_val_views
from splatpipe.bench.metrics import score_views
from splatpipe.bench.render import render_view
from splatpipe.gaussians import read_ply

pytestmark = pytest.mark.gpu

REPO = Path(__file__).resolve().parents[1]
TRUCK_PLY = REPO / "out" / "truck" / "artifacts" / "scene.ply"
TRUCK_SCENE = REPO / "data" / "tandt" / "truck"

# Recorded by the trainer on 2026-08-31, out/truck/train/stats/val_step6999.json.
TRAINER_PSNR = 24.394817
TRAINER_SSIM = 0.8579996
TRAINER_LPIPS = 0.1375573
PSNR_TOLERANCE = 0.05


@pytest.mark.skipif(not TRUCK_PLY.is_file(), reason="truck run not present")
def test_bench_reproduces_the_trainers_own_held_out_metrics():
    cloud = read_ply(TRUCK_PLY)
    views = load_val_views(TRUCK_SCENE, data_factor=1, test_every=8)
    targets = [view.image.astype("float32") / 255.0 for view in views]
    rendered = [render_view(cloud, view) for view in views]
    scores = score_views(rendered, targets)

    assert scores["psnr"] == pytest.approx(TRAINER_PSNR, abs=PSNR_TOLERANCE)
    assert scores["ssim"] == pytest.approx(TRAINER_SSIM, abs=0.002)
    assert scores["lpips"] == pytest.approx(TRAINER_LPIPS, abs=0.005)
```

- [ ] **Step 2: Run the criterion**

Run from PowerShell:

```
cmd /c "call scripts\env.bat >nul 2>&1 && .venv\Scripts\python.exe -m pytest tests/test_bench_truck.py -q -o addopts="
```

Expected: PASS. If it fails, stop. Do not widen the tolerance. Check first that `NORMALIZE_WORLD_SPACE` is `True`, that `data_factor` matches the config that trained the run, and that `test_every` matches. Those three are the ways the poses drift.

- [ ] **Step 3: Produce the real curve**

Run from PowerShell:

```
cmd /c "call scripts\env.bat >nul 2>&1 && .venv\Scripts\splatpipe.exe bench out\truck --scene data\tandt\truck"
```

Expected: three lines of output and `out/truck/curve.json` plus `out/truck/curve.png`. `PngCompression` takes roughly two minutes on this cloud.

- [ ] **Step 4: Record the measured baseline in the README**

Add a section after the existing measured baseline table, filling the table from `out/truck/curve.json` rather than from this plan. The rate column is already known from the preflight probe; the quality columns are what this milestone measured for the first time.

```markdown
## Measured baselines

The first points on the rate-distortion curve, measured on the truck scene's 32
held-out views. Held-out, not training views: measuring on the images the scene
was fitted to would overstate every result.

| Codec | Bytes | Ratio | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|
| Raw `.ply` | 236,001,478 | 1.00x | | | |
| `.splat` | 32,000,000 | 7.38x | | | |
| `PngCompression` | 16,258,005 | 14.52x | | | |

`PngCompression` is both smaller than the `.splat` and keeps the view-dependent
spherical harmonics the `.splat` discards, so it, not the `.splat`, is the
baseline this project's compressor has to beat.
```

- [ ] **Step 5: Update the handoff**

In `HANDOFF.md`, move milestone 2 from Next Steps into Completed, record the measured numbers, and note that the compression target is the `PngCompression` point rather than the `.splat` point.

- [ ] **Step 6: Run both tiers**

Run: `.venv\Scripts\python.exe -m pytest -q`

Expected: PASS, 149 passed, 10 deselected.

Run from PowerShell:

```
cmd /c "call scripts\env.bat >nul 2>&1 && .venv\Scripts\python.exe -m pytest -q -o addopts="
```

Expected: PASS, 159 passed. Confirm pytest output is actually present; a `cmd /c` line launched from Git Bash exits 0 having run nothing.

- [ ] **Step 7: Commit**

```bash
git add tests/test_bench_truck.py README.md HANDOFF.md
git commit -m "Reproduce the trainer's held-out metrics and record the measured baselines"
```

---

## Test counts

The counts in the steps above assume every prior task landed. They are a
checksum, not a contract: if a task adds a test the plan did not anticipate,
update the expected count rather than deleting the test.

| After task | Fast tier | Deselected |
|---|---:|---:|
| 1 | 118 | 3 |
| 2 | 125 | 3 |
| 3 | 130 | 3 |
| 4 | 130 | 6 |
| 5 | 133 | 6 |
| 6 | 133 | 8 |
| 7 | 137 | 8 |
| 8 | 142 | 8 |
| 9 | 149 | 8 |
| 10 | 149 | 9 |
| 11 | 149 | 10 |
