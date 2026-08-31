"""CPU coverage of run_pipeline's orchestration and the CLI's error contract.

The two end-to-end tests in test_pipeline_e2e.py are the real thing, and they
need a GPU, a compiler shell, and over a minute. The fast tier is what has to
survive a two week gap, so stage ordering, the manifest's contents and every
way the CLI can fail are exercised here with the trainer stubbed out.

The stub stands in for exactly what gsplat leaves behind: a zero-padded stats
file and an unpadded ply, which are the two filename spellings that differ
below step 1000.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from splatpipe import cli
from splatpipe.cli import main, run_pipeline
from splatpipe.config import RunConfig
from splatpipe.errors import ArtifactError, ConfigError, SceneError
from splatpipe.formats.splat import BYTES_PER_GAUSSIAN
from splatpipe.gaussians import GaussianCloud, write_ply
from splatpipe.manifest import RunManifest
from splatpipe.paths import RunPaths
from tests.fixtures.tiny_scene import make_tiny_scene

MAX_STEPS = 50
METRICS = {"psnr": 21.5, "ssim": 0.72, "lpips": 0.31, "num_GS": 32}


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


def fake_training(*, write_ply_file: bool = True, write_stats: bool = True, n: int = 32):
    """Build a run_training replacement that leaves gsplat's output behind."""

    def stub(scene, paths, cfg, *args, **kwargs):
        if write_stats:
            stats = paths.train_dir / "stats"
            stats.mkdir(parents=True, exist_ok=True)
            (stats / f"val_step{cfg.train.max_steps - 1:04d}.json").write_text(
                json.dumps(METRICS), encoding="utf-8"
            )
        if write_ply_file:
            ply_dir = paths.train_dir / "ply"
            ply_dir.mkdir(parents=True, exist_ok=True)
            write_ply(a_cloud(n), ply_dir / f"point_cloud_{cfg.train.max_steps - 1}.ply")
        return 1.25

    return stub


@pytest.fixture
def scene(tmp_path):
    return make_tiny_scene(tmp_path / "scene", n_images=24, n_points=64)


@pytest.fixture
def config_path(tmp_path):
    path = tmp_path / "tiny.toml"
    path.write_text(
        f'name = "tiny"\n\n[train]\nmax_steps = {MAX_STEPS}\ncap_max = 5000\n',
        encoding="utf-8",
    )
    return path


@pytest.fixture
def stubbed(monkeypatch):
    """Replace the trainer and the build preflight, which both need a GPU."""

    def install(stub):
        monkeypatch.setattr(cli, "run_training", stub)
        monkeypatch.setattr(cli, "check_build_env", lambda: None)

    return install


def test_run_pipeline_produces_every_artifact_and_a_complete_manifest(
    tmp_path, scene, config_path, stubbed
):
    stubbed(fake_training())
    cfg = RunConfig.from_toml(config_path)

    paths = run_pipeline(scene, cfg, tmp_path / "out")

    assert paths.ply.is_file()
    assert paths.splat.is_file()
    assert paths.config_file.read_text(encoding="utf-8") == config_path.read_text(
        encoding="utf-8"
    )

    manifest = RunManifest.read(paths.manifest)
    assert manifest.name == "tiny"
    assert manifest.config_digest == cfg.digest()
    assert manifest.metrics == METRICS
    assert manifest.timings["train"] == 1.25
    assert manifest.timings["export"] >= 0
    assert [record.path for record in manifest.artifacts] == [
        "artifacts/scene.ply",
        "artifacts/scene.splat",
    ]
    # The recorded size is the file's own size, and the splat is exactly 32
    # bytes per Gaussian because invalid rows fail rather than being dropped.
    ply_record, splat_record = manifest.artifacts
    assert ply_record.bytes == paths.ply.stat().st_size
    assert splat_record.bytes == 32 * BYTES_PER_GAUSSIAN


def test_the_manifest_records_no_absolute_path_and_no_backslash(
    tmp_path, scene, config_path, stubbed
):
    """Recorded paths have to stay portable, which is what as_posix buys."""
    stubbed(fake_training())
    paths = run_pipeline(scene, RunConfig.from_toml(config_path), tmp_path / "out")

    raw = paths.manifest.read_text(encoding="utf-8")
    assert "\\\\" not in raw
    assert str(tmp_path) not in raw
    assert "source_path" not in json.loads(raw)["config"]


def test_skip_train_without_a_previous_run_is_reported(tmp_path, scene, config_path, stubbed):
    stubbed(fake_training())
    with pytest.raises(ArtifactError, match="Run once without it first"):
        run_pipeline(scene, RunConfig.from_toml(config_path), tmp_path / "out", skip_train=True)


def test_skip_train_reuses_the_training_output_without_training_again(
    tmp_path, scene, config_path, stubbed
):
    stubbed(fake_training())
    out_root = tmp_path / "out"
    run_pipeline(scene, RunConfig.from_toml(config_path), out_root)

    def refuse(*args, **kwargs):
        raise AssertionError("skip_train must not call the trainer")

    stubbed(refuse)
    paths = run_pipeline(scene, RunConfig.from_toml(config_path), out_root, skip_train=True)

    assert paths.splat.is_file()
    assert "train" not in RunManifest.read(paths.manifest).timings


def test_training_that_leaves_no_model_is_reported(tmp_path, scene, config_path, stubbed):
    """Stats present, ply missing. Without the guard this is a bare
    FileNotFoundError from shutil.copyfile, straight past the error contract."""
    stubbed(fake_training(write_ply_file=False))
    with pytest.raises(ArtifactError, match="left no final model"):
        run_pipeline(scene, RunConfig.from_toml(config_path), tmp_path / "out")


def test_training_that_produces_no_metrics_is_reported(tmp_path, scene, config_path, stubbed):
    stubbed(fake_training(write_stats=False))
    with pytest.raises(ArtifactError, match="no validation stats"):
        run_pipeline(scene, RunConfig.from_toml(config_path), tmp_path / "out")


def test_main_reports_a_missing_config_file(tmp_path, scene, capsys):
    code = main(
        ["run", str(scene), "--config", str(tmp_path / "absent.toml"), "--out", str(tmp_path)]
    )
    assert code == 1
    error = capsys.readouterr().err
    assert "cannot read config" in error
    assert "absent.toml" in error


def test_main_reports_a_config_that_is_a_directory(tmp_path, scene, capsys):
    directory = tmp_path / "not-a-file"
    directory.mkdir()
    code = main(["run", str(scene), "--config", str(directory), "--out", str(tmp_path)])
    assert code == 1
    assert "cannot read config" in capsys.readouterr().err


def test_main_reports_invalid_toml(tmp_path, scene, config_path, capsys):
    config_path.write_text("name = \n", encoding="utf-8")
    code = main(["run", str(scene), "--config", str(config_path), "--out", str(tmp_path)])
    assert code == 1
    assert "is not valid TOML" in capsys.readouterr().err


def test_main_reports_a_missing_scene(tmp_path, config_path, capsys):
    code = main(
        ["run", str(tmp_path / "absent"), "--config", str(config_path), "--out", str(tmp_path)]
    )
    assert code == 1
    assert "is not a usable scene" in capsys.readouterr().err


def test_main_returns_zero_on_success(tmp_path, scene, config_path, stubbed):
    stubbed(fake_training())
    code = main(
        ["run", str(scene), "--config", str(config_path), "--out", str(tmp_path / "out")]
    )
    assert code == 0


@pytest.mark.parametrize(
    "body, expected",
    [
        ("name = 2024\n", "name must be a string"),
        ('name = "."\n', "path traversal"),
        ('name = ".."\n', "path traversal"),
        ('name = "a b"\n', "usable as a directory name"),
        ('name = "x"\nsource_path = "sneaky"\n', "source_path is set by the loader"),
    ],
)
def test_configs_that_must_not_load(tmp_path, body, expected):
    path = tmp_path / "bad.toml"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(ConfigError, match=expected):
        RunConfig.from_toml(path)


def test_a_scene_error_names_every_problem_at_once(tmp_path):
    """SceneError is aggregate on purpose: one run, one list of what to fix."""
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SceneError) as caught:
        run_pipeline(empty, RunConfig(name="tiny"), tmp_path / "out")
    message = str(caught.value)
    assert "missing image folder" in message
    assert message.count("missing COLMAP file") == 3
