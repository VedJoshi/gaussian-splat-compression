import pytest

from splatpipe.cli import main
from splatpipe.config import RunConfig
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
