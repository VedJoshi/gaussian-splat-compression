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
    assert paths.pruning_dir == Path("out") / "truck" / "pruning"
    assert paths.prune_curve_json == paths.pruning_dir / "curve.json"
    assert paths.prune_curve_png == paths.pruning_dir / "curve.png"
    assert paths.prune_scores == paths.pruning_dir / "scores.npz"
    assert paths.prune_meta == paths.pruning_dir / "meta.json"


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
