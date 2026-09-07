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
    if cfg.source_path is not None:
        shutil.copyfile(cfg.source_path, paths.config_file)

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

    # read_val_metrics runs first, so a train/ directory holding stats but no
    # final ply reaches this line. That combination is unlikely but it is not
    # unreachable, and without this the copy below raises a bare
    # FileNotFoundError straight past the CLI's error contract.
    if not trained_ply.is_file():
        raise ArtifactError(
            f"training left no final model at {trained_ply}. "
            f"See {paths.train_log} for the trainer's output."
        )

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

    bench = subparsers.add_parser("bench", help="measure codecs against a finished run")
    bench.add_argument("run_dir", type=Path, help="an existing splatpipe run directory")
    bench.add_argument("--scene", type=Path, required=True, help="the COLMAP scene it was trained on")
    bench.add_argument("--data-factor", type=int, default=1)
    bench.add_argument("--test-every", type=int, default=8)
    bench.add_argument(
        "--codecs",
        default="ply,splat,png,shvq256,shvq1024,shvq4096",
        help="comma-separated codec names in measurement order. The first is the anchor.",
    )

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


if __name__ == "__main__":
    raise SystemExit(main())
