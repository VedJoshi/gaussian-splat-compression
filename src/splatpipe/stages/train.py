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
        # eval_steps defaults to [7000, 30000] (simple_trainer.py:82) with no
        # max_steps-1 fallback, unlike the checkpoint and ply branches. Without
        # this, a short run such as the end-to-end test's 200 steps never
        # evaluates and read_val_metrics finds nothing. --eval-steps is a
        # tyro List[int] field, so it must be followed by another flag rather
        # than a bare value or tyro keeps consuming tokens as further steps.
        "--eval-steps",
        str(cfg.max_steps),
        "--save-ply",
        "--disable-viewer",
    ]
    if cfg.strategy == "mcmc":
        # cap_max is a field of MCMCStrategy; the default strategy has no such flag.
        command += ["--strategy.cap-max", str(cfg.cap_max)]
    return command


def read_val_metrics(train_dir: Path, max_steps: int) -> dict[str, float]:
    """Read gsplat's held-out evaluation stats for the final step.

    simple_trainer.py writes this file as f"{stage}_step{step:04d}.json"
    (simple_trainer.py:989), zero-padded to four digits, unlike the ply filename
    (point_cloud_{step}.ply, no padding). Below step 1000 the two spellings
    diverge, so the padding has to be applied here explicitly.
    """
    stats = train_dir / "stats" / f"val_step{max_steps - 1:04d}.json"
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
        try:
            subprocess.run(
                command,
                cwd=str(trainer_dir),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        except subprocess.CalledProcessError as error:
            raise ArtifactError(
                f"training exited with code {error.returncode}. "
                f"See {paths.train_log} for the trainer's output."
            ) from error
    return time.time() - started
