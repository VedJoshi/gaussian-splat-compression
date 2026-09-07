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


@dataclass(frozen=True)
class CalibrationView:
    camtoworld: np.ndarray  # (4, 4) float32
    K: np.ndarray  # (3, 3) float32
    height: int
    width: int


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


def load_train_views(
    scene_dir: Path,
    data_factor: int = 1,
    test_every: int = 8,
) -> list[CalibrationView]:
    """Load training-camera geometry without reading every training image."""
    Parser, _ = _import_colmap_dataset()
    parser = Parser(
        data_dir=str(Path(scene_dir).resolve()),
        factor=data_factor,
        normalize=NORMALIZE_WORLD_SPACE,
        test_every=test_every,
    )
    views = []
    indices = np.arange(len(parser.image_names))
    for index in indices[indices % parser.test_every != 0]:
        camera_id = parser.camera_ids[index]
        width, height = parser.imsize_dict[camera_id]
        views.append(
            CalibrationView(
                camtoworld=np.asarray(
                    parser.camtoworlds[index], dtype=np.float32
                ).copy(),
                K=np.asarray(parser.Ks_dict[camera_id], dtype=np.float32).copy(),
                height=int(height),
                width=int(width),
            )
        )
    return views
