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

    @property
    def curve_json(self) -> Path:
        return self.root / "curve.json"

    @property
    def curve_png(self) -> Path:
        return self.root / "curve.png"

    def trained_ply(self, max_steps: int) -> Path:
        """Where gsplat writes the final ply. It names files by step index, so
        a 7000-step run produces point_cloud_6999.ply."""
        return self.train_dir / "ply" / f"point_cloud_{max_steps - 1}.ply"

    def ensure(self) -> None:
        for directory in (self.root, self.train_dir, self.artifacts_dir, self.logs_dir):
            directory.mkdir(parents=True, exist_ok=True)
