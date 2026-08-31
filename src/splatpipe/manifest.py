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
        # Deliberately broad: a manifest must never fail a run over a version
        # string. The cost is that a broken install and an absent one both
        # record "not-imported", which is acceptable for an informational field.
        except Exception:  # noqa: BLE001
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
