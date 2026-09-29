"""Assemble the static site: picker, viewer, and the scenes pinned by site/SHA256SUMS.

    python scripts/build_site.py <scenes-dir> <out-dir>

Stdlib only, so the Pages workflow runs it without the project's environment.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SUMS = REPO / "site" / "SHA256SUMS"


def pinned(sums: Path) -> dict[str, str]:
    lines = sums.read_text(encoding="utf-8").splitlines()
    return {name.strip(): digest for digest, name in (line.split(maxsplit=1) for line in lines if line.strip())}


def build(scenes_dir: Path, out_dir: Path, sums: Path = SUMS) -> None:
    scenes_dir, out_dir = Path(scenes_dir), Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        sys.exit(f"{out_dir} is not empty; remove it to rebuild")
    for name, digest in pinned(sums).items():
        path = scenes_dir / name
        if not path.is_file():
            sys.exit(f"{name} is missing from {scenes_dir}")
        with path.open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != digest:
                sys.exit(f"{name} does not match {sums.name}")

    shutil.copytree(REPO / "viewer", out_dir / "viewer")
    shutil.copy2(REPO / "site" / "index.html", out_dir / "index.html")
    (out_dir / "scenes").mkdir()
    for name in pinned(sums):
        shutil.copy2(scenes_dir / name, out_dir / "scenes" / name)
    print(f"site ready: {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    build(Path(sys.argv[1]), Path(sys.argv[2]))
