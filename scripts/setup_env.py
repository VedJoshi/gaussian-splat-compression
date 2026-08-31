"""Make a fresh checkout runnable. Idempotent; safe to run repeatedly.

    python scripts/setup_env.py --check    report status, change nothing
    python scripts/setup_env.py            clone gsplat at the pin, apply patches

Runs without scripts/env.bat and without a GPU. Resolving a patch target
locates the installed file on disk rather than importing it, so this script can
run before gsplat is able to compile anything, which is what lets it provision a
fresh machine. See resolve_target in scripts/patches/definitions.py.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.patches import apply_patch, patch_status  # noqa: E402
from scripts.patches.definitions import ALL_PATCHES, resolve_target  # noqa: E402

GSPLAT_URL = "https://github.com/nerfstudio-project/gsplat.git"
GSPLAT_PIN = "937e29912570c372bed6747a5c9bf85fed877bae"
GSPLAT_DIR = REPO_ROOT / "_gsplat_repo"


def gsplat_checkout_status() -> str:
    if not (GSPLAT_DIR / ".git").is_dir():
        return "missing"
    head = subprocess.run(
        ["git", "-C", str(GSPLAT_DIR), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return "pinned" if head == GSPLAT_PIN else f"at {head[:12]}, expected {GSPLAT_PIN[:12]}"


def ensure_gsplat_checkout() -> None:
    if not (GSPLAT_DIR / ".git").is_dir():
        print(f"cloning gsplat into {GSPLAT_DIR}")
        subprocess.run(["git", "clone", GSPLAT_URL, str(GSPLAT_DIR)], check=True)
    subprocess.run(["git", "-C", str(GSPLAT_DIR), "checkout", "--detach", GSPLAT_PIN], check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="report status, change nothing")
    args = parser.parse_args()

    print(f"gsplat checkout: {gsplat_checkout_status()}")
    for patch in ALL_PATCHES:
        target = resolve_target(patch)
        print(f"{patch.name}: {patch_status(patch, target)}  ({target})")

    if args.check:
        return 0

    ensure_gsplat_checkout()
    for patch in ALL_PATCHES:
        print(f"{patch.name}: {apply_patch(patch, resolve_target(patch))}")

    print("Environment ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
