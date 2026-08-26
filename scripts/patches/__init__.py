"""Apply small, exact source patches to installed third-party packages.

These are search/replace pairs rather than unified diffs because patch.exe is
not present on a stock Windows install, and because an exact anchor can be
checked in both directions: "is this already applied" and "does this still
apply to the version currently installed".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Replacement:
    """One exact search/replace. `old` must appear exactly once in the file."""

    old: str
    new: str


@dataclass(frozen=True)
class Patch:
    name: str
    module: str
    reason: str
    replacements: tuple[Replacement, ...]


def patch_status(patch: Patch, path: Path) -> str:
    """Return "applied", "appliable", or "stale".

    "stale" means the installed file no longer contains the anchors, which is
    what an upgrade looks like. That is a signal to go and re-derive the patch,
    not to force it.
    """
    text = path.read_text(encoding="utf-8")
    if all(r.new in text for r in patch.replacements):
        return "applied"
    if all(text.count(r.old) == 1 for r in patch.replacements):
        return "appliable"
    return "stale"


def apply_patch(patch: Patch, path: Path) -> str:
    """Apply `patch` to `path` in place. Returns "applied" or "already-applied"."""
    text = path.read_text(encoding="utf-8")

    if all(r.new in text for r in patch.replacements):
        return "already-applied"

    for replacement in patch.replacements:
        count = text.count(replacement.old)
        if count == 0:
            raise RuntimeError(
                f"{patch.name}: anchor does not match {path}. The installed version "
                f"has changed and the patch needs re-deriving. Anchor was:\n{replacement.old}"
            )
        if count > 1:
            raise RuntimeError(
                f"{patch.name}: anchor appears {count} times in {path}, so the "
                f"replacement is ambiguous. Anchor was:\n{replacement.old}"
            )
        text = text.replace(replacement.old, replacement.new)

    path.write_text(text, encoding="utf-8")
    return "applied"
