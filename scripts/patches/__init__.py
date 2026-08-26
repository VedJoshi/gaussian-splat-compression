"""Apply small, exact source patches to installed third-party packages.

These are search/replace pairs rather than unified diffs because patch.exe is
not present on a stock Windows install, and because an exact anchor can be
checked in both directions: "is this already applied" and "does this still
apply to the version currently installed".
"""

from __future__ import annotations

import ast
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
    applied_marker: str
    replacements: tuple[Replacement, ...]


def _anchor_positions(text: str, anchor: str) -> list[int]:
    """Return the start index of every valid occurrence of `anchor` in `text`.

    An anchor that begins with leading whitespace must start at a true line
    boundary (index 0, or immediately after a newline). Without that rule, a
    shorter, less-indented anchor can match as a bare suffix of a longer, more
    indented line ending in the same text: that is exactly what let a fix
    written for an 8-space line silently match, and then corrupt, an
    already-patched 12-space line holding the same code one level deeper. An
    anchor that does not start with whitespace (the pycolmap anchors, which
    intentionally omit indentation so they match at any nesting depth) is
    unaffected by this rule and matches as a plain substring, as before.
    """
    if not anchor:
        return []
    positions = []
    start = 0
    anchor_is_indented = anchor[:1] in (" ", "\t")
    while True:
        idx = text.find(anchor, start)
        if idx == -1:
            break
        if not anchor_is_indented or idx == 0 or text[idx - 1] == "\n":
            positions.append(idx)
        start = idx + 1
    return positions


def patch_status(patch: Patch, path: Path) -> str:
    """Return "applied", "appliable", or "stale".

    "applied" is decided by `applied_marker` alone, a fragment of the patched
    code rather than the full replacement text. A hand-applied fix can carry
    different comment wording than the canonical patch while being the same
    code change; deciding "applied" from the full replacement text (comment
    included) reports a genuinely patched file as merely "appliable", which is
    what let an already-patched file be treated as unpatched and rewritten.

    "stale" means the installed file no longer contains the anchors, which is
    what an upgrade looks like. That is a signal to go and re-derive the patch,
    not to force it.
    """
    text = path.read_text(encoding="utf-8")
    if patch.applied_marker in text:
        return "applied"
    if all(len(_anchor_positions(text, r.old)) == 1 for r in patch.replacements):
        return "appliable"
    return "stale"


def apply_patch(patch: Patch, path: Path) -> str:
    """Apply `patch` to `path` in place. Returns "applied" or "already-applied"."""
    text = path.read_text(encoding="utf-8")

    if patch.applied_marker in text:
        return "already-applied"

    for replacement in patch.replacements:
        positions = _anchor_positions(text, replacement.old)
        if not positions:
            raise RuntimeError(
                f"{patch.name}: anchor does not match {path}. The installed version "
                f"has changed and the patch needs re-deriving. Anchor was:\n{replacement.old}"
            )
        if len(positions) > 1:
            raise RuntimeError(
                f"{patch.name}: anchor appears {len(positions)} times in {path}, so the "
                f"replacement is ambiguous. Anchor was:\n{replacement.old}"
            )
        position = positions[0]
        text = text[:position] + replacement.new + text[position + len(replacement.old) :]

    try:
        ast.parse(text)
    except SyntaxError as error:
        raise RuntimeError(
            f"{patch.name}: applying the patch to {path} produced invalid Python "
            f"({error}). The file has been left unchanged."
        ) from error

    path.write_text(text, encoding="utf-8")
    return "applied"
