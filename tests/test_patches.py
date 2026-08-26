import pytest

from scripts.patches import Patch, Replacement, apply_patch, patch_status

SIMPLE = Patch(
    name="example",
    module="does.not.matter",
    reason="a test patch",
    replacements=(Replacement(old="a = 1", new="a = 2"),),
)

INDENTED = Patch(
    name="indented",
    module="does.not.matter",
    reason="regression guard",
    replacements=(Replacement(old="    value = original\n", new="    value = patched\n"),),
)


def write(tmp_path, text):
    target = tmp_path / "module.py"
    target.write_text(text, encoding="utf-8")
    return target


def test_apply_rewrites_the_anchor(tmp_path):
    target = write(tmp_path, "x = 0\na = 1\ny = 3\n")
    assert apply_patch(SIMPLE, target) == "applied"
    assert target.read_text(encoding="utf-8") == "x = 0\na = 2\ny = 3\n"


def test_apply_is_idempotent(tmp_path):
    target = write(tmp_path, "a = 1\n")
    apply_patch(SIMPLE, target)
    assert apply_patch(SIMPLE, target) == "already-applied"
    assert target.read_text(encoding="utf-8") == "a = 2\n"


def test_status_reports_stale_when_the_anchor_is_gone(tmp_path):
    assert patch_status(SIMPLE, write(tmp_path, "b = 9\n")) == "stale"


def test_apply_refuses_a_stale_anchor(tmp_path):
    with pytest.raises(RuntimeError, match="does not match"):
        apply_patch(SIMPLE, write(tmp_path, "b = 9\n"))


def test_apply_refuses_an_ambiguous_anchor(tmp_path):
    with pytest.raises(RuntimeError, match="2 times"):
        apply_patch(SIMPLE, write(tmp_path, "a = 1\na = 1\n"))


def test_patch_preserves_the_original_four_argument_constructor():
    patch = Patch(
        name="compatibility",
        module="does.not.matter",
        reason="public API regression guard",
        replacements=(Replacement(old="old", new="new"),),
    )

    assert patch.name == "compatibility"


def test_partial_multi_replacement_patch_is_stale_and_refused(tmp_path):
    """One applied replacement must not hide the remaining stale anchors."""
    from scripts.patches.definitions import PYCOLMAP_STRUCT_WIDTHS

    replacements = PYCOLMAP_STRUCT_WIDTHS.replacements
    partial_source = "\n".join(
        (replacements[0].old, replacements[1].new, *(replacement.old for replacement in replacements[2:]))
    )
    target = write(tmp_path, partial_source)

    assert patch_status(PYCOLMAP_STRUCT_WIDTHS, target) == "stale"
    with pytest.raises(RuntimeError, match="does not match"):
        apply_patch(PYCOLMAP_STRUCT_WIDTHS, target)
    assert target.read_text(encoding="utf-8") == partial_source


def test_anchor_does_not_match_a_more_indented_line(tmp_path):
    """A 4-space anchor must not match an 8-space line that ends the same way.

    This is the bug that spliced a nested if/else into gsplat's already-patched
    _backend.py and produced an IndentationError in a live library file.
    """
    target = write(tmp_path, "def f():\n    if x:\n        value = original\n")
    assert patch_status(INDENTED, target) == "stale"


def test_applied_is_decided_by_the_marker_not_the_comment(tmp_path):
    """The installed gsplat patch carries a different comment banner than our
    definition. Same code, different wording, and it is still applied."""
    target = write(tmp_path, "# a completely different comment\n    value = patched\n")
    assert patch_status(INDENTED, target) == "applied"


def test_apply_refuses_to_write_invalid_python(tmp_path):
    breaker = Patch(
        name="breaker",
        module="does.not.matter",
        reason="guard",
        replacements=(Replacement(old="x = 1\n", new="def broken(:\n"),),
    )
    target = write(tmp_path, "x = 1\n")
    with pytest.raises(RuntimeError, match="invalid Python"):
        apply_patch(breaker, target)
    assert target.read_text(encoding="utf-8") == "x = 1\n"


@pytest.mark.gpu
def test_real_patches_apply_to_the_installed_packages():
    """The two upstream bugs from the spike. This is the test that matters.

    Marked `gpu` because resolving the gsplat target imports gsplat, which needs
    a vcvars shell. Run with: scripts\\env.bat then pytest -m ""
    """
    from scripts.patches.definitions import ALL_PATCHES, resolve_target

    assert len(ALL_PATCHES) == 2
    for patch in ALL_PATCHES:
        status = patch_status(patch, resolve_target(patch))
        assert status in {"applied", "appliable"}, f"{patch.name} is {status}"
