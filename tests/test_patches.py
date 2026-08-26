import pytest

from scripts.patches import Patch, Replacement, apply_patch, patch_status

SIMPLE = Patch(
    name="example",
    module="does.not.matter",
    reason="a test patch",
    replacements=(Replacement(old="a = 1", new="a = 2"),),
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
