"""The two upstream bugs the spike hit. See README.md for the full diagnosis."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from scripts.patches import Patch, Replacement

GSPLAT_MSVC_FLAGS = Patch(
    name="gsplat-msvc-flags",
    module="gsplat.cuda._backend",
    reason=(
        "gsplat passes GCC/Clang flags to the host compiler unconditionally. "
        "cl.exe reads '-Wno-attributes' as /W followed by 'no-attributes' and "
        "fails with D8021, and '-O3' is not MSVC syntax either."
    ),
    replacements=(
        Replacement(
            old='        extra_cflags = [opt_level, "-Wno-attributes"]\n',
            new=(
                "        # PATCHED by scripts/setup_env.py: MSVC flag portability.\n"
                '        if os.name == "nt":\n'
                '            extra_cflags = ["/Od" if FAST_COMPILE else "/O2"]\n'
                "        else:\n"
                '            extra_cflags = [opt_level, "-Wno-attributes"]\n'
            ),
            applied_marker='extra_cflags = ["/Od" if FAST_COMPILE else "/O2"]',
        ),
    ),
)

PYCOLMAP_STRUCT_WIDTHS = Patch(
    name="pycolmap-struct-widths",
    module="pycolmap.scene_manager",
    reason=(
        "The reader uses struct formats with no byte-order prefix, so 'L' takes "
        "native width: 8 bytes on Linux LP64, 4 on Windows LLP64. It reads 8 "
        "bytes from the file and then tries to unpack 4 of them."
    ),
    replacements=(
        Replacement(
            old="num_cameras = struct.unpack('L', f.read(8))[0]",
            new="num_cameras = struct.unpack('<Q', f.read(8))[0]",
        ),
        Replacement(
            old="camera_id, camera_type, w, h = struct.unpack('IiLL', f.read(24))",
            new="camera_id, camera_type, w, h = struct.unpack('<IiQQ', f.read(24))",
        ),
        Replacement(
            old="num_images = struct.unpack('L', f.read(8))[0]",
            new="num_images = struct.unpack('<Q', f.read(8))[0]",
        ),
        # Not a bug on either platform ('Q' is 8 bytes natively), but made
        # explicit so that grepping this file for native formats turns up only
        # the still-broken write path at lines 313-421.
        Replacement(
            old="num_points2D = struct.unpack('Q', f.read(8))[0]",
            new="num_points2D = struct.unpack('<Q', f.read(8))[0]",
        ),
        Replacement(
            old="num_points3D = struct.unpack('L', f.read(8))[0]",
            new="num_points3D = struct.unpack('<Q', f.read(8))[0]",
        ),
    ),
)

ALL_PATCHES = (GSPLAT_MSVC_FLAGS, PYCOLMAP_STRUCT_WIDTHS)


def resolve_target(patch: Patch) -> Path:
    """Find the installed source file a patch targets, without executing it.

    This used to call importlib.import_module, which reads better and cannot
    bootstrap. Importing gsplat.cuda._backend runs gsplat's __init__, which
    JIT-compiles the CUDA extension, which is the very thing the MSVC flag
    patch exists to make possible: on a fresh install the import died with
    `cl : Command line error D8021 : invalid numeric argument '/Wno-attributes'`
    before it could report which file to patch. Verified by provisioning a
    clean venv on 2026-08-31.

    find_spec on a top-level name locates the package without running it. The
    submodule path is then walked on disk. find_spec is not used on the dotted
    name directly, because for a dotted name it imports the parent package,
    which is the deadlock again.
    """
    top_level, _, submodule = patch.module.partition(".")
    spec = importlib.util.find_spec(top_level)
    if spec is None:
        raise RuntimeError(f"{top_level} is not installed, so {patch.name} cannot be resolved")

    if not submodule:
        if spec.origin is None:
            raise RuntimeError(f"{patch.module} has no file on disk and cannot be patched")
        return Path(spec.origin)

    if not spec.submodule_search_locations:
        raise RuntimeError(f"{top_level} is not a package, so {patch.module} cannot be resolved")

    root = Path(next(iter(spec.submodule_search_locations)))
    parts = submodule.split(".")
    candidates = (root.joinpath(*parts).with_suffix(".py"), root.joinpath(*parts, "__init__.py"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RuntimeError(f"{patch.module} was not found under {root}, tried {candidates}")
