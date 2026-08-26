"""The two upstream bugs the spike hit. See README.md for the full diagnosis."""

from __future__ import annotations

import importlib
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
    applied_marker='extra_cflags = ["/Od" if FAST_COMPILE else "/O2"]',
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
    applied_marker="struct.unpack('<IiQQ', f.read(24))",
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
    """Find the installed source file a patch targets.

    Imports the module rather than guessing at site-packages layout, so this
    works for a venv, a user install, or an editable checkout.
    """
    module = importlib.import_module(patch.module)
    if module.__file__ is None:
        raise RuntimeError(f"{patch.module} has no __file__ and cannot be patched")
    return Path(module.__file__)
