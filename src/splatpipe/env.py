"""Preflight checks for the gsplat build environment.

gsplat's CUDA backend runs `where cl` on import even when the extension is
already compiled and cached, so an ordinary shell cannot import gsplat at all.
Everything that touches gsplat has to run inside scripts/env.bat.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable, Mapping

from splatpipe.errors import BuildEnvError

REQUIRED_CUDA_VERSION = "v12.8"
REQUIRED_ARCH_LIST = "8.9"

_HINT = "Run scripts\\env.bat in this shell first."


def check_build_env(
    env: Mapping[str, str] | None = None,
    which: Callable[..., str | None] = shutil.which,
) -> None:
    """Raise BuildEnvError if this shell cannot build or load gsplat.

    Reports every problem at once rather than stopping at the first one.
    """
    env = os.environ if env is None else env
    problems: list[str] = []

    if which("cl", path=env.get("PATH")) is None:
        problems.append(f"cl.exe is not on PATH. {_HINT}")

    cuda_home = env.get("CUDA_HOME", "")
    if REQUIRED_CUDA_VERSION not in cuda_home:
        problems.append(
            f"CUDA_HOME is {cuda_home or '(unset)'}, but torch is built against cu128, "
            f"so it must point at a {REQUIRED_CUDA_VERSION} install. {_HINT}"
        )

    arch_list = env.get("TORCH_CUDA_ARCH_LIST")
    if arch_list != REQUIRED_ARCH_LIST:
        problems.append(
            f"TORCH_CUDA_ARCH_LIST is {arch_list or '(unset)'}, expected "
            f"{REQUIRED_ARCH_LIST}. Without it a rebuild targets every architecture "
            f"and takes roughly 8x longer. {_HINT}"
        )

    if problems:
        raise BuildEnvError("Build environment is not set up:\n  - " + "\n  - ".join(problems))
