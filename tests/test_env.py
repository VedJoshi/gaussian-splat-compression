import pytest

from splatpipe.env import REQUIRED_ARCH_LIST, REQUIRED_CUDA_VERSION, check_build_env
from splatpipe.errors import BuildEnvError

GOOD_ENV = {
    "PATH": r"C:\fake\msvc\bin",
    "CUDA_HOME": rf"C:\CUDA\{REQUIRED_CUDA_VERSION}",
    "TORCH_CUDA_ARCH_LIST": REQUIRED_ARCH_LIST,
}


def found(cmd, path=None):
    return r"C:\fake\msvc\bin\cl.exe"


def missing(cmd, path=None):
    return None


def test_good_environment_passes():
    check_build_env(GOOD_ENV, which=found)


def test_missing_compiler_is_reported():
    with pytest.raises(BuildEnvError) as excinfo:
        check_build_env(GOOD_ENV, which=missing)
    assert "env.bat" in str(excinfo.value)


def test_wrong_cuda_version_is_reported():
    env = GOOD_ENV | {"CUDA_HOME": r"C:\CUDA\v12.9"}
    with pytest.raises(BuildEnvError) as excinfo:
        check_build_env(env, which=found)
    assert REQUIRED_CUDA_VERSION in str(excinfo.value)


def test_all_problems_are_reported_at_once():
    with pytest.raises(BuildEnvError) as excinfo:
        check_build_env({}, which=missing)
    assert str(excinfo.value).count("\n  - ") == 3, str(excinfo.value)
