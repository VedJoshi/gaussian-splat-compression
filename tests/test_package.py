"""The package installs and its error types are importable from one place."""


def test_package_version():
    import splatpipe

    assert splatpipe.__version__ == "0.1.0"


def test_numpy_is_below_2():
    """gsplat 1.5.3's examples pin numpy<2. The venv has drifted off this before."""
    import numpy as np

    assert np.__version__.startswith("1."), f"numpy {np.__version__} breaks the gsplat examples"


def test_error_types_share_a_base():
    from splatpipe.errors import BuildEnvError, ConfigError, SceneError, SplatpipeError

    for exc in (BuildEnvError, ConfigError, SceneError):
        assert issubclass(exc, SplatpipeError)
