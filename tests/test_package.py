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


def test_png_compression_dependencies_are_importable():
    """PngCompression's three undeclared dependencies must be installed.

    torchpq declares only numpy and torch in its metadata but imports cupy at
    runtime inside torchpq.clustering.KMeans, so importing torchpq alone proves
    nothing. All three are checked explicitly.
    """
    import cupy
    import plas
    import torchpq

    assert cupy.__version__.startswith("13."), (
        f"cupy must stay on 13.x; 14 and above require numpy>=2.0 and this "
        f"project pins numpy<2.0.0. Found {cupy.__version__}"
    )


def test_numpy_stays_below_2():
    import numpy

    assert numpy.__version__.startswith("1."), (
        f"numpy must stay below 2.0.0, found {numpy.__version__}"
    )
