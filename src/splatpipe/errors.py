"""Every exception this package raises deliberately.

Kept in one module so that no module imports another only to catch its errors.
"""


class SplatpipeError(Exception):
    """Base class. Catch this to catch anything the pipeline raises deliberately."""


class BuildEnvError(SplatpipeError):
    """The vcvars64 / CUDA environment is not set up. See scripts/env.bat."""


class ConfigError(SplatpipeError):
    """A run config is malformed, has unknown keys, or has out-of-range values."""


class SceneError(SplatpipeError):
    """An input scene directory is missing something the trainer needs."""


class ArtifactError(SplatpipeError):
    """An expected artifact is missing or does not have the shape we expect."""
