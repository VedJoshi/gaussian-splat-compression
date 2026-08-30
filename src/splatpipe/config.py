"""Run configuration.

Frozen dataclasses with validation in __post_init__, loaded from TOML. Unknown
keys are an error rather than a silent no-op: a typo in a config file that is
quietly ignored costs a whole training run to notice.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

from splatpipe.errors import ConfigError

STRATEGIES = ("mcmc", "default")
ORDERS = ("morton", "size_opacity", "none")


def _reject_unknown(cls: type, data: dict[str, Any]) -> None:
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ConfigError(
            f"{cls.__name__}: unknown key(s) {', '.join(unknown)}. Known keys are: "
            f"{', '.join(sorted(known))}"
        )


@dataclass(frozen=True)
class TrainConfig:
    max_steps: int = 7000
    cap_max: int = 1_000_000
    data_factor: int = 1
    test_every: int = 8
    strategy: str = "mcmc"

    def __post_init__(self) -> None:
        for field_name in ("max_steps", "cap_max", "data_factor", "test_every"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ConfigError(f"{field_name} must be an integer, got {value!r}")
        if self.max_steps <= 0:
            raise ConfigError(f"max_steps must be positive, got {self.max_steps}")
        if self.cap_max <= 0:
            raise ConfigError(f"cap_max must be positive, got {self.cap_max}")
        if self.data_factor < 1:
            raise ConfigError(f"data_factor must be at least 1, got {self.data_factor}")
        if self.test_every < 2:
            raise ConfigError(
                f"test_every must be at least 2, got {self.test_every}. Below 2 there "
                f"are no training images left, and at 1 every image is held out."
            )
        if self.strategy not in STRATEGIES:
            raise ConfigError(f"strategy must be one of {STRATEGIES}, got {self.strategy!r}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainConfig:
        _reject_unknown(cls, data)
        return cls(**data)


@dataclass(frozen=True)
class ExportConfig:
    order: str = "morton"

    def __post_init__(self) -> None:
        if self.order not in ORDERS:
            raise ConfigError(f"order must be one of {ORDERS}, got {self.order!r}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExportConfig:
        _reject_unknown(cls, data)
        return cls(**data)


@dataclass(frozen=True)
class RunConfig:
    name: str
    train: TrainConfig = field(default_factory=TrainConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    source_path: Path | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not self.name or any(c in self.name for c in r'\/:*?"<>| '):
            raise ConfigError(
                f"name must be a non-empty string usable as a directory name, got {self.name!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("source_path")
        return data

    def digest(self) -> str:
        """Twelve hex characters identifying the parameters, ignoring the name."""
        payload = self.to_dict()
        payload.pop("name")
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunConfig:
        data = dict(data)
        train = TrainConfig.from_dict(data.pop("train", {}))
        export = ExportConfig.from_dict(data.pop("export", {}))
        _reject_unknown(cls, data)
        return cls(train=train, export=export, **data)

    @classmethod
    def from_toml(cls, path: str | Path) -> RunConfig:
        path = Path(path)
        # utf-8-sig strips a leading byte order mark if present and is a no-op
        # otherwise, so one decode handles both. PowerShell's
        # Set-Content -Encoding utf8 and several Windows editors write a BOM
        # by default, and tomllib.load rejects it outright.
        text = path.read_bytes().decode("utf-8-sig")
        try:
            data = tomllib.loads(text)
        except tomllib.TOMLDecodeError as error:
            raise ConfigError(f"{path} is not valid TOML: {error}") from error
        return replace(cls.from_dict(data), source_path=path)
