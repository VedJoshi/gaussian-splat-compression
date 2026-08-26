import pytest

from splatpipe.config import ExportConfig, RunConfig, TrainConfig
from splatpipe.errors import ConfigError


def test_defaults_match_the_spike():
    cfg = RunConfig(name="truck")
    assert cfg.train.max_steps == 7000
    assert cfg.train.cap_max == 1_000_000
    assert cfg.train.data_factor == 1
    assert cfg.train.strategy == "mcmc"
    assert cfg.export.order == "morton"
    assert not hasattr(cfg.train, "seed"), "gsplat hardcodes the seed; do not pretend otherwise"


def test_round_trips_through_a_dict():
    cfg = RunConfig(name="truck", train=TrainConfig(max_steps=50))
    assert RunConfig.from_dict(cfg.to_dict()) == cfg


def test_digest_is_stable_and_short():
    cfg = RunConfig(name="truck")
    assert cfg.digest() == cfg.digest()
    assert len(cfg.digest()) == 12


def test_digest_tracks_parameters_but_not_the_name():
    base = RunConfig(name="truck")
    assert RunConfig(name="lorry").digest() == base.digest()
    assert RunConfig(name="truck", train=TrainConfig(max_steps=50)).digest() != base.digest()
    assert RunConfig(name="truck", export=ExportConfig(order="none")).digest() != base.digest()


def test_unknown_keys_are_rejected():
    with pytest.raises(ConfigError, match="max_step"):
        RunConfig.from_dict({"name": "truck", "train": {"max_step": 50}})


def test_out_of_range_values_are_rejected():
    with pytest.raises(ConfigError, match="max_steps"):
        RunConfig(name="truck", train=TrainConfig(max_steps=0))
    with pytest.raises(ConfigError, match="test_every"):
        RunConfig(name="truck", train=TrainConfig(test_every=1))
    with pytest.raises(ConfigError, match="strategy"):
        RunConfig(name="truck", train=TrainConfig(strategy="magic"))
    with pytest.raises(ConfigError, match="order"):
        RunConfig(name="truck", export=ExportConfig(order="sideways"))


@pytest.mark.parametrize("value", [2.5, True], ids=["float", "bool"])
@pytest.mark.parametrize("field", ["max_steps", "cap_max", "data_factor", "test_every"])
def test_non_integer_training_values_are_rejected(field, value):
    with pytest.raises(ConfigError, match=f"{field} must be an integer"):
        TrainConfig.from_dict({field: value})


def test_loads_from_toml(tmp_path):
    path = tmp_path / "run.toml"
    path.write_text('name = "tiny"\n\n[train]\nmax_steps = 50\ncap_max = 5000\n', encoding="utf-8")
    cfg = RunConfig.from_toml(path)
    assert cfg.name == "tiny"
    assert cfg.train.max_steps == 50
    assert cfg.train.cap_max == 5000
    assert cfg.train.data_factor == 1  # unspecified keys keep their default


def test_the_checked_in_truck_config_loads():
    cfg = RunConfig.from_toml("configs/truck.toml")
    assert cfg.name == "truck"
    assert cfg.train.max_steps == 7000


def test_source_path_is_recorded_but_is_not_a_parameter(tmp_path):
    """The CLI copies the config next to the output, so it needs to know where
    the file came from. That is provenance, so it stays out of the digest."""
    path = tmp_path / "run.toml"
    path.write_text('name = "tiny"\n', encoding="utf-8")
    cfg = RunConfig.from_toml(path)
    assert cfg.source_path == path
    assert "source_path" not in cfg.to_dict()
    assert cfg.digest() == RunConfig(name="tiny").digest()
    assert cfg == RunConfig(name="tiny")
