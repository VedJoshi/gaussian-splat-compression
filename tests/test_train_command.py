from pathlib import Path

import pytest

from splatpipe.config import TrainConfig
from splatpipe.stages.train import build_train_command, read_val_metrics


def command_for(**overrides):
    return build_train_command(
        python=Path("py.exe"),
        trainer_dir=Path("gsplat/examples"),
        scene=Path("data/truck"),
        result_dir=Path("out/truck/train"),
        cfg=TrainConfig(**overrides),
    )


def test_strategy_is_a_positional_subcommand():
    assert command_for(strategy="mcmc")[:3] == ["py.exe", "simple_trainer.py", "mcmc"]


def test_every_flag_uses_hyphens_not_underscores():
    """tyro derives flags from field names with hyphens. Underscores fail."""
    for token in command_for():
        if token.startswith("--"):
            assert "_" not in token, token


def test_values_come_from_the_config():
    command = command_for(max_steps=50, cap_max=5000, data_factor=2, test_every=4)
    for flag, value in (
        ("--max-steps", "50"),
        ("--strategy.cap-max", "5000"),
        ("--data-factor", "2"),
        ("--test-every", "4"),
        ("--eval-steps", "50"),
    ):
        assert command[command.index(flag) + 1] == value


def test_no_seed_flag_is_passed():
    """simple_trainer.py has no seed field; passing one would make tyro exit 2."""
    assert not [token for token in command_for() if "seed" in token]


def test_viewer_is_disabled_and_ply_is_saved():
    command = command_for()
    assert "--disable-viewer" in command
    assert "--save-ply" in command


def test_cap_max_is_only_passed_to_the_mcmc_strategy():
    """--strategy.cap-max is an MCMC field; the default strategy rejects it."""
    assert "--strategy.cap-max" not in command_for(strategy="default")


def test_eval_steps_value_is_followed_by_a_flag_not_a_bare_value():
    """--eval-steps takes a variable number of ints (tyro List[int]), so the next
    token must start with -- or tyro would try to consume it as another step."""
    command = command_for()
    next_token = command[command.index("--eval-steps") + 2]
    assert next_token.startswith("--")


def test_read_val_metrics(tmp_path):
    stats = tmp_path / "stats"
    stats.mkdir()
    (stats / "val_step6999.json").write_text(
        '{"psnr": 24.4, "ssim": 0.858, "lpips": 0.137, "num_GS": 1000000}', encoding="utf-8"
    )
    metrics = read_val_metrics(tmp_path, max_steps=7000)
    assert metrics["psnr"] == pytest.approx(24.4)
    assert metrics["num_GS"] == 1000000


def test_read_val_metrics_pads_step_to_four_digits(tmp_path):
    """simple_trainer.py writes stats as f"{stage}_step{step:04d}.json"
    (simple_trainer.py:989), zero-padded to four digits, unlike the unpadded ply
    filename. Below step 1000 the two spellings differ, so this pins the padding."""
    stats = tmp_path / "stats"
    stats.mkdir()
    (stats / "val_step0199.json").write_text(
        '{"psnr": 18.1, "ssim": 0.5, "lpips": 0.4, "num_GS": 5000}', encoding="utf-8"
    )
    metrics = read_val_metrics(tmp_path, max_steps=200)
    assert metrics["psnr"] == pytest.approx(18.1)
    assert metrics["num_GS"] == 5000
