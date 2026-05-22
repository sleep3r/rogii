from pathlib import Path

import pytest
import yaml

from mtpnet.config import load_config


def write_config(tmp_path: Path, k_wells: int) -> Path:
    path = tmp_path / "config.yml"
    path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "data_dir": "data",
                    "train_dir": "data/train",
                    "test_dir": "data/test",
                    "k_wells": k_wells,
                },
                "run": {"name": "unit"},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_k_wells_minus_one_is_all(tmp_path: Path) -> None:
    cfg = load_config(write_config(tmp_path, -1))
    assert cfg.data.k_wells == -1
    assert cfg.data.use_all_wells is True


def test_positive_k_wells_is_limited(tmp_path: Path) -> None:
    cfg = load_config(write_config(tmp_path, 7))
    assert cfg.data.k_wells == 7
    assert cfg.data.use_all_wells is False


def test_window_train_valid_modes_are_configurable(tmp_path: Path) -> None:
    path = tmp_path / "config.yml"
    path.write_text(
        yaml.safe_dump(
            {
                "window": {
                    "train_history_mode": "teacher_forcing",
                    "valid_history_mode": "known_tail_only",
                    "train_center_source": "true_tvt",
                    "valid_center_source": "tvt_input_tail",
                    "channels": ["gr_diff", "gr_z_diff", "dgr_diff"],
                }
            }
        ),
        encoding="utf-8",
    )
    cfg = load_config(path)

    assert cfg.window.train_history_mode == "teacher_forcing"
    assert cfg.window.valid_history_mode == "known_tail_only"
    assert cfg.window.train_center_source == "true_tvt"
    assert cfg.window.valid_center_source == "tvt_input_tail"
    assert cfg.window.channels == ("gr_diff", "gr_z_diff", "dgr_diff")


def test_v0_1_diversity_knobs_are_configurable(tmp_path: Path) -> None:
    path = tmp_path / "config.yml"
    path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "bounded_output": True,
                    "mode_bias_init": True,
                    "mode_bias_span_bins": 12.0,
                },
                "loss": {
                    "alpha_cls": 0.05,
                    "cls_warmup_epochs": 2,
                    "alpha_cls_warmup_value": 0.0,
                    "entropy_lambda": 0.01,
                    "entropy_warmup_epochs": 3,
                    "entropy_final_lambda": 0.002,
                    "diversity_lambda": 0.02,
                    "diversity_margin_bins": 3.0,
                },
            }
        ),
        encoding="utf-8",
    )

    cfg = load_config(path)

    assert cfg.model.bounded_output is True
    assert cfg.model.mode_bias_init is True
    assert cfg.model.mode_bias_span_bins == pytest.approx(12.0)
    assert cfg.loss.alpha_cls == pytest.approx(0.05)
    assert cfg.loss.cls_warmup_epochs == 2
    assert cfg.loss.alpha_cls_warmup_value == pytest.approx(0.0)
    assert cfg.loss.entropy_lambda == pytest.approx(0.01)
    assert cfg.loss.entropy_warmup_epochs == 3
    assert cfg.loss.entropy_final_lambda == pytest.approx(0.002)
    assert cfg.loss.diversity_lambda == pytest.approx(0.02)
    assert cfg.loss.diversity_margin_bins == pytest.approx(3.0)


def test_mtp_v0_1_diverse_config_loads() -> None:
    cfg = load_config(Path("configs/mtp_v0_1_diverse.yml"))

    assert cfg.run.name == "mtp_v0_1_diverse"
    assert cfg.model.bounded_output is True
    assert cfg.model.mode_bias_init is True
    assert cfg.loss.cls_warmup_epochs == 2
    assert cfg.loss.entropy_lambda > 0.0
    assert cfg.loss.diversity_lambda > 0.0


def test_zero_k_wells_fails(tmp_path: Path) -> None:
    with pytest.raises(
        ValueError, match="data.k_wells must be -1 or a positive integer"
    ):
        load_config(write_config(tmp_path, 0))
