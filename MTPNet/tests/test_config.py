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


def test_validation_explicit_fold_wells_are_configurable(tmp_path: Path) -> None:
    path = tmp_path / "config.yml"
    path.write_text(
        yaml.safe_dump(
            {
                "validation": {
                    "train_wells": ["a", "b"],
                    "valid_wells": ["c"],
                }
            }
        ),
        encoding="utf-8",
    )

    cfg = load_config(path)

    assert cfg.validation.train_wells == ("a", "b")
    assert cfg.validation.valid_wells == ("c",)


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


def test_mtp_v0_2_mixed_config_loads() -> None:
    cfg = load_config(Path("configs/mtp_v0_2_mixed.yml"))

    assert cfg.run.name == "mtp_v0_2_mixed"
    assert cfg.window.train_sample_mix == {
        "teacher_forcing_hidden": 0.5,
        "known_tail_start": 0.25,
        "base_center_hidden": 0.25,
    }
    assert cfg.window.valid_sample_types == (
        "known_tail_start",
        "base_center_hidden",
    )


def test_v1_prior_and_soft_probability_config_loads(tmp_path: Path) -> None:
    path = tmp_path / "config.yml"
    path.write_text(
        yaml.safe_dump(
            {
                "priors": {
                    "enabled": True,
                    "base_path": "../old/artifacts/oof_baseline/schema10_oof.parquet",
                    "b2_path": "../old/artifacts/formation_b2_danger_guard_a2_full_schema10/guarded_predictions.parquet",
                    "a_path": "../old/artifacts/formation_plane_knn/oof_candidates.parquet",
                    "base_column": "schema10_oof_pp",
                    "b2_column": "b2_guarded_submit",
                    "a_p50_column": "formation_sample_median",
                    "a_p10_column": "formation_sample_p10",
                    "a_p90_column": "formation_sample_p90",
                },
                "window": {
                    "vertical_bins": 96,
                    "vertical_radius_ft": 240.0,
                    "channels": ["base_sdf", "b2_sdf", "a_density"],
                },
                "loss": {
                    "alpha_cls": 0.0,
                    "soft_prob_alpha": 0.2,
                    "soft_prob_tau_bins": 2.0,
                },
            }
        ),
        encoding="utf-8",
    )

    cfg = load_config(path)

    assert cfg.priors.enabled is True
    assert cfg.priors.base_column == "schema10_oof_pp"
    assert cfg.priors.b2_column == "b2_guarded_submit"
    assert cfg.window.vertical_bins == 96
    assert cfg.window.vertical_radius_ft == pytest.approx(240.0)
    assert cfg.loss.soft_prob_alpha == pytest.approx(0.2)
    assert cfg.loss.soft_prob_tau_bins == pytest.approx(2.0)


def test_train_time_selection_loss_knobs_are_configurable(tmp_path: Path) -> None:
    path = tmp_path / "config.yml"
    path.write_text(
        yaml.safe_dump(
            {
                "loss": {
                    "top3_margin_alpha": 0.05,
                    "top3_margin": 0.1,
                    "continuation_alpha": 0.02,
                    "continuation_tau_bins": 3.0,
                }
            }
        ),
        encoding="utf-8",
    )

    cfg = load_config(path)

    assert cfg.loss.top3_margin_alpha == pytest.approx(0.05)
    assert cfg.loss.top3_margin == pytest.approx(0.1)
    assert cfg.loss.continuation_alpha == pytest.approx(0.02)
    assert cfg.loss.continuation_tau_bins == pytest.approx(3.0)


def test_mtp_v1_prior_conditioned_config_loads() -> None:
    cfg = load_config(Path("configs/mtp_v1_prior_conditioned.yml"))

    assert cfg.run.name == "mtp_v1_prior_conditioned"
    assert cfg.priors.enabled is True
    assert cfg.window.vertical_bins == 96
    assert cfg.window.vertical_radius_ft == pytest.approx(240.0)
    assert "base_sdf" in cfg.window.channels
    assert "b2_sdf" in cfg.window.channels
    assert "a_density" in cfg.window.channels
    assert cfg.loss.alpha_cls == pytest.approx(0.0)
    assert cfg.loss.soft_prob_alpha == pytest.approx(0.2)


def test_v2_anchor_dropout_config_loads(tmp_path: Path) -> None:
    path = tmp_path / "config.yml"
    path.write_text(
        yaml.safe_dump(
            {
                "augmentation": {
                    "enabled": True,
                    "drop_anchor_sdf_prob": 0.15,
                    "drop_b2_sdf_prob": 0.30,
                    "drop_a_density_prob": 0.20,
                    "drop_all_priors_prob": 0.10,
                    "anchor_jitter_ft": [10.0, 20.0, 30.0],
                    "anchor_swap_prob": 0.25,
                    "anchor_swap": [
                        "schema10",
                        "B2",
                        "A_p50",
                        "A_weighted_mean",
                        "noisy_anchor",
                    ],
                    "wrong_anchor_prob": 0.10,
                    "wrong_anchor_shift_ft": [40.0, 80.0],
                    "batch_mix": {
                        "normal": 0.55,
                        "no_priors": 0.20,
                        "jittered_priors": 0.15,
                        "wrong_priors": 0.10,
                    },
                },
                "window": {
                    "channels": ["anchor_sdf", "b2_sdf", "a_density", "anchor_offset_value"]
                },
                "loss": {
                    "contrastive_alpha": 0.02,
                    "contrastive_margin_bins": 0.3,
                },
            }
        ),
        encoding="utf-8",
    )

    cfg = load_config(path)

    assert cfg.augmentation.enabled is True
    assert cfg.augmentation.drop_anchor_sdf_prob == pytest.approx(0.15)
    assert cfg.augmentation.drop_b2_sdf_prob == pytest.approx(0.30)
    assert cfg.augmentation.drop_a_density_prob == pytest.approx(0.20)
    assert cfg.augmentation.drop_all_priors_prob == pytest.approx(0.10)
    assert cfg.augmentation.anchor_jitter_ft == (10.0, 20.0, 30.0)
    assert cfg.augmentation.anchor_swap_prob == pytest.approx(0.25)
    assert cfg.augmentation.anchor_swap == (
        "schema10",
        "B2",
        "A_p50",
        "A_weighted_mean",
        "noisy_anchor",
    )
    assert cfg.augmentation.wrong_anchor_prob == pytest.approx(0.10)
    assert cfg.augmentation.wrong_anchor_shift_ft == (40.0, 80.0)
    assert cfg.augmentation.batch_mix == {
        "normal": 0.55,
        "no_priors": 0.20,
        "jittered_priors": 0.15,
        "wrong_priors": 0.10,
    }
    assert cfg.loss.contrastive_alpha == pytest.approx(0.02)
    assert cfg.loss.contrastive_margin_bins == pytest.approx(0.3)
    assert "anchor_sdf" in cfg.window.channels


def test_mtp_v2_anchor_dropout_config_loads() -> None:
    cfg = load_config(Path("configs/mtp_v2_anchor_dropout.yml"))

    assert cfg.run.name == "mtp_v2_anchor_dropout"
    assert cfg.augmentation.enabled is True
    assert cfg.augmentation.drop_all_priors_prob > 0.0
    assert "anchor_sdf" in cfg.window.channels


def test_mtp_v2_train_time_selection_config_loads() -> None:
    cfg = load_config(Path("configs/mtp_v2_train_time_selection.yml"))

    assert cfg.run.name == "mtp_v2_train_time_selection"
    assert cfg.loss.soft_prob_alpha == pytest.approx(0.2)
    assert cfg.loss.top3_margin_alpha == pytest.approx(0.05)
    assert cfg.loss.continuation_alpha == pytest.approx(0.02)
    assert cfg.augmentation.enabled is True
    assert "anchor_sdf" in cfg.window.channels


def test_mtp_v3_gr_forced_config_loads() -> None:
    cfg = load_config(Path("configs/mtp_v3_gr_forced.yml"))

    assert cfg.run.name == "mtp_v3_gr_forced"
    assert cfg.augmentation.drop_all_priors_prob == pytest.approx(0.25)
    assert cfg.augmentation.wrong_anchor_prob == pytest.approx(0.10)
    assert cfg.augmentation.wrong_anchor_shift_ft == (40.0, 80.0)
    assert cfg.augmentation.batch_mix["no_priors"] == pytest.approx(0.20)
    assert cfg.loss.contrastive_alpha == pytest.approx(0.02)
    assert cfg.loss.top3_margin_alpha == pytest.approx(0.05)


def test_zero_k_wells_fails(tmp_path: Path) -> None:
    with pytest.raises(
        ValueError, match="data.k_wells must be -1 or a positive integer"
    ):
        load_config(write_config(tmp_path, 0))
