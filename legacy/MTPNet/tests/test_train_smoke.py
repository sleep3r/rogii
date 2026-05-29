from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml

from mtpnet import train as train_module
from mtpnet.train import _selection_score, prepare_sample_splits, train_from_config
from mtpnet.stitch import write_mode_windows


def make_well(root: Path, well_id: str, shift: float) -> None:
    root.mkdir(parents=True, exist_ok=True)
    n = 160
    tvt = [1000.0 + shift + i * 0.5 for i in range(n)]
    gr = [80.0 + ((i % 20) - 10) * 1.5 for i in range(n)]
    tvt_input = [value if i < 80 else None for i, value in enumerate(tvt)]
    pd.DataFrame(
        {
            "id": [f"{well_id}_{i}" for i in range(n)],
            "TVT": tvt,
            "TVT_input": tvt_input,
            "GR": gr,
        }
    ).to_csv(root / f"{well_id}__horizontal_well.csv", index=False)
    type_tvt = [970.0 + shift + i * 0.5 for i in range(260)]
    type_gr = [80.0 + ((i % 20) - 10) * 1.5 for i in range(260)]
    pd.DataFrame({"TVT": type_tvt, "GR": type_gr}).to_csv(
        root / f"{well_id}__typewell.csv", index=False
    )


def test_train_from_config_writes_metrics(tmp_path: Path) -> None:
    train_dir = tmp_path / "data" / "train"
    make_well(train_dir, "well_a", 0.0)
    make_well(train_dir, "well_b", 5.0)
    config = {
        "data": {"data_dir": str(tmp_path / "data"), "train_dir": str(train_dir), "k_wells": -1},
        "window": {
            "rows_per_step": 4,
            "history_steps": 4,
            "future_steps": 6,
            "vertical_bins": 32,
            "vertical_radius_ft": 60.0,
            "stride_steps": 4,
            "max_windows_per_well": 4,
            "channels": ["gr_diff", "gr_z_diff", "dgr_diff", "abs_gr_diff", "history_mask", "history_sdf", "finite_mask"],
            "train_history_mode": "teacher_forcing",
            "valid_history_mode": "known_tail_only",
            "train_center_source": "true_tvt",
            "valid_center_source": "tvt_input_tail",
        },
        "model": {"k_modes": 3, "conv_channels": [8, 16], "hidden_dims": [32], "dropout": 0.0},
        "train": {
            "batch_size": 4,
            "epochs": 2,
            "learning_rate": 0.001,
            "device": "cpu",
            "num_workers": 0,
            "seed": 42,
        },
        "validation": {"valid_fraction": 0.5, "seed": 42},
        "run": {"name": "unit", "output_dir": str(tmp_path / "artifacts" / "unit")},
    }
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    summary = train_from_config(config_path)
    assert summary["valid"]["num_windows"] > 0
    assert summary["best_epoch"] >= 1
    assert summary["best_valid_score"] == summary["valid"]["checkpoint_score"]
    assert summary["best_valid_oracle_topk_rmse_bins"] == summary["valid"]["oracle_topk_rmse_bins"]
    assert summary["sanity"]["shuffled_gr"]["oracle_topk_rmse_ft"] >= 0.0
    assert summary["sanity"]["no_history"]["oracle_topk_rmse_ft"] >= 0.0
    assert (tmp_path / "artifacts" / "unit" / "metrics.json").exists()
    assert (tmp_path / "artifacts" / "unit" / "checkpoints" / "best.pt").exists()
    report = tmp_path / "artifacts" / "unit" / "geometry_report.md"
    assert "MTP_V0_GEOMETRY_REPORT" in report.read_text(encoding="utf-8")


def test_prepare_sample_splits_builds_v0_2_mixed_sets(tmp_path: Path) -> None:
    train_dir = tmp_path / "data" / "train"
    make_well(train_dir, "well_a", 0.0)
    make_well(train_dir, "well_b", 5.0)
    config = {
        "data": {"data_dir": str(tmp_path / "data"), "train_dir": str(train_dir), "k_wells": -1},
        "window": {
            "rows_per_step": 4,
            "history_steps": 4,
            "future_steps": 6,
            "vertical_bins": 32,
            "vertical_radius_ft": 60.0,
            "stride_steps": 4,
            "max_windows_per_well": 4,
            "channels": ["gr_diff", "abs_gr_diff", "history_mask", "history_sdf", "finite_mask"],
            "train_sample_mix": {
                "teacher_forcing_hidden": 0.5,
                "known_tail_start": 0.25,
                "base_center_hidden": 0.25,
            },
            "valid_sample_types": ["known_tail_start", "base_center_hidden"],
        },
        "validation": {"valid_fraction": 0.5, "seed": 42},
    }
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    from mtpnet.config import load_config

    splits = prepare_sample_splits(load_config(config_path))

    assert set(splits.train_buckets) == {
        "teacher_forcing_hidden",
        "known_tail_start",
        "base_center_hidden",
    }
    assert set(splits.valid_sets) == {
        "valid_first_chunk_known_tail",
        "valid_base_center_all_hidden",
    }
    assert splits.primary_valid_name == "valid_base_center_all_hidden"
    assert len(splits.valid_sets["valid_base_center_all_hidden"]) > len(
        splits.valid_sets["valid_first_chunk_known_tail"]
    )


def test_tiny_synthetic_pretrain_writes_corr_artifacts(tmp_path: Path) -> None:
    train_dir = tmp_path / "data" / "train"
    make_well(train_dir, "well_a", 0.0)
    make_well(train_dir, "well_b", 5.0)
    run_dir = tmp_path / "artifacts" / "v4_tiny"
    config = {
        "data": {"data_dir": str(tmp_path / "data"), "train_dir": str(train_dir), "k_wells": -1},
        "synthetic": {
            "enabled": True,
            "windows_per_epoch": 6,
            "valid_windows": 3,
            "real_fraction": 0.0,
            "noise_std_range": [0.0, 0.0],
            "amplitude_scale_range": [1.0, 1.0],
            "baseline_shift_range": [0.0, 0.0],
            "dropout_max_frac": 0.0,
            "stretch_range": [1.0, 1.0],
            "bad_prior_prob": 0.0,
            "no_prior_prob": 0.0,
            "seed": 7,
        },
        "corr_head": {
            "enabled": True,
            "target_tau_bins": 1.5,
            "alpha_synth": 0.5,
            "alpha_real": 0.25,
        },
        "window": {
            "rows_per_step": 4,
            "history_steps": 4,
            "future_steps": 6,
            "vertical_bins": 32,
            "vertical_radius_ft": 60.0,
            "stride_steps": 4,
            "max_windows_per_well": 4,
            "channels": [
                "gr_diff",
                "gr_z_diff",
                "dgr_diff",
                "abs_gr_diff",
                "history_mask",
                "history_sdf",
                "finite_mask",
            ],
            "train_history_mode": "teacher_forcing",
            "valid_history_mode": "known_tail_only",
            "train_center_source": "true_tvt",
            "valid_center_source": "tvt_input_tail",
        },
        "model": {
            "k_modes": 3,
            "conv_channels": [8, 16],
            "hidden_dims": [32],
            "dropout": 0.0,
            "bounded_output": True,
        },
        "loss": {"alpha_cls": 0.0, "smooth_lambda": 0.0},
        "train": {
            "batch_size": 3,
            "epochs": 1,
            "learning_rate": 0.001,
            "device": "cpu",
            "num_workers": 0,
            "seed": 42,
            "selection_source": "synthetic",
        },
        "validation": {"valid_fraction": 0.5, "seed": 42},
        "run": {"name": "mtp_v4_synth_pretrain_tiny", "output_dir": str(run_dir)},
    }
    config_path = tmp_path / "v4_tiny.yml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    summary = train_from_config(config_path)
    mode_windows = write_mode_windows(run_dir)

    assert summary["primary_valid_name"] == "valid_synthetic"
    assert summary["valid"]["corr_target_top3_rate"] >= 0.0
    assert "no_corr_head" not in summary["sanity"]
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "window_predictions.parquet").exists()
    assert (run_dir / "checkpoints" / "best.pt").exists()
    assert "corr_scores" in mode_windows.columns


def test_synthetic_train_samples_regenerate_each_epoch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train_dir = tmp_path / "data" / "train"
    make_well(train_dir, "well_a", 0.0)
    make_well(train_dir, "well_b", 5.0)
    run_dir = tmp_path / "artifacts" / "v4_regen"
    config = {
        "data": {"data_dir": str(tmp_path / "data"), "train_dir": str(train_dir), "k_wells": -1},
        "synthetic": {
            "enabled": True,
            "windows_per_epoch": 4,
            "valid_windows": 2,
            "real_fraction": 0.0,
            "noise_std_range": [0.0, 0.0],
            "amplitude_scale_range": [1.0, 1.0],
            "baseline_shift_range": [0.0, 0.0],
            "dropout_max_frac": 0.0,
            "stretch_range": [1.0, 1.0],
            "path_families": ["real_residual"],
            "seed": 7,
        },
        "corr_head": {"enabled": True, "target_tau_bins": 1.5},
        "window": {
            "rows_per_step": 4,
            "history_steps": 4,
            "future_steps": 6,
            "vertical_bins": 32,
            "vertical_radius_ft": 60.0,
            "stride_steps": 4,
            "max_windows_per_well": 4,
            "channels": ["gr_diff", "gr_z_diff", "dgr_diff", "abs_gr_diff", "history_mask", "history_sdf", "finite_mask"],
        },
        "model": {"k_modes": 3, "conv_channels": [8, 16], "hidden_dims": [32], "dropout": 0.0},
        "loss": {"alpha_cls": 0.0, "smooth_lambda": 0.0},
        "train": {
            "batch_size": 2,
            "epochs": 2,
            "learning_rate": 0.001,
            "device": "cpu",
            "num_workers": 0,
            "seed": 42,
            "selection_source": "synthetic",
        },
        "validation": {"valid_fraction": 0.5, "seed": 42},
        "run": {"name": "mtp_v4_regen_tiny", "output_dir": str(run_dir)},
    }
    config_path = tmp_path / "v4_regen.yml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    seed_offsets: list[int] = []
    original = train_module.generate_synthetic_samples

    def recording_generate(*args, **kwargs):
        seed_offsets.append(int(kwargs.get("seed_offset", 0)))
        return original(*args, **kwargs)

    monkeypatch.setattr(train_module, "generate_synthetic_samples", recording_generate)

    train_module.train_from_config(config_path)

    train_offsets = [offset for offset in seed_offsets if 100_000 <= offset < 1_000_000]
    assert len(train_offsets) == 2
    assert len(set(train_offsets)) == 2
    assert train_offsets[1] - train_offsets[0] == config["synthetic"]["windows_per_epoch"]


def test_init_checkpoint_rejects_unexpected_incompatible_keys(tmp_path: Path) -> None:
    train_dir = tmp_path / "data" / "train"
    make_well(train_dir, "well_a", 0.0)
    make_well(train_dir, "well_b", 5.0)
    checkpoint = tmp_path / "bad.pt"
    torch.save({"model": {"not_a_model.weight": torch.ones(1)}}, checkpoint)
    config = {
        "data": {"data_dir": str(tmp_path / "data"), "train_dir": str(train_dir), "k_wells": -1},
        "window": {
            "rows_per_step": 4,
            "history_steps": 4,
            "future_steps": 6,
            "vertical_bins": 32,
            "vertical_radius_ft": 60.0,
            "stride_steps": 4,
            "max_windows_per_well": 4,
            "channels": ["gr_diff", "abs_gr_diff", "history_mask", "history_sdf", "finite_mask"],
        },
        "model": {"k_modes": 3, "conv_channels": [8, 16], "hidden_dims": [32], "dropout": 0.0},
        "train": {
            "batch_size": 4,
            "epochs": 1,
            "learning_rate": 0.001,
            "device": "cpu",
            "num_workers": 0,
            "seed": 42,
            "init_checkpoint": str(checkpoint),
        },
        "validation": {"valid_fraction": 0.5, "seed": 42},
        "run": {"name": "bad_init", "output_dir": str(tmp_path / "artifacts" / "bad_init")},
    }
    config_path = tmp_path / "bad_init.yml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="Incompatible init_checkpoint"):
        train_from_config(config_path)


def test_selection_score_includes_corr_metrics_for_v4_synthetic_stage() -> None:
    weak_corr = {
        "oracle_topk_rmse_bins": 1.0,
        "weighted_mean_rmse_bins": 1.0,
        "top1_rmse_bins": 1.0,
        "corr_nll": 3.0,
        "corr_target_top3_rate": 0.1,
    }
    strong_corr = {
        **weak_corr,
        "corr_nll": 0.5,
        "corr_target_top3_rate": 0.9,
    }

    assert _selection_score(strong_corr, selection_source="synthetic") < _selection_score(
        weak_corr,
        selection_source="synthetic",
    )


def test_selection_score_is_nan_safe_when_corr_nll_is_missing_or_nan() -> None:
    path_metrics = {
        "oracle_topk_rmse_bins": 2.0,
        "weighted_mean_rmse_bins": 3.0,
        "top1_rmse_bins": 4.0,
    }
    base_score = _selection_score(path_metrics)
    with_nan_corr = {**path_metrics, "corr_nll": float("nan"), "corr_target_top3_rate": float("nan")}
    with_valid_corr = {**path_metrics, "corr_nll": 0.5, "corr_target_top3_rate": 0.5}

    assert _selection_score(with_nan_corr) == pytest.approx(base_score)
    assert _selection_score(with_nan_corr, selection_source="synthetic") == pytest.approx(
        base_score
    )
    assert _selection_score(with_valid_corr) != pytest.approx(base_score)
