import json
from pathlib import Path

import pandas as pd

from mtpnet.eval import evaluate_run
from mtpnet.train import write_geometry_report


def test_evaluate_run_writes_geometry_report_from_metrics_and_predictions(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    metrics = {
        "num_train_wells": 2,
        "num_valid_wells": 1,
        "train": {"num_windows": 4},
        "valid": {
            "num_windows": 1,
            "target_in_crop_rate": 1.0,
            "target_at_crop_edge_frac": 0.0,
            "top1_rmse_bins": 1.0,
            "weighted_mean_rmse_bins": 1.1,
            "oracle_top3_rmse_bins": 0.9,
            "oracle_topk_rmse_bins": 0.8,
            "top1_rmse_ft": 10.0,
            "weighted_mean_rmse_ft": 11.0,
            "oracle_top3_rmse_ft": 9.0,
            "oracle_topk_rmse_ft": 8.0,
            "classification_accuracy_best_mode": 0.5,
            "mode_entropy_mean": 0.7,
            "mode_usage_histogram": {"0": 1},
            "pred_bin_oob_frac": 0.2,
            "raw_path_oob_frac_before_bound": 0.3,
            "bounded_output": True,
            "top1_pred_bin_oob_frac": 0.1,
            "weighted_pred_bin_oob_frac": 0.0,
            "pred_bin_min": -1.0,
            "pred_bin_max": 65.0,
        },
        "sanity": {
            "shuffled_gr": {"oracle_topk_rmse_ft": 12.0},
            "no_history": {"oracle_topk_rmse_ft": 13.0},
        },
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    pd.DataFrame({"top1_rmse_ft": [10.0], "oracle_rmse_ft": [8.0]}).to_parquet(
        run_dir / "window_predictions.parquet", index=False
    )

    loaded = evaluate_run(run_dir)

    report = (run_dir / "geometry_report.md").read_text(encoding="utf-8")
    assert loaded["valid"]["oracle_topk_rmse_ft"] == 8.0
    assert "MTP_V0_GEOMETRY_REPORT" in report
    assert "pred_bin_oob_frac: 0.2" in report
    assert "raw_path_oob_frac_before_bound: 0.3" in report
    assert "bounded_output: True" in report
    assert "parquet rows: 1" in report


def test_geometry_report_uses_v0_1_diversity_title(tmp_path: Path) -> None:
    metrics = {
        "run_name": "mtp_v0_1_diverse",
        "valid": {"num_windows": 0},
        "train": {"num_windows": 0},
    }

    write_geometry_report(metrics, tmp_path)

    report = (tmp_path / "geometry_report.md").read_text(encoding="utf-8")
    assert report.startswith("MTP_V0_1_DIVERSITY_REPORT")


def test_geometry_report_uses_v0_2_mixed_title_and_validation_sections(
    tmp_path: Path,
) -> None:
    metrics = {
        "run_name": "mtp_v0_2_mixed",
        "train": {"num_windows": 3},
        "valid": {"num_windows": 2},
        "valid_first_chunk_known_tail": {
            "num_windows": 1,
            "top1_rmse_ft": 4.0,
            "weighted_mean_rmse_ft": 3.5,
            "oracle_top3_rmse_ft": 2.5,
            "oracle_topk_rmse_ft": 2.0,
            "mode_entropy_mean": 0.4,
            "mode_usage_histogram": {"0": 1},
        },
        "valid_base_center_all_hidden": {
            "num_windows": 2,
            "top1_rmse_ft": 5.0,
            "weighted_mean_rmse_ft": 4.5,
            "oracle_top3_rmse_ft": 3.5,
            "oracle_topk_rmse_ft": 3.0,
            "mode_entropy_mean": 0.5,
            "mode_usage_histogram": {"1": 2},
        },
    }

    write_geometry_report(metrics, tmp_path)

    report = (tmp_path / "geometry_report.md").read_text(encoding="utf-8")
    assert report.startswith("MTP_V0_2_MIXED_REPORT")
    assert "valid_first_chunk_known_tail:" in report
    assert "valid_base_center_all_hidden:" in report
