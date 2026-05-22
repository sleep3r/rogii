import json
from pathlib import Path

import pandas as pd

from mtpnet.eval import evaluate_run


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
    assert "parquet rows: 1" in report
