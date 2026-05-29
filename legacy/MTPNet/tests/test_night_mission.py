from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _write_tiny_train(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for well_idx in range(2):
        well_id = f"w{well_idx}"
        rows = []
        for row_idx in range(6):
            tvt = 100.0 + 10.0 * well_idx + row_idx
            rows.append(
                {
                    "MD": 1000.0 + row_idx,
                    "X": float(row_idx),
                    "Y": float(well_idx),
                    "Z": -float(row_idx),
                    "GR": 80.0 + row_idx,
                    "TVT": tvt,
                    "TVT_input": tvt if row_idx < 3 else np.nan,
                }
            )
        pd.DataFrame(rows).to_csv(root / f"{well_id}__horizontal_well.csv", index=False)


def test_load_hidden_truth_builds_ids_and_buckets(tmp_path: Path) -> None:
    from mtpnet.night_mission import load_hidden_truth

    data_dir = tmp_path / "train"
    _write_tiny_train(data_dir)

    truth = load_hidden_truth(data_dir)

    assert len(truth) == 6
    assert {"id", "well_id", "row_idx", "TVT", "hidden_len", "hidden_len_bucket"}.issubset(truth.columns)
    assert truth["id"].tolist()[:2] == ["w0_3", "w0_4"]
    assert set(truth["hidden_len_bucket"]) == {"short"}


def test_normalize_prediction_artifact_joins_truth(tmp_path: Path) -> None:
    from mtpnet.night_mission import load_hidden_truth, normalize_prediction_artifact

    data_dir = tmp_path / "train"
    _write_tiny_train(data_dir)
    truth = load_hidden_truth(data_dir)
    pred_path = tmp_path / "artifact" / "row_predictions.parquet"
    pred_path.parent.mkdir()
    pd.DataFrame(
        {
            "well_id": ["w0", "w0", "w1"],
            "row_idx": [3, 4, 5],
            "pred_tvt": [103.5, 104.5, 115.5],
            "candidate": ["model_a", "model_a", "model_a"],
        }
    ).to_parquet(pred_path, index=False)

    normalized = normalize_prediction_artifact(pred_path, truth=truth, artifact_root=tmp_path)

    assert len(normalized) == 3
    assert normalized["true_tvt"].notna().all()
    assert normalized["candidate"].unique().tolist() == ["model_a"]
    assert normalized["experiment"].str.contains("artifact").all()


def test_score_predictions_reports_pooled_and_worst(tmp_path: Path) -> None:
    from mtpnet.night_mission import score_predictions

    frame = pd.DataFrame(
        {
            "experiment": ["e"] * 4,
            "candidate": ["a"] * 4,
            "well_id": ["w0", "w0", "w1", "w1"],
            "row_idx": [3, 4, 3, 4],
            "pred_tvt": [1.0, 2.0, 10.0, 10.0],
            "true_tvt": [1.0, 4.0, 8.0, 14.0],
            "hidden_len_bucket": ["short", "short", "medium", "medium"],
        }
    )

    scoreboard, worst = score_predictions(frame, top_n=1)

    assert len(scoreboard) == 1
    assert scoreboard.iloc[0]["rows"] == 4
    assert scoreboard.iloc[0]["pooled_rmse"] > 0
    assert worst.iloc[0]["well_id"] == "w1"


def test_run_path_bank_writes_wide_bank_and_oracle(tmp_path: Path) -> None:
    from mtpnet.night_mission import NightPathBankConfig, run_path_bank

    data_dir = tmp_path / "train"
    _write_tiny_train(data_dir)
    output_dir = tmp_path / "night"
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True)
    rows = []
    for well_idx in range(2):
        well_id = f"w{well_idx}"
        for row_idx in range(3, 6):
            true = 100.0 + 10.0 * well_idx + row_idx
            rows.append(
                {
                    "experiment": "exp",
                    "candidate": "low_bias",
                    "artifact_path": "a.parquet",
                    "id": f"{well_id}_{row_idx}",
                    "well_id": well_id,
                    "row_idx": row_idx,
                    "pred_tvt": true - (0.1 if row_idx % 2 else 5.0),
                    "true_tvt": true,
                    "hidden_len": 3,
                    "hidden_len_bucket": "short",
                    "gr_valid_frac": 1.0,
                }
            )
            rows.append(
                {
                    "experiment": "exp",
                    "candidate": "high_bias",
                    "artifact_path": "b.parquet",
                    "id": f"{well_id}_{row_idx}",
                    "well_id": well_id,
                    "row_idx": row_idx,
                    "pred_tvt": true + (5.0 if row_idx % 2 else 0.2),
                    "true_tvt": true,
                    "hidden_len": 3,
                    "hidden_len_bucket": "short",
                    "gr_valid_frac": 1.0,
                }
            )
    pd.DataFrame(rows).to_parquet(pred_dir / "norm.parquet", index=False)
    pd.DataFrame(
        [
            {"experiment": "exp", "candidate": "low_bias", "rows": 6, "wells": 2, "pooled_rmse": 3.0},
            {"experiment": "exp", "candidate": "high_bias", "rows": 6, "wells": 2, "pooled_rmse": 3.0},
        ]
    ).to_csv(output_dir / "oof_scoreboard.csv", index=False)

    metrics = run_path_bank(
        NightPathBankConfig(
            data_dir=data_dir,
            output_dir=output_dir,
            min_rows_fraction=1.0,
            max_candidates=2,
        )
    )

    assert metrics["selected_candidates"] == 2
    assert metrics["oracle_rows"] == 6
    assert metrics["oracle_pooled_rmse"] < 1.0
    bank = pd.read_parquet(output_dir / "path_bank_oof.parquet")
    assert {"p000__exp__low_bias", "p001__exp__high_bias"}.issubset(bank.columns)
    assert (output_dir / "best_of_bank_report.md").exists()


def test_run_path_bank_excludes_target_oracle_candidates(tmp_path: Path) -> None:
    from mtpnet.night_mission import NightPathBankConfig, run_path_bank

    data_dir = tmp_path / "train"
    _write_tiny_train(data_dir)
    output_dir = tmp_path / "night"
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True)
    rows = []
    for row_idx in range(3, 6):
        true = 100.0 + row_idx
        rows.append(
            {
                "experiment": "exp",
                "candidate": "grid_oracle",
                "artifact_path": "oracle.parquet",
                "id": f"w0_{row_idx}",
                "well_id": "w0",
                "row_idx": row_idx,
                "pred_tvt": true,
                "true_tvt": true,
                "hidden_len": 3,
                "hidden_len_bucket": "short",
                "gr_valid_frac": 1.0,
            }
        )
        rows.append(
            {
                "experiment": "exp",
                "candidate": "deployable_path",
                "artifact_path": "deployable.parquet",
                "id": f"w0_{row_idx}",
                "well_id": "w0",
                "row_idx": row_idx,
                "pred_tvt": true + 1.0,
                "true_tvt": true,
                "hidden_len": 3,
                "hidden_len_bucket": "short",
                "gr_valid_frac": 1.0,
            }
        )
    pd.DataFrame(rows).to_parquet(pred_dir / "norm.parquet", index=False)
    pd.DataFrame(
        [
            {"experiment": "exp", "candidate": "grid_oracle", "rows": 3, "wells": 1, "pooled_rmse": 0.0},
            {"experiment": "exp", "candidate": "deployable_path", "rows": 3, "wells": 1, "pooled_rmse": 1.0},
        ]
    ).to_csv(output_dir / "oof_scoreboard.csv", index=False)

    metrics = run_path_bank(
        NightPathBankConfig(
            data_dir=data_dir,
            output_dir=output_dir,
            min_rows_fraction=0.5,
            max_candidates=2,
        )
    )

    assert metrics["selected_candidates"] == 1
    assert metrics["oracle_pooled_rmse"] == 1.0
    manifest = pd.read_csv(output_dir / "path_bank_manifest.csv")
    assert manifest["candidate"].tolist() == ["deployable_path"]
