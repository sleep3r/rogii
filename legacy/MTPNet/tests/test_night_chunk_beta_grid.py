from pathlib import Path

import numpy as np
import pandas as pd

from mtpnet.night_chunk_beta_grid import NightChunkBetaGridConfig, make_value_target, run_chunk_beta_grid


def test_make_value_target_uses_rows_left_weighted_endpoint_error() -> None:
    table = pd.DataFrame(
        {
            "chunk_mse": [4.0, 4.0],
            "endpoint_error": [2.0, 2.0],
            "feat_rows_left_log": [np.log1p(0.0), np.log1p(9.0)],
        }
    )
    target = make_value_target(table, beta=0.5)
    assert np.isclose(target.iloc[0], np.log1p(4.0))
    assert np.isclose(target.iloc[1], np.log1p(4.0 + 0.5 * 9.0 * 4.0))


def test_run_chunk_beta_grid_writes_beta_metrics(tmp_path: Path) -> None:
    table_rows = []
    bank_rows = []
    for well in range(4):
        for row in range(4):
            true = float(row)
            bank_rows.append(
                {
                    "well_id": f"w{well}",
                    "row_idx": row,
                    "true_tvt": true,
                    "p0": true if well % 2 == 0 else true + 3.0,
                    "p1": true + 3.0 if well % 2 == 0 else true,
                }
            )
        for chunk in range(2):
            for cand_id, col in enumerate(["p0", "p1"]):
                good = (well % 2 == 0 and cand_id == 0) or (well % 2 == 1 and cand_id == 1)
                table_rows.append(
                    {
                        "well_id": f"w{well}",
                        "chunk_id": chunk,
                        "candidate_id": cand_id,
                        "candidate_col": col,
                        "chunk_mse": 0.0 if good else 9.0,
                        "endpoint_error": 0.0 if good else 3.0,
                        "feat_rows_left_log": np.log1p(2.0),
                        "feat_candidate_id": float(cand_id),
                    }
                )
    table_path = tmp_path / "table.parquet"
    pd.DataFrame(table_rows).to_parquet(table_path, index=False)
    bank_path = tmp_path / "bank.parquet"
    pd.DataFrame(bank_rows).to_parquet(bank_path, index=False)

    metrics = run_chunk_beta_grid(
        NightChunkBetaGridConfig(
            chunk_table_path=table_path,
            path_bank_path=bank_path,
            output_dir=tmp_path / "night",
            mission_path=tmp_path / "missing.md",
            chunk_size=2,
            betas=(0.0, 0.5),
            n_folds=2,
            iterations=5,
        )
    )

    assert metrics["betas"] == [0.0, 0.5]
    assert (tmp_path / "night" / "chunk_beta_grid.csv").exists()
    assert (tmp_path / "night" / "chunk_beta_best_oof_predictions.parquet").exists()
