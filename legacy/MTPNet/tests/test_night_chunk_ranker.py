from pathlib import Path

import numpy as np
import pandas as pd

from mtpnet.night_chunk_ranker import NightChunkRankerConfig, run_chunk_ranker


def test_run_chunk_ranker_trains_grouped_ranker_and_writes_predictions(tmp_path: Path) -> None:
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
                    "p0": true if well % 2 == 0 else true + 4.0,
                    "p1": true + 4.0 if well % 2 == 0 else true,
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
                        "chunk_mse": 0.0 if good else 16.0,
                        "target_log_cost": 0.0 if good else np.log1p(16.0),
                        "feat_parity": float(well % 2),
                        "feat_candidate_id": float(cand_id),
                    }
                )
    table_path = tmp_path / "table.parquet"
    pd.DataFrame(table_rows).to_parquet(table_path, index=False)
    bank_path = tmp_path / "bank.parquet"
    pd.DataFrame(bank_rows).to_parquet(bank_path, index=False)

    metrics = run_chunk_ranker(
        NightChunkRankerConfig(
            chunk_table_path=table_path,
            path_bank_path=bank_path,
            output_dir=tmp_path / "night",
            mission_path=tmp_path / "missing.md",
            n_folds=2,
            iterations=5,
        )
    )

    assert metrics["candidate_count"] == 2
    assert np.isfinite(metrics["ranker_row_rmse"])
    assert (tmp_path / "night" / "chunk_ranker_oof_predictions.parquet").exists()
    assert (tmp_path / "night" / "chunk_ranker_metrics.json").exists()
