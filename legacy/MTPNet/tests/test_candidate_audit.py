from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _tiny_baseline_dataset() -> pd.DataFrame:
    """Two candidates ({b2, alt}) across 4 wells × 3 chunks each."""
    rows = []
    rng = np.random.default_rng(3)
    for well_idx in range(4):
        wid = f"w{well_idx:02d}"
        for chunk_id in range(3):
            for cand, mse in [
                ("b2", float(10.0 + rng.uniform(0, 5))),
                ("alt", float(8.0 + rng.uniform(0, 5))),
            ]:
                rows.append(
                    {
                        "well_id": wid,
                        "run_id": 0,
                        "chunk_id": chunk_id,
                        "group_id": f"{wid}:{chunk_id}",
                        "candidate": cand,
                        "row_count": 8,
                        "target_mse": mse,
                        "target_rmse": float(np.sqrt(mse)),
                        "target_b2_mse": float(10.0),
                        "target_gain_vs_b2_mse": 0.0,
                    }
                )
    return pd.DataFrame(rows)


def _tiny_well_csvs(tmp_path: Path) -> Path:
    """Per-well CSV with TVT/TVT_input/Z matching the baseline dataset."""
    data_dir = tmp_path / "data" / "train"
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    for well_idx in range(4):
        wid = f"w{well_idx:02d}"
        # 4 known + 24 hidden -> 24/8 = 3 chunks of 8 rows each
        n = 4 + 24
        rows = []
        for i in range(n):
            tvt = 100.0 + 0.5 * i + rng.normal(scale=0.05)
            tvt_in = tvt if i < 4 else np.nan
            rows.append(
                {
                    "row_idx": i,
                    "MD": 1000.0 + i * 10.0,
                    "X": float(i),
                    "Y": 0.0,
                    "Z": -float(i),
                    "GR": 80.0,
                    "TVT": tvt,
                    "TVT_input": tvt_in,
                }
            )
        pd.DataFrame(rows).to_csv(data_dir / f"{wid}__horizontal_well.csv", index=False)
    return data_dir


def test_candidate_audit_runs_and_compares(tmp_path: Path) -> None:
    from mtpnet.candidate_audit import _CandidateInput, audit_candidates

    baseline_path = tmp_path / "chunk_policy_dataset.parquet"
    _tiny_baseline_dataset().to_parquet(baseline_path, index=False)
    data_dir = _tiny_well_csvs(tmp_path)

    # Build a candidate parquet: predictions that approximate truth → should
    # beat b2 strongly because b2_mse is hand-set to ~10.
    rows = []
    for well_idx in range(4):
        wid = f"w{well_idx:02d}"
        well = pd.read_csv(data_dir / f"{wid}__horizontal_well.csv")
        hidden = well[well["TVT_input"].isna()]
        for _, r in hidden.iterrows():
            rows.append(
                {
                    "id": f"{wid}_{int(r['row_idx'])}",
                    "well_id": wid,
                    "row_idx": int(r["row_idx"]),
                    "pred_tvt": float(r["TVT"]) + 0.1,  # tiny offset
                }
            )
    cand_path = tmp_path / "k_offset_oof_predictions.parquet"
    pd.DataFrame(rows).to_parquet(cand_path, index=False)

    out_metrics = tmp_path / "audit.json"
    metrics = audit_candidates(
        baseline=baseline_path,
        data_dir=data_dir,
        chunk_size=8,
        candidates=[_CandidateInput("k_segment_offset_v0", cand_path)],
        output=out_metrics,
    )

    assert "k_segment_offset_v0" in metrics["candidates_labels"]
    assert metrics["n_chunks"] == 4 * 3
    assert metrics["n_wells"] == 4
    assert out_metrics.exists()
    # An accurate candidate should beat b2 most of the time
    assert "+ k_segment_offset_v0" in metrics["report"]
