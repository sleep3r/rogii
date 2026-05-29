from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _tiny_dtvt_frame(n_wells: int = 6, well_len: int = 48) -> pd.DataFrame:
    """Synthetic horizontal-well frame with closed-form ``TVT = -Z + C``.

    Wells share the structural property the model exploits: between control
    points the formation is flat (``dC = 0``), and at sparse "dips" ``C``
    makes small step jumps.
    """
    rng = np.random.default_rng(13)
    records = []
    for well_idx in range(n_wells):
        well_id = f"w{well_idx:02d}"
        z = np.linspace(-100.0, -200.0, well_len) + rng.normal(scale=0.3, size=well_len)
        c = np.zeros(well_len)
        c[0] = 50.0 + 5 * well_idx
        # Three plateaus with two step jumps
        for i in range(1, well_len):
            c[i] = c[i - 1]
            if i in {well_len // 3, 2 * well_len // 3}:
                c[i] += rng.uniform(-2.0, 2.0)
            c[i] += rng.normal(scale=0.005)
        tvt = -z + c
        hidden_start = int(0.7 * well_len)
        for row_idx in range(well_len):
            tvt_in = tvt[row_idx] if row_idx < hidden_start else np.nan
            records.append(
                {
                    "id": f"{well_id}_{row_idx}",
                    "well_id": well_id,
                    "row_idx": row_idx,
                    "MD": 1000.0 + row_idx * 10.0,
                    "X": float(row_idx),
                    "Y": float(well_idx) * 50.0,
                    "Z": float(z[row_idx]),
                    "GR": 80.0 + 10.0 * np.sin(row_idx / 4.0),
                    "TVT": float(tvt[row_idx]),
                    "TVT_input": float(tvt_in) if np.isfinite(tvt_in) else np.nan,
                    "ANCC": float(c[row_idx]),
                }
            )
    return pd.DataFrame(records)


def test_dtvt_state_features_are_schema_safe() -> None:
    from mtpnet.dtvt_state_model import PER_ROW_FEATURES
    from mtpnet.schema_safe import FORBIDDEN_INFERENCE_COLUMNS, assert_schema_safe_columns

    assert FORBIDDEN_INFERENCE_COLUMNS.isdisjoint(set(PER_ROW_FEATURES))
    # raises if any forbidden column slipped in
    assert_schema_safe_columns(list(PER_ROW_FEATURES), context="test")


def test_dtvt_state_cumsum_is_consistent() -> None:
    import torch

    from mtpnet.dtvt_state_model import _cumsum_tvt

    n = 10
    anchor = 4
    anchor_tvt = 100.0
    r_pred = torch.zeros(n)
    dz = torch.tensor([0.0, 0.1, -0.1, 0.0, 0.2, -0.3, 0.0, 0.4, -0.2, 0.1])
    tvt_pred = _cumsum_tvt(r_pred, dz, anchor, anchor_tvt)
    # With r=0, the recurrence is tvt[i] = tvt[i-1] - dz[i], so tvt should
    # equal anchor_tvt - cumulative dz past anchor.
    expected_post = anchor_tvt - torch.cumsum(dz[anchor + 1 :], 0)
    assert torch.allclose(tvt_pred[anchor + 1 :], expected_post, atol=1e-5)


def test_dtvt_state_model_smoke_writes_oof_artifact(tmp_path: Path) -> None:
    from mtpnet.dtvt_state_model import (
        DTVTStateModelConfig,
        run_dtvt_state_model_from_frame,
    )

    frame = _tiny_dtvt_frame(n_wells=6, well_len=48)
    metrics = run_dtvt_state_model_from_frame(
        frame,
        config=DTVTStateModelConfig(
            output_dir=tmp_path,
            n_folds=2,
            epochs=3,
            hidden_dim=16,
            dropout=0.0,
            learning_rate=5e-3,
            weight_decay=0.0,
            alpha_local=1.0,
            beta_global=1e-3,
            rows_per_step=8,
            top_state_path=None,
            device="cpu",
        ),
        top_state_lookup={},
    )

    assert metrics["candidate"] == "dtvt_state_model_v0"
    assert metrics["wells"] == 6
    assert np.isfinite(metrics["hidden_pooled_rmse"])
    preds = pd.read_parquet(tmp_path / "dtvt_state_oof_predictions.parquet")
    assert {"id", "well_id", "row_idx", "pred_tvt"}.issubset(preds.columns)
    assert preds["pred_tvt"].notna().all()
    # Predictions must cover every well's hidden range
    assert preds["well_id"].nunique() == 6
    assert (tmp_path / "dtvt_state_metrics.json").exists()
    assert (tmp_path / "dtvt_state_report.md").exists()


def test_dtvt_state_predictions_feed_candidate_bank() -> None:
    import tempfile

    from mtpnet.candidate_bank import build_candidate_bank_from_frames
    from mtpnet.dtvt_state_model import (
        DTVTStateModelConfig,
        run_dtvt_state_model_from_frame,
    )

    frame = _tiny_dtvt_frame(n_wells=6, well_len=48)
    with tempfile.TemporaryDirectory() as td:
        run_dtvt_state_model_from_frame(
            frame,
            config=DTVTStateModelConfig(
                output_dir=Path(td),
                n_folds=2,
                epochs=2,
                hidden_dim=8,
                dropout=0.0,
                learning_rate=5e-3,
                rows_per_step=8,
                top_state_path=None,
                device="cpu",
            ),
            top_state_lookup={},
        )
        preds = pd.read_parquet(Path(td) / "dtvt_state_oof_predictions.parquet")

    # The same parquet schema as residual_stack / k_segment_offset, so it
    # can ride through ``k_offset_predictions`` slot. We test the simpler
    # property: bank accepts it as residual_predictions too.
    bank = build_candidate_bank_from_frames(frame, residual_predictions=preds)
    assert "residual_stack_v0" in set(bank["candidate"].astype(str))
