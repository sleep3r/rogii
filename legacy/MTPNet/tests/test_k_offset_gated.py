from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _tiny_frame(n_wells: int = 8, well_len: int = 64) -> pd.DataFrame:
    """Synthetic well frame with one contiguous hidden block at the end.

    Wells are designed so that ``TVT = -Z + C`` with piecewise-linear C, plus
    a known sign per well to give the gate classifier a learnable signal.
    """
    rng = np.random.default_rng(11)
    rows = []
    for well_idx in range(n_wells):
        well_id = f"w{well_idx:02d}"
        z = np.linspace(-100.0, -200.0, well_len) + rng.normal(scale=0.3, size=well_len)
        sign = 1.0 if well_idx % 2 == 0 else -1.0
        c = np.zeros(well_len)
        c[0] = 50.0 + 5 * well_idx
        for i in range(1, well_len):
            c[i] = c[i - 1] + sign * 0.03 + rng.normal(scale=0.005)
        tvt = -z + c
        hidden_start = int(0.6 * well_len)
        for row_idx in range(well_len):
            tvt_in = tvt[row_idx] if row_idx < hidden_start else np.nan
            rows.append(
                {
                    "id": f"{well_id}_{row_idx}",
                    "well_id": well_id,
                    "row_idx": row_idx,
                    "MD": 1000.0 + row_idx * 10.0,
                    "X": float(row_idx),
                    "Y": float(well_idx) * 50.0,
                    "Z": float(z[row_idx]),
                    "GR": 80.0 + 5.0 * np.sin(row_idx / 4.0),
                    "TVT": float(tvt[row_idx]),
                    "TVT_input": float(tvt_in) if np.isfinite(tvt_in) else np.nan,
                    "ANCC": float(c[row_idx]),
                }
            )
    return pd.DataFrame(rows)


def _synth_k_offset(frame: pd.DataFrame) -> pd.DataFrame:
    """Cheating k_offset predictions: tvt_true + small offset, only on hidden rows."""
    hidden = frame[frame["TVT_input"].isna()].copy()
    rng = np.random.default_rng(5)
    hidden["pred_tvt"] = hidden["TVT"].values + rng.normal(scale=0.5, size=len(hidden))
    return hidden[["id", "well_id", "row_idx", "pred_tvt"]]


def _synth_b2(frame: pd.DataFrame, column: str = "b2_guarded_submit") -> pd.DataFrame:
    """Synth b2: tvt_true + biased noise. Half the wells get small noise (b2
    wins → gate target 0), the other half get large noise (k_offset wins →
    gate target 1). This gives the classifier a balanced learnable signal.
    """
    rng = np.random.default_rng(7)
    out = frame[["id", "well_id"]].copy()
    well_ids = sorted(frame["well_id"].unique())
    noisy_wells = set(well_ids[::2])  # half noisy → k_offset will win on those
    noise_scale = np.where(frame["well_id"].isin(noisy_wells), 4.0, 0.05)
    out[column] = frame["TVT"].values + rng.normal(scale=noise_scale, size=len(frame))
    return out


def test_gated_features_are_schema_safe() -> None:
    from mtpnet.k_offset_gated import FEATURE_COLUMNS
    from mtpnet.schema_safe import FORBIDDEN_INFERENCE_COLUMNS, assert_schema_safe_columns

    assert FORBIDDEN_INFERENCE_COLUMNS.isdisjoint(set(FEATURE_COLUMNS))
    assert_schema_safe_columns(list(FEATURE_COLUMNS), context="test")


def test_binary_auc_matches_known_value() -> None:
    from mtpnet.k_offset_gated import _binary_auc

    auc = _binary_auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8]))
    # Perfect minus one swap = AUC 0.75
    assert abs(auc - 0.75) < 1e-9
    perfect = _binary_auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.3, 0.4]))
    assert abs(perfect - 1.0) < 1e-9


def test_gated_pipeline_smoke(tmp_path: Path) -> None:
    from mtpnet.k_offset_gated import KOffsetGatedConfig, run_k_offset_gated_from_frame

    frame = _tiny_frame(n_wells=10, well_len=64)
    k_offset_preds = _synth_k_offset(frame)
    b2_preds = _synth_b2(frame).rename(columns={"b2_guarded_submit": "b2_tvt"})

    metrics = run_k_offset_gated_from_frame(
        frame,
        config=KOffsetGatedConfig(
            output_dir=tmp_path,
            n_folds=2,
            chunk_size=8,
            iterations=40,
            depth=3,
            learning_rate=0.2,
            seed=11,
            rows_per_step=8,
            top_state_path=None,
        ),
        k_offset_preds=k_offset_preds,
        b2_preds=b2_preds,
        top_state_lookup={},
    )

    assert metrics["candidate"] == "k_offset_gated_v0"
    # Gate target should fire often because k_offset is near-perfect by construction
    assert metrics.get("gate_above_0.5_pct", 0.0) > 0.3
    preds_path = tmp_path / "k_offset_gated_oof_predictions.parquet"
    assert preds_path.exists()
    preds = pd.read_parquet(preds_path)
    assert {"id", "well_id", "row_idx", "pred_tvt"}.issubset(preds.columns)
    assert preds["pred_tvt"].notna().all()
    # one prediction per hidden row
    hidden_rows = frame["TVT_input"].isna().sum()
    assert len(preds) == hidden_rows
    chunks_path = tmp_path / "k_offset_gated_chunks.parquet"
    assert chunks_path.exists()
    assert (tmp_path / "k_offset_gated_metrics.json").exists()


def test_gated_predictions_feed_candidate_bank() -> None:
    import tempfile

    from mtpnet.candidate_bank import build_candidate_bank_from_frames
    from mtpnet.k_offset_gated import KOffsetGatedConfig, run_k_offset_gated_from_frame

    frame = _tiny_frame(n_wells=8, well_len=48)
    k_offset_preds = _synth_k_offset(frame)
    b2_preds = _synth_b2(frame).rename(columns={"b2_guarded_submit": "b2_tvt"})
    with tempfile.TemporaryDirectory() as td:
        run_k_offset_gated_from_frame(
            frame,
            config=KOffsetGatedConfig(
                output_dir=Path(td),
                n_folds=2,
                chunk_size=8,
                iterations=30,
                depth=3,
                learning_rate=0.2,
                seed=11,
                rows_per_step=8,
                top_state_path=None,
            ),
            k_offset_preds=k_offset_preds,
            b2_preds=b2_preds,
            top_state_lookup={},
        )
        preds = pd.read_parquet(Path(td) / "k_offset_gated_oof_predictions.parquet")

    # Ride through the k_offset_predictions slot so the bank registers it as a candidate
    bank = build_candidate_bank_from_frames(frame, k_offset_predictions=preds)
    assert "k_segment_offset_v0" in set(bank["candidate"].astype(str))
