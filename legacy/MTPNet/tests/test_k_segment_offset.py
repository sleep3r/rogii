from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _tiny_k_segment_frame(n_wells: int = 6, well_len: int = 64) -> pd.DataFrame:
    """Synthetic frame with one contiguous hidden lateral block at the end.

    Wells are stylised so that ``TVT = -Z + C`` holds with a piecewise-constant
    drift in C. That gives the K-segment offset model a learnable structure.
    """
    rng = np.random.default_rng(7)
    records = []
    for well_idx in range(n_wells):
        well_id = f"w{well_idx:02d}"
        # Z is roughly linear with mild noise
        z = np.linspace(-100.0, -200.0, well_len) + rng.normal(scale=0.5, size=well_len)
        # piecewise C: 3 segments with small but distinct slopes
        drift = np.zeros(well_len)
        boundaries = [0, well_len // 3, 2 * well_len // 3, well_len]
        slopes = rng.uniform(-0.03, 0.03, size=3)
        c0 = 50.0 + well_idx
        c = np.zeros(well_len)
        c[0] = c0
        for k in range(3):
            lo, hi = boundaries[k], boundaries[k + 1]
            for i in range(lo + 1 if k == 0 else lo, hi):
                c[i] = c[i - 1] + slopes[k] + rng.normal(scale=0.005)
        tvt = -z + c
        # First two-thirds are "known", last third hidden (mimics real data shape)
        hidden_start = int(0.66 * well_len)
        for row_idx in range(well_len):
            tvt_in = tvt[row_idx] if row_idx < hidden_start else np.nan
            records.append(
                {
                    "id": f"{well_id}_{row_idx}",
                    "well_id": well_id,
                    "row_idx": row_idx,
                    "MD": 1000.0 + row_idx * 10.0,
                    "X": float(row_idx),
                    "Y": float(well_idx) * 100.0,
                    "Z": float(z[row_idx]),
                    "GR": 80.0 + 10.0 * np.sin(row_idx / 4.0),
                    "TVT": float(tvt[row_idx]),
                    "TVT_input": float(tvt_in) if np.isfinite(tvt_in) else np.nan,
                    "ANCC": float(c[row_idx]),
                }
            )
    return pd.DataFrame(records)


def test_k_segment_dataset_is_schema_safe() -> None:
    from mtpnet.k_segment_offset import build_k_segment_dataset
    from mtpnet.schema_safe import FORBIDDEN_INFERENCE_COLUMNS

    frame = _tiny_k_segment_frame()
    fold_of_well = {wid: idx % 2 for idx, wid in enumerate(frame["well_id"].unique())}

    df, geometries, feat_cols = build_k_segment_dataset(
        frame,
        K=3,
        rows_per_step=8,
        fold_of_well=fold_of_well,
        top_state_lookup={},
    )

    assert FORBIDDEN_INFERENCE_COLUMNS.isdisjoint(feat_cols)
    assert {"well_id", "fold", "segment_k", "target_c"}.issubset(df.columns)
    # Three segments per well, 6 wells = 18 rows
    assert len(df) == 6 * 3
    assert df["segment_k"].between(0, 2).all()
    assert len(geometries) == 6
    # Geometry preserves hidden range starts at first hidden row index
    for geom in geometries:
        assert geom.hidden_row_idx.size > 0
        # boundaries must cover hidden range exactly
        assert int(geom.bounds[0]) == geom.last_known_row + 1
        # designed matrix is finite
        from mtpnet.k_segment_offset import _segment_design_matrix

        M = _segment_design_matrix(geom.bounds, geom.hidden_row_idx)
        assert np.isfinite(M).all()


def test_k_segment_oracle_matches_lstsq_property() -> None:
    """If we use the *oracle* per-segment c, hidden TVT reconstruction is exact-up-to-noise."""
    from mtpnet.k_segment_offset import (
        _oracle_segment_offsets,
        _segment_boundaries,
        _segment_design_matrix,
    )

    frame = _tiny_k_segment_frame()
    well = frame[frame["well_id"] == "w00"].sort_values("row_idx")
    z = well["Z"].to_numpy(dtype=np.float64)
    tvt = well["TVT"].to_numpy(dtype=np.float64)
    tvt_in = well["TVT_input"].to_numpy(dtype=np.float64)

    bounds = _segment_boundaries(int(np.flatnonzero(np.isfinite(tvt_in))[-1]), len(z), K=3)
    result = _oracle_segment_offsets(z, tvt, tvt_in, bounds)
    assert result is not None
    c, hidden_idx = result
    M = _segment_design_matrix(bounds, hidden_idx)
    anchor = int(np.flatnonzero(np.isfinite(tvt_in))[-1])
    pred = float(tvt_in[anchor]) - (z[hidden_idx] - z[anchor]) + (M @ c)
    rmse = float(np.sqrt(np.mean((pred - tvt[hidden_idx]) ** 2)))
    # Synthetic data has noise std 0.005 cumulating over ~20 rows
    assert rmse < 0.5, f"oracle reconstruction RMSE too large: {rmse}"


def test_k_segment_pipeline_writes_oof_artifact(tmp_path: Path) -> None:
    from mtpnet.k_segment_offset import (
        KSegmentOffsetConfig,
        run_k_segment_offset_from_frame,
    )

    frame = _tiny_k_segment_frame(n_wells=8, well_len=64)
    metrics = run_k_segment_offset_from_frame(
        frame,
        config=KSegmentOffsetConfig(
            output_dir=tmp_path,
            n_folds=2,
            K=3,
            iterations=40,
            depth=3,
            learning_rate=0.2,
            seed=11,
            rows_per_step=8,
            top_state_path=None,
        ),
        top_state_lookup={},
    )

    assert metrics["candidate"] == "k_segment_offset_v0"
    assert metrics["K"] == 3
    assert (tmp_path / "k_offset_oof_predictions.parquet").exists()
    assert (tmp_path / "k_offset_metrics.json").exists()
    assert (tmp_path / "k_offset_report.md").exists()

    preds = pd.read_parquet(tmp_path / "k_offset_oof_predictions.parquet")
    assert {"id", "well_id", "row_idx", "pred_tvt"}.issubset(preds.columns)
    assert preds["pred_tvt"].notna().all()
    # one row per hidden row in each well (8 wells * ~22 hidden ≈ 176)
    assert len(preds) > 8 * 5


def test_k_segment_offset_feeds_candidate_bank() -> None:
    """The OOF parquet has the schema candidate_bank.build_candidate_bank_from_frames expects."""
    from mtpnet.candidate_bank import build_candidate_bank_from_frames
    from mtpnet.k_segment_offset import (
        KSegmentOffsetConfig,
        run_k_segment_offset_from_frame,
    )

    import tempfile

    frame = _tiny_k_segment_frame(n_wells=6, well_len=48)
    with tempfile.TemporaryDirectory() as td:
        run_k_segment_offset_from_frame(
            frame,
            config=KSegmentOffsetConfig(
                output_dir=Path(td),
                n_folds=2,
                K=3,
                iterations=30,
                depth=3,
                learning_rate=0.2,
                seed=11,
                rows_per_step=8,
                top_state_path=None,
            ),
            top_state_lookup={},
        )
        preds = pd.read_parquet(Path(td) / "k_offset_oof_predictions.parquet")

    bank = build_candidate_bank_from_frames(frame, k_offset_predictions=preds)
    assert "k_segment_offset_v0" in set(bank["candidate"].astype(str))
