from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _write_tiny_offset_data(root: Path, n_wells: int = 6, n_rows: int = 64) -> pd.DataFrame:
    """Write tiny train-like CSVs where one discrete offset is exactly correct."""
    rng = np.random.default_rng(31)
    rows = []
    offsets = np.asarray([-0.02, 0.0, 0.02], dtype=float)
    for well_idx in range(n_wells):
        well_id = f"w{well_idx:02d}"
        offset = float(offsets[well_idx % len(offsets)])
        z = -100.0 - np.linspace(0.0, 60.0, n_rows) + rng.normal(scale=0.05, size=n_rows)
        tvt = np.zeros(n_rows, dtype=float)
        tvt[0] = 500.0 + well_idx
        for i in range(1, n_rows):
            tvt[i] = tvt[i - 1] - (z[i] - z[i - 1]) + offset
        gr = 90.0 + 20.0 * np.sin(tvt / 18.0)
        hidden_start = int(0.65 * n_rows)
        horizontal = pd.DataFrame(
            {
                "MD": 1000.0 + np.arange(n_rows) * 10.0,
                "X": np.arange(n_rows, dtype=float),
                "Y": np.full(n_rows, well_idx, dtype=float),
                "Z": z,
                "GR": gr,
                "TVT": tvt,
                "TVT_input": np.where(np.arange(n_rows) < hidden_start, tvt, np.nan),
                "ANCC": tvt + z,  # train-only, must not enter features
            }
        )
        typewell_tvt = np.linspace(float(tvt.min()) - 20.0, float(tvt.max()) + 20.0, 180)
        typewell = pd.DataFrame(
            {
                "TVT": typewell_tvt,
                "GR": 90.0 + 20.0 * np.sin(typewell_tvt / 18.0),
            }
        )
        horizontal.to_csv(root / f"{well_id}__horizontal_well.csv", index=False)
        typewell.to_csv(root / f"{well_id}__typewell.csv", index=False)
        for row_idx in range(n_rows):
            rows.append(
                {
                    "id": f"{well_id}_{row_idx}",
                    "well_id": well_id,
                    "row_idx": row_idx,
                    **horizontal.iloc[row_idx].to_dict(),
                }
            )
    return pd.DataFrame(rows)


def test_parse_offset_grid_range_and_list() -> None:
    from mtpnet.discrete_offset import parse_offset_grid

    grid = parse_offset_grid("-0.02:0.02:0.01")
    assert np.allclose(grid, [-0.02, -0.01, 0.0, 0.01, 0.02])
    listed = parse_offset_grid("0.02,-0.01,0.02,0")
    assert np.allclose(listed, [-0.01, 0.0, 0.02])


def test_discrete_offset_features_are_schema_safe(tmp_path: Path) -> None:
    from mtpnet.discrete_offset import (
        build_discrete_offset_dataset,
        parse_offset_grid,
        _load_wells,
    )
    from mtpnet.residual_stack import make_group_folds
    from mtpnet.schema_safe import FORBIDDEN_INFERENCE_COLUMNS

    frame = _write_tiny_offset_data(tmp_path)
    well_ids = sorted(frame["well_id"].unique())
    folds = make_group_folds(well_ids, n_folds=2, seed=1)
    fold_of_well = {str(w): idx for idx, (_, valid) in enumerate(folds) for w in valid}
    wells = _load_wells(tmp_path, k_wells=-1, fold_of_well=fold_of_well)
    rows, feature_columns = build_discrete_offset_dataset(
        wells,
        offset_grid=parse_offset_grid("-0.02:0.02:0.02"),
        seed=1,
    )

    assert FORBIDDEN_INFERENCE_COLUMNS.isdisjoint(feature_columns)
    assert {"well_id", "offset", "candidate_mse", "is_best_offset"}.issubset(rows.columns)
    assert rows["well_id"].nunique() == len(well_ids)
    assert rows.groupby("well_id")["is_best_offset"].sum().ge(1).all()


def test_discrete_offset_selfcal_features_prefer_correct_known_offset(tmp_path: Path) -> None:
    from mtpnet.discrete_offset import (
        build_discrete_offset_dataset,
        parse_offset_grid,
        _load_wells,
    )
    from mtpnet.residual_stack import make_group_folds

    frame = _write_tiny_offset_data(tmp_path, n_wells=3, n_rows=96)
    well_ids = sorted(frame["well_id"].unique())
    folds = make_group_folds(well_ids, n_folds=2, seed=2)
    fold_of_well = {str(w): idx for idx, (_, valid) in enumerate(folds) for w in valid}
    wells = _load_wells(tmp_path, k_wells=-1, fold_of_well=fold_of_well)

    rows, feature_columns = build_discrete_offset_dataset(
        wells,
        offset_grid=parse_offset_grid("-0.02:0.02:0.02"),
        seed=2,
    )

    assert "feat_selfcal_rmse_mean" in feature_columns
    for well_id, group in rows.groupby("well_id"):
        best_selfcal = group.sort_values("feat_selfcal_rmse_mean").iloc[0]
        best_oracle = group.sort_values("candidate_mse").iloc[0]
        assert float(best_selfcal["offset"]) == float(best_oracle["offset"]), well_id


def test_discrete_offset_smoke_writes_artifacts(tmp_path: Path) -> None:
    from mtpnet.discrete_offset import DiscreteOffsetConfig, run_discrete_offset

    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    _write_tiny_offset_data(data_dir, n_wells=8, n_rows=56)
    metrics = run_discrete_offset(
        DiscreteOffsetConfig(
            data_dir=data_dir,
            output_dir=out_dir,
            offset_grid="-0.02:0.02:0.02",
            n_folds=2,
            iterations=20,
            learning_rate=0.2,
            depth=3,
            seed=3,
            include_shuffled=True,
        )
    )

    assert metrics["experiment"] == "discrete_offset_v0"
    assert metrics["grid_oracle"]["row_rmse"] < 0.2
    assert (out_dir / "discrete_offset_metrics.json").exists()
    assert (out_dir / "discrete_offset_report.md").exists()
    assert (out_dir / "discrete_offset_candidates.parquet").exists()
    preds = pd.read_parquet(out_dir / "discrete_offset_oof_predictions.parquet")
    assert {"id", "well_id", "row_idx", "pred_tvt", "selected_offset"}.issubset(preds.columns)
    assert preds["pred_tvt"].notna().all()
