from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _tiny_hidden_rows() -> pd.DataFrame:
    rows = []
    for well_id, offset in [("a", 0.0), ("b", 20.0), ("c", -10.0), ("d", 35.0)]:
        for row_idx in range(8):
            is_hidden = row_idx >= 3
            tvt = 100.0 + offset + row_idx * 2.0
            anchor = tvt - (5.0 if well_id in {"a", "b"} else -4.0)
            rows.append(
                {
                    "id": f"{well_id}_{row_idx}",
                    "well_id": well_id,
                    "row_idx": row_idx,
                    "step": row_idx // 2,
                    "MD": 1000.0 + row_idx,
                    "X": 10.0 + row_idx,
                    "Y": 20.0,
                    "Z": -100.0 - row_idx,
                    "GR": 80.0 + row_idx,
                    "TVT": tvt,
                    "TVT_input": np.nan if is_hidden else tvt,
                    "anchor_tvt": anchor,
                    "b2_tvt": anchor,
                    "base_tvt": anchor + 1.0,
                    "a_p50_tvt": anchor - 1.0,
                    "tail_class": "G_all_candidates_fail" if well_id == "d" else "OK_or_mixed",
                }
            )
    return pd.DataFrame(rows)


def test_residual_feature_builder_is_schema_safe_and_finite() -> None:
    from mtpnet.residual_stack import build_residual_step_dataset
    from mtpnet.schema_safe import FORBIDDEN_INFERENCE_COLUMNS

    dataset = build_residual_step_dataset(_tiny_hidden_rows(), rows_per_step=2)

    assert dataset.feature_columns
    assert FORBIDDEN_INFERENCE_COLUMNS.isdisjoint(dataset.feature_columns)
    assert np.isfinite(dataset.features.to_numpy(dtype=np.float64)).all()
    assert dataset.rows["well_id"].nunique() == 4


def test_residual_group_folds_are_disjoint() -> None:
    from mtpnet.residual_stack import make_group_folds

    folds = make_group_folds(["a", "b", "c", "d"], n_folds=2, seed=7)

    assert len(folds) == 2
    for train_wells, valid_wells in folds:
        assert set(train_wells).isdisjoint(valid_wells)
        assert train_wells
        assert valid_wells


def test_residual_stack_smoke_writes_artifacts(tmp_path: Path) -> None:
    from mtpnet.residual_stack import ResidualStackConfig, run_residual_stack_from_frames

    metrics = run_residual_stack_from_frames(
        _tiny_hidden_rows(),
        config=ResidualStackConfig(
            output_dir=tmp_path,
            rows_per_step=2,
            n_folds=2,
            iterations=5,
            learning_rate=0.1,
            depth=2,
            seed=11,
        ),
    )

    assert metrics["rows"] == 20
    assert metrics["folds"] == 2
    assert set(metrics["feature_columns"]).isdisjoint({"TVT", "Geology", "ANCC"})
    assert (tmp_path / "oof_predictions.parquet").exists()
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "report.md").exists()
    saved = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert saved["candidate"] == "residual_stack_v0"
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "RESIDUAL_STACK_V0_REPORT" in report


def test_residual_stack_rejects_forbidden_feature_columns() -> None:
    from mtpnet.residual_stack import ResidualStepDataset, train_fold_model

    dataset = ResidualStepDataset(
        rows=pd.DataFrame({"well_id": ["a"], "target_residual": [1.0]}),
        features=pd.DataFrame({"TVT": [100.0]}),
        feature_columns=["TVT"],
    )

    with pytest.raises(ValueError, match="forbidden"):
        train_fold_model(dataset, train_wells=["a"], config=None)
