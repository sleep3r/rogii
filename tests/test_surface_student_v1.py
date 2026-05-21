from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rogii.constants import FORMATIONS
from rogii.runlog import RunLogger
from rogii.surface_student_v1 import (
    SURFACE_TARGET_PREFIX,
    TARGET_TEACHER_DELTA,
    TARGET_TRUE_RESIDUAL,
    add_v1_derived_features,
    select_v1_feature_columns,
    teacher_quality_weights,
    train_heads_oof,
)


def _v1_frame(rows: int = 160) -> pd.DataFrame:
    idx = np.arange(rows, dtype=float)
    fold = np.where(idx < rows / 2, 1, 2)
    well = np.where(idx < rows / 2, "well_a", "well_b")
    schema = 1100.0 + 0.05 * idx
    true = schema + 1.5 * np.sin(idx / 11.0)
    last = np.where(idx < rows / 2, 1098.0, 1101.0)
    z = 2500.0 - 0.08 * idx
    frame = pd.DataFrame(
        {
            "id": [f"{well_name}_{int(i)}" for well_name, i in zip(well, idx, strict=False)],
            "well": well,
            "row_index": idx.astype(int),
            "fold": fold.astype(int),
            "tvt_true": true,
            "schema10_oof_raw": schema,
            "flat_tvt": last + 0.04 * idx,
            "last_known_tvt": last,
            "idx": idx,
            "idx_frac": idx / rows,
            "md": idx * 2.0,
            "x": 100.0 + idx,
            "y": 200.0 + 0.5 * idx,
            "z": z,
            "gr": 80.0 + np.sin(idx / 8.0),
            "dxdmd": np.full(rows, 0.5),
            "dydmd": np.full(rows, 0.25),
            "dzdmd": np.full(rows, -0.04),
            "kg_pf_ancc_tvt": schema + 0.8,
            "kg_dtw_tvt": schema + 1.1,
            "kg_dwt_tvt": schema + 0.6,
            "kg_beam_mean_tvt": schema + 0.7,
            "kg_ncc_mean_tvt": schema + 0.9,
            "kg_path_leaky_tvt": true,
            "geo_teacher_tvt": true + 0.2,
            "geo_teacher_conf": np.linspace(0.1, 0.9, rows),
            "teacher_abs_error": np.linspace(0.1, 8.0, rows),
            "teacher_best_surface_id": np.zeros(rows),
        }
    )
    frame[TARGET_TRUE_RESIDUAL] = frame["tvt_true"] - frame["schema10_oof_raw"]
    frame[TARGET_TEACHER_DELTA] = frame["geo_teacher_tvt"] - frame["last_known_tvt"]
    for pos, formation in enumerate(FORMATIONS):
        z_minus = 1000.0 + pos * 5.0 + 0.01 * idx
        frame[f"z_minus_{formation}_true"] = z_minus
        frame[f"{SURFACE_TARGET_PREFIX}{formation}"] = z_minus
    return frame


def test_v1_feature_selector_keeps_alignment_and_excludes_privileged() -> None:
    frame = add_v1_derived_features(_v1_frame())
    columns = select_v1_feature_columns(frame)

    assert "kg_pf_ancc_tvt" in columns
    assert "kg_dtw_tvt" in columns
    assert "candidate_tvt_std" in columns
    assert "azimuth_sin" in columns
    assert "kg_path_leaky_tvt" not in columns
    assert "geo_teacher_tvt" not in columns
    assert TARGET_TRUE_RESIDUAL not in columns
    assert "z_minus_ANCC_true" not in columns


def test_teacher_quality_weights_downweight_teacher_outliers() -> None:
    frame = pd.DataFrame({"teacher_abs_error": [0.0, 1.0, 10.0, np.nan]})
    weights = teacher_quality_weights(
        frame,
        {"teacher_error_threshold": 5.0, "teacher_min_weight": 0.1},
    )

    assert weights[0] > weights[1] > weights[2]
    assert weights[3] == 0.1


def test_train_heads_oof_writes_v1_outputs(tmp_path: Path) -> None:
    frame = add_v1_derived_features(_v1_frame())
    feature_columns = select_v1_feature_columns(frame)

    oof, rows = train_heads_oof(
        frame,
        feature_columns=feature_columns,
        cfg={
            "seed": 7,
            "model": {
                "params": {
                    "iterations": 5,
                    "early_stopping_rounds": 2,
                    "learning_rate": 0.1,
                    "depth": 3,
                    "thread_count": 1,
                    "task_type": "CPU",
                }
            },
        },
        output_dir=tmp_path,
        logger=RunLogger(),
    )

    assert rows
    for column in [
        "geo_student_v1_true_tvt",
        "geo_student_v1_teacher_tvt",
        "geo_student_v1_blend_tvt",
        "surface_hat_ANCC",
        "z_minus_surface_hat_BUDA",
        "student_ensemble_std",
        "student_vs_schema10",
    ]:
        assert column in oof.columns
    assert np.isfinite(oof["geo_student_v1_blend_tvt"]).all()
    assert (tmp_path / "models" / TARGET_TRUE_RESIDUAL / "fold_1.pkl").exists()
