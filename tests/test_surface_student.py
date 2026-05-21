from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from rogii.constants import FORMATIONS
from rogii.surface_student import (
    build_surface_student_features_for_well,
    train_surface_student,
)


def _synthetic_well(
    *,
    n: int = 90,
    hidden_start: int = 55,
    well_shift: float = 0.0,
) -> pd.DataFrame:
    md = np.arange(n, dtype=float)
    z = 2500.0 - 0.05 * md + well_shift * 0.01
    tvt = 1100.0 + well_shift + 0.08 * md + 0.00015 * (md - 40.0) ** 2
    tvt_input = tvt.copy()
    tvt_input[hidden_start:] = np.nan
    frame = pd.DataFrame(
        {
            "MD": md,
            "X": 100.0 + well_shift + 0.12 * md,
            "Y": 200.0 - well_shift + 0.08 * md,
            "Z": z,
            "GR": 85.0 + np.sin(md / 7.0) + 0.02 * well_shift,
            "TVT_input": tvt_input,
            "TVT": tvt,
        }
    )
    for idx, formation in enumerate(FORMATIONS):
        offset = 1010.0 + idx * 9.0 + 0.1 * well_shift
        frame[formation] = z - (tvt - offset)
    return frame


def _typewell() -> pd.DataFrame:
    tvt = np.linspace(1000.0, 1250.0, 120)
    return pd.DataFrame({"TVT": tvt, "GR": 84.0 + np.sin(tvt / 15.0)})


def _student_config(*, spatial_impute: bool = False) -> dict:
    return {
        "seed": 7,
        "surface_teacher": {"tail_rows": 40, "min_fit_rows": 12, "ridge": 1e-2},
        "surface_student": {
            "seed": 7,
            "n_splits": 2,
            "tail_rows": 40,
            "rolling_windows": [5, 15],
            "spatial_impute": spatial_impute,
            "model": {
                "name": "catboost",
                "params": {
                    "iterations": 8,
                    "early_stopping_rounds": 3,
                    "learning_rate": 0.08,
                    "depth": 3,
                    "thread_count": 1,
                    "task_type": "CPU",
                },
            },
        },
    }


def test_surface_student_features_do_not_read_raw_surfaces() -> None:
    frame = _synthetic_well()
    changed = frame.copy()
    for formation in FORMATIONS:
        changed[formation] = changed[formation] + np.linspace(10.0, 20.0, len(changed))

    features_a = build_surface_student_features_for_well(frame, _typewell(), _student_config())
    features_b = build_surface_student_features_for_well(changed, _typewell(), _student_config())

    assert len(features_a) == int(frame["TVT_input"].isna().sum())
    assert not any(formation in features_a.columns for formation in FORMATIONS)
    pd.testing.assert_frame_equal(features_a, features_b, check_dtype=False)


def test_surface_student_adds_spatial_imputed_surface_features() -> None:
    class DummyContext:
        def impute_formations(
            self,
            xy: np.ndarray,
            *,
            self_well: str | None = None,
        ) -> tuple[np.ndarray, np.ndarray]:
            del self_well
            base = xy[:, :1] * 0.0 + 2400.0
            surfaces = np.hstack([base + idx for idx, _ in enumerate(FORMATIONS)])
            return surfaces, np.full(len(xy), 12.5)

    features = build_surface_student_features_for_well(
        _synthetic_well(),
        _typewell(),
        _student_config(spatial_impute=True),
        context=DummyContext(),  # type: ignore[arg-type]
        well="well0",
    )

    assert features["spatial_surface_dist"].eq(12.5).all()
    assert features["spatial_surface_hat_ANCC"].notna().all()
    assert features["z_minus_spatial_surface_hat_BUDA"].notna().all()


def test_surface_student_train_smoke_writes_oof_and_models(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    train_dir = data_dir / "train"
    train_dir.mkdir(parents=True)
    for idx in range(4):
        well = f"well{idx:04d}"
        _synthetic_well(well_shift=float(idx)).to_csv(
            train_dir / f"{well}__horizontal_well.csv",
            index=False,
        )
        _typewell().to_csv(train_dir / f"{well}__typewell.csv", index=False)

    metrics = train_surface_student(
        config=_student_config(),
        data_dir=data_dir,
        output_dir=tmp_path / "student",
        max_wells=4,
    )

    assert metrics["model_backend"] == "catboost"
    assert metrics["rows"] == 140
    assert metrics["wells"] == 4
    assert metrics["feature_count"] > 10
    assert np.isfinite(metrics["student_vs_teacher_rmse"])
    assert (tmp_path / "student" / "models" / "target_geo_teacher_delta_last").exists()
    assert (tmp_path / "student" / "surface_student_report.md").exists()
    artifact = Path(metrics["prediction_artifact"]["path"])
    assert artifact.exists()
    if artifact.suffix == ".parquet":
        oof = pd.read_parquet(artifact)
    else:
        oof = pd.read_csv(artifact)
    for column in [
        "geo_student_tvt",
        "geo_student_delta_last",
        "geo_student_minus_flat",
        "geo_student_uncertainty",
        "surface_hat_ANCC",
        "surface_unc_BUDA",
        "student_vs_pf_ancc",
        "student_vs_dtw",
        "student_vs_dwt",
        "student_vs_schema10",
    ]:
        assert column in oof.columns
    written_metrics = json.loads((tmp_path / "student" / "surface_student_metrics.json").read_text())
    assert written_metrics["model_backend"] == "catboost"
