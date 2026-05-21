from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from rogii.surface_teacher import (
    FORMATIONS,
    build_geo_teacher_for_well,
    build_surface_teacher_dataset,
)


def _synthetic_well(
    *,
    n: int = 160,
    hidden_start: int = 110,
    noisy_hidden_surfaces: bool = False,
) -> pd.DataFrame:
    md = np.arange(n, dtype=float)
    z = 2400.0 - 0.04 * md
    tvt = 1100.0 + 0.07 * md + 0.0001 * (md - 80.0) ** 2
    tvt_input = tvt.copy()
    tvt_input[hidden_start:] = np.nan
    frame = pd.DataFrame(
        {
            "MD": md,
            "X": md * 0.1,
            "Y": md * 0.2,
            "Z": z,
            "GR": 80.0 + np.sin(md / 8.0),
            "TVT_input": tvt_input,
            "TVT": tvt,
        }
    )
    for idx, formation in enumerate(FORMATIONS):
        offset = 1040.0 + idx * 10.0
        surface = z - (tvt - offset)
        if noisy_hidden_surfaces and idx >= 2:
            surface = surface.copy()
            surface[hidden_start:] += 20.0 + idx * np.sin(md[hidden_start:] / 5.0)
        frame[formation] = surface
    return frame


def _teacher_config() -> dict:
    return {"surface_teacher": {"tail_rows": 60, "min_fit_rows": 16, "ridge": 1e-2}}


def test_surface_teacher_produces_finite_hidden_outputs() -> None:
    frame = _synthetic_well()

    teacher = build_geo_teacher_for_well(frame, None, _teacher_config())

    assert teacher["row_index"].tolist()[0] == 110
    assert teacher["row_index"].tolist()[-1] == 159
    assert teacher["geo_teacher_tvt"].notna().all()
    assert teacher["geo_teacher_conf"].between(0.0, 1.0).all()
    assert "z_minus_ANCC_true" in teacher.columns
    assert teacher["teacher_best_surface_id"].notna().all()


def test_surface_teacher_missing_surfaces_has_zero_coverage() -> None:
    frame = _synthetic_well().drop(columns=FORMATIONS)

    teacher = build_geo_teacher_for_well(frame, None, _teacher_config())

    assert len(teacher) == int(frame["TVT_input"].isna().sum())
    assert teacher["row_index"].tolist()[0] == 110
    assert teacher["geo_teacher_tvt"].isna().all()
    assert teacher["teacher_missing_surfaces"].eq(1.0).all()


def test_surface_teacher_does_not_read_hidden_true_tvt() -> None:
    frame = _synthetic_well()
    altered = frame.copy()
    hidden_mask = altered["TVT_input"].isna()
    altered.loc[hidden_mask, "TVT"] = altered.loc[hidden_mask, "TVT"] + 10_000.0

    teacher_a = build_geo_teacher_for_well(frame, None, _teacher_config())
    teacher_b = build_geo_teacher_for_well(altered, None, _teacher_config())

    np.testing.assert_allclose(
        teacher_a["geo_teacher_tvt"].to_numpy(float),
        teacher_b["geo_teacher_tvt"].to_numpy(float),
        equal_nan=True,
    )


def test_surface_teacher_confidence_drops_when_surfaces_disagree() -> None:
    clean = build_geo_teacher_for_well(_synthetic_well(), None, _teacher_config())
    noisy = build_geo_teacher_for_well(
        _synthetic_well(noisy_hidden_surfaces=True), None, _teacher_config()
    )

    assert float(clean["geo_teacher_conf"].mean()) > float(noisy["geo_teacher_conf"].mean())
    assert float(clean["teacher_surface_spread"].median()) < float(
        noisy["teacher_surface_spread"].median()
    )


def test_surface_teacher_dataset_writer_outputs_reports(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    train_dir = data_dir / "train"
    train_dir.mkdir(parents=True)
    frame = _synthetic_well(n=80, hidden_start=50)
    frame.to_csv(train_dir / "abcd1234__horizontal_well.csv", index=False)
    pd.DataFrame({"TVT": np.linspace(1000, 1200, 20), "GR": np.linspace(70, 90, 20)}).to_csv(
        train_dir / "abcd1234__typewell.csv", index=False
    )

    metrics = build_surface_teacher_dataset(
        data_dir=data_dir,
        output_dir=tmp_path / "teacher",
        config=_teacher_config(),
        progress_interval=1,
    )

    assert metrics["teacher_coverage"] == 1.0
    assert (tmp_path / "teacher" / "teacher_rows.csv").exists()
    assert (tmp_path / "teacher" / "teacher_by_well.csv").exists()
    metrics_json = json.loads((tmp_path / "teacher" / "teacher_metrics.json").read_text())
    assert metrics_json["rows"] == 30
    assert (tmp_path / "teacher" / "teacher_report.md").exists()
