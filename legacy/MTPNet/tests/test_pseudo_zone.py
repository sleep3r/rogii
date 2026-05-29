from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mtpnet.pseudo_zone import (
    PseudoZoneTemplate,
    fit_pseudo_zone_template,
    predict_typewell_geology,
    run_pseudo_zone_audit,
)


def _typewell(labels: list[str]) -> pd.DataFrame:
    tvt = np.arange(len(labels), dtype=np.float32) * 10.0
    gr = np.linspace(50.0, 100.0, len(labels), dtype=np.float32)
    return pd.DataFrame({"TVT": tvt, "GR": gr, "Geology": labels})


def test_template_predicts_geology_from_normalized_tvt_bins() -> None:
    train_typewells = [
        _typewell(["A", "A", "A", "B", "B", "B"]),
        _typewell(["A", "A", "A", "B", "B", "B"]),
    ]
    model = fit_pseudo_zone_template(train_typewells, n_bins=6)

    pred = predict_typewell_geology(_typewell(["A", "A", "A", "B", "B", "B"]), model)

    assert pred[:3].tolist() == ["A", "A", "A"]
    assert pred[3:].tolist() == ["B", "B", "B"]


def test_template_fills_empty_bins_from_nearest_known_label() -> None:
    model = PseudoZoneTemplate(
        labels_by_bin=np.array(["A", "__unknown__", "__unknown__", "B"], dtype=object),
        n_bins=4,
    ).filled()

    assert model.labels_by_bin.tolist() == ["A", "A", "B", "B"]


def test_run_pseudo_zone_audit_writes_artifacts(tmp_path: Path) -> None:
    train = tmp_path / "data" / "train"
    train.mkdir(parents=True)
    for well_id in ["well_a", "well_b", "well_c"]:
        tvt = np.arange(8, dtype=np.float32) * 5.0
        gr = np.linspace(40.0, 80.0, len(tvt), dtype=np.float32)
        tvt_input = tvt.copy()
        tvt_input[4:] = np.nan
        labels = ["A"] * 4 + ["B"] * 4
        pd.DataFrame({"TVT": tvt, "TVT_input": tvt_input, "GR": gr}).to_csv(
            train / f"{well_id}__horizontal_well.csv", index=False
        )
        pd.DataFrame({"TVT": tvt, "GR": gr, "Geology": labels}).to_csv(
            train / f"{well_id}__typewell.csv", index=False
        )

    output = tmp_path / "pseudo"
    run_pseudo_zone_audit(
        data_dir=tmp_path / "data",
        output_dir=output,
        rows_per_step=1,
        n_bins=8,
        n_folds=3,
    )

    metrics = json.loads((output / "pseudo_zone_metrics.json").read_text())
    assert metrics["aggregate"]["hidden_zone_match_rate"] == pytest.approx(1.0)
    assert (output / "pseudo_zone_by_well.csv").exists()
    assert (output / "pseudo_zone_steps.parquet").exists()
    assert (output / "pseudo_zone_report.md").exists()
