from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mtpnet.pseudo_zone_v1 import (
    COARSE_ZONE_LABELS,
    FORBIDDEN_INFERENCE_COLUMNS,
    TEST_SAFE_LATERAL_COLUMNS,
    PseudoZoneV1Config,
    build_lateral_zone_features,
    build_typewell_zone_features,
    collapse_geology_label,
    run_pseudo_zone_corr,
    run_pseudo_zone_v1,
    zone_mask_from_predictions,
)


def _write_well(root: Path, well_id: str, *, shift: float = 0.0) -> None:
    train = root / "train"
    train.mkdir(parents=True, exist_ok=True)
    tvt = np.arange(12, dtype=np.float32) * 10.0 + shift
    gr = np.array([10, 11, 12, 20, 21, 22, 40, 41, 42, 70, 71, 72], dtype=np.float32)
    tvt_input = tvt.copy()
    tvt_input[6:] = np.nan
    horizontal = pd.DataFrame(
        {
            "MD": np.arange(len(tvt), dtype=np.float32),
            "X": np.linspace(0, 1, len(tvt), dtype=np.float32),
            "Y": np.linspace(1, 2, len(tvt), dtype=np.float32),
            "Z": -tvt,
            "GR": gr,
            "TVT": tvt,
            "TVT_input": tvt_input,
            "ANCC": -100.0,
            "ASTNU": -200.0,
            "Geology": "SHOULD_NOT_BE_USED",
        }
    )
    typewell = pd.DataFrame(
        {
            "TVT": tvt,
            "GR": gr,
            "Geology": ["ANCC"] * 3 + ["ASTNU"] * 3 + ["EGFDL"] * 3 + ["LTHL"] * 3,
        }
    )
    horizontal.to_csv(train / f"{well_id}__horizontal_well.csv", index=False)
    typewell.to_csv(train / f"{well_id}__typewell.csv", index=False)


def test_coarse_label_collapse_groups_rare_labels() -> None:
    assert collapse_geology_label("ANCC") == "ANCC"
    assert collapse_geology_label("OLMOS") == "OLMOS"
    assert collapse_geology_label("LTHL") == "EGFDL_SUB"
    assert collapse_geology_label("AC_UEF_THL") == "EGFDL_SUB"
    assert collapse_geology_label("Clay Rich Interval") == "OTHER"
    assert collapse_geology_label(np.nan) == "__unknown__"
    assert "EGFDL_SUB" in COARSE_ZONE_LABELS


def test_inference_feature_builders_exclude_train_only_columns() -> None:
    horizontal = pd.DataFrame(
        {
            "MD": [0.0, 1.0, 2.0, 3.0],
            "X": [0.0, 1.0, 2.0, 3.0],
            "Y": [3.0, 2.0, 1.0, 0.0],
            "Z": [-1.0, -2.0, -3.0, -4.0],
            "GR": [10.0, np.nan, 12.0, 13.0],
            "TVT": [100.0, 101.0, 102.0, 103.0],
            "TVT_input": [100.0, 101.0, np.nan, np.nan],
            "ANCC": [1.0] * 4,
            "ASTNU": [2.0] * 4,
            "Geology": ["A"] * 4,
        }
    )
    typewell = pd.DataFrame(
        {"TVT": [100.0, 101.0, 102.0], "GR": [10.0, 11.0, 12.0], "Geology": ["A", "B", "C"]}
    )

    lateral_features, lateral_columns = build_lateral_zone_features(horizontal, rows_per_step=1)
    typewell_features, typewell_columns, _tvt = build_typewell_zone_features(typewell, vertical_step_ft=1.0)

    assert lateral_features.shape[1] == len(TEST_SAFE_LATERAL_COLUMNS)
    assert tuple(lateral_columns) == TEST_SAFE_LATERAL_COLUMNS
    assert typewell_features.shape[0] == 3
    assert np.isfinite(lateral_features).all()
    assert np.isfinite(typewell_features).all()
    assert not set(lateral_columns).intersection(FORBIDDEN_INFERENCE_COLUMNS)
    assert not set(typewell_columns).intersection(FORBIDDEN_INFERENCE_COLUMNS)


def test_zone_mask_uses_predictions_not_true_hidden_tvt() -> None:
    typewell_zones = np.array(["ANCC", "ASTNU", "EGFDL", "EGFDL_SUB"], dtype=object)
    probs = np.zeros((2, len(COARSE_ZONE_LABELS)), dtype=np.float32)
    probs[:, COARSE_ZONE_LABELS.index("EGFDL")] = 0.7
    probs[:, COARSE_ZONE_LABELS.index("ASTNU")] = 0.2

    mask_a = zone_mask_from_predictions(typewell_zones, probs, top_n=1)
    mask_b = zone_mask_from_predictions(typewell_zones, probs, top_n=1)

    assert mask_a.tolist() == mask_b.tolist()
    assert mask_a[0].tolist() == [False, False, True, False]


def test_run_pseudo_zone_v1_writes_test_safe_oof_artifacts(tmp_path: Path) -> None:
    data = tmp_path / "data"
    for idx in range(4):
        _write_well(data, f"well_{idx}", shift=float(idx))

    output = tmp_path / "pz"
    run_pseudo_zone_v1(
        data_dir=data,
        output_dir=output,
        cfg=PseudoZoneV1Config(rows_per_step=1, vertical_step_ft=10.0, n_folds=2, neural_epochs=1, seed=7),
    )

    metrics = json.loads((output / "pseudo_zone_metrics.json").read_text())
    pred = pd.read_parquet(output / "pseudo_zone_predictions.parquet")
    typewell = pd.read_parquet(output / "pseudo_zone_typewell_bins.parquet")

    assert metrics["template_baseline"]["hidden_zone_top1_rate"] >= 0.5
    assert metrics["neural_zone"]["hidden_zone_top3_rate"] >= 0.0
    assert (output / "pseudo_zone_report.md").exists()
    assert not {"true_geology", "target_zone", "Geology", "ANCC", "ASTNU"}.intersection(pred.columns)
    assert not {"Geology", "ANCC", "ASTNU"}.intersection(typewell.columns)


def test_pseudo_zone_corr_writes_shuffled_variants_and_train_only_oracle(tmp_path: Path) -> None:
    data = tmp_path / "data"
    for idx in range(4):
        _write_well(data, f"well_{idx}", shift=float(idx))
    output = tmp_path / "pz"
    cfg = PseudoZoneV1Config(rows_per_step=1, vertical_step_ft=10.0, n_folds=2, neural_epochs=1, seed=11)
    run_pseudo_zone_v1(data_dir=data, output_dir=output, cfg=cfg)
    run_pseudo_zone_corr(data_dir=data, output_dir=output, cfg=cfg)

    steps = pd.read_parquet(output / "pseudo_zone_corr_steps.parquet")
    metrics = json.loads((output / "pseudo_zone_metrics.json").read_text())

    assert {"normal", "shuffled_gr"}.issubset(set(steps["gr_variant"]))
    assert {"global", "template_zone_top1", "neural_zone_top1", "oracle_geology"}.issubset(
        set(steps["zone_variant"])
    )
    assert metrics["correlation"]["oracle_geology"]["train_only"] is True
    assert "normal_vs_shuffled_top10_gap" in metrics["correlation"]["neural_zone_top1"]
