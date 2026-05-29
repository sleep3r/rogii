from __future__ import annotations

import numpy as np
import pandas as pd

from rogii.validation import (
    artificial_hidden_masks,
    bucket_metrics,
    grouped_well_folds,
    mask_validation_surfaces,
    path_shift_metrics,
)


def test_grouped_well_folds_keep_wells_disjoint() -> None:
    wells = np.array(["a", "a", "b", "b", "c", "c", "d", "d"])

    folds = grouped_well_folds(wells, n_splits=2, seed=7)

    assert len(folds) == 2
    seen_valid: set[str] = set()
    for train_idx, valid_idx in folds:
        train_wells = set(wells[train_idx])
        valid_wells = set(wells[valid_idx])
        assert train_wells.isdisjoint(valid_wells)
        seen_valid.update(valid_wells)
    assert seen_valid == {"a", "b", "c", "d"}


def test_mask_validation_surfaces_masks_formations_and_hidden_tvt_input() -> None:
    frame = pd.DataFrame(
        {
            "TVT_input": [1.0, 2.0, 3.0],
            "TVT": [1.0, 2.0, 3.0],
            "ANCC": [10.0, 11.0, 12.0],
            "GR": [80.0, 81.0, 82.0],
        }
    )

    masked = mask_validation_surfaces(frame, hidden_mask=[False, True, True])

    assert masked["ANCC"].isna().all()
    assert masked["TVT_input"].tolist()[0] == 1.0
    assert masked["TVT_input"].isna().tolist()[1:] == [True, True]
    assert masked["TVT"].tolist() == [1.0, 2.0, 3.0]
    assert frame["ANCC"].notna().all()


def test_artificial_hidden_masks_hide_suffixes() -> None:
    masks = artificial_hidden_masks(10, hidden_fractions=(0.3, 0.5), min_known_rows=2)

    assert masks["hide_last_30pct"].tolist() == [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
    ]
    assert masks["hide_last_50pct"].sum() == 5


def test_bucket_and_path_shift_metrics() -> None:
    y_true = np.array([1.0, 2.0, 10.0, 12.0])
    y_pred = np.array([1.0, 4.0, 11.0, 14.0])
    wells = np.array(["a", "a", "b", "b"])

    metrics = bucket_metrics(
        y_true,
        y_pred,
        wells,
        typewell_available=np.array([True, True, False, False]),
        hidden_row_counts={"a": 2, "b": 5},
    )
    assert metrics["well_count"] == 2
    assert metrics["worst_wells"][0]["well"] == "b"
    assert np.isfinite(metrics["typewell_rmse"])
    assert np.isfinite(metrics["no_typewell_rmse"])
    assert np.isfinite(metrics["long_hidden_rmse"])
    assert np.isfinite(metrics["short_hidden_rmse"])

    shifts = path_shift_metrics(
        candidate=np.array([2.0, 3.0, 14.0, 16.0]),
        anchor=np.array([1.0, 2.0, 10.0, 12.0]),
        well_ids=wells,
    )
    assert shifts["wells"] == 2
    assert shifts["median_abs_shift"] == 2.5
    assert shifts["per_well"][0]["well"] == "a"
