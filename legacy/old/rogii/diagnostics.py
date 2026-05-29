from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def rmse_value(pred: np.ndarray, true: np.ndarray) -> float | None:
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    valid = np.isfinite(pred) & np.isfinite(true)
    if not valid.any():
        return None
    return float(np.sqrt(np.mean((pred[valid] - true[valid]) ** 2)))


def regression_diagnostics(
    pred: np.ndarray,
    y_true: np.ndarray,
    groups: np.ndarray,
    features: pd.DataFrame | None = None,
) -> dict[str, Any]:
    pred = np.asarray(pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    groups = np.asarray(groups)
    frame = pd.DataFrame(
        {
            "group": groups,
            "pred": pred,
            "true": y_true,
            "sqerr": (pred - y_true) ** 2,
        }
    )
    grouped = frame.groupby("group", sort=True)
    per_well = grouped.agg(rows=("sqerr", "size"), mse=("sqerr", "mean")).reset_index()
    per_well["rmse"] = np.sqrt(per_well["mse"])
    well_rmse = per_well["rmse"].to_numpy(dtype=float)

    diagnostics: dict[str, Any] = {
        "global_rmse": rmse_value(pred, y_true),
        "well_count": int(len(per_well)),
        "mean_well_rmse": finite_float(np.nanmean(well_rmse)),
        "median_well_rmse": finite_float(np.nanmedian(well_rmse)),
        "p90_well_rmse": finite_float(np.nanpercentile(well_rmse, 90)),
        "p95_well_rmse": finite_float(np.nanpercentile(well_rmse, 95)),
        "worst_well_rmse": finite_float(np.nanmax(well_rmse)),
        "worst_wells": per_well.sort_values("rmse", ascending=False)
        .head(10)[["group", "rows", "rmse"]]
        .to_dict("records"),
    }

    row_counts = dict(zip(per_well["group"], per_well["rows"], strict=True))
    median_rows = float(np.median(list(row_counts.values()))) if row_counts else 0.0
    long_groups = {group for group, rows in row_counts.items() if rows > median_rows}
    short_groups = set(row_counts) - long_groups
    diagnostics["hidden_rows_median"] = median_rows
    diagnostics["long_hidden_rmse"] = slice_rmse(frame, long_groups)
    diagnostics["short_hidden_rmse"] = slice_rmse(frame, short_groups)

    diagnostics["no_typewell_rmse"] = None
    diagnostics["typewell_rmse"] = None
    if features is not None and "typewell_tvt_range" in features.columns:
        typewell_range = features["typewell_tvt_range"].to_numpy(dtype=float)
        by_group = (
            pd.DataFrame({"group": groups, "has_typewell": typewell_range > 0.0})
            .groupby("group", sort=False)["has_typewell"]
            .max()
        )
        no_typewell_groups = set(by_group.index[~by_group.to_numpy(dtype=bool)])
        typewell_groups = set(by_group.index[by_group.to_numpy(dtype=bool)])
        diagnostics["no_typewell_rmse"] = slice_rmse(frame, no_typewell_groups)
        diagnostics["typewell_rmse"] = slice_rmse(frame, typewell_groups)

    return diagnostics


def slice_rmse(frame: pd.DataFrame, groups: set[Any]) -> float | None:
    if not groups:
        return None
    sliced = frame[frame["group"].isin(groups)]
    if sliced.empty:
        return None
    return finite_float(np.sqrt(float(sliced["sqerr"].mean())))


def git_hash(repo: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        )
    except Exception:
        return ""
    return completed.stdout.strip()
