from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import FORMATIONS


def _well_name(item: Any) -> str:
    if isinstance(item, Path):
        return item.name.split("__", 1)[0]
    text = str(item)
    if "__" in text:
        return Path(text).name.split("__", 1)[0]
    return text


def grouped_well_folds(
    wells: Sequence[Any] | np.ndarray,
    *,
    n_splits: int = 5,
    seed: int = 42,
    shuffle: bool = True,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return deterministic GroupKFold-style row indices grouped by well id.

    The input can be a row-level well-id array or a list of well paths/names.
    The returned arrays index the original input order.
    """

    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    groups = np.asarray([_well_name(item) for item in wells], dtype=object)
    if groups.size == 0:
        raise ValueError("wells must not be empty")
    unique = np.array(sorted(pd.unique(groups).astype(str)), dtype=object)
    if n_splits > len(unique):
        raise ValueError(
            f"n_splits={n_splits} cannot exceed unique wells={len(unique)}"
        )
    if shuffle:
        unique = np.random.default_rng(seed).permutation(unique)
    split_wells = np.array_split(unique, n_splits)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for valid_wells in split_wells:
        valid_set = set(valid_wells.tolist())
        valid_mask = np.array([str(group) in valid_set for group in groups], dtype=bool)
        valid_idx = np.flatnonzero(valid_mask)
        train_idx = np.flatnonzero(~valid_mask)
        folds.append((train_idx, valid_idx))
    return folds


def mask_validation_surfaces(
    frame: pd.DataFrame,
    *,
    hidden_mask: Iterable[bool] | np.ndarray | None = None,
    surface_columns: Sequence[str] = tuple(FORMATIONS),
    mask_tvt_input: bool = True,
    tvt_input_column: str = "TVT_input",
) -> pd.DataFrame:
    """Return a validation copy that looks like test input.

    Raw validation formation surfaces are masked because the real test CSVs do
    not contain them. If a hidden mask is provided, `TVT_input` is also masked
    on those rows to simulate the competition hidden interval.
    """

    masked = frame.copy()
    for column in surface_columns:
        if column in masked.columns:
            masked[column] = np.nan
    if hidden_mask is not None and mask_tvt_input and tvt_input_column in masked.columns:
        mask = np.asarray(list(hidden_mask), dtype=bool)
        if len(mask) != len(masked):
            raise ValueError(
                f"hidden_mask length {len(mask)} does not match frame length {len(masked)}"
            )
        masked.loc[mask, tvt_input_column] = np.nan
    return masked


def artificial_hidden_masks(
    frame_or_length: pd.DataFrame | int,
    *,
    hidden_fractions: Sequence[float] = (0.3, 0.5, 0.7),
    min_known_rows: int = 25,
) -> dict[str, np.ndarray]:
    """Build suffix masks for fake-hidden backtests.

    `hidden_fractions=(0.3, 0.5, 0.7)` means hide the last 30%, 50%, and 70% of
    rows. The first `min_known_rows` rows are always kept visible.
    """

    n = int(frame_or_length if isinstance(frame_or_length, int) else len(frame_or_length))
    if n <= 0:
        raise ValueError("frame_or_length must contain at least one row")
    if min_known_rows < 1:
        raise ValueError("min_known_rows must be >= 1")
    masks: dict[str, np.ndarray] = {}
    for fraction in hidden_fractions:
        if not 0.0 < float(fraction) < 1.0:
            raise ValueError(f"hidden fraction must be between 0 and 1: {fraction}")
        start = int(np.floor(n * (1.0 - float(fraction))))
        start = min(max(start, min_known_rows), n - 1)
        mask = np.zeros(n, dtype=bool)
        mask[start:] = True
        pct = int(round(float(fraction) * 100))
        masks[f"hide_last_{pct}pct"] = mask
    return masks


def _rmse(pred: np.ndarray, true: np.ndarray) -> float:
    mask = np.isfinite(pred) & np.isfinite(true)
    if not np.any(mask):
        return float("nan")
    err = pred[mask] - true[mask]
    return float(np.sqrt(np.mean(err * err)))


def bucket_metrics(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
    well_ids: Sequence[Any] | np.ndarray,
    *,
    typewell_available: Sequence[bool] | np.ndarray | None = None,
    hidden_row_counts: dict[str, int] | Sequence[int] | np.ndarray | None = None,
    long_hidden_threshold: float | None = None,
) -> dict[str, Any]:
    """Compute global, per-well, typewell, and hidden-length RMSE buckets."""

    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    wells = np.asarray([str(item) for item in well_ids], dtype=object)
    if not (len(true) == len(pred) == len(wells)):
        raise ValueError("y_true, y_pred, and well_ids must have the same length")

    per_well: list[dict[str, Any]] = []
    for well in sorted(pd.unique(wells).astype(str)):
        mask = wells == well
        per_well.append({"well": well, "rows": int(mask.sum()), "rmse": _rmse(pred[mask], true[mask])})
    well_rmse = np.array([item["rmse"] for item in per_well], dtype=float)
    finite_well_rmse = well_rmse[np.isfinite(well_rmse)]

    result: dict[str, Any] = {
        "global_rmse": _rmse(pred, true),
        "well_count": int(len(per_well)),
        "mean_well_rmse": float(np.mean(finite_well_rmse)) if len(finite_well_rmse) else float("nan"),
        "median_well_rmse": float(np.median(finite_well_rmse)) if len(finite_well_rmse) else float("nan"),
        "p90_well_rmse": float(np.percentile(finite_well_rmse, 90)) if len(finite_well_rmse) else float("nan"),
        "p95_well_rmse": float(np.percentile(finite_well_rmse, 95)) if len(finite_well_rmse) else float("nan"),
        "worst_well_rmse": float(np.max(finite_well_rmse)) if len(finite_well_rmse) else float("nan"),
        "worst_wells": sorted(per_well, key=lambda item: item["rmse"], reverse=True)[:10],
    }

    if typewell_available is not None:
        available = np.asarray(typewell_available, dtype=bool)
        if len(available) != len(wells):
            raise ValueError("typewell_available must match y_true length")
        result["typewell_rmse"] = _rmse(pred[available], true[available])
        result["no_typewell_rmse"] = _rmse(pred[~available], true[~available])

    if hidden_row_counts is not None:
        if isinstance(hidden_row_counts, dict):
            counts = np.asarray([hidden_row_counts[str(well)] for well in wells], dtype=float)
        else:
            counts = np.asarray(hidden_row_counts, dtype=float)
            if len(counts) != len(wells):
                raise ValueError("hidden_row_counts sequence must match y_true length")
        threshold = (
            float(np.nanmedian(counts))
            if long_hidden_threshold is None
            else float(long_hidden_threshold)
        )
        long_mask = counts >= threshold
        result["hidden_rows_median"] = threshold
        result["long_hidden_rmse"] = _rmse(pred[long_mask], true[long_mask])
        result["short_hidden_rmse"] = _rmse(pred[~long_mask], true[~long_mask])

    return result


def _roughness(values: np.ndarray) -> tuple[float, float]:
    if len(values) < 2:
        return 0.0, 0.0
    slope = np.diff(values)
    slope_abs = float(np.nanmean(np.abs(slope))) if len(slope) else 0.0
    if len(slope) < 2:
        return slope_abs, 0.0
    curvature = np.diff(slope)
    return slope_abs, float(np.nanmean(np.abs(curvature))) if len(curvature) else 0.0


def _ratio(num: float, den: float) -> float:
    if not np.isfinite(num):
        return float("nan")
    if not np.isfinite(den) or abs(den) < 1e-12:
        return float("inf") if abs(num) > 1e-12 else 1.0
    return float(num / den)


def path_shift_metrics(
    candidate: Sequence[float] | np.ndarray,
    anchor: Sequence[float] | np.ndarray,
    well_ids: Sequence[Any] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Summarize per-well path shift against a safe anchor prediction."""

    cand = np.asarray(candidate, dtype=float)
    base = np.asarray(anchor, dtype=float)
    if len(cand) != len(base):
        raise ValueError("candidate and anchor must have the same length")
    wells = (
        np.asarray(["__all__"] * len(cand), dtype=object)
        if well_ids is None
        else np.asarray([str(item) for item in well_ids], dtype=object)
    )
    if len(wells) != len(cand):
        raise ValueError("well_ids must match candidate length")

    shift = cand - base
    abs_shift = np.abs(shift)
    per_well: list[dict[str, Any]] = []
    for well in sorted(pd.unique(wells).astype(str)):
        mask = wells == well
        w_cand = cand[mask]
        w_base = base[mask]
        w_shift = w_cand - w_base
        w_abs = np.abs(w_shift)
        cand_slope, cand_curv = _roughness(w_cand)
        base_slope, base_curv = _roughness(w_base)
        per_well.append(
            {
                "well": well,
                "rows": int(mask.sum()),
                "median_shift": float(np.nanmedian(w_shift)),
                "median_abs_shift": float(np.nanmedian(w_abs)),
                "p95_abs_shift": float(np.nanpercentile(w_abs, 95)),
                "max_abs_shift": float(np.nanmax(w_abs)),
                "endpoint_shift": float(w_cand[-1] - w_base[-1]) if len(w_cand) else float("nan"),
                "endpoint_abs_shift": float(abs(w_cand[-1] - w_base[-1])) if len(w_cand) else float("nan"),
                "slope_ratio": _ratio(cand_slope, base_slope),
                "curvature_ratio": _ratio(cand_curv, base_curv),
                "same_sign_fraction": float(max(np.mean(w_shift >= 0), np.mean(w_shift <= 0))),
            }
        )

    return {
        "rows": int(len(cand)),
        "wells": int(len(per_well)),
        "median_shift": float(np.nanmedian(shift)),
        "median_abs_shift": float(np.nanmedian(abs_shift)),
        "mean_abs_shift": float(np.nanmean(abs_shift)),
        "p95_abs_shift": float(np.nanpercentile(abs_shift, 95)),
        "max_abs_shift": float(np.nanmax(abs_shift)),
        "per_well": per_well,
    }
