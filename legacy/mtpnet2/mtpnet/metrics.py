"""Evaluation metrics for the offset-state TVT decoder.

Primary metric: pooled row-level RMSE (in feet) over all hidden rows across
all wells.  This is the competition metric and must NOT be replaced by offset
MAE or per-well RMSE.

Usage:
    from mtpnet.metrics import row_rmse, make_scoreboard

    rmse = row_rmse(preds_flat, trues_flat)
    board = make_scoreboard(results_dict)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core metric
# ---------------------------------------------------------------------------

def row_rmse(
    predictions: np.ndarray | list[float],
    targets: np.ndarray | list[float],
) -> float:
    """Pooled row-level RMSE (ft).

    Both arrays are 1-D and correspond to hidden rows across all wells.
    NaN entries in either array are excluded.

    Args:
        predictions : predicted TVT values (H_total,)
        targets     : true TVT values (H_total,)

    Returns:
        RMSE in feet (float).  Returns nan if no valid rows.
    """
    p = np.asarray(predictions, dtype=np.float64).ravel()
    t = np.asarray(targets, dtype=np.float64).ravel()
    if len(p) != len(t):
        raise ValueError(f"Length mismatch: predictions={len(p)}, targets={len(t)}")
    valid = np.isfinite(p) & np.isfinite(t)
    if not np.any(valid):
        return float("nan")
    return float(np.sqrt(np.mean((p[valid] - t[valid]) ** 2)))


def per_well_rmse(
    predictions: list[np.ndarray],
    targets: list[np.ndarray],
    well_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Compute per-well RMSE and return a sorted DataFrame.

    Args:
        predictions : list of (H_i,) arrays (one per well)
        targets     : list of (H_i,) arrays (one per well)
        well_ids    : optional list of well ID strings

    Returns:
        DataFrame with columns [well_id, n_hidden, rmse], sorted by rmse desc.
    """
    rows = []
    for i, (p, t) in enumerate(zip(predictions, targets)):
        wid = well_ids[i] if well_ids is not None else str(i)
        p_ = np.asarray(p, dtype=np.float64)
        t_ = np.asarray(t, dtype=np.float64)
        valid = np.isfinite(p_) & np.isfinite(t_)
        n = int(np.sum(valid))
        rmse = float(np.sqrt(np.mean((p_[valid] - t_[valid]) ** 2))) if n > 0 else float("nan")
        rows.append({"well_id": wid, "n_hidden": n, "rmse": rmse})
    df = pd.DataFrame(rows).sort_values("rmse", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Top-K oracle RMSE (upper bound from model's soft distribution)
# ---------------------------------------------------------------------------

def top_k_oracle_rmse(
    proba: np.ndarray,
    offset_grid: np.ndarray,
    z: np.ndarray,
    anchor_row: int,
    anchor_tvt: float,
    hidden_rows: np.ndarray,
    tvt_true: np.ndarray,
    k: int = 20,
) -> float:
    """Oracle RMSE achievable if we pick the best offset from the top-K predicted bins.

    This is an upper-bound diagnostic: if the true offset bin is consistently
    in the top-K, the model has learned the right distribution.

    Args:
        proba        : (N_OFFSET_BINS,) soft probability over offset bins for this well
        offset_grid  : offset values (N_OFFSET_BINS,)
        z, anchor_row, anchor_tvt, hidden_rows : well geometry
        tvt_true     : (H,) true TVT for hidden rows
        k            : how many top bins to check

    Returns:
        Best RMSE over top-K bins (float).
    """
    from .offsets import tvt_from_global_offset

    top_bins = np.argsort(proba)[::-1][:k]
    best = float("inf")
    for b in top_bins:
        pred = tvt_from_global_offset(z, anchor_row, anchor_tvt, hidden_rows, offset_grid[b])
        r = row_rmse(pred, tvt_true)
        if r < best:
            best = r
    return best


def posterior_mean_rmse(
    proba: np.ndarray,
    offset_grid: np.ndarray,
    z: np.ndarray,
    anchor_row: int,
    anchor_tvt: float,
    hidden_rows: np.ndarray,
    tvt_true: np.ndarray,
) -> float:
    """RMSE using the posterior-mean offset (E[c | proba]).

    Args:
        proba : (N_OFFSET_BINS,) soft probability distribution
        (rest as above)

    Returns:
        RMSE from posterior-mean offset (float).
    """
    from .offsets import tvt_from_global_offset

    p = np.asarray(proba, dtype=np.float64)
    p = p / (p.sum() + 1e-30)
    c_mean = float(np.dot(p, offset_grid))
    pred = tvt_from_global_offset(z, anchor_row, anchor_tvt, hidden_rows, c_mean)
    return row_rmse(pred, tvt_true)


# ---------------------------------------------------------------------------
# Scoreboard
# ---------------------------------------------------------------------------

def make_scoreboard(results: dict[str, float | dict]) -> pd.DataFrame:
    """Format a scoreboard DataFrame from a results dict.

    Args:
        results : mapping of experiment_name → rmse (float) or dict with 'rmse' key

    Returns:
        DataFrame with columns [experiment, rmse] sorted by rmse ascending.
    """
    rows = []
    for name, val in results.items():
        if isinstance(val, dict):
            rmse = val.get("rmse", float("nan"))
        else:
            rmse = float(val)
        rows.append({"experiment": name, "rmse": rmse})
    df = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    return df


def print_scoreboard(results: dict[str, float | dict], title: str = "Scoreboard") -> None:
    """Pretty-print the scoreboard."""
    df = make_scoreboard(results)
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")
    for _, row in df.iterrows():
        print(f"  {row['experiment']:<40s}  {row['rmse']:.4f} ft")
    print(f"{'='*50}\n")


# ---------------------------------------------------------------------------
# Aggregate helpers
# ---------------------------------------------------------------------------

def collect_predictions_and_targets(
    samples: list,   # list[OffsetSample]
    predictions: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten per-well predictions and targets into 1-D arrays.

    Args:
        samples     : list of OffsetSample (must have tvt_true set)
        predictions : list of (H_i,) predicted TVT arrays

    Returns:
        (all_preds, all_trues) — concatenated float64 arrays
    """
    all_preds: list[np.ndarray] = []
    all_trues: list[np.ndarray] = []
    for s, p in zip(samples, predictions):
        t = s.tvt_hidden_true
        if t is None:
            continue
        all_preds.append(np.asarray(p, dtype=np.float64))
        all_trues.append(np.asarray(t, dtype=np.float64))
    return np.concatenate(all_preds), np.concatenate(all_trues)
