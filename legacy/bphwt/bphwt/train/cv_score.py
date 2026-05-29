"""OOF scoring utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_oof_rmse(oof_df: pd.DataFrame) -> float:
    """RMSE on hidden rows only (where TVT_input was NaN)."""
    hidden = oof_df[oof_df["is_hidden"] == 1]
    if len(hidden) == 0:
        return float("inf")
    return float(np.sqrt(np.mean((hidden["tvt_pred"] - hidden["tvt_true"]) ** 2)))


def score_by_buckets(oof_df: pd.DataFrame) -> dict:
    """Break down OOF RMSE by diagnostic buckets."""
    results = {}

    # Overall hidden
    results["overall"] = compute_oof_rmse(oof_df)

    # By fold
    if "fold" in oof_df.columns:
        for fold in sorted(oof_df["fold"].unique()):
            sub = oof_df[oof_df["fold"] == fold]
            results[f"fold_{fold}"] = compute_oof_rmse(sub)

    # By GR valid ratio bucket
    if "gr_valid_ratio" in oof_df.columns:
        for lo, hi, label in [(0, 0.2, "gr_lt20"), (0.2, 0.6, "gr_20_60"), (0.6, 1.01, "gr_gt60")]:
            mask = (oof_df["gr_valid_ratio"] >= lo) & (oof_df["gr_valid_ratio"] < hi)
            sub = oof_df[mask]
            results[label] = compute_oof_rmse(sub) if len(sub) > 0 else float("nan")

    # By hidden ratio
    if "hidden_ratio" in oof_df.columns:
        for lo, hi, label in [
            (0, 0.5, "hidden_lt50"),
            (0.5, 0.8, "hidden_50_80"),
            (0.8, 1.01, "hidden_gt80"),
        ]:
            mask = (oof_df["hidden_ratio"] >= lo) & (oof_df["hidden_ratio"] < hi)
            sub = oof_df[mask]
            results[label] = compute_oof_rmse(sub) if len(sub) > 0 else float("nan")

    return results


def compare_with_baseline(oof_df: pd.DataFrame) -> pd.DataFrame:
    """Compare NN predictions against simple baselines."""
    rows = []
    for label in ["tvt_pred", "tvt_linear", "tvt_hmm"]:
        if label not in oof_df.columns:
            continue
        hidden = oof_df[oof_df["is_hidden"] == 1]
        rmse = float(np.sqrt(np.mean((hidden[label] - hidden["tvt_true"]) ** 2)))
        rows.append({"model": label, "rmse_hidden": rmse})
    return pd.DataFrame(rows)
