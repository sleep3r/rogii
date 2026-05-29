"""Full-length OOF diagnostics and per-well error analysis."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SOURCE_KEYS = {
    "base": "tvt_base",
    "linear": "tvt_linear",
    "last": "tvt_last",
    "hmm": "tvt_hmm",
    "dtw": "tvt_dtw",
    "neighbor": "tvt_neighbor",
}


def rmse_sse_n(pred: np.ndarray, true: np.ndarray, mask: np.ndarray) -> tuple[float, float, int]:
    """Return RMSE, squared-error sum, and row count under a finite-value mask."""
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool) & np.isfinite(pred) & np.isfinite(true)
    n = int(mask.sum())
    if n == 0:
        return float("nan"), 0.0, 0
    diff = pred[mask] - true[mask]
    sse = float(np.sum(diff**2))
    return float(math.sqrt(sse / n)), sse, n


def row_weighted_rmse(df: pd.DataFrame, source: str) -> float:
    """Compute source RMSE weighted by hidden rows, not by number of wells."""
    if {"source", "sse", "n"}.issubset(df.columns):
        sub = df[df["source"] == source]
        sse = float(sub["sse"].sum())
        n = int(sub["n"].sum())
    else:
        sse_col = f"{source}_sse"
        n_col = f"{source}_n"
        if sse_col not in df.columns or n_col not in df.columns:
            return float("nan")
        sse = float(df[sse_col].sum())
        n = int(df[n_col].sum())
    return float(math.sqrt(sse / n)) if n > 0 else float("nan")


def summarize_well_errors(
    *,
    fold: int,
    well_id: str,
    data: Mapping[str, np.ndarray],
    nn_pred: np.ndarray,
) -> dict[str, float | int | str]:
    """Build one per-well diagnostics row from a cache npz/dict and NN prediction."""
    true = np.asarray(data["tvt_true"], dtype=np.float32)
    hidden = np.asarray(data["hidden_mask"], dtype=np.float32) > 0.5
    gr_valid = np.asarray(data.get("gr_valid", np.ones_like(true)), dtype=np.float32) > 0.5

    row: dict[str, float | int | str] = {
        "fold": int(fold),
        "well_id": str(well_id),
        "n_rows": int(len(true)),
        "hidden_rows": int(hidden.sum()),
        "hidden_ratio": float(hidden.mean()) if len(hidden) else float("nan"),
        "gr_valid_ratio": float(gr_valid.mean()) if len(gr_valid) else float("nan"),
    }
    if hidden.any():
        row["true_range_hidden"] = float(np.nanmax(true[hidden]) - np.nanmin(true[hidden]))
        row["true_std_hidden"] = float(np.nanstd(true[hidden]))
    else:
        row["true_range_hidden"] = float("nan")
        row["true_std_hidden"] = float("nan")

    nn_rmse, nn_sse, nn_n = rmse_sse_n(nn_pred, true, hidden)
    row.update({"nn_rmse": nn_rmse, "nn_sse": nn_sse, "nn_n": nn_n})

    best_prior = "-"
    best_prior_rmse = float("inf")
    for source, key in SOURCE_KEYS.items():
        if key not in data:
            continue
        rmse, sse, n = rmse_sse_n(np.asarray(data[key]), true, hidden)
        row[f"{source}_rmse"] = rmse
        row[f"{source}_sse"] = sse
        row[f"{source}_n"] = n
        if math.isfinite(rmse) and rmse < best_prior_rmse:
            best_prior = source
            best_prior_rmse = rmse

    row["best_prior"] = best_prior
    row["best_prior_rmse"] = best_prior_rmse
    row["nn_minus_best_prior"] = nn_rmse - best_prior_rmse if math.isfinite(best_prior_rmse) else float("nan")
    if "tvt_base" in data:
        row["nn_vs_base_rmse"] = rmse_sse_n(nn_pred, np.asarray(data["tvt_base"]), hidden)[0]

    for key in ["hmm_entropy", "hmm_gr_mismatch", "dtw_score"]:
        if key in data:
            vals = np.asarray(data[key], dtype=np.float64)
            m = hidden & np.isfinite(vals)
            row[f"{key}_hidden_mean"] = float(np.nanmean(vals[m])) if m.any() else float("nan")

    return row


def make_oof_diagnostics(
    cfg,
    output_dir: Path,
    diag_path: Path,
    summary_path: Path,
) -> pd.DataFrame:
    """Run full-length OOF prediction and write per-well diagnostics artifacts."""
    from bphwt.infer.predict_nn import load_model, predict_well
    from bphwt.train.train_fold import resolve_device

    output_dir = Path(output_dir)
    diag_path = Path(diag_path)
    summary_path = Path(summary_path)
    cache_dir = cfg.resolved_cache_dir() / "train"
    cv_summary_path = output_dir / "cv_summary.json"
    if not cv_summary_path.exists():
        raise FileNotFoundError(f"Missing CV summary: {cv_summary_path}")

    with open(cv_summary_path) as f:
        cv_summary = json.load(f)

    device = resolve_device(cfg.infer.device)
    rows = []
    for fold_result in cv_summary.get("fold_results", []):
        fold = int(fold_result["fold"])
        ckpt = output_dir / f"fold_{fold}" / "best_ema.pt"
        if not ckpt.exists():
            logger.warning("No checkpoint for fold %s at %s", fold, ckpt)
            continue

        model = load_model(ckpt, device)
        for well_id in fold_result.get("val_ids", []):
            npz_path = cache_dir / f"{well_id}.npz"
            if not npz_path.exists():
                logger.warning("No train cache for validation well %s", well_id)
                continue

            pred = predict_well(model, npz_path, device, cfg)
            data = np.load(npz_path, allow_pickle=True)
            row = summarize_well_errors(
                fold=fold,
                well_id=well_id,
                data=data,
                nn_pred=pred["tvt_pred"],
            )
            row["cv_val_rmse"] = float(fold_result.get("val_rmse", float("nan")))
            row["cv_val_loss"] = float(fold_result.get("val_loss", float("nan")))
            rows.append(row)

    df = pd.DataFrame(rows)
    diag_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(diag_path, index=False)

    summary = build_diagnostics_summary(df)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    _log_diagnostics_tables(df, diag_path, summary_path)
    return df


def build_diagnostics_summary(df: pd.DataFrame) -> dict:
    """Build JSON-friendly row-weighted OOF summary."""
    sources = ["nn", *SOURCE_KEYS.keys()]
    overall = {
        source: row_weighted_rmse(df, source)
        for source in sources
        if f"{source}_sse" in df.columns or source == "nn"
    }

    by_fold = []
    if "fold" in df.columns:
        for fold, sub in df.groupby("fold", sort=True):
            row = {
                "fold": int(fold),
                "wells": int(len(sub)),
                "hidden_rows": int(sub["hidden_rows"].sum()) if "hidden_rows" in sub else 0,
                "cv_val_rmse": float(sub["cv_val_rmse"].iloc[0]) if "cv_val_rmse" in sub and len(sub) else float("nan"),
            }
            for source in sources:
                rmse = row_weighted_rmse(sub, source)
                if math.isfinite(rmse):
                    row[f"{source}_rmse"] = rmse
            by_fold.append(row)

    worst_cols = [
        "fold",
        "well_id",
        "hidden_rows",
        "gr_valid_ratio",
        "nn_rmse",
        "base_rmse",
        "linear_rmse",
        "hmm_rmse",
        "best_prior",
        "best_prior_rmse",
        "nn_minus_best_prior",
    ]
    worst_cols = [c for c in worst_cols if c in df.columns]
    worst = (
        df.sort_values("nn_rmse", ascending=False)[worst_cols].head(20).replace({np.nan: None}).to_dict("records")
        if len(df)
        else []
    )

    return {
        "overall_row_weighted_rmse": overall,
        "by_fold": by_fold,
        "worst_wells": worst,
    }


def _log_diagnostics_tables(df: pd.DataFrame, diag_path: Path, summary_path: Path) -> None:
    if df.empty:
        logger.warning("OOF diagnostics produced no rows")
        return

    sources = ["nn", *SOURCE_KEYS.keys()]
    source_rows = []
    for source in sources:
        rmse = row_weighted_rmse(df, source)
        if math.isfinite(rmse):
            source_rows.append({"source": source, "rmse": rmse})
    source_df = pd.DataFrame(source_rows).sort_values("rmse")

    fold_rows = []
    for fold, sub in df.groupby("fold", sort=True):
        row = {
            "fold": int(fold),
            "cv": float(sub["cv_val_rmse"].iloc[0]) if "cv_val_rmse" in sub else float("nan"),
            "full_nn": row_weighted_rmse(sub, "nn"),
            "base": row_weighted_rmse(sub, "base"),
            "linear": row_weighted_rmse(sub, "linear"),
            "hmm": row_weighted_rmse(sub, "hmm"),
            "hidden": int(sub["hidden_rows"].sum()),
        }
        fold_rows.append(row)
    fold_df = pd.DataFrame(fold_rows)

    worst_cols = [
        "fold",
        "well_id",
        "hidden_rows",
        "gr_valid_ratio",
        "nn_rmse",
        "base_rmse",
        "linear_rmse",
        "hmm_rmse",
        "best_prior",
        "best_prior_rmse",
    ]
    worst_cols = [c for c in worst_cols if c in df.columns]
    worst_df = df.sort_values("nn_rmse", ascending=False)[worst_cols].head(10)

    def float_fmt(x: float) -> str:
        return f"{x:9.4f}"

    logger.info("\nOOF SOURCE RMSE (full hidden rows)\n%s", source_df.to_string(index=False, float_format=float_fmt))
    logger.info("\nOOF FOLD RMSE: window validation vs full hidden rows\n%s", fold_df.to_string(index=False, float_format=float_fmt))
    logger.info("\nWORST VALIDATION WELLS\n%s", worst_df.to_string(index=False, float_format=float_fmt))
    logger.info("Saved OOF diagnostics: %s", diag_path)
    logger.info("Saved OOF diagnostics summary: %s", summary_path)
