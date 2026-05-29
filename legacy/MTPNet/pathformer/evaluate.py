"""PathFormer evaluation: standalone row-level hidden path metrics."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .dataset import WellSample
from .model import PathFormer


TAIL_CLASSES = [
    "G_all_candidates_fail",
    "A_base_b2_level_shift",
    "D_alignment_ambiguity",
    "C_GR_missing_or_noisy",
    "B_long_well_drift",
    "OK_or_mixed",
    "unknown",
]


# ---------------------------------------------------------------------------
# Inference: predict TVT for one batch of wells
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_well(
    model: PathFormer,
    sample: WellSample,
    max_seq_len: int,
    device: torch.device,
) -> np.ndarray:
    """Return predicted TVT (in ft, absolute) for all steps of the well."""
    from .dataset import _pad_sample

    item = _pad_sample(sample, max_seq_len)
    features = item["features"].unsqueeze(0).to(device)    # (1, L, F)
    pad_mask = item["pad_mask"].unsqueeze(0).to(device)    # (1, L)

    model.eval()
    pred_delta = model(features, pad_mask)                  # (1, L)
    pred_delta = pred_delta[0, : sample.seq_len].cpu().numpy()  # (T,)
    pred_tvt = sample.last_known_tvt + pred_delta
    return pred_tvt.astype(np.float32)


def predict_row_predictions(
    model: PathFormer,
    samples: list[WellSample],
    max_seq_len: int,
    device: torch.device,
    *,
    candidate: str = "pathformer_direct",
) -> pd.DataFrame:
    """Return hidden-row PathFormer predictions for artifact/candidate-bank use."""
    rows: list[dict[str, Any]] = []
    for sample in samples:
        pred_tvt = predict_well(model, sample, max_seq_len, device)
        if len(sample.hidden_row_steps) == 0:
            hidden_steps = np.flatnonzero(sample.hidden_mask)
            for step in hidden_steps:
                true_tvt = sample.last_known_tvt + sample.target_delta[int(step)]
                if not np.isfinite(true_tvt):
                    continue
                rows.append(
                    {
                        "id": f"{sample.well_id}_step{int(step)}",
                        "well_id": sample.well_id,
                        "row_idx": int(step),
                        "step": int(step),
                        "TVT": float(true_tvt),
                        "GR": np.nan,
                        "pred_tvt": float(pred_tvt[int(step)]),
                        "candidate": candidate,
                        "tail_class": sample.tail_class,
                    }
                )
            continue
        for row_id, row_idx, step, true_tvt, gr in zip(
            sample.hidden_row_ids,
            sample.hidden_row_idx,
            sample.hidden_row_steps,
            sample.hidden_row_tvt,
            sample.hidden_row_gr,
            strict=True,
        ):
            step_int = int(step)
            if step_int < 0 or step_int >= len(pred_tvt) or not np.isfinite(true_tvt):
                continue
            rows.append(
                {
                    "id": str(row_id),
                    "well_id": sample.well_id,
                    "row_idx": int(row_idx),
                    "step": step_int,
                    "TVT": float(true_tvt),
                    "GR": float(gr) if np.isfinite(gr) else np.nan,
                    "pred_tvt": float(pred_tvt[step_int]),
                    "candidate": candidate,
                    "tail_class": sample.tail_class,
                }
            )
    columns = [
        "id",
        "well_id",
        "row_idx",
        "step",
        "TVT",
        "GR",
        "pred_tvt",
        "candidate",
        "tail_class",
    ]
    return pd.DataFrame(rows, columns=columns)


def _rmse(values: np.ndarray | pd.Series) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(arr))))


# ---------------------------------------------------------------------------
# Evaluate over a list of WellSamples
# ---------------------------------------------------------------------------

def evaluate_samples(
    model: PathFormer,
    samples: list[WellSample],
    max_seq_len: int,
    device: torch.device,
) -> dict[str, Any]:
    """Run model on samples and return standalone PathFormer row-level metrics."""
    rows = predict_row_predictions(
        model, samples, max_seq_len, device, candidate="pathformer_direct"
    )
    if rows.empty:
        return {"n_wells": 0}

    rows = rows[
        np.isfinite(pd.to_numeric(rows["TVT"], errors="coerce"))
        & np.isfinite(pd.to_numeric(rows["pred_tvt"], errors="coerce"))
    ].copy()
    rows["_err"] = pd.to_numeric(rows["pred_tvt"], errors="coerce") - pd.to_numeric(
        rows["TVT"], errors="coerce"
    )
    well_rmse = rows.groupby("well_id")["_err"].apply(_rmse)

    metrics: dict[str, Any] = {
        "n_wells": int(rows["well_id"].nunique()),
        "rows": int(len(rows)),
        "row_rmse_ft": _rmse(rows["_err"]),
        "global_rmse": _rmse(rows["_err"]),
        "mean_well_rmse": float(well_rmse.mean()),
        "p50_well_rmse": float(well_rmse.quantile(0.50)),
        "p90_well_rmse": float(well_rmse.quantile(0.90)),
        "p95_well_rmse": float(well_rmse.quantile(0.95)),
        "worst_well_rmse": float(well_rmse.max()),
    }

    # Per tail-class breakdown
    tail_metrics: dict[str, Any] = {}
    for cls in TAIL_CLASSES:
        sub = rows[rows["tail_class"] == cls]
        if len(sub) == 0:
            continue
        sub_well_rmse = sub.groupby("well_id")["_err"].apply(_rmse)
        tail_metrics[cls] = {
            "n_wells": int(sub["well_id"].nunique()),
            "rows": int(len(sub)),
            "row_rmse_ft": _rmse(sub["_err"]),
            "mean_well_rmse": float(sub_well_rmse.mean()),
            "p90_well_rmse": float(sub_well_rmse.quantile(0.90))
            if len(sub_well_rmse) >= 2
            else float(sub_well_rmse.mean()),
        }
    metrics["tail_class"] = tail_metrics

    return metrics


def log_metrics(metrics: dict[str, Any], prefix: str = "") -> None:
    """Print metrics in a compact human-readable format."""
    p = f"[{prefix}] " if prefix else ""
    print(f"{p}n_wells={metrics.get('n_wells', '?')}  "
          f"row_rmse={metrics.get('row_rmse_ft', float('nan')):.3f}  "
          f"mean_well_rmse={metrics.get('mean_well_rmse', float('nan')):.3f}  "
          f"p50={metrics.get('p50_well_rmse', float('nan')):.3f}  "
          f"p90={metrics.get('p90_well_rmse', float('nan')):.3f}  "
          f"p95={metrics.get('p95_well_rmse', float('nan')):.3f}  "
          f"worst={metrics.get('worst_well_rmse', float('nan')):.2f}")
    tc = metrics.get("tail_class", {})
    for cls in ["G_all_candidates_fail", "A_base_b2_level_shift", "D_alignment_ambiguity"]:
        if cls in tc:
            info = tc[cls]
            print(
                f"  {cls[:30]:30s}  n={info['n_wells']:3d}  "
                f"row_rmse={info['row_rmse_ft']:.3f}  "
                f"mean_well={info['mean_well_rmse']:.3f}"
            )


def save_metrics(metrics: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
