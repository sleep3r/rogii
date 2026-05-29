from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from tqdm import tqdm

from bphwt.features.candidate_features import _backward_fill, _forward_fill
from bphwt.infer.optimize_curve import optimize_well_curve


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if pred.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def fill_tvt_input(tvt_input: np.ndarray) -> np.ndarray:
    filled = _forward_fill(tvt_input.astype(np.float32).copy())
    filled = _backward_fill(filled)
    return filled.astype(np.float32)


def choose_base(d: dict[str, np.ndarray]) -> np.ndarray:
    linear = d["tvt_linear"].astype(np.float32)
    if "tvt_dtw" not in d:
        return linear

    dtw = d["tvt_dtw"].astype(np.float32)
    if "dtw_score" in d:
        score = d["dtw_score"].astype(np.float32)
        finite = np.isfinite(score)
        if not finite.any():
            return linear
        threshold = float(np.nanpercentile(score[finite], 70))
        good = finite & (score < threshold)
        if float(good.mean()) < 0.20:
            return linear

    return (0.85 * linear + 0.15 * dtw).astype(np.float32)


def predict_one(npz_path: Path, args: argparse.Namespace | SimpleNamespace) -> dict[str, Any]:
    d = _load_npz(npz_path)
    meta = d["_meta"].item() if "_meta" in d else {}
    well_id = str(meta.get("well_id", npz_path.stem)) if isinstance(meta, dict) else npz_path.stem

    tvt_base = choose_base(d)
    tvt_input = d["tvt_input"].astype(np.float32)
    tvt_input_filled = fill_tvt_input(tvt_input)
    known_mask = d["known_mask"].astype(np.float32)
    hidden_mask = d["hidden_mask"].astype(np.float32)
    gr_valid = d["gr_valid"].astype(np.float32)
    gr_valid_ratio = float(gr_valid.mean())

    if gr_valid_ratio < args.min_gr_valid:
        clip_correction = min(float(args.clip_correction), 4.0)
        lambda_prior = max(float(args.lambda_prior), 2.0)
    else:
        clip_correction = float(args.clip_correction)
        lambda_prior = float(args.lambda_prior)

    log_sigma = np.log(np.full_like(tvt_base, float(args.prior_sigma), dtype=np.float32))
    pred = optimize_well_curve(
        tvt_nn=tvt_base,
        log_sigma=log_sigma,
        md=d["md"].astype(np.float32),
        known_mask=known_mask,
        tvt_input=tvt_input,
        tvt_input_filled=tvt_input_filled,
        gr_obs=d["gr_obs"].astype(np.float32),
        gr_valid=gr_valid,
        tw_tvt=d["tw_tvt"].astype(np.float32),
        tw_gr=d["tw_gr"].astype(np.float32),
        tvt_hmm=None,
        n_knots=int(args.n_knots),
        n_steps=int(args.n_steps),
        lr=float(args.lr),
        lambda_prior=lambda_prior,
        lambda_hmm=0.0,
        lambda_smooth=float(args.lambda_smooth),
        lambda_anchor=float(args.lambda_anchor),
        gr_huber_delta=float(args.gr_huber_delta),
        clip_correction=clip_correction,
    )
    correction = pred - tvt_base
    hm = hidden_mask > 0.5
    gr_rmse_base = forward_gr_rmse(
        tvt_base,
        d["gr_obs"].astype(np.float32),
        gr_valid,
        d["tw_tvt"].astype(np.float32),
        d["tw_gr"].astype(np.float32),
    )
    gr_rmse_bpi = forward_gr_rmse(
        pred,
        d["gr_obs"].astype(np.float32),
        gr_valid,
        d["tw_tvt"].astype(np.float32),
        d["tw_gr"].astype(np.float32),
    )

    row: dict[str, Any] = {
        "well_id": well_id,
        "n_rows": int(len(pred)),
        "n_hidden": int(hidden_mask.sum()),
        "gr_valid_ratio": gr_valid_ratio,
        "correction_abs_mean": float(np.mean(np.abs(correction[hm]))) if hm.any() else float("nan"),
        "correction_abs_p95": float(np.percentile(np.abs(correction[hm]), 95)) if hm.any() else float("nan"),
        "correction_abs_max": float(np.max(np.abs(correction[hm]))) if hm.any() else float("nan"),
        "forward_gr_rmse_base": gr_rmse_base,
        "forward_gr_rmse_bpi": gr_rmse_bpi,
        "forward_gr_rmse_gain": float(gr_rmse_base - gr_rmse_bpi),
        "pred": pred,
        "base": tvt_base,
        "hidden_mask": hidden_mask,
    }

    if "tvt_true" in d:
        tvt_true = d["tvt_true"].astype(np.float32)
        hm = hidden_mask > 0.5
        row["rmse_base"] = rmse(tvt_base[hm], tvt_true[hm])
        row["rmse_bpi"] = rmse(pred[hm], tvt_true[hm])
        row["gain"] = float(row["rmse_base"] - row["rmse_bpi"])

    return row


def forward_gr_rmse(
    tvt_pred: np.ndarray,
    gr_obs: np.ndarray,
    gr_valid: np.ndarray,
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
) -> float:
    gv = (gr_valid > 0.5) & np.isfinite(gr_obs)
    if int(gv.sum()) < 5:
        return float("nan")
    gr_fwd = np.interp(tvt_pred[gv], tw_tvt, tw_gr).astype(np.float32)
    go = gr_obs[gv].astype(np.float32)
    A = np.column_stack([gr_fwd, np.ones_like(gr_fwd)])
    try:
        sol, _, _, _ = np.linalg.lstsq(A, go, rcond=None)
        beta1, beta0 = float(sol[0]), float(sol[1])
        if not (0.2 < abs(beta1) < 5.0):
            beta0, beta1 = 0.0, 1.0
    except Exception:
        beta0, beta1 = 0.0, 1.0
    residual = beta1 * gr_fwd + beta0 - go
    return float(np.sqrt(np.mean(residual**2)))


def run(args: argparse.Namespace) -> dict[str, Any]:
    cache_dir = Path(args.cache_dir)
    npz_paths = sorted(cache_dir.glob("*.npz"))
    if args.limit and args.limit > 0:
        npz_paths = npz_paths[: args.limit]
    if not npz_paths:
        raise FileNotFoundError(f"No .npz files found in {cache_dir}")

    rows: list[dict[str, Any]] = []
    preds_all: list[np.ndarray] = []
    base_all: list[np.ndarray] = []
    y_all: list[np.ndarray] = []

    iterator = tqdm(npz_paths, desc="bpi_spline", disable=args.no_progress)
    for npz_path in iterator:
        result = predict_one(npz_path, args)
        rows.append(_row_without_arrays(result))

        d = _load_npz(npz_path)
        if "tvt_true" in d:
            hm = result["hidden_mask"] > 0.5
            preds_all.append(result["pred"][hm])
            base_all.append(result["base"][hm])
            y_all.append(d["tvt_true"].astype(np.float32)[hm])

    summary: dict[str, Any] = {"cache_dir": str(cache_dir), "n_wells": len(rows), "per_well": rows}
    if y_all:
        pred = np.concatenate(preds_all)
        base = np.concatenate(base_all)
        y = np.concatenate(y_all)
        rmse_base = rmse(base, y)
        rmse_bpi = rmse(pred, y)
        summary["overall"] = {
            "rmse_base": rmse_base,
            "rmse_bpi_spline": rmse_bpi,
            "gain": float(rmse_base - rmse_bpi),
            "n_hidden": int(len(y)),
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(out_path.with_suffix(".csv"), index=False)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run no-NN BPI-Spline OOF diagnostics on a feature cache.")
    parser.add_argument("--cache-dir", required=True, help="Path to cache/train")
    parser.add_argument("--out", default="artifacts/bpi_spline/oof_summary.json")
    parser.add_argument("--limit", type=int, default=0, help="Use first N wells for a quick local smoke run")
    parser.add_argument("--n-knots", type=int, default=32)
    parser.add_argument("--n-steps", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.35)
    parser.add_argument("--prior-sigma", type=float, default=10.0)
    parser.add_argument("--lambda-prior", type=float, default=0.6)
    parser.add_argument("--lambda-smooth", type=float, default=0.03)
    parser.add_argument("--lambda-anchor", type=float, default=100.0)
    parser.add_argument("--clip-correction", type=float, default=10.0)
    parser.add_argument("--gr-huber-delta", type=float, default=15.0)
    parser.add_argument("--min-gr-valid", type=float, default=0.20)
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run(args)
    print(json.dumps(summary.get("overall", {}), indent=2))
    print(f"wrote {args.out} and {Path(args.out).with_suffix('.csv')}")
    return 0


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _row_without_arrays(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not isinstance(value, np.ndarray)}


if __name__ == "__main__":
    raise SystemExit(main())
