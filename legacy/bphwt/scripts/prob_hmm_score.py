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

from bphwt.priors.prob_residual_hmm import make_base_tvt


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    pred_arr = np.asarray(pred, dtype=np.float64)
    target_arr = np.asarray(target, dtype=np.float64)
    valid = np.isfinite(pred_arr) & np.isfinite(target_arr)
    if not valid.any():
        return float("nan")
    return float(np.sqrt(np.mean((pred_arr[valid] - target_arr[valid]) ** 2)))


def run(args: argparse.Namespace | SimpleNamespace) -> dict[str, Any]:
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(data_dir.glob("*__horizontal_well.csv"))
    limit = int(getattr(args, "limit", 0) or 0)
    if limit > 0:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"No horizontal well CSV files found in {data_dir}")

    rows: list[dict[str, Any]] = []
    iterator = tqdm(paths, desc="prob_hmm_base", disable=bool(getattr(args, "no_progress", False)))
    for hw_path in iterator:
        row = score_one_well(hw_path)
        rows.append(row)

    oof = pd.DataFrame(rows)
    oof_path = out_dir / "base_oof.csv"
    oof.to_csv(oof_path, index=False)

    summary = {
        "rmse_base": _aggregate_rmse(rows, "sse_base"),
        "n_wells": int(len(rows)),
        "n_hidden_rows": int(sum(int(r["n_hidden"]) for r in rows)),
        "by_gr_valid_ratio": _bucket_summary(rows, "gr_valid_bucket"),
        "by_hidden_len": _bucket_summary(rows, "hidden_len_bucket"),
    }
    summary_path = out_dir / "base_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def score_one_well(hw_path: Path) -> dict[str, Any]:
    well_id = hw_path.name.removesuffix("__horizontal_well.csv")
    hw = pd.read_csv(hw_path)
    required = {"MD", "TVT_input", "TVT"}
    missing = sorted(required.difference(hw.columns))
    if missing:
        raise ValueError(f"{hw_path} is missing required columns: {missing}")

    md = hw["MD"].to_numpy(dtype=np.float64)
    tvt_input = hw["TVT_input"].to_numpy(dtype=np.float64)
    tvt_true = hw["TVT"].to_numpy(dtype=np.float64)
    base, _ = make_base_tvt(tvt_input, md)
    hidden = ~np.isfinite(tvt_input) & np.isfinite(tvt_true)
    pred_hidden = base[hidden]
    true_hidden = tvt_true[hidden]
    diff = pred_hidden - true_hidden
    sse = float(np.sum(diff * diff)) if diff.size else 0.0
    gr_valid_ratio = _gr_valid_ratio(hw)
    hidden_max_run = _max_true_run(hidden)

    return {
        "well_id": well_id,
        "n_rows": int(len(hw)),
        "n_hidden": int(hidden.sum()),
        "hidden_max_run": int(hidden_max_run),
        "gr_valid_ratio": gr_valid_ratio,
        "gr_valid_bucket": _gr_valid_bucket(gr_valid_ratio),
        "hidden_len_bucket": _hidden_len_bucket(hidden_max_run),
        "rmse_base": rmse(pred_hidden, true_hidden),
        "sse_base": sse,
    }


def _aggregate_rmse(rows: list[dict[str, Any]], sse_key: str) -> float:
    n = sum(int(r["n_hidden"]) for r in rows)
    if n <= 0:
        return float("nan")
    sse = sum(float(r[sse_key]) for r in rows)
    return float(np.sqrt(sse / n))


def _bucket_summary(rows: list[dict[str, Any]], bucket_key: str) -> dict[str, dict[str, float | int]]:
    out: dict[str, dict[str, float | int]] = {}
    for label in sorted({str(r[bucket_key]) for r in rows}):
        sub = [r for r in rows if str(r[bucket_key]) == label]
        out[label] = {
            "rmse_base": _aggregate_rmse(sub, "sse_base"),
            "n_wells": int(len(sub)),
            "n_hidden_rows": int(sum(int(r["n_hidden"]) for r in sub)),
        }
    return out


def _gr_valid_ratio(hw: pd.DataFrame) -> float:
    if "GR" not in hw.columns:
        return 0.0
    gr = hw["GR"].to_numpy(dtype=np.float64)
    return float(np.isfinite(gr).mean()) if gr.size else 0.0


def _gr_valid_bucket(value: float) -> str:
    if value < 0.20:
        return "low_<0.2"
    if value < 0.60:
        return "medium_0.2_0.6"
    return "high_>=0.6"


def _hidden_len_bucket(hidden_max_run: int) -> str:
    if hidden_max_run < 250:
        return "short_<250"
    if hidden_max_run < 1000:
        return "medium_250_999"
    return "long_>=1000"


def _max_true_run(mask: np.ndarray) -> int:
    best = 0
    cur = 0
    for value in np.asarray(mask, dtype=bool):
        if value:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score the simple TVT_input baseline on train hidden rows.")
    parser.add_argument("--data-dir", default="data/train")
    parser.add_argument("--out-dir", default="artifacts/prob_hmm")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run(args)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
