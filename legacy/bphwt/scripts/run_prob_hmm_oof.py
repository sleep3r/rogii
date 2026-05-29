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

from bphwt.priors.prob_residual_hmm import (
    ProbResidualHMMConfig,
    blend_predictions,
    infer_residual_hmm,
    make_base_tvt,
)

HMM_CONFIGS: list[dict[str, Any]] = [
    {
        "name": "safe_60",
        "residual_range_ft": 60.0,
        "sigma_rw": 0.8,
        "sigma_base": 25.0,
        "sigma_gr_z": 0.8,
        "blend_alpha": 0.5,
    },
    {
        "name": "mid_80",
        "residual_range_ft": 80.0,
        "sigma_rw": 1.25,
        "sigma_base": 35.0,
        "sigma_gr_z": 0.75,
        "blend_alpha": 0.7,
    },
    {
        "name": "aggr_100",
        "residual_range_ft": 100.0,
        "sigma_rw": 1.8,
        "sigma_base": 50.0,
        "sigma_gr_z": 0.65,
        "blend_alpha": 0.85,
    },
]

DEFAULT_BLEND_ALPHAS = [0.25, 0.5, 0.7, 0.85, 1.0]


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

    blend_alphas = _parse_blend_alphas(getattr(args, "blend_alphas", ""))
    per_rows: list[dict[str, Any]] = []

    iterator = tqdm(paths, desc="prob_hmm_oof", disable=bool(getattr(args, "no_progress", False)))
    for hw_path in iterator:
        well_id = hw_path.name.removesuffix("__horizontal_well.csv")
        tw_path = hw_path.with_name(f"{well_id}__typewell.csv")
        if not tw_path.exists():
            raise FileNotFoundError(f"Missing typewell CSV for {well_id}: {tw_path}")
        well = _load_train_well(hw_path, tw_path)

        for cfg_spec in HMM_CONFIGS:
            cfg_name = str(cfg_spec["name"])
            cfg_kwargs = {k: v for k, v in cfg_spec.items() if k != "name"}
            cfg = ProbResidualHMMConfig(**cfg_kwargs)
            result = infer_residual_hmm(
                well["md"],
                well["gr"],
                well["tvt_input"],
                well["tw_tvt"],
                well["tw_gr"],
                cfg,
            )
            for alpha in blend_alphas:
                per_rows.append(_score_result(well_id, cfg_name, alpha, well, result))

    per_well = pd.DataFrame(per_rows)
    per_well_path = out_dir / "per_well.csv"
    per_well.to_csv(per_well_path, index=False)

    experiments = _experiment_summary(per_well)
    experiments_path = out_dir / "experiments.csv"
    experiments.to_csv(experiments_path, index=False)

    per_well_best = _best_per_well(per_well)
    best_path = out_dir / "per_well_best.csv"
    per_well_best.to_csv(best_path, index=False)

    summary = {
        "n_wells": int(per_well["well_id"].nunique()),
        "n_experiments": int(len(experiments)),
        "best_experiment": _best_experiment(experiments),
        "outputs": {
            "per_well": str(per_well_path),
            "experiments": str(experiments_path),
            "per_well_best": str(best_path),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _load_train_well(hw_path: Path, tw_path: Path) -> dict[str, np.ndarray]:
    hw = pd.read_csv(hw_path)
    tw = pd.read_csv(tw_path)
    required_hw = {"MD", "GR", "TVT_input", "TVT"}
    required_tw = {"TVT", "GR"}
    missing_hw = sorted(required_hw.difference(hw.columns))
    missing_tw = sorted(required_tw.difference(tw.columns))
    if missing_hw:
        raise ValueError(f"{hw_path} is missing required columns: {missing_hw}")
    if missing_tw:
        raise ValueError(f"{tw_path} is missing required columns: {missing_tw}")

    md = hw["MD"].to_numpy(dtype=np.float64)
    tvt_input = hw["TVT_input"].to_numpy(dtype=np.float64)
    tvt_true = hw["TVT"].to_numpy(dtype=np.float64)
    hidden = ~np.isfinite(tvt_input) & np.isfinite(tvt_true)
    base, _ = make_base_tvt(tvt_input, md)
    return {
        "md": md,
        "gr": hw["GR"].to_numpy(dtype=np.float64),
        "tvt_input": tvt_input,
        "tvt_true": tvt_true,
        "hidden": hidden,
        "base": base,
        "tw_tvt": tw["TVT"].to_numpy(dtype=np.float64),
        "tw_gr": tw["GR"].to_numpy(dtype=np.float64),
        "gr_valid_ratio": np.array([_finite_ratio(hw["GR"].to_numpy(dtype=np.float64))]),
        "hidden_max_run": np.array([_max_true_run(hidden)]),
    }


def _score_result(
    well_id: str,
    cfg_name: str,
    alpha: float,
    well: dict[str, np.ndarray],
    result: dict[str, Any],
) -> dict[str, Any]:
    hidden = well["hidden"].astype(bool)
    y = well["tvt_true"][hidden]
    base = result["pred_base"]
    pred_mean = result["pred_mean"]
    pred_viterbi = result["pred_viterbi"]
    pred_blend = blend_predictions(
        base,
        pred_mean,
        alpha=alpha,
        confidence=float(result["confidence"]),
        tvt_input=well["tvt_input"],
    )

    posterior_std = np.asarray(result["posterior_std"], dtype=np.float64)
    std_hidden = posterior_std[hidden]
    gr_valid_ratio = float(well["gr_valid_ratio"][0])
    hidden_max_run = int(well["hidden_max_run"][0])
    experiment = f"{cfg_name}_a{str(alpha).replace('.', 'p')}"
    rmse_base, sse_base = _rmse_sse(base[hidden], y)
    rmse_mean, sse_mean = _rmse_sse(pred_mean[hidden], y)
    rmse_viterbi, sse_viterbi = _rmse_sse(pred_viterbi[hidden], y)
    rmse_blend, sse_blend = _rmse_sse(pred_blend[hidden], y)

    return {
        "experiment": experiment,
        "hmm_config": cfg_name,
        "blend_alpha": float(alpha),
        "well_id": well_id,
        "n_rows": int(len(well["md"])),
        "n_hidden": int(hidden.sum()),
        "hidden_max_run": hidden_max_run,
        "gr_valid_ratio": gr_valid_ratio,
        "posterior_std_hidden_mean": float(np.mean(std_hidden)) if std_hidden.size else float("nan"),
        "ll_gain_vs_base": float(result["ll_gain_vs_base"]),
        "log_evidence": float(result["log_evidence"]),
        "confidence": float(result["confidence"]),
        "rmse_base": rmse_base,
        "rmse_hmm_mean": rmse_mean,
        "rmse_hmm_viterbi": rmse_viterbi,
        "rmse_blend": rmse_blend,
        "gain_vs_base": float(rmse_base - rmse_blend) if np.isfinite(rmse_base + rmse_blend) else float("nan"),
        "improved": bool(np.isfinite(rmse_base + rmse_blend) and rmse_blend < rmse_base),
        "sse_base": sse_base,
        "sse_hmm_mean": sse_mean,
        "sse_hmm_viterbi": sse_viterbi,
        "sse_blend": sse_blend,
    }


def _experiment_summary(per_well: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for experiment, sub in per_well.groupby("experiment", sort=True):
        rows.append(
            {
                "experiment": experiment,
                "hmm_config": str(sub["hmm_config"].iloc[0]),
                "blend_alpha": float(sub["blend_alpha"].iloc[0]),
                "n_wells": int(len(sub)),
                "n_hidden_rows": int(sub["n_hidden"].sum()),
                "rmse_base": _group_rmse(sub, "sse_base"),
                "rmse_hmm_mean": _group_rmse(sub, "sse_hmm_mean"),
                "rmse_hmm_viterbi": _group_rmse(sub, "sse_hmm_viterbi"),
                "rmse_blend": _group_rmse(sub, "sse_blend"),
                "gain_vs_base": _group_rmse(sub, "sse_base") - _group_rmse(sub, "sse_blend"),
                "wells_improved_frac": float(sub["improved"].mean()) if len(sub) else float("nan"),
                "rmse_base_high_gr_valid": _group_rmse(sub[sub["gr_valid_ratio"] > 0.6], "sse_base"),
                "rmse_blend_high_gr_valid": _group_rmse(sub[sub["gr_valid_ratio"] > 0.6], "sse_blend"),
                "rmse_base_low_gr_valid": _group_rmse(sub[sub["gr_valid_ratio"] < 0.2], "sse_base"),
                "rmse_blend_low_gr_valid": _group_rmse(sub[sub["gr_valid_ratio"] < 0.2], "sse_blend"),
                "rmse_base_long_hidden": _group_rmse(sub[sub["hidden_max_run"] >= 1000], "sse_base"),
                "rmse_blend_long_hidden": _group_rmse(sub[sub["hidden_max_run"] >= 1000], "sse_blend"),
                "rmse_blend_low_posterior_std": _group_rmse(
                    sub[sub["posterior_std_hidden_mean"] <= 20.0],
                    "sse_blend",
                ),
                "rmse_blend_high_posterior_std": _group_rmse(
                    sub[sub["posterior_std_hidden_mean"] > 20.0],
                    "sse_blend",
                ),
                "rmse_blend_ll_gain_positive": _group_rmse(sub[sub["ll_gain_vs_base"] > 0.0], "sse_blend"),
                "rmse_blend_ll_gain_negative": _group_rmse(sub[sub["ll_gain_vs_base"] <= 0.0], "sse_blend"),
            }
        )
    return pd.DataFrame(rows).sort_values("rmse_blend", na_position="last").reset_index(drop=True)


def _best_per_well(per_well: pd.DataFrame) -> pd.DataFrame:
    ordered = per_well.sort_values(["well_id", "rmse_blend"], na_position="last")
    return ordered.groupby("well_id", as_index=False, sort=True).head(1).reset_index(drop=True)


def _best_experiment(experiments: pd.DataFrame) -> dict[str, Any]:
    if experiments.empty:
        return {}
    row = experiments.sort_values("rmse_blend", na_position="last").iloc[0]
    return {
        "experiment": str(row["experiment"]),
        "rmse_blend": float(row["rmse_blend"]),
        "gain_vs_base": float(row["gain_vs_base"]),
    }


def _parse_blend_alphas(value: str) -> list[float]:
    if not value:
        return list(DEFAULT_BLEND_ALPHAS)
    alphas = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not alphas:
        raise ValueError("At least one blend alpha is required")
    return alphas


def _rmse_sse(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    pred_arr = np.asarray(pred, dtype=np.float64)
    target_arr = np.asarray(target, dtype=np.float64)
    valid = np.isfinite(pred_arr) & np.isfinite(target_arr)
    if not valid.any():
        return float("nan"), 0.0
    diff = pred_arr[valid] - target_arr[valid]
    sse = float(np.sum(diff * diff))
    return float(np.sqrt(np.mean(diff * diff))), sse


def _group_rmse(df: pd.DataFrame, sse_col: str) -> float:
    if df.empty:
        return float("nan")
    n = int(df["n_hidden"].sum())
    if n <= 0:
        return float("nan")
    return float(np.sqrt(float(df[sse_col].sum()) / n))


def _finite_ratio(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float64)
    return float(np.isfinite(arr).mean()) if arr.size else 0.0


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
    parser = argparse.ArgumentParser(description="Run standalone Bayesian residual HMM OOF diagnostics.")
    parser.add_argument("--data-dir", default="data/train")
    parser.add_argument("--out-dir", default="artifacts/prob_hmm")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--blend-alphas", default=",".join(str(x) for x in DEFAULT_BLEND_ALPHAS))
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run(args)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
