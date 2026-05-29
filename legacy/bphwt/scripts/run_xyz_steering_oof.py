from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

from bphwt.features.xyz_steering import XYZSteeringConfig, build_xyz_steering_features
from bphwt.priors.prob_residual_hmm import make_base_tvt

DEFAULT_BLEND_ALPHAS = [0.1, 0.25, 0.5, 0.7, 1.0]


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

    modes = _parse_modes(getattr(args, "modes", "past,full"))
    blend_alphas = _parse_float_list(getattr(args, "blend_alphas", ""), DEFAULT_BLEND_ALPHAS)
    cfg = XYZSteeringConfig()
    wells = _load_wells(paths, cfg, modes=modes, no_progress=bool(getattr(args, "no_progress", False)))
    if len(wells) < 2:
        raise ValueError("Need at least two wells for GroupKFold OOF")

    base_rows = [_score_base(well) for well in wells]
    per_well_rows: list[dict[str, Any]] = [{"experiment": "base", **row} for row in base_rows]

    n_splits = max(2, min(int(getattr(args, "n_splits", 5)), len(wells)))
    for mode in modes:
        mode_rows = _run_mode_oof(mode, wells, args, n_splits, blend_alphas)
        per_well_rows.extend(mode_rows)

    per_well = pd.DataFrame(per_well_rows)
    per_well_path = out_dir / "per_well.csv"
    per_well.to_csv(per_well_path, index=False)

    experiments = _summarize_experiments(per_well)
    experiments_path = out_dir / "experiments.csv"
    experiments.to_csv(experiments_path, index=False)

    summary = {
        "n_wells": int(len(wells)),
        "n_hidden_rows": int(sum(int(w["hidden"].sum()) for w in wells)),
        "n_experiments": int(len(experiments)),
        "best_experiment": _best_experiment(experiments),
        "config": {
            "xyz_features": asdict(cfg),
            "modes": modes,
            "blend_alphas": blend_alphas,
            "n_splits": n_splits,
            "train_stride": int(getattr(args, "train_stride", 5)),
            "max_train_rows": int(getattr(args, "max_train_rows", 500_000)),
        },
        "outputs": {
            "per_well": str(per_well_path),
            "experiments": str(experiments_path),
        },
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _load_wells(paths: list[Path], cfg: XYZSteeringConfig, modes: list[str], no_progress: bool) -> list[dict[str, Any]]:
    wells: list[dict[str, Any]] = []
    iterator = tqdm(paths, desc="xyz_load", disable=no_progress)
    for path in iterator:
        hw = pd.read_csv(path)
        required = {"MD", "X", "Y", "Z", "TVT_input", "TVT"}
        missing = sorted(required.difference(hw.columns))
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")

        md = hw["MD"].to_numpy(dtype=np.float64)
        tvt_input = hw["TVT_input"].to_numpy(dtype=np.float64)
        tvt_true = hw["TVT"].to_numpy(dtype=np.float64)
        base, known = make_base_tvt(tvt_input, md)
        hidden = ~known & np.isfinite(tvt_true)
        if not hidden.any():
            continue

        features = {
            mode: build_xyz_steering_features(
                md,
                hw["X"].to_numpy(dtype=np.float64),
                hw["Y"].to_numpy(dtype=np.float64),
                hw["Z"].to_numpy(dtype=np.float64),
                known_mask=known,
                cfg=cfg,
                mode=mode,
            )
            for mode in modes
        }
        for mode, frame in features.items():
            frame["base_tvt"] = base.astype(np.float32)
            frame["base_slope"] = np.gradient(base, md, edge_order=1).astype(np.float32) if len(base) > 1 else 0.0

        well_id = path.name.removesuffix("__horizontal_well.csv")
        wells.append(
            {
                "well_id": well_id,
                "md": md,
                "tvt_true": tvt_true,
                "tvt_input": tvt_input,
                "base": base,
                "known": known,
                "hidden": hidden,
                "target_residual": tvt_true - base,
                "features": features,
                "gr_valid_ratio": _gr_valid_ratio(hw),
                "hidden_max_run": _max_true_run(hidden),
            }
        )
    return wells


def _run_mode_oof(
    mode: str,
    wells: list[dict[str, Any]],
    args: argparse.Namespace | SimpleNamespace,
    n_splits: int,
    blend_alphas: list[float],
) -> list[dict[str, Any]]:
    groups = np.arange(len(wells))
    splitter = GroupKFold(n_splits=n_splits)
    residual_pred = {well["well_id"]: np.zeros_like(well["base"], dtype=np.float64) for well in wells}

    for fold, (train_idx, val_idx) in enumerate(splitter.split(groups, groups=groups)):
        X_train, y_train = _make_train_matrix(
            [wells[i] for i in train_idx],
            mode=mode,
            train_stride=int(getattr(args, "train_stride", 5)),
            max_train_rows=int(getattr(args, "max_train_rows", 500_000)),
            seed=13_337 + fold,
        )
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=int(getattr(args, "max_iter", 180)),
            learning_rate=float(getattr(args, "learning_rate", 0.06)),
            l2_regularization=float(getattr(args, "l2_regularization", 0.02)),
            max_leaf_nodes=int(getattr(args, "max_leaf_nodes", 31)),
            random_state=44_000 + fold,
        )
        model.fit(X_train, y_train)
        for i in val_idx:
            well = wells[i]
            frame = well["features"][mode]
            pred = model.predict(frame).astype(np.float64)
            residual_pred[well["well_id"]] = pred

    rows: list[dict[str, Any]] = []
    clip = float(getattr(args, "clip_residual", 80.0))
    for well in wells:
        pred_r = np.clip(residual_pred[well["well_id"]], -clip, clip)
        for alpha in blend_alphas:
            experiment = f"{mode}_a{str(alpha).replace('.', 'p')}"
            pred = well["base"] + float(alpha) * pred_r
            pred[well["known"]] = well["tvt_input"][well["known"]]
            rows.append(_score_prediction(experiment, well, pred))
    return rows


def _make_train_matrix(
    wells: list[dict[str, Any]],
    mode: str,
    train_stride: int,
    max_train_rows: int,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    X_parts: list[pd.DataFrame] = []
    y_parts: list[np.ndarray] = []
    stride = max(1, int(train_stride))
    for well in wells:
        hidden_idx = np.flatnonzero(well["hidden"])
        hidden_idx = hidden_idx[::stride]
        if hidden_idx.size == 0:
            continue
        X_parts.append(well["features"][mode].iloc[hidden_idx])
        y_parts.append(well["target_residual"][hidden_idx].astype(np.float64))
    if not X_parts:
        raise ValueError("No hidden training rows available")

    X = pd.concat(X_parts, axis=0, ignore_index=True)
    y = np.concatenate(y_parts)
    if max_train_rows > 0 and len(y) > max_train_rows:
        rng = np.random.default_rng(seed)
        take = np.sort(rng.choice(len(y), size=max_train_rows, replace=False))
        X = X.iloc[take].reset_index(drop=True)
        y = y[take]
    return X, y


def _score_base(well: dict[str, Any]) -> dict[str, Any]:
    return _score_prediction("base", well, well["base"])


def _score_prediction(experiment: str, well: dict[str, Any], pred: np.ndarray) -> dict[str, Any]:
    hidden = well["hidden"].astype(bool)
    y = well["tvt_true"][hidden]
    p = np.asarray(pred, dtype=np.float64)[hidden]
    rmse, sse = _rmse_sse(p, y)
    base_rmse, _ = _rmse_sse(well["base"][hidden], y)
    return {
        "experiment": experiment,
        "well_id": well["well_id"],
        "n_rows": int(len(well["md"])),
        "n_hidden": int(hidden.sum()),
        "hidden_max_run": int(well["hidden_max_run"]),
        "gr_valid_ratio": float(well["gr_valid_ratio"]),
        "rmse": rmse,
        "sse": sse,
        "base_rmse": base_rmse,
        "gain_vs_base": float(base_rmse - rmse) if np.isfinite(base_rmse + rmse) else float("nan"),
        "improved": bool(np.isfinite(base_rmse + rmse) and rmse < base_rmse),
    }


def _summarize_experiments(per_well: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for experiment, sub in per_well.groupby("experiment", sort=True):
        rmse = _group_rmse(sub)
        base_rmse = _group_base_rmse(sub)
        rows.append(
            {
                "experiment": experiment,
                "rmse": rmse,
                "base_rmse": base_rmse,
                "gain_vs_base": float(base_rmse - rmse) if np.isfinite(base_rmse + rmse) else float("nan"),
                "n_wells": int(len(sub)),
                "n_hidden_rows": int(sub["n_hidden"].sum()),
                "wells_improved_frac": float(sub["improved"].mean()) if len(sub) else float("nan"),
                "rmse_high_gr_valid": _group_rmse(sub[sub["gr_valid_ratio"] > 0.6]),
                "rmse_medium_gr_valid": _group_rmse(
                    sub[(sub["gr_valid_ratio"] >= 0.2) & (sub["gr_valid_ratio"] <= 0.6)]
                ),
                "rmse_low_gr_valid": _group_rmse(sub[sub["gr_valid_ratio"] < 0.2]),
                "rmse_long_hidden": _group_rmse(sub[sub["hidden_max_run"] >= 1000]),
            }
        )
    return pd.DataFrame(rows).sort_values("rmse", na_position="last").reset_index(drop=True)


def _best_experiment(experiments: pd.DataFrame) -> dict[str, Any]:
    if experiments.empty:
        return {}
    row = experiments.sort_values("rmse", na_position="last").iloc[0]
    return {
        "experiment": str(row["experiment"]),
        "rmse": float(row["rmse"]),
        "gain_vs_base": float(row["gain_vs_base"]),
    }


def _group_rmse(df: pd.DataFrame) -> float:
    if df.empty:
        return float("nan")
    n = int(df["n_hidden"].sum())
    if n <= 0:
        return float("nan")
    return float(np.sqrt(float(df["sse"].sum()) / n))


def _group_base_rmse(df: pd.DataFrame) -> float:
    if df.empty:
        return float("nan")
    n = int(df["n_hidden"].sum())
    if n <= 0:
        return float("nan")
    sse = float(((df["base_rmse"].to_numpy(dtype=np.float64) ** 2) * df["n_hidden"].to_numpy(dtype=np.float64)).sum())
    return float(np.sqrt(sse / n))


def _rmse_sse(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    pred_arr = np.asarray(pred, dtype=np.float64)
    target_arr = np.asarray(target, dtype=np.float64)
    valid = np.isfinite(pred_arr) & np.isfinite(target_arr)
    if not valid.any():
        return float("nan"), 0.0
    diff = pred_arr[valid] - target_arr[valid]
    sse = float(np.sum(diff * diff))
    return float(np.sqrt(np.mean(diff * diff))), sse


def _parse_modes(value: str) -> list[str]:
    modes = [part.strip() for part in str(value).split(",") if part.strip()]
    if not modes:
        raise ValueError("At least one mode is required")
    bad = sorted(set(modes).difference({"past", "full"}))
    if bad:
        raise ValueError(f"Unsupported modes: {bad}")
    return modes


def _parse_float_list(value: str, default: list[float]) -> list[float]:
    if not value:
        return list(default)
    out = [float(part.strip()) for part in str(value).split(",") if part.strip()]
    if not out:
        raise ValueError("At least one float value is required")
    return out


def _gr_valid_ratio(hw: pd.DataFrame) -> float:
    if "GR" not in hw.columns:
        return 0.0
    gr = hw["GR"].to_numpy(dtype=np.float64)
    return float(np.isfinite(gr).mean()) if gr.size else 0.0


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
    parser = argparse.ArgumentParser(description="OOF test for post-drilling XYZ steering trajectory signal.")
    parser.add_argument("--data-dir", default="data/train")
    parser.add_argument("--out-dir", default="artifacts/xyz_steering")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--modes", default="past,full")
    parser.add_argument("--blend-alphas", default=",".join(str(v) for v in DEFAULT_BLEND_ALPHAS))
    parser.add_argument("--train-stride", type=int, default=5)
    parser.add_argument("--max-train-rows", type=int, default=500_000)
    parser.add_argument("--clip-residual", type=float, default=80.0)
    parser.add_argument("--max-iter", type=int, default=180)
    parser.add_argument("--learning-rate", type=float, default=0.06)
    parser.add_argument("--l2-regularization", type=float, default=0.02)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run(args)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
