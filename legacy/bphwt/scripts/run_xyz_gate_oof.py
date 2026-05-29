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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from bphwt.features.xyz_steering import XYZSteeringConfig
from bphwt.infer.xyz_gate import XYZGateConfig, choose_gate_alpha, residual_gate_stats
from scripts.run_xyz_steering_oof import (
    _best_experiment,
    _load_wells,
    _make_train_matrix,
    _parse_float_list,
    _score_base,
    _score_prediction,
    _summarize_experiments,
)

DEFAULT_FALLBACK_ALPHAS = [0.0, 0.25, 0.5]
DEFAULT_MAX_ABS_P95 = [20.0, 30.0, 40.0, 60.0]
DEFAULT_MAX_DISAGREEMENT = [5.0, 10.0, 15.0, 25.0]


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

    fallback_alphas = _parse_float_list(getattr(args, "fallback_alphas", ""), DEFAULT_FALLBACK_ALPHAS)
    max_abs_p95_values = _parse_float_list(getattr(args, "max_abs_p95_values", ""), DEFAULT_MAX_ABS_P95)
    max_disagreement_values = _parse_float_list(
        getattr(args, "max_disagreement_values", ""),
        DEFAULT_MAX_DISAGREEMENT,
    )

    feature_cfg = XYZSteeringConfig()
    wells = _load_wells(paths, feature_cfg, modes=["past", "full"], no_progress=bool(getattr(args, "no_progress", False)))
    if len(wells) < 2:
        raise ValueError("Need at least two wells for GroupKFold OOF")

    residual_full = _oof_residual_prior("full", wells, args)
    residual_past = _oof_residual_prior("past", wells, args)

    clip = float(getattr(args, "clip_residual", 80.0))
    per_well_rows: list[dict[str, Any]] = [{"experiment": "base", **_score_base(well)} for well in wells]
    for alpha, label in [(1.0, "full_a1p0"), (0.5, "full_a0p5"), (0.25, "full_a0p25")]:
        for well in wells:
            full = np.clip(residual_full[well["well_id"]], -clip, clip)
            pred = well["base"] + alpha * full
            pred[well["known"]] = well["tvt_input"][well["known"]]
            row = _score_prediction(label, well, pred)
            row.update(residual_gate_stats(full, residual_past[well["well_id"]], well["hidden"]))
            row["used_alpha"] = alpha
            per_well_rows.append(row)

    for fallback_alpha in fallback_alphas:
        for max_abs_p95 in max_abs_p95_values:
            for max_disagreement in max_disagreement_values:
                cfg = XYZGateConfig(
                    high_alpha=1.0,
                    fallback_alpha=float(fallback_alpha),
                    max_abs_p95=float(max_abs_p95),
                    max_disagreement_rmse=float(max_disagreement),
                )
                experiment = _experiment_name(fallback_alpha, max_abs_p95, max_disagreement)
                for well in wells:
                    full = np.clip(residual_full[well["well_id"]], -clip, clip)
                    past = np.clip(residual_past[well["well_id"]], -clip, clip)
                    alpha = choose_gate_alpha(full, past, well["hidden"], cfg)
                    pred = well["base"] + alpha * full
                    pred[well["known"]] = well["tvt_input"][well["known"]]
                    row = _score_prediction(experiment, well, pred)
                    row.update(residual_gate_stats(full, past, well["hidden"]))
                    row["used_alpha"] = float(alpha)
                    row["fallback_alpha"] = float(fallback_alpha)
                    row["max_abs_p95"] = float(max_abs_p95)
                    row["max_disagreement_rmse"] = float(max_disagreement)
                    per_well_rows.append(row)

    per_well = pd.DataFrame(per_well_rows)
    per_well_path = out_dir / "per_well.csv"
    per_well.to_csv(per_well_path, index=False)

    experiments = _summarize_experiments(per_well)
    alpha_summary = per_well.groupby("experiment", as_index=False)["used_alpha"].mean()
    experiments = experiments.merge(alpha_summary, on="experiment", how="left")
    experiments_path = out_dir / "experiments.csv"
    experiments.to_csv(experiments_path, index=False)

    summary = {
        "n_wells": int(len(wells)),
        "n_hidden_rows": int(sum(int(w["hidden"].sum()) for w in wells)),
        "n_experiments": int(len(experiments)),
        "best_experiment": _best_experiment(experiments),
        "config": {
            "fallback_alphas": fallback_alphas,
            "max_abs_p95_values": max_abs_p95_values,
            "max_disagreement_values": max_disagreement_values,
            "n_splits": int(getattr(args, "n_splits", 5)),
            "train_stride": int(getattr(args, "train_stride", 10)),
            "max_train_rows": int(getattr(args, "max_train_rows", 350_000)),
        },
        "outputs": {
            "per_well": str(per_well_path),
            "experiments": str(experiments_path),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _oof_residual_prior(
    mode: str,
    wells: list[dict[str, Any]],
    args: argparse.Namespace | SimpleNamespace,
) -> dict[str, np.ndarray]:
    n_splits = max(2, min(int(getattr(args, "n_splits", 5)), len(wells)))
    groups = np.arange(len(wells))
    splitter = GroupKFold(n_splits=n_splits)
    residual_pred = {well["well_id"]: np.zeros_like(well["base"], dtype=np.float64) for well in wells}
    for fold, (train_idx, val_idx) in enumerate(splitter.split(groups, groups=groups)):
        X_train, y_train = _make_train_matrix(
            [wells[i] for i in train_idx],
            mode=mode,
            train_stride=int(getattr(args, "train_stride", 10)),
            max_train_rows=int(getattr(args, "max_train_rows", 350_000)),
            seed=72_000 + fold,
        )
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=int(getattr(args, "max_iter", 100)),
            learning_rate=float(getattr(args, "learning_rate", 0.06)),
            l2_regularization=float(getattr(args, "l2_regularization", 0.02)),
            max_leaf_nodes=int(getattr(args, "max_leaf_nodes", 31)),
            random_state=73_000 + fold,
        )
        model.fit(X_train, y_train)
        for i in val_idx:
            well = wells[i]
            residual_pred[well["well_id"]] = model.predict(well["features"][mode]).astype(np.float64)
    return residual_pred


def _experiment_name(fallback_alpha: float, max_abs_p95: float, max_disagreement: float) -> str:
    return (
        f"gate_fb{_float_token(fallback_alpha)}"
        f"_p95{_float_token(max_abs_p95)}"
        f"_d{_float_token(max_disagreement)}"
    )


def _float_token(value: float) -> str:
    return f"{float(value):g}".replace(".", "p").replace("-", "m")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OOF gate search for full XYZ trajectory residual corrections.")
    parser.add_argument("--data-dir", default="data/train")
    parser.add_argument("--out-dir", default="artifacts/xyz_gate")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--fallback-alphas", default=",".join(str(v) for v in DEFAULT_FALLBACK_ALPHAS))
    parser.add_argument("--max-abs-p95-values", default=",".join(str(v) for v in DEFAULT_MAX_ABS_P95))
    parser.add_argument("--max-disagreement-values", default=",".join(str(v) for v in DEFAULT_MAX_DISAGREEMENT))
    parser.add_argument("--train-stride", type=int, default=10)
    parser.add_argument("--max-train-rows", type=int, default=350_000)
    parser.add_argument("--clip-residual", type=float, default=80.0)
    parser.add_argument("--max-iter", type=int, default=100)
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
