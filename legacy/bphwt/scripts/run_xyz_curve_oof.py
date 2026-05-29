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

from bphwt.infer.xyz_curve import XYZCurveConfig, smooth_residual_curve
from scripts.run_xyz_steering_oof import (
    _best_experiment,
    _load_wells,
    _make_train_matrix,
    _parse_float_list,
    _score_base,
    _score_prediction,
    _summarize_experiments,
)

DEFAULT_BLEND_ALPHAS = [0.25, 0.5, 0.7, 1.0]
DEFAULT_SMOOTH_LAMBDAS = [0.0, 5.0, 20.0, 80.0]
DEFAULT_BASE_LAMBDAS = [0.0, 0.02, 0.08]


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

    mode = str(getattr(args, "mode", "full"))
    if mode not in {"past", "full"}:
        raise ValueError("mode must be 'past' or 'full'")
    blend_alphas = _parse_float_list(getattr(args, "blend_alphas", ""), DEFAULT_BLEND_ALPHAS)
    smooth_lambdas = _parse_float_list(getattr(args, "smooth_lambdas", ""), DEFAULT_SMOOTH_LAMBDAS)
    base_lambdas = _parse_float_list(getattr(args, "base_lambdas", ""), DEFAULT_BASE_LAMBDAS)

    from bphwt.features.xyz_steering import XYZSteeringConfig

    feature_cfg = XYZSteeringConfig()
    wells = _load_wells(paths, feature_cfg, modes=[mode], no_progress=bool(getattr(args, "no_progress", False)))
    if len(wells) < 2:
        raise ValueError("Need at least two wells for GroupKFold OOF")

    base_rows = [{"experiment": "base", **_score_base(well)} for well in wells]
    residual_prior = _oof_residual_prior(mode, wells, args)

    per_well_rows: list[dict[str, Any]] = list(base_rows)
    clip = float(getattr(args, "clip_residual", 80.0))
    anchor_lambda = float(getattr(args, "anchor_lambda", 1.0e6))
    for well in wells:
        prior = np.clip(residual_prior[well["well_id"]], -clip, clip)
        for alpha in blend_alphas:
            scaled_prior = float(alpha) * prior
            for smooth in smooth_lambdas:
                for base_lambda in base_lambdas:
                    curve_cfg = XYZCurveConfig(
                        lambda_smooth=float(smooth),
                        lambda_base=float(base_lambda),
                        lambda_anchor=anchor_lambda,
                    )
                    residual = smooth_residual_curve(well["md"], scaled_prior, well["known"], curve_cfg)
                    pred = well["base"] + residual
                    pred[well["known"]] = well["tvt_input"][well["known"]]
                    experiment = _experiment_name(mode, alpha, smooth, base_lambda)
                    row = _score_prediction(experiment, well, pred)
                    row["alpha"] = float(alpha)
                    row["lambda_smooth"] = float(smooth)
                    row["lambda_base"] = float(base_lambda)
                    per_well_rows.append(row)

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
            "mode": mode,
            "blend_alphas": blend_alphas,
            "smooth_lambdas": smooth_lambdas,
            "base_lambdas": base_lambdas,
            "anchor_lambda": anchor_lambda,
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
            seed=52_000 + fold,
        )
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=int(getattr(args, "max_iter", 100)),
            learning_rate=float(getattr(args, "learning_rate", 0.06)),
            l2_regularization=float(getattr(args, "l2_regularization", 0.02)),
            max_leaf_nodes=int(getattr(args, "max_leaf_nodes", 31)),
            random_state=61_000 + fold,
        )
        model.fit(X_train, y_train)
        for i in val_idx:
            well = wells[i]
            residual_pred[well["well_id"]] = model.predict(well["features"][mode]).astype(np.float64)
    return residual_pred


def _experiment_name(mode: str, alpha: float, smooth: float, base_lambda: float) -> str:
    return (
        f"curve_{mode}"
        f"_a{_float_token(alpha)}"
        f"_s{_float_token(smooth)}"
        f"_b{_float_token(base_lambda)}"
    )


def _float_token(value: float) -> str:
    return f"{float(value):g}".replace(".", "p").replace("-", "m")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OOF test for smoothed XYZ residual curve priors.")
    parser.add_argument("--data-dir", default="data/train")
    parser.add_argument("--out-dir", default="artifacts/xyz_curve")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--mode", default="full", choices=["past", "full"])
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--blend-alphas", default=",".join(str(v) for v in DEFAULT_BLEND_ALPHAS))
    parser.add_argument("--smooth-lambdas", default=",".join(str(v) for v in DEFAULT_SMOOTH_LAMBDAS))
    parser.add_argument("--base-lambdas", default=",".join(str(v) for v in DEFAULT_BASE_LAMBDAS))
    parser.add_argument("--anchor-lambda", type=float, default=1.0e6)
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
