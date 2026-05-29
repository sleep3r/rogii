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
from scripts.run_xyz_steering_oof import (
    _best_experiment,
    _load_wells,
    _make_train_matrix,
    _parse_float_list,
    _score_base,
    _score_prediction,
    _summarize_experiments,
)


MODEL_SPECS: list[dict[str, float | int | str]] = [
    {"name": "base31", "max_iter": 100, "learning_rate": 0.06, "l2_regularization": 0.02, "max_leaf_nodes": 31},
    {"name": "shallow15", "max_iter": 140, "learning_rate": 0.045, "l2_regularization": 0.05, "max_leaf_nodes": 15},
    {"name": "wide63", "max_iter": 90, "learning_rate": 0.05, "l2_regularization": 0.01, "max_leaf_nodes": 63},
]
DEFAULT_BLEND_ALPHAS = [0.25, 0.5, 0.7, 1.0]


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

    blend_alphas = _parse_float_list(getattr(args, "blend_alphas", ""), DEFAULT_BLEND_ALPHAS)
    feature_cfg = XYZSteeringConfig()
    wells = _load_wells(paths, feature_cfg, modes=["full"], no_progress=bool(getattr(args, "no_progress", False)))
    if len(wells) < 2:
        raise ValueError("Need at least two wells for GroupKFold OOF")

    residual_pred = _oof_ensemble_prior(wells, args)
    clip = float(getattr(args, "clip_residual", 80.0))
    per_well_rows: list[dict[str, Any]] = [{"experiment": "base", **_score_base(well)} for well in wells]
    for well in wells:
        residual = np.clip(residual_pred[well["well_id"]], -clip, clip)
        for alpha in blend_alphas:
            experiment = f"ens_a{_float_token(alpha)}"
            pred = well["base"] + float(alpha) * residual
            pred[well["known"]] = well["tvt_input"][well["known"]]
            row = _score_prediction(experiment, well, pred)
            row["residual_abs_mean"] = float(np.mean(np.abs(residual[well["hidden"]])))
            row["residual_abs_p95"] = float(np.percentile(np.abs(residual[well["hidden"]]), 95.0))
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
        "model_specs": MODEL_SPECS,
        "outputs": {
            "per_well": str(per_well_path),
            "experiments": str(experiments_path),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _oof_ensemble_prior(
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
            mode="full",
            train_stride=int(getattr(args, "train_stride", 10)),
            max_train_rows=int(getattr(args, "max_train_rows", 350_000)),
            seed=91_000 + fold,
        )
        fold_preds = {wells[i]["well_id"]: [] for i in val_idx}
        for spec_idx, spec in enumerate(MODEL_SPECS):
            model = HistGradientBoostingRegressor(
                loss="squared_error",
                max_iter=int(spec["max_iter"]),
                learning_rate=float(spec["learning_rate"]),
                l2_regularization=float(spec["l2_regularization"]),
                max_leaf_nodes=int(spec["max_leaf_nodes"]),
                random_state=92_000 + 100 * fold + spec_idx,
            )
            model.fit(X_train, y_train)
            for i in val_idx:
                well = wells[i]
                fold_preds[well["well_id"]].append(model.predict(well["features"]["full"]).astype(np.float64))
        for well_id, preds in fold_preds.items():
            residual_pred[well_id] = np.mean(np.stack(preds, axis=0), axis=0)
    return residual_pred


def _float_token(value: float) -> str:
    return f"{float(value):g}".replace(".", "p").replace("-", "m")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OOF ensemble of full XYZ steering residual models.")
    parser.add_argument("--data-dir", default="data/train")
    parser.add_argument("--out-dir", default="artifacts/xyz_ensemble")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--blend-alphas", default=",".join(str(v) for v in DEFAULT_BLEND_ALPHAS))
    parser.add_argument("--train-stride", type=int, default=10)
    parser.add_argument("--max-train-rows", type=int, default=350_000)
    parser.add_argument("--clip-residual", type=float, default=80.0)
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run(args)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
