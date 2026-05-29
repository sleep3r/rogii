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
from tqdm import tqdm

from bphwt.features.xyz_steering import XYZSteeringConfig, build_xyz_steering_features
from bphwt.priors.prob_residual_hmm import make_base_tvt


def run(args: argparse.Namespace | SimpleNamespace) -> dict[str, Any]:
    train_dir = Path(args.train_dir)
    test_dir = Path(args.test_dir)
    sample_path = Path(args.sample_submission)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mode = str(getattr(args, "mode", "full"))
    if mode not in {"past", "full"}:
        raise ValueError("mode must be 'past' or 'full'")

    cfg = XYZSteeringConfig()
    train_paths = sorted(train_dir.glob("*__horizontal_well.csv"))
    if not train_paths:
        raise FileNotFoundError(f"No train horizontal well CSV files found in {train_dir}")
    train_wells = _load_train_wells(train_paths, cfg, mode, no_progress=bool(getattr(args, "no_progress", False)))
    X_train, y_train = _make_train_matrix(
        train_wells,
        train_stride=int(getattr(args, "train_stride", 10)),
        max_train_rows=int(getattr(args, "max_train_rows", 350_000)),
        seed=82_001,
    )
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=int(getattr(args, "max_iter", 100)),
        learning_rate=float(getattr(args, "learning_rate", 0.06)),
        l2_regularization=float(getattr(args, "l2_regularization", 0.02)),
        max_leaf_nodes=int(getattr(args, "max_leaf_nodes", 31)),
        random_state=82_101,
    )
    model.fit(X_train, y_train)

    sample = pd.read_csv(sample_path)
    sample["well_id"] = sample["id"].astype(str).str.rsplit("_", n=1).str[0]
    sample["row_idx"] = sample["id"].astype(str).str.rsplit("_", n=1).str[1].astype(int)
    required_wells = sorted(sample["well_id"].unique())

    predictions: dict[str, np.ndarray] = {}
    iterator = tqdm(required_wells, desc="xyz_submit", disable=bool(getattr(args, "no_progress", False)))
    for well_id in iterator:
        hw_path = test_dir / f"{well_id}__horizontal_well.csv"
        if not hw_path.exists():
            continue
        pred = _predict_test_well(hw_path, cfg, mode, model, args)
        predictions[well_id] = pred

    rows: list[dict[str, Any]] = []
    missing = 0
    for row in sample.itertuples(index=False):
        well_id = str(row.well_id)
        row_idx = int(row.row_idx)
        pred = predictions.get(well_id)
        if pred is None or row_idx < 0 or row_idx >= len(pred):
            value = 0.0
            missing += 1
        else:
            value = float(pred[row_idx])
        rows.append({"id": row.id, "tvt": value})

    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    summary = {
        "n_rows": int(len(out)),
        "n_wells": int(len(required_wells)),
        "missing_rows": int(missing),
        "mode": mode,
        "blend_alpha": float(getattr(args, "blend_alpha", 1.0)),
        "out": str(out_path),
    }
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _load_train_wells(paths: list[Path], cfg: XYZSteeringConfig, mode: str, no_progress: bool) -> list[dict[str, Any]]:
    wells: list[dict[str, Any]] = []
    iterator = tqdm(paths, desc="xyz_train_load", disable=no_progress)
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
        features = _features_from_frame(hw, known, cfg, mode)
        features["base_tvt"] = base.astype(np.float32)
        features["base_slope"] = np.gradient(base, md, edge_order=1).astype(np.float32) if len(base) > 1 else 0.0
        wells.append(
            {
                "features": features,
                "hidden": hidden,
                "target_residual": tvt_true - base,
            }
        )
    if not wells:
        raise ValueError("No usable train wells with hidden targets")
    return wells


def _make_train_matrix(
    wells: list[dict[str, Any]],
    train_stride: int,
    max_train_rows: int,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    X_parts: list[pd.DataFrame] = []
    y_parts: list[np.ndarray] = []
    stride = max(1, int(train_stride))
    for well in wells:
        idx = np.flatnonzero(well["hidden"])[::stride]
        if idx.size == 0:
            continue
        X_parts.append(well["features"].iloc[idx])
        y_parts.append(well["target_residual"][idx].astype(np.float64))
    X = pd.concat(X_parts, axis=0, ignore_index=True)
    y = np.concatenate(y_parts)
    if max_train_rows > 0 and len(y) > max_train_rows:
        rng = np.random.default_rng(seed)
        take = np.sort(rng.choice(len(y), size=max_train_rows, replace=False))
        X = X.iloc[take].reset_index(drop=True)
        y = y[take]
    return X, y


def _predict_test_well(
    hw_path: Path,
    cfg: XYZSteeringConfig,
    mode: str,
    model: HistGradientBoostingRegressor,
    args: argparse.Namespace | SimpleNamespace,
) -> np.ndarray:
    hw = pd.read_csv(hw_path)
    required = {"MD", "X", "Y", "Z", "TVT_input"}
    missing = sorted(required.difference(hw.columns))
    if missing:
        raise ValueError(f"{hw_path} is missing required columns: {missing}")
    md = hw["MD"].to_numpy(dtype=np.float64)
    tvt_input = hw["TVT_input"].to_numpy(dtype=np.float64)
    base, known = make_base_tvt(tvt_input, md)
    features = _features_from_frame(hw, known, cfg, mode)
    features["base_tvt"] = base.astype(np.float32)
    features["base_slope"] = np.gradient(base, md, edge_order=1).astype(np.float32) if len(base) > 1 else 0.0
    residual = model.predict(features).astype(np.float64)
    clip = float(getattr(args, "clip_residual", 80.0))
    residual = np.clip(residual, -clip, clip)
    pred = base + float(getattr(args, "blend_alpha", 1.0)) * residual
    pred[known] = tvt_input[known]
    return pred.astype(np.float64)


def _features_from_frame(
    hw: pd.DataFrame,
    known: np.ndarray,
    cfg: XYZSteeringConfig,
    mode: str,
) -> pd.DataFrame:
    return build_xyz_steering_features(
        hw["MD"].to_numpy(dtype=np.float64),
        hw["X"].to_numpy(dtype=np.float64),
        hw["Y"].to_numpy(dtype=np.float64),
        hw["Z"].to_numpy(dtype=np.float64),
        known_mask=known,
        cfg=cfg,
        mode=mode,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Make a submission using the XYZ steering trajectory residual model.")
    parser.add_argument("--train-dir", default="data/train")
    parser.add_argument("--test-dir", default="data/test")
    parser.add_argument("--sample-submission", default="data/sample_submission.csv")
    parser.add_argument("--out", default="artifacts/xyz_steering/submission.csv")
    parser.add_argument("--mode", default="full", choices=["past", "full"])
    parser.add_argument("--blend-alpha", type=float, default=1.0)
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
