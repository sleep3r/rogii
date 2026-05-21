from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import load_config
from .constants import FORMATIONS
from .io import well_name
from .pipeline import shuffled_path_folds
from .surface_student import _format_duration
from .validation import grouped_well_folds


SAFE_STUDENT_FEATURES: tuple[str, ...] = (
    "geo_student_tvt",
    "geo_student_delta_last",
    "geo_student_minus_schema10",
    "geo_student_minus_flat",
    *(f"surface_hat_{formation}" for formation in FORMATIONS),
    *(f"z_minus_surface_hat_{formation}" for formation in FORMATIONS),
)


def _rmse(pred: np.ndarray | pd.Series, true: np.ndarray | pd.Series) -> float:
    pred_arr = np.asarray(pred, dtype=float)
    true_arr = np.asarray(true, dtype=float)
    mask = np.isfinite(pred_arr) & np.isfinite(true_arr)
    if not np.any(mask):
        return float("nan")
    err = pred_arr[mask] - true_arr[mask]
    return float(np.sqrt(np.mean(err * err)))


def _well_rmse(frame: pd.DataFrame, pred_col: str, true_col: str) -> pd.DataFrame:
    return (
        frame.assign(_err=frame[pred_col].to_numpy(float) - frame[true_col].to_numpy(float))
        .groupby("well")
        .agg(
            rows=("well", "size"),
            rmse=("_err", lambda x: float(np.sqrt(np.nanmean(np.asarray(x, dtype=float) ** 2)))),
        )
        .reset_index()
        .sort_values("rmse", ascending=False)
    )


def _summarize_wells(by_well: pd.DataFrame) -> dict[str, Any]:
    return {
        "mean": float(by_well["rmse"].mean()),
        "median": float(by_well["rmse"].median()),
        "p90": float(by_well["rmse"].quantile(0.90)),
        "p95": float(by_well["rmse"].quantile(0.95)),
        "worst": float(by_well["rmse"].max()),
        "worst_well": str(by_well.iloc[0]["well"]) if len(by_well) else None,
    }


def reconstruct_schema10_oof(
    *,
    model_path: Path,
    config_path: Path,
    data_dir: Path,
) -> pd.DataFrame:
    """Reconstruct schema10 raw OOF row order from the persisted model."""

    config = load_config(config_path)
    seed = int(config.get("seed", 42))
    n_splits = int(config.get("validation", {}).get("n_splits", 5))
    train_dir = data_dir / "train"
    paths = sorted(train_dir.glob("*__horizontal_well.csv"), key=well_name)
    if not paths:
        raise ValueError(f"No train wells found in {train_dir}")
    with model_path.open("rb") as file:
        model = pickle.load(file)
    residual = np.asarray(getattr(model, "oof_residual_", None), dtype=float)
    if residual.ndim != 1 or len(residual) == 0:
        raise ValueError(f"Model at {model_path} does not contain oof_residual_")

    rows: list[pd.DataFrame] = []
    for _fold_id, _train_paths, valid_paths in shuffled_path_folds(paths, n_splits, seed):
        for path in valid_paths:
            well = well_name(path)
            df = pd.read_csv(path, usecols=lambda column: column in {"TVT_input", "TVT", "Z"})
            tvt_input = pd.to_numeric(df["TVT_input"], errors="coerce").to_numpy(dtype=float)
            tvt_true = pd.to_numeric(df["TVT"], errors="coerce").to_numpy(dtype=float)
            z = pd.to_numeric(df["Z"], errors="coerce").to_numpy(dtype=float)
            hidden_idx = np.flatnonzero(~np.isfinite(tvt_input) & np.isfinite(tvt_true))
            known_idx = np.flatnonzero(np.isfinite(tvt_input))
            last_known = float(tvt_input[known_idx[-1]]) if len(known_idx) else 0.0
            rows.append(
                pd.DataFrame(
                    {
                        "id": [f"{well}_{int(idx)}" for idx in hidden_idx],
                        "well": well,
                        "row_index": hidden_idx.astype(int),
                        "tvt_true_schema10": tvt_true[hidden_idx],
                        "z": z[hidden_idx],
                        "schema10_baseline": np.full(len(hidden_idx), last_known, dtype=float),
                    }
                )
            )
    frame = pd.concat(rows, ignore_index=True)
    if len(frame) != len(residual):
        raise ValueError(
            f"Schema10 row count mismatch: reconstructed={len(frame)} residual={len(residual)}"
        )
    frame["schema10_residual_oof"] = residual
    frame["schema10_oof_raw"] = frame["schema10_baseline"] + residual
    raw_rmse = _rmse(frame["schema10_oof_raw"], frame["tvt_true_schema10"])
    metrics = getattr(model, "metrics_", {}) or {}
    expected = float(metrics.get("oof_rmse", np.nan))
    if np.isfinite(expected) and abs(raw_rmse - expected) > 1e-3:
        raise ValueError(
            "Reconstructed schema10 OOF RMSE does not match persisted model: "
            f"reconstructed={raw_rmse:.6f} expected={expected:.6f}"
        )
    return frame


def build_integration_frame(student_oof_path: Path, schema_oof: pd.DataFrame) -> pd.DataFrame:
    student = pd.read_parquet(student_oof_path)
    keep = [
        "id",
        "well",
        "row_index",
        "tvt_true",
        "flat_tvt",
        "geo_student_tvt",
        "geo_student_delta_last",
        "geo_student_minus_flat",
        "geo_student_uncertainty",
        *(f"surface_hat_{formation}" for formation in FORMATIONS),
    ]
    missing = [column for column in keep if column not in student.columns]
    if missing:
        raise ValueError(f"Surface student OOF missing columns: {missing}")
    merged = schema_oof.merge(student[keep], on=["id", "well", "row_index"], how="inner")
    if len(merged) != len(schema_oof):
        raise ValueError(
            f"Student/schema OOF merge lost rows: schema={len(schema_oof)} merged={len(merged)}"
        )
    if not np.allclose(
        merged["tvt_true_schema10"].to_numpy(float),
        merged["tvt_true"].to_numpy(float),
        equal_nan=False,
    ):
        raise ValueError("Schema10 and student true TVT columns are not aligned")
    merged["geo_student_minus_schema10"] = merged["geo_student_tvt"] - merged["schema10_oof_raw"]
    for formation in FORMATIONS:
        merged[f"z_minus_surface_hat_{formation}"] = merged["z"] - merged[f"surface_hat_{formation}"]
    return merged


def fit_student_meta_oof(
    frame: pd.DataFrame,
    *,
    seed: int,
    n_splits: int,
    iterations: int,
    allow_teacher_uncertainty: bool,
) -> tuple[pd.DataFrame, list[str], list[dict[str, Any]]]:
    from catboost import CatBoostRegressor

    feature_columns = [column for column in SAFE_STUDENT_FEATURES if column in frame.columns]
    excluded_columns: list[str] = []
    if allow_teacher_uncertainty and "geo_student_uncertainty" in frame.columns:
        feature_columns.append("geo_student_uncertainty")
    elif "geo_student_uncertainty" in frame.columns:
        excluded_columns.append("geo_student_uncertainty")
    if not feature_columns:
        raise ValueError("No student integration feature columns found")

    target = frame["tvt_true_schema10"].to_numpy(float) - frame["schema10_oof_raw"].to_numpy(float)
    pred = np.full(len(frame), np.nan, dtype=float)
    fold_rows: list[dict[str, Any]] = []
    folds = grouped_well_folds(
        frame["well"].to_numpy(object),
        n_splits=min(int(n_splits), int(frame["well"].nunique())),
        seed=seed,
    )
    for fold_id, (train_idx, valid_idx) in enumerate(folds, start=1):
        valid_train = np.isfinite(target[train_idx])
        train_rows = train_idx[valid_train]
        valid_rows = valid_idx[np.isfinite(target[valid_idx])]
        model = CatBoostRegressor(
            iterations=int(iterations),
            early_stopping_rounds=max(20, min(100, int(iterations) // 4)),
            learning_rate=0.05,
            depth=5,
            l2_leaf_reg=8.0,
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=seed + fold_id,
            allow_writing_files=False,
            verbose=False,
        )
        model.fit(
            frame.loc[train_rows, feature_columns],
            target[train_rows],
            eval_set=(frame.loc[valid_rows, feature_columns], target[valid_rows]),
            use_best_model=True,
            verbose=False,
        )
        pred[valid_rows] = model.predict(frame.loc[valid_rows, feature_columns])
        fold_frame = frame.iloc[valid_rows].copy()
        fold_frame["schema10_plus_student_raw"] = fold_frame["schema10_oof_raw"] + pred[valid_rows]
        fold_rows.append(
            {
                "fold": int(fold_id),
                "valid_rows": int(len(valid_rows)),
                "valid_wells": int(fold_frame["well"].nunique()),
                "schema10_rmse": _rmse(fold_frame["schema10_oof_raw"], fold_frame["tvt_true_schema10"]),
                "plus_student_rmse": _rmse(
                    fold_frame["schema10_plus_student_raw"],
                    fold_frame["tvt_true_schema10"],
                ),
                "best_iteration": int(model.get_best_iteration() or 0),
            }
        )
    out = frame.copy()
    out["student_meta_residual"] = pred
    out["schema10_plus_student_raw"] = out["schema10_oof_raw"] + pred
    return out, feature_columns + [f"excluded:{column}" for column in excluded_columns], fold_rows


def run_smoke(
    *,
    student_oof_path: Path,
    schema_model_path: Path,
    schema_config_path: Path,
    data_dir: Path,
    output_dir: Path,
    seed: int,
    n_splits: int,
    iterations: int,
    allow_teacher_uncertainty: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    schema_oof = reconstruct_schema10_oof(
        model_path=schema_model_path,
        config_path=schema_config_path,
        data_dir=data_dir,
    )
    frame = build_integration_frame(student_oof_path, schema_oof)
    frame, feature_columns, fold_rows = fit_student_meta_oof(
        frame,
        seed=seed,
        n_splits=n_splits,
        iterations=iterations,
        allow_teacher_uncertainty=allow_teacher_uncertainty,
    )
    schema_rmse = _rmse(frame["schema10_oof_raw"], frame["tvt_true_schema10"])
    plus_rmse = _rmse(frame["schema10_plus_student_raw"], frame["tvt_true_schema10"])
    student_rmse = _rmse(frame["geo_student_tvt"], frame["tvt_true_schema10"])
    flat_rmse = _rmse(frame["flat_tvt"], frame["tvt_true_schema10"])
    schema_well = _summarize_wells(_well_rmse(frame, "schema10_oof_raw", "tvt_true_schema10"))
    plus_well = _summarize_wells(
        _well_rmse(frame, "schema10_plus_student_raw", "tvt_true_schema10")
    )
    metrics = {
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "features": feature_columns,
        "allow_teacher_derived_uncertainty": bool(allow_teacher_uncertainty),
        "schema10_raw_rmse": float(schema_rmse),
        "schema10_plus_student_raw_rmse": float(plus_rmse),
        "gain": float(schema_rmse - plus_rmse),
        "geo_student_standalone_rmse": float(student_rmse),
        "flat_rmse": float(flat_rmse),
        "schema10_well": schema_well,
        "plus_student_well": plus_well,
        "p95_delta": float(schema_well["p95"] - plus_well["p95"]),
        "worst_delta": float(schema_well["worst"] - plus_well["worst"]),
        "folds": fold_rows,
    }
    frame[
        [
            "id",
            "well",
            "row_index",
            "tvt_true_schema10",
            "schema10_oof_raw",
            "geo_student_tvt",
            "schema10_plus_student_raw",
            "student_meta_residual",
        ]
    ].to_parquet(output_dir / "integration_oof_predictions.parquet", index=False)
    (output_dir / "integration_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    _write_report(output_dir, metrics)
    return metrics


def _decision(gain: float) -> str:
    if gain < 0.05:
        return "DROP from main features; keep only for candidate_bank"
    if gain < 0.15:
        return "KEEP as weak feature/candidate; no submit"
    return "RUN proper schema10 + v0 integration"


def _write_report(output_dir: Path, metrics: dict[str, Any]) -> None:
    lines = [
        "# Surface Student Integration Smoke",
        "",
        "Cheap Experiment 0.5: CatBoost meta-residual on top of schema10 raw OOF, using only Surface Student v0 outputs.",
        "",
        "## Summary",
        "",
        f"- Rows: `{metrics['rows']}`",
        f"- Wells: `{metrics['wells']}`",
        f"- Schema10 raw RMSE: `{metrics['schema10_raw_rmse']:.6f}`",
        f"- Schema10 + student raw RMSE: `{metrics['schema10_plus_student_raw_rmse']:.6f}`",
        f"- Gain: `{metrics['gain']:.6f}`",
        f"- Geo student standalone RMSE: `{metrics['geo_student_standalone_rmse']:.6f}`",
        f"- Flat RMSE: `{metrics['flat_rmse']:.6f}`",
        f"- Decision: **{_decision(float(metrics['gain']))}**",
        "",
        "## Feature Policy",
        "",
        f"- Teacher-derived uncertainty allowed: `{metrics['allow_teacher_derived_uncertainty']}`",
        "- `geo_student_uncertainty` is excluded by default because the current v0 column is `abs(student - teacher)`, which is not available at test time.",
        "",
        "## Well Metrics",
        "",
        "| metric | schema10 | schema10+student | delta positive=better |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key in ("mean", "median", "p90", "p95", "worst"):
        base = float(metrics["schema10_well"][key])
        plus = float(metrics["plus_student_well"][key])
        lines.append(f"| `{key}` | `{base:.6f}` | `{plus:.6f}` | `{base - plus:.6f}` |")
    lines.extend(
        [
            "",
            "## Fold Metrics",
            "",
            "| fold | rows | wells | schema10 | schema10+student | gain | best_iter |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in metrics["folds"]:
        gain = float(row["schema10_rmse"]) - float(row["plus_student_rmse"])
        lines.append(
            f"| `{row['fold']}` | `{row['valid_rows']}` | `{row['valid_wells']}` | "
            f"`{row['schema10_rmse']:.6f}` | `{row['plus_student_rmse']:.6f}` | "
            f"`{gain:.6f}` | `{row['best_iteration']}` |"
        )
    lines.append("")
    (output_dir / "integration_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cheap Surface Student v0 integration smoke.")
    parser.add_argument(
        "--student-oof",
        type=Path,
        default=Path("artifacts/surface_student/oof_predictions.parquet"),
    )
    parser.add_argument(
        "--schema-model",
        type=Path,
        default=Path("artifacts/clearml/ed4d9dc6c7cb479881f087fee1217253/model.pkl"),
    )
    parser.add_argument(
        "--schema-config",
        type=Path,
        default=Path("artifacts/clearml/ed4d9dc6c7cb479881f087fee1217253/config.yml"),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/surface_student_integration"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--allow-teacher-derived-uncertainty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = pd.Timestamp.now()
    metrics = run_smoke(
        student_oof_path=args.student_oof,
        schema_model_path=args.schema_model,
        schema_config_path=args.schema_config,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        n_splits=args.n_splits,
        iterations=args.iterations,
        allow_teacher_uncertainty=bool(args.allow_teacher_derived_uncertainty),
    )
    elapsed = (pd.Timestamp.now() - started).total_seconds()
    print(
        "Surface student integration smoke complete | "
        f"rows={metrics['rows']} schema10={metrics['schema10_raw_rmse']:.6f} "
        f"plus_student={metrics['schema10_plus_student_raw_rmse']:.6f} "
        f"gain={metrics['gain']:.6f} duration={_format_duration(elapsed)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
