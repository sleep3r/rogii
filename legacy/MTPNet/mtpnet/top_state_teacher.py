from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from .annotation_top_audit import FORMATION_TOP_COLUMNS
from .residual_stack import make_group_folds
from .schema_safe import assert_schema_safe_columns


@dataclass(frozen=True)
class TopStateTeacherConfig:
    data_dir: Path = Path("data/train")
    output_dir: Path = Path("artifacts/top_state_teacher_v0")
    rows_per_step: int = 32
    n_folds: int = 5
    seed: int = 42
    k_wells: int = -1
    target_top_column: str = "ANCC"
    flat_epsilon_ft: float = 0.05
    iterations: int = 400
    learning_rate: float = 0.05
    depth: int = 5
    l2_leaf_reg: float = 8.0
    n_panel_wells: int = 6
    progress_every: int = 100


@dataclass
class TopStateDataset:
    rows: pd.DataFrame
    features: pd.DataFrame
    feature_columns: list[str]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _safe_mean(values: pd.Series | np.ndarray) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def _safe_std(values: pd.Series | np.ndarray) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.std()) if arr.size > 1 else 0.0


def _safe_slope(values: pd.Series | np.ndarray) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=np.float64)
    finite = np.isfinite(arr)
    if finite.sum() < 2:
        return 0.0
    x = np.arange(arr.size, dtype=np.float64)[finite]
    y = arr[finite]
    return float(np.polyfit(x, y, 1)[0])


def _load_horizontal_frame(data_dir: Path, k_wells: int) -> pd.DataFrame:
    paths = sorted(Path(data_dir).glob("*__horizontal_well.csv"))
    if k_wells > 0:
        paths = paths[:k_wells]
    rows: list[pd.DataFrame] = []
    for path in paths:
        well_id = path.name.replace("__horizontal_well.csv", "")
        frame = pd.read_csv(path)
        frame["well_id"] = well_id
        frame["row_idx"] = np.arange(len(frame), dtype=np.int32)
        frame["id"] = [f"{well_id}_{idx}" for idx in frame["row_idx"]]
        rows.append(frame)
    if not rows:
        raise FileNotFoundError(f"No horizontal wells found in {data_dir}")
    return pd.concat(rows, ignore_index=True)


def _state_from_delta(delta: np.ndarray, eps: float) -> np.ndarray:
    out = np.full(delta.shape, 1, dtype=np.int8)
    out[delta < -float(eps)] = 0
    out[delta > float(eps)] = 2
    return out


def _prepare_well_steps(
    well: pd.DataFrame,
    *,
    rows_per_step: int,
    target_top_column: str,
    flat_epsilon_ft: float,
) -> pd.DataFrame:
    ordered = well.sort_values("row_idx").copy()
    if "step" not in ordered.columns:
        ordered["step"] = (pd.to_numeric(ordered["row_idx"], errors="coerce") // rows_per_step).astype(int)
    records: list[dict[str, Any]] = []
    known_mask_all = pd.to_numeric(ordered.get("TVT_input", pd.Series(np.nan, index=ordered.index)), errors="coerce").notna()
    known_rows = ordered.loc[known_mask_all]
    if known_rows.empty:
        last_known_row_global = int(pd.to_numeric(ordered["row_idx"], errors="coerce").min())
        last_known_tvt_global = 0.0
        known_slope_global = 0.0
    else:
        last_known = known_rows.iloc[-1]
        last_known_row_global = int(last_known["row_idx"])
        last_known_tvt_global = float(pd.to_numeric(pd.Series([last_known["TVT_input"]]), errors="coerce").iloc[0])
        known_slope_global = _safe_slope(known_rows.tail(8)["TVT_input"])
    comp_top: list[float] = []
    grouped = list(ordered.groupby("step", sort=True))
    for _, group in grouped:
        comp_top.append(_safe_mean(group[target_top_column]))
    comp_top_arr = np.asarray(comp_top, dtype=np.float64)
    deltas = np.diff(comp_top_arr, prepend=comp_top_arr[0])
    states = _state_from_delta(deltas, flat_epsilon_ft)
    total_steps = max(len(grouped) - 1, 1)
    for idx, (step, group) in enumerate(grouped):
        tvt_input = pd.to_numeric(group.get("TVT_input", pd.Series(np.nan, index=group.index)), errors="coerce")
        known_frac = float(tvt_input.notna().mean())
        hidden_frac = float(tvt_input.isna().mean())
        row_idx_mean = _safe_mean(group["row_idx"])
        md = pd.to_numeric(group.get("MD", pd.Series(np.nan, index=group.index)), errors="coerce")
        z = pd.to_numeric(group.get("Z", pd.Series(np.nan, index=group.index)), errors="coerce")
        gr = pd.to_numeric(group.get("GR", pd.Series(np.nan, index=group.index)), errors="coerce")
        records.append(
            {
                "well_id": str(group["well_id"].iloc[0]),
                "step": int(step),
                "row_idx_mean": row_idx_mean,
                "n_rows": int(len(group)),
                "known_frac": known_frac,
                "hidden_frac": hidden_frac,
                "target_top": comp_top_arr[idx],
                "target_delta": deltas[idx],
                "target_state": int(states[idx]),
                "feat_step_progress": float(idx / total_steps),
                "feat_row_idx_rel": float((row_idx_mean - last_known_row_global) / 1000.0),
                "feat_last_known_tvt": float(last_known_tvt_global / 10000.0)
                if np.isfinite(last_known_tvt_global)
                else 0.0,
                "feat_known_slope": known_slope_global,
                "feat_known_frac": known_frac,
                "feat_hidden_frac": hidden_frac,
                "feat_md_mean": _safe_mean(md) / 10000.0,
                "feat_md_slope": _safe_slope(md) / 100.0,
                "feat_x_mean": _safe_mean(group.get("X", pd.Series(np.nan, index=group.index))) / 10000.0,
                "feat_y_mean": _safe_mean(group.get("Y", pd.Series(np.nan, index=group.index))) / 10000.0,
                "feat_z_mean": _safe_mean(z) / 10000.0,
                "feat_z_delta": _safe_slope(z),
                "feat_neg_z_delta": -_safe_slope(z),
                "feat_gr_mean": _safe_mean(gr) / 100.0,
                "feat_gr_std": _safe_std(gr) / 50.0,
                "feat_gr_valid_frac": float(gr.notna().mean()),
                "feat_tvt_input_mean": _safe_mean(tvt_input) / 10000.0
                if tvt_input.notna().any()
                else 0.0,
            }
        )
    return pd.DataFrame(records)


def build_top_state_dataset(
    frame: pd.DataFrame,
    *,
    rows_per_step: int = 32,
    target_top_column: str = "ANCC",
    flat_epsilon_ft: float = 0.05,
) -> TopStateDataset:
    if target_top_column not in frame.columns:
        available = [column for column in FORMATION_TOP_COLUMNS if column in frame.columns]
        raise ValueError(f"{target_top_column} not found; available formation tops: {available}")
    step_parts = [
        _prepare_well_steps(
            well,
            rows_per_step=rows_per_step,
            target_top_column=target_top_column,
            flat_epsilon_ft=flat_epsilon_ft,
        )
        for _, well in frame.groupby("well_id", sort=True)
    ]
    rows = pd.concat(step_parts, ignore_index=True)
    feature_columns = [column for column in rows.columns if column.startswith("feat_")]
    assert_schema_safe_columns(feature_columns, context="TopStateTeacher features")
    features = rows[feature_columns].apply(pd.to_numeric, errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    return TopStateDataset(rows=rows, features=features, feature_columns=feature_columns)


def _predict_proba_fixed(model: CatBoostClassifier, x: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(x)
    out = np.zeros((len(x), 3), dtype=np.float64)
    classes = [int(c) for c in model.classes_]
    for idx, cls in enumerate(classes):
        if 0 <= cls <= 2:
            out[:, cls] = raw[:, idx]
    row_sum = out.sum(axis=1)
    missing = row_sum <= 0
    if missing.any():
        out[missing, :] = 1.0 / 3.0
        row_sum = out.sum(axis=1)
    return out / row_sum[:, None]


def _train_classifier(dataset: TopStateDataset, train_wells: list[str], config: TopStateTeacherConfig) -> CatBoostClassifier:
    mask = dataset.rows["well_id"].astype(str).isin(set(train_wells)).to_numpy()
    y = pd.to_numeric(dataset.rows.loc[mask, "target_state"], errors="coerce").astype(int)
    model = CatBoostClassifier(
        loss_function="MultiClass",
        iterations=config.iterations,
        learning_rate=config.learning_rate,
        depth=config.depth,
        l2_leaf_reg=config.l2_leaf_reg,
        random_seed=config.seed,
        allow_writing_files=False,
        verbose=False,
    )
    model.fit(dataset.features.loc[mask], y)
    return model


def _state_metrics(predictions: pd.DataFrame, mask: pd.Series) -> dict[str, float | int]:
    subset = predictions.loc[mask].copy()
    if subset.empty:
        return {"rows": 0, "accuracy": float("nan")}
    target = pd.to_numeric(subset["target_state"], errors="coerce").to_numpy(dtype=np.float64)
    pred = pd.to_numeric(subset["pred_state"], errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(target) & np.isfinite(pred)
    if not valid.any():
        return {"rows": 0, "accuracy": float("nan")}
    expected = pd.to_numeric(subset["pred_expected_sign"], errors="coerce").to_numpy(dtype=np.float64)
    target_sign = target - 1.0
    return {
        "rows": int(valid.sum()),
        "wells": int(subset.loc[valid, "well_id"].nunique()),
        "accuracy": float((target[valid] == pred[valid]).mean()),
        "updown_sign_accuracy": float((np.sign(expected[valid]) == np.sign(target_sign[valid])).mean()),
        "mean_prob_target": float(
            np.mean(
                np.choose(
                    target[valid].astype(int),
                    [
                        subset.loc[valid, "prob_down"].to_numpy(dtype=np.float64),
                        subset.loc[valid, "prob_flat"].to_numpy(dtype=np.float64),
                        subset.loc[valid, "prob_up"].to_numpy(dtype=np.float64),
                    ],
                )
            )
        ),
    }


def _write_report(output_dir: Path, metrics: dict[str, Any], predictions: pd.DataFrame) -> None:
    lines = [
        "# TOP_STATE_TEACHER_V0",
        "",
        "Fold-safe classifier that uses train-only formation top derivatives as labels,",
        "but only test-safe `MD/X/Y/Z/GR/TVT_input` features as inputs.",
        "",
        "Formation tops are **not** deployable features. This artifact is a teacher/proxy experiment.",
        "",
        "## summary",
        "```json",
        json.dumps(_json_safe(metrics), indent=2),
        "```",
        "",
        "## predicted state counts",
        predictions["pred_state_name"].value_counts().to_string(),
    ]
    (output_dir / "top_state_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_figures(output_dir: Path, predictions: pd.DataFrame, *, n_panel_wells: int) -> None:
    import matplotlib.pyplot as plt

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    names = ["down", "flat", "up"]
    confusion = pd.crosstab(
        predictions["target_state"].map(lambda x: names[int(x)]),
        predictions["pred_state"].map(lambda x: names[int(x)]),
        normalize="index",
    ).reindex(index=names, columns=names).fillna(0.0)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(confusion.to_numpy(dtype=np.float64), cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(3), names)
    ax.set_yticks(range(3), names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("target")
    ax.set_title("Top-state OOF confusion", fontweight="bold")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{confusion.iloc[i, j]:.2f}", ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(fig_dir / "top_state_confusion.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    for well_id, well in list(predictions.groupby("well_id", sort=True))[: max(int(n_panel_wells), 0)]:
        ordered = well.sort_values("step")
        fig, ax = plt.subplots(figsize=(12, 3.8))
        x = ordered["step"].to_numpy(dtype=np.float64)
        ax.plot(x, ordered["target_state"].to_numpy(dtype=np.float64) - 1.0, color="black", label="target state")
        ax.plot(x, ordered["pred_expected_sign"].to_numpy(dtype=np.float64), color="#d62728", label="pred expected sign")
        ax.fill_between(x, -1.1, 1.1, where=ordered["hidden_frac"].to_numpy(dtype=np.float64) > 0.5, color="gray", alpha=0.08, step="mid")
        ax.set_ylim(-1.15, 1.15)
        ax.set_title(f"{well_id}: top-state teacher OOF", fontweight="bold")
        ax.set_xlabel("compressed step")
        ax.set_ylabel("state")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(fig_dir / f"{well_id}_top_state_panel.png", dpi=160, bbox_inches="tight")
        plt.close(fig)


def run_top_state_teacher_from_frame(
    frame: pd.DataFrame,
    *,
    config: TopStateTeacherConfig,
) -> dict[str, Any]:
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dataset = build_top_state_dataset(
        frame,
        rows_per_step=config.rows_per_step,
        target_top_column=config.target_top_column,
        flat_epsilon_ft=config.flat_epsilon_ft,
    )
    folds = make_group_folds(dataset.rows["well_id"], n_folds=config.n_folds, seed=config.seed)
    parts: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, Any]] = []
    for fold_idx, (train_wells, valid_wells) in enumerate(folds):
        print(
            f"[top-state] fold {fold_idx + 1}/{len(folds)} train_wells={len(train_wells)} "
            f"valid_wells={len(valid_wells)}",
            file=sys.stderr,
            flush=True,
        )
        model = _train_classifier(dataset, train_wells, config)
        valid_mask = dataset.rows["well_id"].astype(str).isin(set(valid_wells)).to_numpy()
        proba = _predict_proba_fixed(model, dataset.features.loc[valid_mask])
        local = dataset.rows.loc[valid_mask].copy()
        local["fold"] = fold_idx
        local["prob_down"] = proba[:, 0]
        local["prob_flat"] = proba[:, 1]
        local["prob_up"] = proba[:, 2]
        local["pred_state"] = np.argmax(proba, axis=1).astype(np.int8)
        local["pred_expected_sign"] = proba[:, 2] - proba[:, 0]
        local["pred_state_name"] = local["pred_state"].map({0: "down", 1: "flat", 2: "up"})
        fold_metric = _state_metrics(local, pd.Series(True, index=local.index))
        fold_metric.update(
            {
                "fold": fold_idx,
                "train_wells": int(len(train_wells)),
                "valid_wells": int(len(valid_wells)),
            }
        )
        fold_metrics.append(fold_metric)
        parts.append(local)
    predictions = pd.concat(parts, ignore_index=True)
    hidden_mask = pd.to_numeric(predictions["hidden_frac"], errors="coerce") > 0.5
    known_mask = pd.to_numeric(predictions["known_frac"], errors="coerce") > 0.5
    metrics: dict[str, Any] = {
        "candidate": "top_state_teacher_v0",
        "rows": int(len(predictions)),
        "wells": int(predictions["well_id"].nunique()),
        "folds": int(len(folds)),
        "target_top_column": config.target_top_column,
        "feature_columns": dataset.feature_columns,
        "config": asdict(config),
        "all": _state_metrics(predictions, pd.Series(True, index=predictions.index)),
        "hidden": _state_metrics(predictions, hidden_mask),
        "known": _state_metrics(predictions, known_mask),
        "fold_metrics": fold_metrics,
    }
    keep = [
        "well_id",
        "step",
        "row_idx_mean",
        "known_frac",
        "hidden_frac",
        "target_delta",
        "target_state",
        "pred_state",
        "pred_state_name",
        "pred_expected_sign",
        "prob_down",
        "prob_flat",
        "prob_up",
        "fold",
    ]
    predictions[keep].to_parquet(out / "top_state_oof_predictions.parquet", index=False)
    (out / "top_state_metrics.json").write_text(
        json.dumps(_json_safe(metrics), indent=2),
        encoding="utf-8",
    )
    _write_report(out, metrics, predictions)
    _write_figures(out, predictions, n_panel_wells=config.n_panel_wells)
    return metrics


def run_top_state_teacher(config: TopStateTeacherConfig) -> dict[str, Any]:
    frame = _load_horizontal_frame(config.data_dir, config.k_wells)
    return run_top_state_teacher_from_frame(frame, config=config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train fold-safe top-state teacher from train-only formation labels")
    parser.add_argument("--data-dir", type=Path, default=TopStateTeacherConfig.data_dir)
    parser.add_argument("--output-dir", type=Path, default=TopStateTeacherConfig.output_dir)
    parser.add_argument("--rows-per-step", type=int, default=TopStateTeacherConfig.rows_per_step)
    parser.add_argument("--n-folds", type=int, default=TopStateTeacherConfig.n_folds)
    parser.add_argument("--seed", type=int, default=TopStateTeacherConfig.seed)
    parser.add_argument("--k-wells", type=int, default=TopStateTeacherConfig.k_wells)
    parser.add_argument("--target-top-column", default=TopStateTeacherConfig.target_top_column)
    parser.add_argument("--flat-epsilon-ft", type=float, default=TopStateTeacherConfig.flat_epsilon_ft)
    parser.add_argument("--iterations", type=int, default=TopStateTeacherConfig.iterations)
    parser.add_argument("--learning-rate", type=float, default=TopStateTeacherConfig.learning_rate)
    parser.add_argument("--depth", type=int, default=TopStateTeacherConfig.depth)
    parser.add_argument("--l2-leaf-reg", type=float, default=TopStateTeacherConfig.l2_leaf_reg)
    parser.add_argument("--n-panel-wells", type=int, default=TopStateTeacherConfig.n_panel_wells)
    parser.add_argument("--progress-every", type=int, default=TopStateTeacherConfig.progress_every)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metrics = run_top_state_teacher(
        TopStateTeacherConfig(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            rows_per_step=args.rows_per_step,
            n_folds=args.n_folds,
            seed=args.seed,
            k_wells=args.k_wells,
            target_top_column=args.target_top_column,
            flat_epsilon_ft=args.flat_epsilon_ft,
            iterations=args.iterations,
            learning_rate=args.learning_rate,
            depth=args.depth,
            l2_leaf_reg=args.l2_leaf_reg,
            n_panel_wells=args.n_panel_wells,
            progress_every=args.progress_every,
        )
    )
    compact = {
        "candidate": metrics["candidate"],
        "wells": metrics["wells"],
        "rows": metrics["rows"],
        "hidden": metrics["hidden"],
        "report": str(Path(args.output_dir) / "top_state_report.md"),
    }
    print(json.dumps(_json_safe(compact), indent=2))


if __name__ == "__main__":
    main()
