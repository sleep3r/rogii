from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from .schema_safe import assert_schema_safe_columns


DEFAULT_BASE_PATH = Path("../old/artifacts/oof_baseline/schema10_oof.parquet")
DEFAULT_B2_PATH = Path(
    "../old/artifacts/formation_b2_danger_guard_a2_full_schema10/guarded_predictions.parquet"
)
DEFAULT_A_PATH = Path("../old/artifacts/formation_plane_knn/oof_candidates.parquet")
DEFAULT_PSEUDO_ZONE_PATH = Path("artifacts/pseudo_zone_v1/pseudo_zone_predictions.parquet")


@dataclass(frozen=True)
class ResidualStackConfig:
    data_dir: Path = Path("data/train")
    output_dir: Path = Path("artifacts/residual_stack_v0")
    rows_per_step: int = 32
    n_folds: int = 5
    seed: int = 42
    k_wells: int = -1
    iterations: int = 700
    learning_rate: float = 0.04
    depth: int = 6
    l2_leaf_reg: float = 8.0
    base_path: Path | None = DEFAULT_BASE_PATH
    base_column: str = "schema10_oof_pp"
    b2_path: Path | None = DEFAULT_B2_PATH
    b2_column: str = "b2_guarded_submit"
    a_path: Path | None = DEFAULT_A_PATH
    a_p50_column: str = "formation_sample_median"
    a_p10_column: str = "formation_sample_p10"
    a_p90_column: str = "formation_sample_p90"
    pseudo_zone_path: Path | None = DEFAULT_PSEUDO_ZONE_PATH
    tail_audit_path: Path | None = Path("artifacts/tail_audit_v1/well_tail_audit.csv")


@dataclass
class ResidualStepDataset:
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
    return value


def _rmse(values: np.ndarray | pd.Series) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(arr))))


def _mean_or_nan(values: pd.Series | np.ndarray) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def _slope(values: pd.Series | np.ndarray) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=np.float64)
    finite = np.isfinite(arr)
    if finite.sum() < 2:
        return 0.0
    x = np.arange(arr.size, dtype=np.float64)[finite]
    y = arr[finite]
    return float(np.polyfit(x, y, 1)[0])


def _std_or_zero(values: pd.Series | np.ndarray) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return 0.0
    return float(arr.std())


def _ensure_ids(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "well_id" not in out.columns:
        raise ValueError("frame must contain well_id")
    if "row_idx" not in out.columns:
        out["row_idx"] = out.groupby("well_id").cumcount()
    if "id" not in out.columns:
        out["id"] = [
            f"{well_id}_{int(row_idx)}"
            for well_id, row_idx in zip(out["well_id"], out["row_idx"], strict=False)
        ]
    out["id"] = out["id"].astype(str)
    out["well_id"] = out["well_id"].astype(str)
    return out


def _choose_anchor(frame: pd.DataFrame) -> pd.Series:
    if "anchor_tvt" in frame.columns:
        anchor = pd.to_numeric(frame["anchor_tvt"], errors="coerce")
    else:
        anchor = pd.Series(np.nan, index=frame.index, dtype=np.float64)
    for column in ("b2_tvt", "base_tvt", "a_p50_tvt"):
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            anchor = anchor.where(anchor.notna(), values)
    return anchor


def _add_missing_prior_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in ("b2_tvt", "base_tvt", "a_p50_tvt", "a_p10_tvt", "a_p90_tvt"):
        if column not in out.columns:
            out[column] = np.nan
    out["anchor_tvt"] = _choose_anchor(out)
    return out


def _load_one_prior(path: Path | None, source_column: str, dest_column: str) -> pd.DataFrame:
    if path is None or not Path(path).exists():
        return pd.DataFrame(columns=["id", dest_column]).set_index("id")
    p = Path(path)
    if p.suffix.lower() == ".csv":
        frame = pd.read_csv(p, usecols=["id", source_column])
    else:
        frame = pd.read_parquet(p, columns=["id", source_column])
    frame = frame.rename(columns={source_column: dest_column})
    frame["id"] = frame["id"].astype(str)
    return frame.drop_duplicates("id").set_index("id")


def load_prior_frame(config: ResidualStackConfig) -> pd.DataFrame:
    parts = [
        _load_one_prior(config.base_path, config.base_column, "base_tvt"),
        _load_one_prior(config.b2_path, config.b2_column, "b2_tvt"),
        _load_one_prior(config.a_path, config.a_p50_column, "a_p50_tvt"),
        _load_one_prior(config.a_path, config.a_p10_column, "a_p10_tvt"),
        _load_one_prior(config.a_path, config.a_p90_column, "a_p90_tvt"),
    ]
    result = parts[0]
    for part in parts[1:]:
        result = result.join(part, how="outer")
    return result


def load_training_frame(config: ResidualStackConfig) -> pd.DataFrame:
    data_dir = Path(config.data_dir)
    prior_frame = load_prior_frame(config)
    tail_map: dict[str, str] = {}
    if config.tail_audit_path is not None and Path(config.tail_audit_path).exists():
        tail = pd.read_csv(config.tail_audit_path)
        tail_map = dict(zip(tail["well_id"].astype(str), tail["tail_class"].astype(str)))

    paths = sorted(data_dir.glob("*__horizontal_well.csv"))
    if config.k_wells > 0:
        paths = paths[: config.k_wells]
    rows: list[pd.DataFrame] = []
    for path in paths:
        well_id = path.name.replace("__horizontal_well.csv", "")
        frame = pd.read_csv(path)
        frame["well_id"] = well_id
        frame["row_idx"] = np.arange(len(frame), dtype=np.int32)
        frame["id"] = [f"{well_id}_{idx}" for idx in frame["row_idx"]]
        for column in ("MD", "X", "Y", "Z", "GR", "TVT", "TVT_input"):
            if column not in frame.columns:
                frame[column] = np.nan
        joined = frame.join(prior_frame, on="id")
        joined["tail_class"] = tail_map.get(well_id, "unknown")
        rows.append(joined)
    if not rows:
        raise FileNotFoundError(f"No horizontal wells found in {data_dir}")
    return pd.concat(rows, ignore_index=True)


def _zone_features(frame: pd.DataFrame) -> pd.DataFrame:
    zone_cols = [
        col
        for col in frame.columns
        if col.startswith("neural_prob_") or col.startswith("template_prob_")
    ]
    if not zone_cols:
        return pd.DataFrame(index=frame.index)
    return frame[["well_id", "step", *zone_cols]].drop_duplicates(["well_id", "step"])


def _attach_pseudo_zone(step_rows: pd.DataFrame, pseudo_zone_path: Path | None) -> pd.DataFrame:
    if pseudo_zone_path is None or not Path(pseudo_zone_path).exists():
        return step_rows
    zones = pd.read_parquet(pseudo_zone_path)
    zone_features = _zone_features(zones)
    if zone_features.empty:
        return step_rows
    return step_rows.merge(zone_features, on=["well_id", "step"], how="left")


def build_residual_step_dataset(
    frame: pd.DataFrame,
    *,
    rows_per_step: int = 32,
    pseudo_zone_path: Path | None = None,
) -> ResidualStepDataset:
    raw = _add_missing_prior_columns(_ensure_ids(frame))
    if "step" not in raw.columns:
        raw["step"] = (pd.to_numeric(raw["row_idx"], errors="coerce") // rows_per_step).astype(int)
    rows: list[dict[str, Any]] = []
    for well_id, well in raw.sort_values(["well_id", "row_idx"]).groupby("well_id", sort=True):
        hidden_mask = (
            pd.to_numeric(well["TVT_input"], errors="coerce").isna()
            if "TVT_input" in well.columns
            else pd.Series(True, index=well.index)
        )
        hidden_steps = set(well.loc[hidden_mask, "step"].astype(int).tolist())
        if not hidden_steps:
            continue
        known = well.loc[~hidden_mask]
        if known.empty:
            last_known_tvt = _mean_or_nan(well["anchor_tvt"])
            last_known_row = int(pd.to_numeric(well["row_idx"], errors="coerce").min())
            known_slope = 0.0
        else:
            last_known = known.iloc[-1]
            last_known_tvt = float(pd.to_numeric(pd.Series([last_known["TVT_input"]]), errors="coerce").iloc[0])
            last_known_row = int(last_known["row_idx"])
            known_slope = _slope(known.tail(5)["TVT_input"])
        total_hidden_rows = int(hidden_mask.sum())
        for step, group in well.groupby("step", sort=True):
            if int(step) not in hidden_steps:
                continue
            group_hidden = group.loc[hidden_mask.reindex(group.index).fillna(False)]
            if group_hidden.empty:
                continue
            anchor = _mean_or_nan(group_hidden["anchor_tvt"])
            target_tvt = _mean_or_nan(group_hidden["TVT"]) if "TVT" in group_hidden.columns else np.nan
            row_idx_mean = _mean_or_nan(group_hidden["row_idx"])
            b2 = _mean_or_nan(group_hidden["b2_tvt"])
            base = _mean_or_nan(group_hidden["base_tvt"])
            a50 = _mean_or_nan(group_hidden["a_p50_tvt"])
            a10 = _mean_or_nan(group_hidden["a_p10_tvt"])
            a90 = _mean_or_nan(group_hidden["a_p90_tvt"])
            rec: dict[str, Any] = {
                "well_id": str(well_id),
                "step": int(step),
                "row_idx_mean": row_idx_mean,
                "n_hidden_rows": int(len(group_hidden)),
                "target_residual": target_tvt - anchor if np.isfinite(target_tvt) and np.isfinite(anchor) else np.nan,
                "target_tvt": target_tvt,
                "anchor_tvt": anchor,
                "b2_tvt": b2,
                "base_tvt": base,
                "a_p50_tvt": a50,
                "a_spread": (a90 - a10) if np.isfinite(a90) and np.isfinite(a10) else 0.0,
                "tail_class": str(group_hidden.get("tail_class", pd.Series(["unknown"])).iloc[0]),
                "feat_hidden_progress": (row_idx_mean - last_known_row) / max(total_hidden_rows, 1),
                "feat_distance_from_last_known_rows": row_idx_mean - last_known_row,
                "feat_last_known_tvt": last_known_tvt / 10000.0 if np.isfinite(last_known_tvt) else 0.0,
                "feat_known_slope": known_slope,
                "feat_md_rel": (_mean_or_nan(group_hidden.get("MD", pd.Series(np.nan, index=group_hidden.index))) - _mean_or_nan(known.get("MD", pd.Series(np.nan)))) / 1000.0,
                "feat_x_rel": (_mean_or_nan(group_hidden.get("X", pd.Series(np.nan, index=group_hidden.index))) - _mean_or_nan(known.get("X", pd.Series(np.nan)))) / 1000.0,
                "feat_y_rel": (_mean_or_nan(group_hidden.get("Y", pd.Series(np.nan, index=group_hidden.index))) - _mean_or_nan(known.get("Y", pd.Series(np.nan)))) / 1000.0,
                "feat_z_rel": (_mean_or_nan(group_hidden.get("Z", pd.Series(np.nan, index=group_hidden.index))) - _mean_or_nan(known.get("Z", pd.Series(np.nan)))) / 100.0,
                "feat_gr_mean": _mean_or_nan(group_hidden.get("GR", pd.Series(np.nan, index=group_hidden.index))) / 100.0,
                "feat_gr_std": _std_or_zero(group_hidden.get("GR", pd.Series(np.nan, index=group_hidden.index))) / 50.0,
                "feat_gr_valid_frac": float(pd.to_numeric(group_hidden.get("GR", pd.Series(np.nan, index=group_hidden.index)), errors="coerce").notna().mean()),
                "feat_anchor_delta_last": (anchor - last_known_tvt) / 100.0 if np.isfinite(anchor) and np.isfinite(last_known_tvt) else 0.0,
                "feat_b2_delta_anchor": (b2 - anchor) / 50.0 if np.isfinite(b2) and np.isfinite(anchor) else 0.0,
                "feat_base_delta_anchor": (base - anchor) / 50.0 if np.isfinite(base) and np.isfinite(anchor) else 0.0,
                "feat_a50_delta_anchor": (a50 - anchor) / 50.0 if np.isfinite(a50) and np.isfinite(anchor) else 0.0,
                "feat_b2_base_gap": (b2 - base) / 50.0 if np.isfinite(b2) and np.isfinite(base) else 0.0,
                "feat_a_spread": ((a90 - a10) / 100.0) if np.isfinite(a90) and np.isfinite(a10) else 0.0,
                "feat_b2_available": float(np.isfinite(b2)),
                "feat_base_available": float(np.isfinite(base)),
                "feat_a_available": float(np.isfinite(a50)),
            }
            rows.append(rec)
    if not rows:
        raise ValueError("No hidden step rows available for residual stack")
    step_rows = pd.DataFrame(rows)
    step_rows = _attach_pseudo_zone(step_rows, pseudo_zone_path)
    feature_columns = [
        col
        for col in step_rows.columns
        if col.startswith("feat_") or col.startswith("neural_prob_") or col.startswith("template_prob_")
    ]
    assert_schema_safe_columns(feature_columns, context="ResidualStack features")
    features = step_rows[feature_columns].apply(pd.to_numeric, errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    return ResidualStepDataset(rows=step_rows, features=features, feature_columns=feature_columns)


def make_group_folds(
    well_ids: list[str] | np.ndarray | pd.Series,
    *,
    n_folds: int,
    seed: int,
) -> list[tuple[list[str], list[str]]]:
    unique = np.asarray(sorted({str(well_id) for well_id in well_ids}), dtype=object)
    if unique.size < 2:
        raise ValueError("Need at least two wells for grouped folds")
    n = min(max(2, int(n_folds)), int(unique.size))
    rng = np.random.default_rng(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    valid_splits = np.array_split(shuffled, n)
    folds: list[tuple[list[str], list[str]]] = []
    all_set = set(unique.tolist())
    for split in valid_splits:
        valid = [str(item) for item in split.tolist()]
        train = sorted(all_set.difference(valid))
        folds.append((train, valid))
    return folds


def train_fold_model(
    dataset: ResidualStepDataset,
    *,
    train_wells: list[str],
    config: ResidualStackConfig | None,
) -> CatBoostRegressor:
    assert_schema_safe_columns(dataset.feature_columns, context="ResidualStack features")
    cfg = config or ResidualStackConfig()
    train_mask = dataset.rows["well_id"].astype(str).isin(set(train_wells)).to_numpy()
    if not train_mask.any():
        raise ValueError("No training rows selected for residual fold")
    y = pd.to_numeric(dataset.rows.loc[train_mask, "target_residual"], errors="coerce")
    finite = np.isfinite(y.to_numpy(dtype=np.float64))
    if not finite.any():
        raise ValueError("No finite residual targets in residual fold")
    x_train = dataset.features.loc[train_mask].iloc[finite]
    y_train = y.iloc[finite]
    model = CatBoostRegressor(
        loss_function="RMSE",
        iterations=cfg.iterations,
        learning_rate=cfg.learning_rate,
        depth=cfg.depth,
        l2_leaf_reg=cfg.l2_leaf_reg,
        random_seed=cfg.seed,
        allow_writing_files=False,
        verbose=False,
    )
    model.fit(x_train, y_train)
    return model


def _row_level_predictions(
    hidden_frame: pd.DataFrame,
    step_predictions: pd.DataFrame,
    *,
    rows_per_step: int,
) -> pd.DataFrame:
    raw = _add_missing_prior_columns(_ensure_ids(hidden_frame))
    if "step" not in raw.columns:
        raw["step"] = (pd.to_numeric(raw["row_idx"], errors="coerce") // rows_per_step).astype(int)
    hidden_mask = (
        pd.to_numeric(raw["TVT_input"], errors="coerce").isna()
        if "TVT_input" in raw.columns
        else pd.Series(True, index=raw.index)
    )
    rows = raw.loc[hidden_mask].copy()
    merged = rows.merge(
        step_predictions[["well_id", "step", "pred_residual", "fold"]],
        on=["well_id", "step"],
        how="left",
    )
    merged["pred_tvt"] = pd.to_numeric(merged["anchor_tvt"], errors="coerce") + pd.to_numeric(
        merged["pred_residual"], errors="coerce"
    )
    merged["candidate"] = "residual_stack_v0"
    keep = [
        "id",
        "well_id",
        "row_idx",
        "step",
        "candidate",
        "pred_tvt",
        "TVT",
        "TVT_input",
        "GR",
        "anchor_tvt",
        "b2_tvt",
        "base_tvt",
        "a_p50_tvt",
        "tail_class",
        "fold",
    ]
    return merged[[col for col in keep if col in merged.columns]].reset_index(drop=True)


def _metrics_from_predictions(predictions: pd.DataFrame, fold_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    pred = pd.to_numeric(predictions["pred_tvt"], errors="coerce")
    true = pd.to_numeric(predictions["TVT"], errors="coerce")
    anchor = pd.to_numeric(predictions["anchor_tvt"], errors="coerce")
    valid = np.isfinite(pred) & np.isfinite(true)
    anchor_valid = np.isfinite(anchor) & np.isfinite(true)
    by_well = []
    for well_id, group in predictions.loc[valid].groupby("well_id"):
        err = pd.to_numeric(group["pred_tvt"], errors="coerce") - pd.to_numeric(group["TVT"], errors="coerce")
        anchor_err = pd.to_numeric(group["anchor_tvt"], errors="coerce") - pd.to_numeric(group["TVT"], errors="coerce")
        by_well.append(
            {
                "well_id": str(well_id),
                "rmse": _rmse(err),
                "anchor_rmse": _rmse(anchor_err),
                "tail_class": str(group["tail_class"].iloc[0]) if "tail_class" in group.columns else "unknown",
            }
        )
    well_df = pd.DataFrame(by_well)
    tail_rows: list[dict[str, Any]] = []
    if not well_df.empty:
        for tail_class, group in well_df.groupby("tail_class"):
            tail_rows.append(
                {
                    "tail_class": str(tail_class),
                    "wells": int(len(group)),
                    "mean_rmse": float(group["rmse"].mean()),
                    "mean_anchor_rmse": float(group["anchor_rmse"].mean()),
                    "mean_gain": float(group["anchor_rmse"].mean() - group["rmse"].mean()),
                }
            )
    return {
        "candidate": "residual_stack_v0",
        "rows": int(valid.sum()),
        "wells": int(predictions.loc[valid, "well_id"].nunique()),
        "folds": int(len(fold_metrics)),
        "row_rmse_ft": _rmse(pred[valid].to_numpy() - true[valid].to_numpy()),
        "anchor_row_rmse_ft": _rmse(anchor[anchor_valid].to_numpy() - true[anchor_valid].to_numpy()),
        "row_gain_vs_anchor_ft": _rmse(anchor[anchor_valid].to_numpy() - true[anchor_valid].to_numpy())
        - _rmse(pred[valid].to_numpy() - true[valid].to_numpy()),
        "mean_well_rmse_ft": float(well_df["rmse"].mean()) if not well_df.empty else float("nan"),
        "mean_anchor_well_rmse_ft": float(well_df["anchor_rmse"].mean()) if not well_df.empty else float("nan"),
        "p95_well_rmse_ft": float(well_df["rmse"].quantile(0.95)) if not well_df.empty else float("nan"),
        "worst_well_rmse_ft": float(well_df["rmse"].max()) if not well_df.empty else float("nan"),
        "fold_metrics": fold_metrics,
        "tail_class_metrics": tail_rows,
    }


def _write_svg_bars(path: Path, title: str, labels: list[str], values: list[float]) -> None:
    width = 820
    height = max(220, 80 + 32 * len(labels))
    finite = [abs(v) for v in values if np.isfinite(v)]
    scale = max(finite) if finite else 1.0
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="24" y="32" font-family="Arial" font-size="18" font-weight="700">{title}</text>',
    ]
    x0 = 260
    zero = x0 + 220
    lines.append(f'<line x1="{zero}" y1="50" x2="{zero}" y2="{height - 24}" stroke="#777"/>')
    for i, (label, value) in enumerate(zip(labels, values, strict=False)):
        y = 70 + i * 32
        bar = 200 * (abs(value) / scale if scale else 0.0)
        color = "#2c7fb8" if value >= 0 else "#d95f0e"
        x = zero if value >= 0 else zero - bar
        lines.append(f'<text x="24" y="{y + 14}" font-family="Arial" font-size="12">{label}</text>')
        lines.append(f'<rect x="{x}" y="{y}" width="{bar}" height="18" fill="{color}"/>')
        lines.append(f'<text x="{zero + (bar + 8 if value >= 0 else -bar - 54)}" y="{y + 14}" font-family="Arial" font-size="12">{value:.3f}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_report(output_dir: Path, metrics: dict[str, Any]) -> None:
    lines = [
        "RESIDUAL_STACK_V0_REPORT",
        "",
        "summary:",
        json.dumps(_json_safe({k: v for k, v in metrics.items() if k != "feature_columns"}), indent=2),
        "",
        "feature_columns:",
        "\n".join(f"- {col}" for col in metrics.get("feature_columns", [])),
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_residual_stack_from_frames(
    frame: pd.DataFrame,
    *,
    config: ResidualStackConfig,
) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_residual_step_dataset(
        frame,
        rows_per_step=config.rows_per_step,
        pseudo_zone_path=config.pseudo_zone_path,
    )
    folds = make_group_folds(dataset.rows["well_id"], n_folds=config.n_folds, seed=config.seed)
    step_pred_parts: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, Any]] = []
    for fold_idx, (train_wells, valid_wells) in enumerate(folds):
        model = train_fold_model(dataset, train_wells=train_wells, config=config)
        valid_mask = dataset.rows["well_id"].astype(str).isin(set(valid_wells)).to_numpy()
        pred = model.predict(dataset.features.loc[valid_mask])
        local = dataset.rows.loc[valid_mask, ["well_id", "step", "target_residual"]].copy()
        local["pred_residual"] = pred.astype(np.float32)
        local["fold"] = fold_idx
        step_pred_parts.append(local)
        finite = np.isfinite(local["target_residual"])
        fold_metrics.append(
            {
                "fold": fold_idx,
                "train_wells": int(len(train_wells)),
                "valid_wells": int(len(valid_wells)),
                "valid_steps": int(finite.sum()),
                "residual_rmse_ft": _rmse(
                    local.loc[finite, "pred_residual"].to_numpy()
                    - local.loc[finite, "target_residual"].to_numpy()
                ),
            }
        )
    step_predictions = pd.concat(step_pred_parts, ignore_index=True)
    row_predictions = _row_level_predictions(frame, step_predictions, rows_per_step=config.rows_per_step)
    row_predictions.to_parquet(output_dir / "oof_predictions.parquet", index=False)
    step_predictions.to_parquet(output_dir / "oof_step_predictions.parquet", index=False)
    pd.DataFrame(fold_metrics).to_csv(output_dir / "fold_metrics.csv", index=False)
    metrics = _metrics_from_predictions(row_predictions, fold_metrics)
    metrics["feature_columns"] = dataset.feature_columns
    (output_dir / "metrics.json").write_text(
        json.dumps(_json_safe(metrics), indent=2), encoding="utf-8"
    )
    _write_report(output_dir, metrics)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    _write_svg_bars(
        fig_dir / "fold_residual_rmse.svg",
        "Residual RMSE by fold",
        [f"fold {item['fold']}" for item in fold_metrics],
        [float(item["residual_rmse_ft"]) for item in fold_metrics],
    )
    tail = metrics.get("tail_class_metrics", [])
    _write_svg_bars(
        fig_dir / "tail_class_gain.svg",
        "Mean gain vs anchor by tail class",
        [str(item["tail_class"]) for item in tail],
        [float(item["mean_gain"]) for item in tail],
    )
    return metrics


def run_residual_stack(config: ResidualStackConfig) -> dict[str, Any]:
    frame = load_training_frame(config)
    metrics = run_residual_stack_from_frames(frame, config=config)
    print(json.dumps(_json_safe({k: v for k, v in metrics.items() if k != "feature_columns"}), indent=2))
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train fold-safe schema-safe residual stack")
    parser.add_argument("--data-dir", type=Path, default=ResidualStackConfig.data_dir)
    parser.add_argument("--output-dir", type=Path, default=ResidualStackConfig.output_dir)
    parser.add_argument("--rows-per-step", type=int, default=ResidualStackConfig.rows_per_step)
    parser.add_argument("--n-folds", type=int, default=ResidualStackConfig.n_folds)
    parser.add_argument("--seed", type=int, default=ResidualStackConfig.seed)
    parser.add_argument("--k-wells", type=int, default=ResidualStackConfig.k_wells)
    parser.add_argument("--iterations", type=int, default=ResidualStackConfig.iterations)
    parser.add_argument("--learning-rate", type=float, default=ResidualStackConfig.learning_rate)
    parser.add_argument("--depth", type=int, default=ResidualStackConfig.depth)
    parser.add_argument("--l2-leaf-reg", type=float, default=ResidualStackConfig.l2_leaf_reg)
    parser.add_argument("--pseudo-zone-path", type=Path, default=ResidualStackConfig.pseudo_zone_path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_residual_stack(
        ResidualStackConfig(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            rows_per_step=args.rows_per_step,
            n_folds=args.n_folds,
            seed=args.seed,
            k_wells=args.k_wells,
            iterations=args.iterations,
            learning_rate=args.learning_rate,
            depth=args.depth,
            l2_leaf_reg=args.l2_leaf_reg,
            pseudo_zone_path=args.pseudo_zone_path,
        )
    )


if __name__ == "__main__":
    main()
