from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from .numeric import flat_tvt_prediction

FORMATIONS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
TEACHER_COLUMNS: tuple[str, ...] = (
    "row_index",
    "geo_teacher_tvt",
    "geo_teacher_delta_last",
    "geo_teacher_minus_flat",
    "geo_teacher_conf",
    "teacher_best_surface_id",
    "teacher_surface_agreement",
    "teacher_surface_spread",
    "teacher_surface_fit_rmse_min",
    "teacher_surface_fit_rmse_gap",
    *(f"z_minus_{formation}_true" for formation in FORMATIONS),
)


def _finite(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def _full_nan_teacher(row_index: np.ndarray, n_rows: int) -> pd.DataFrame:
    data: dict[str, np.ndarray] = {"row_index": row_index.astype(int)}
    for column in TEACHER_COLUMNS:
        if column != "row_index":
            data[column] = np.full(len(row_index), np.nan, dtype=float)
    data["teacher_missing_surfaces"] = np.ones(len(row_index), dtype=float)
    data["teacher_hidden_rows"] = np.full(len(row_index), int(len(row_index)), dtype=float)
    data["teacher_total_rows"] = np.full(len(row_index), int(n_rows), dtype=float)
    return pd.DataFrame(data)


def _ridge_fit(x: np.ndarray, y: np.ndarray, ridge: float = 1e-2) -> np.ndarray | None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(y) & np.isfinite(x).all(axis=1)
    if int(mask.sum()) < x.shape[1] + 2:
        return None
    xs = x[mask]
    ys = y[mask]
    scale = np.nanstd(xs, axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
    xs_scaled = xs / scale
    xtx = xs_scaled.T @ xs_scaled
    penalty = np.eye(xtx.shape[0]) * float(ridge)
    penalty[0, 0] = 0.0
    try:
        beta_scaled = np.linalg.solve(xtx + penalty, xs_scaled.T @ ys)
    except np.linalg.LinAlgError:
        beta_scaled = np.linalg.lstsq(xtx + penalty, xs_scaled.T @ ys, rcond=None)[0]
    return (beta_scaled / scale).astype(float)


def _robust_line_slope(x: np.ndarray, y: np.ndarray, default: float = 0.0) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float(default)
    xx = x[mask]
    yy = y[mask]
    dx = xx - np.median(xx)
    dy = yy - np.median(yy)
    denom = float(np.dot(dx, dx))
    if denom <= 1e-9:
        return float(default)
    slope = float(np.dot(dx, dy) / denom)
    return slope if np.isfinite(slope) else float(default)


def _clip_path_steps(
    path: np.ndarray,
    md: np.ndarray,
    hidden_idx: np.ndarray,
    last_idx: int,
    last_tvt: float,
    tail_slope_abs: float,
) -> np.ndarray:
    path = np.asarray(path, dtype=float).copy()
    if len(hidden_idx) == 0:
        return path
    max_slope = max(0.015, min(0.35, 4.0 * float(tail_slope_abs)))
    prev_idx = int(last_idx)
    prev_value = float(last_tvt)
    for idx in hidden_idx:
        step_md = abs(float(md[int(idx)] - md[prev_idx]))
        max_step = max(0.75, max_slope * max(step_md, 1.0))
        current = float(path[int(idx)])
        if not np.isfinite(current):
            current = prev_value
        current = float(np.clip(current, prev_value - max_step, prev_value + max_step))
        path[int(idx)] = current
        prev_idx = int(idx)
        prev_value = current
    return path


def _known_tail_mask(tvt_input: np.ndarray, last_idx: int, tail_rows: int) -> np.ndarray:
    known = np.isfinite(tvt_input)
    start = max(0, int(last_idx) - int(tail_rows) + 1)
    mask = np.zeros(len(tvt_input), dtype=bool)
    mask[start : int(last_idx) + 1] = True
    return mask & known


def _weighted_median_paths(paths: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if paths.ndim != 2 or paths.shape[0] == 0:
        return np.full(paths.shape[1] if paths.ndim == 2 else 0, np.nan, dtype=float)
    finite_weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    if float(finite_weights.sum()) <= 0.0:
        return np.nanmedian(paths, axis=0)
    normalized = finite_weights / float(finite_weights.sum())
    repeats = np.maximum(np.round(normalized * 100).astype(int), 1)
    replicated = np.vstack(
        [paths[index] for index, count in enumerate(repeats) for _ in range(int(count))]
    )
    return np.nanmedian(replicated, axis=0)


def _config_value(config: dict[str, Any] | None, name: str, default: Any) -> Any:
    config = config or {}
    teacher_cfg = config.get("surface_teacher") or {}
    feature_cfg = (config.get("features") or {}).get("surface_teacher") or {}
    if name in teacher_cfg:
        return teacher_cfg[name]
    if name in feature_cfg:
        return feature_cfg[name]
    return default


def build_geo_teacher_for_well(
    horizontal_df: pd.DataFrame,
    typewell_df: pd.DataFrame | None,
    config: dict[str, Any] | None,
) -> pd.DataFrame:
    """Build privileged train-only formation teacher labels for hidden rows.

    The teacher intentionally reads raw formation surfaces from the train
    dataframe. It must not be called as an inference-time feature builder.
    """

    _ = typewell_df
    n = len(horizontal_df)
    if n == 0:
        return _full_nan_teacher(np.array([], dtype=int), 0)

    md = pd.to_numeric(horizontal_df.get("MD"), errors="coerce").to_numpy(dtype=float)
    z = pd.to_numeric(horizontal_df.get("Z"), errors="coerce").to_numpy(dtype=float)
    tvt_input = pd.to_numeric(
        horizontal_df.get("TVT_input", pd.Series(np.full(n, np.nan))),
        errors="coerce",
    ).to_numpy(dtype=float)
    hidden_idx = np.flatnonzero(~np.isfinite(tvt_input))
    if len(hidden_idx) == 0:
        return _full_nan_teacher(np.array([], dtype=int), n)

    available_surfaces = [formation for formation in FORMATIONS if formation in horizontal_df.columns]
    if not available_surfaces:
        return _full_nan_teacher(hidden_idx, n)

    known_before = np.flatnonzero(np.isfinite(tvt_input) & (np.arange(n) < int(hidden_idx[0])))
    if len(known_before) == 0:
        known_before = np.flatnonzero(np.isfinite(tvt_input))
    if len(known_before) == 0:
        return _full_nan_teacher(hidden_idx, n)

    tail_rows = int(_config_value(config, "tail_rows", 384))
    ridge = float(_config_value(config, "ridge", 1e-2))
    min_fit_rows = int(_config_value(config, "min_fit_rows", 16))
    last_idx = int(known_before[-1])
    last_tvt = float(tvt_input[last_idx])
    last_md = float(md[last_idx]) if np.isfinite(md[last_idx]) else 0.0
    dmd = md - last_md
    dmd_scale = max(float(np.nanmax(np.abs(dmd))) if np.isfinite(dmd).any() else 1.0, 1.0)
    dmd_norm = dmd / dmd_scale
    tail_mask = _known_tail_mask(tvt_input, last_idx, tail_rows)
    tail_slope_abs = abs(_robust_line_slope(md[tail_mask], tvt_input[tail_mask], default=0.02))
    flat_pred = flat_tvt_prediction(md, tvt_input, tail_rows)

    surface_paths: list[np.ndarray] = []
    surface_rmse: list[float] = []
    surface_ids: list[int] = []
    z_minus: dict[str, np.ndarray] = {}

    for surface in FORMATIONS:
        if surface not in horizontal_df.columns:
            z_minus[surface] = np.full(n, np.nan, dtype=float)
            continue
        surface_values = pd.to_numeric(horizontal_df[surface], errors="coerce").to_numpy(dtype=float)
        rel = z - surface_values
        z_minus[surface] = rel
        xmat = np.column_stack([np.ones(n, dtype=float), rel, dmd_norm])
        valid = tail_mask & np.isfinite(rel)
        if int(valid.sum()) < min_fit_rows:
            continue
        beta = _ridge_fit(xmat[valid], tvt_input[valid], ridge=ridge)
        if beta is None:
            continue
        pred = xmat @ beta
        if np.isfinite(pred[last_idx]):
            pred = pred + (last_tvt - float(pred[last_idx]))
        rmse = float(np.sqrt(np.nanmean((pred[valid] - tvt_input[valid]) ** 2)))
        if not np.isfinite(rmse):
            continue
        pred = _clip_path_steps(pred, md, hidden_idx, last_idx, last_tvt, tail_slope_abs)
        surface_paths.append(pred[hidden_idx])
        surface_rmse.append(rmse)
        surface_ids.append(FORMATIONS.index(surface))

    if not surface_paths:
        frame = _full_nan_teacher(hidden_idx, n)
        for surface in FORMATIONS:
            frame[f"z_minus_{surface}_true"] = z_minus[surface][hidden_idx]
        return frame

    stack = np.vstack(surface_paths)
    rmse_arr = np.asarray(surface_rmse, dtype=float)
    finite_rmse = np.where(np.isfinite(rmse_arr) & (rmse_arr > 1e-6), rmse_arr, np.inf)
    weights = 1.0 / (finite_rmse + 1e-3)
    teacher = _weighted_median_paths(stack, weights)
    spread = np.nanstd(stack, axis=0) if stack.shape[0] > 1 else np.zeros(len(hidden_idx), dtype=float)
    agreement = 1.0 / (1.0 + spread)

    order = np.argsort(rmse_arr)
    best_pos = int(order[0])
    best_rmse = float(rmse_arr[best_pos])
    second_rmse = float(rmse_arr[int(order[1])]) if len(order) > 1 else float("nan")
    fit_gap = second_rmse - best_rmse if np.isfinite(second_rmse) else float("nan")
    rmse_spread = (
        float(np.nanmax(rmse_arr) - np.nanmin(rmse_arr))
        if len(rmse_arr)
        else float("nan")
    )
    conf = 1.0 / (1.0 + best_rmse + spread)

    output = pd.DataFrame(
        {
            "row_index": hidden_idx.astype(int),
            "geo_teacher_tvt": teacher,
            "geo_teacher_delta_last": teacher - last_tvt,
            "geo_teacher_minus_flat": teacher - flat_pred[hidden_idx],
            "geo_teacher_conf": conf,
            "teacher_best_surface_id": np.full(len(hidden_idx), surface_ids[best_pos], dtype=float),
            "teacher_surface_agreement": agreement,
            "teacher_surface_spread": spread,
            "teacher_surface_fit_rmse_min": np.full(len(hidden_idx), best_rmse, dtype=float),
            "teacher_surface_fit_rmse_gap": np.full(len(hidden_idx), fit_gap, dtype=float),
            "teacher_missing_surfaces": np.zeros(len(hidden_idx), dtype=float),
            "teacher_hidden_rows": np.full(len(hidden_idx), int(len(hidden_idx)), dtype=float),
            "teacher_total_rows": np.full(len(hidden_idx), int(n), dtype=float),
        }
    )
    for surface in FORMATIONS:
        output[f"z_minus_{surface}_true"] = z_minus[surface][hidden_idx]
    output["teacher_surface_rmse_spread"] = np.full(len(hidden_idx), rmse_spread, dtype=float)
    output["teacher_surface_count"] = np.full(len(hidden_idx), len(surface_paths), dtype=float)
    return output


def _rmse(pred: np.ndarray, true: np.ndarray) -> float:
    mask = np.isfinite(pred) & np.isfinite(true)
    if not np.any(mask):
        return float("nan")
    err = pred[mask] - true[mask]
    return float(np.sqrt(np.mean(err * err)))


def _safe_quantile(values: np.ndarray, q: float) -> float:
    finite = _finite(values)
    return float(np.quantile(finite, q)) if len(finite) else float("nan")


def _summarize_by_well(rows: pd.DataFrame) -> pd.DataFrame:
    grouped: list[dict[str, Any]] = []
    for well, frame in rows.groupby("well", sort=True):
        grouped.append(
            {
                "well": str(well),
                "rows": int(len(frame)),
                "coverage": float(frame["geo_teacher_tvt"].notna().mean()) if len(frame) else 0.0,
                "rmse": _rmse(
                    frame["geo_teacher_tvt"].to_numpy(dtype=float),
                    frame["tvt_true"].to_numpy(dtype=float),
                ),
                "mean_conf": float(np.nanmean(frame["geo_teacher_conf"].to_numpy(dtype=float))),
                "best_surface_id": float(np.nanmedian(frame["teacher_best_surface_id"].to_numpy(dtype=float))),
                "fit_rmse_min": float(np.nanmedian(frame["teacher_surface_fit_rmse_min"].to_numpy(dtype=float))),
                "surface_spread": float(np.nanmedian(frame["teacher_surface_spread"].to_numpy(dtype=float))),
            }
        )
    return pd.DataFrame(grouped)


def _confidence_calibration(rows: pd.DataFrame) -> list[dict[str, Any]]:
    valid = rows[
        rows["geo_teacher_conf"].notna()
        & rows["geo_teacher_tvt"].notna()
        & rows["tvt_true"].notna()
    ].copy()
    if valid.empty:
        return []
    try:
        valid["confidence_bucket"] = pd.qcut(
            valid["geo_teacher_conf"], q=4, labels=False, duplicates="drop"
        )
    except ValueError:
        valid["confidence_bucket"] = 0
    output: list[dict[str, Any]] = []
    for bucket, frame in valid.groupby("confidence_bucket", sort=True):
        output.append(
            {
                "bucket": int(bucket),
                "rows": int(len(frame)),
                "conf_min": float(frame["geo_teacher_conf"].min()),
                "conf_max": float(frame["geo_teacher_conf"].max()),
                "conf_mean": float(frame["geo_teacher_conf"].mean()),
                "rmse": _rmse(
                    frame["geo_teacher_tvt"].to_numpy(dtype=float),
                    frame["tvt_true"].to_numpy(dtype=float),
                ),
            }
        )
    return output


def _surface_name(surface_id: float) -> str:
    if np.isfinite(surface_id):
        idx = int(surface_id)
        if 0 <= idx < len(FORMATIONS):
            return FORMATIONS[idx]
    return "none"


def _metrics(rows: pd.DataFrame, by_well: pd.DataFrame) -> dict[str, Any]:
    finite_teacher = rows["geo_teacher_tvt"].notna()
    well_rmse = by_well["rmse"].to_numpy(dtype=float) if not by_well.empty else np.array([])
    finite_well_rmse = _finite(well_rmse)
    hidden_counts = by_well.set_index("well")["rows"].to_dict() if not by_well.empty else {}
    hidden_row_count = rows["well"].map(hidden_counts).to_numpy(dtype=float) if len(rows) else np.array([])
    threshold = float(np.nanmedian(hidden_row_count)) if len(hidden_row_count) else float("nan")
    long_mask = hidden_row_count >= threshold if len(hidden_row_count) else np.array([], dtype=bool)

    by_surface: list[dict[str, Any]] = []
    if len(rows):
        for surface_id, frame in rows.groupby("teacher_best_surface_id", dropna=False, sort=True):
            by_surface.append(
                {
                    "surface_id": float(surface_id) if pd.notna(surface_id) else float("nan"),
                    "surface": _surface_name(float(surface_id)) if pd.notna(surface_id) else "none",
                    "rows": int(len(frame)),
                    "rmse": _rmse(
                        frame["geo_teacher_tvt"].to_numpy(dtype=float),
                        frame["tvt_true"].to_numpy(dtype=float),
                    ),
                }
            )

    return {
        "teacher_vs_true_rmse": _rmse(
            rows["geo_teacher_tvt"].to_numpy(dtype=float),
            rows["tvt_true"].to_numpy(dtype=float),
        )
        if len(rows)
        else float("nan"),
        "teacher_coverage": float(finite_teacher.mean()) if len(rows) else 0.0,
        "rows": int(len(rows)),
        "covered_rows": int(finite_teacher.sum()),
        "well_count": int(by_well["well"].nunique()) if not by_well.empty else 0,
        "mean_well_rmse": float(np.mean(finite_well_rmse)) if len(finite_well_rmse) else float("nan"),
        "median_well_rmse": float(np.median(finite_well_rmse)) if len(finite_well_rmse) else float("nan"),
        "p90_well_rmse": _safe_quantile(finite_well_rmse, 0.90),
        "p95_well_rmse": _safe_quantile(finite_well_rmse, 0.95),
        "worst_well_rmse": float(np.max(finite_well_rmse)) if len(finite_well_rmse) else float("nan"),
        "hidden_rows_median": threshold,
        "long_hidden_rmse": _rmse(
            rows.loc[long_mask, "geo_teacher_tvt"].to_numpy(dtype=float),
            rows.loc[long_mask, "tvt_true"].to_numpy(dtype=float),
        )
        if len(rows)
        else float("nan"),
        "short_hidden_rmse": _rmse(
            rows.loc[~long_mask, "geo_teacher_tvt"].to_numpy(dtype=float),
            rows.loc[~long_mask, "tvt_true"].to_numpy(dtype=float),
        )
        if len(rows)
        else float("nan"),
        "by_formation": by_surface,
        "confidence_calibration": _confidence_calibration(rows),
        "worst_wells": by_well.sort_values("rmse", ascending=False).head(10).to_dict("records")
        if not by_well.empty
        else [],
    }


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_empty_"
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame[columns].iterrows():
        values = []
        for value in row.tolist():
            if isinstance(value, float):
                values.append(f"{value:.6f}" if np.isfinite(value) else "nan")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def write_report(output_dir: Path, metrics: dict[str, Any], by_well: pd.DataFrame) -> None:
    report = output_dir / "teacher_report.md"
    lines = [
        "# Surface Teacher Report",
        "",
        "Privileged train-only teacher using true formation surfaces. Do not use as inference feature.",
        "",
        "## Summary",
        "",
        f"- Rows: `{metrics['rows']}`",
        f"- Covered rows: `{metrics['covered_rows']}`",
        f"- Teacher coverage: `{metrics['teacher_coverage']:.6f}`",
        f"- Teacher vs true RMSE: `{metrics['teacher_vs_true_rmse']:.6f}`",
        f"- Mean well RMSE: `{metrics['mean_well_rmse']:.6f}`",
        f"- Median well RMSE: `{metrics['median_well_rmse']:.6f}`",
        f"- P90/P95 well RMSE: `{metrics['p90_well_rmse']:.6f}` / `{metrics['p95_well_rmse']:.6f}`",
        f"- Worst well RMSE: `{metrics['worst_well_rmse']:.6f}`",
        f"- Long/short hidden RMSE: `{metrics['long_hidden_rmse']:.6f}` / `{metrics['short_hidden_rmse']:.6f}`",
        "",
        "## Confidence Calibration",
        "",
        _markdown_table(pd.DataFrame(metrics["confidence_calibration"]), ["bucket", "rows", "conf_min", "conf_max", "conf_mean", "rmse"]),
        "",
        "## By Best Formation",
        "",
        _markdown_table(pd.DataFrame(metrics["by_formation"]), ["surface_id", "surface", "rows", "rmse"]),
        "",
        "## Worst Wells",
        "",
        _markdown_table(by_well.sort_values("rmse", ascending=False).head(10), ["well", "rows", "coverage", "rmse", "mean_conf", "best_surface_id", "fit_rmse_min", "surface_spread"]),
        "",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_surface_teacher_dataset(
    *,
    data_dir: Path,
    output_dir: Path,
    config: dict[str, Any] | None = None,
    max_wells: int | None = None,
    progress_interval: int = 25,
) -> dict[str, Any]:
    train_dir = data_dir / "train"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    row_path = output_dir / "teacher_rows.csv"
    by_well_path = output_dir / "teacher_by_well.csv"
    metrics_path = output_dir / "teacher_metrics.json"

    paths = sorted(train_dir.glob("*__horizontal_well.csv"))
    if max_wells is not None:
        paths = paths[: int(max_wells)]
    if not paths:
        raise ValueError(f"No train wells found in {train_dir}")

    started = perf_counter()
    metric_parts: list[pd.DataFrame] = []
    header_written = False
    total_rows = 0
    for current, path in enumerate(paths, start=1):
        well = path.name.split("__", 1)[0]
        df = pd.read_csv(path)
        typewell_path = path.with_name(path.name.replace("__horizontal_well.csv", "__typewell.csv"))
        typewell_df = pd.read_csv(typewell_path) if typewell_path.exists() else None
        teacher = build_geo_teacher_for_well(df, typewell_df, config)
        teacher.insert(0, "well", well)
        teacher.insert(1, "id", [f"{well}_{int(idx)}" for idx in teacher["row_index"].to_numpy(dtype=int)])
        if "TVT" in df.columns and len(teacher):
            row_idx = teacher["row_index"].to_numpy(dtype=int)
            teacher["tvt_true"] = pd.to_numeric(df.loc[row_idx, "TVT"], errors="coerce").to_numpy(dtype=float)
            teacher["teacher_error"] = teacher["geo_teacher_tvt"] - teacher["tvt_true"]
            teacher["teacher_abs_error"] = np.abs(teacher["teacher_error"])
        else:
            teacher["tvt_true"] = np.nan
            teacher["teacher_error"] = np.nan
            teacher["teacher_abs_error"] = np.nan
        teacher.to_csv(
            row_path,
            mode="w" if not header_written else "a",
            index=False,
            header=not header_written,
            float_format="%.6f",
        )
        header_written = True
        metric_parts.append(
            teacher[
                [
                    "well",
                    "geo_teacher_tvt",
                    "geo_teacher_conf",
                    "teacher_best_surface_id",
                    "teacher_surface_fit_rmse_min",
                    "teacher_surface_spread",
                    "tvt_true",
                ]
            ].copy()
        )
        total_rows += len(teacher)
        if current == 1 or current % progress_interval == 0 or current == len(paths):
            elapsed = perf_counter() - started
            rate = total_rows / max(elapsed, 1e-9)
            print(
                "Surface teacher progress | "
                f"current={current} total={len(paths)} well={well} "
                f"rows={total_rows} elapsed={elapsed:.1f}s rows_per_sec={rate:.1f}",
                flush=True,
            )

    rows = pd.concat(metric_parts, ignore_index=True) if metric_parts else pd.DataFrame()
    by_well = _summarize_by_well(rows)
    metrics = _metrics(rows, by_well)
    by_well.to_csv(by_well_path, index=False, float_format="%.6f")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_report(output_dir, metrics, by_well)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build privileged surface-teacher labels.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/surface_teacher"))
    parser.add_argument("--tail-rows", type=int, default=384)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--min-fit-rows", type=int, default=16)
    parser.add_argument("--max-wells", type=int, default=None)
    parser.add_argument("--progress-interval", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = {
        "surface_teacher": {
            "tail_rows": args.tail_rows,
            "ridge": args.ridge,
            "min_fit_rows": args.min_fit_rows,
        }
    }
    metrics = build_surface_teacher_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        config=config,
        max_wells=args.max_wells,
        progress_interval=args.progress_interval,
    )
    print(
        "Surface teacher complete | "
        f"rows={metrics['rows']} coverage={metrics['teacher_coverage']:.6f} "
        f"rmse={metrics['teacher_vs_true_rmse']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
