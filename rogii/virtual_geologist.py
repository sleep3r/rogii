from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from .config import load_config
from .constants import FORMATIONS
from .io import horizontal_files, resolve_data_dir, resolve_train_dir, typewell_path, well_name
from .runlog import RunLogger, format_duration
from .surface_student_v1 import (
    _prepare_feature_config,
    attach_schema10_oof,
    build_fold_safe_feature_frame,
)


VG_OUTPUT_COLUMNS: tuple[str, ...] = (
    "vg_best_tvt",
    "vg_mean_tvt",
    "vg_p10_tvt",
    "vg_p50_tvt",
    "vg_p90_tvt",
    "vg_entropy",
    "vg_top2_gap",
    "vg_min_cost",
    "vg_shift",
    "vg_dip",
    "vg_stretch",
    "vg_known_fit_rmse",
    "vg_gr_cost",
    "vg_dgr_cost",
    "vg_surface_cost",
    "vg_tail_cost",
    "vg_tiepoint_count",
    "vg_tiepoint_conf",
)


def _rmse(pred: np.ndarray | pd.Series, true: np.ndarray | pd.Series) -> float:
    pred_arr = np.asarray(pred, dtype=float)
    true_arr = np.asarray(true, dtype=float)
    mask = np.isfinite(pred_arr) & np.isfinite(true_arr)
    if not np.any(mask):
        return float("nan")
    err = pred_arr[mask] - true_arr[mask]
    return float(np.sqrt(np.mean(err * err)))


def _mae(pred: np.ndarray | pd.Series, true: np.ndarray | pd.Series) -> float:
    pred_arr = np.asarray(pred, dtype=float)
    true_arr = np.asarray(true, dtype=float)
    mask = np.isfinite(pred_arr) & np.isfinite(true_arr)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(pred_arr[mask] - true_arr[mask])))


def _safe_std(values: np.ndarray, default: float = 1.0) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) < 2:
        return float(default)
    value = float(np.nanstd(finite))
    return value if np.isfinite(value) and value > 1e-6 else float(default)


def _read_typewell(path: Path | None) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if path is None or not path.exists():
        return None
    frame = pd.read_csv(path)
    if "TVT" not in frame.columns or "GR" not in frame.columns:
        return None
    tvt = pd.to_numeric(frame["TVT"], errors="coerce").to_numpy(dtype=float)
    gr = pd.to_numeric(frame["GR"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(tvt) & np.isfinite(gr)
    if int(valid.sum()) < 5:
        return None
    tvt = tvt[valid]
    gr = gr[valid]
    order = np.argsort(tvt)
    tvt = tvt[order]
    gr = gr[order]
    unique_tvt, inverse = np.unique(tvt, return_inverse=True)
    counts = np.bincount(inverse)
    gr_unique = np.bincount(inverse, weights=gr) / counts
    dgr = np.gradient(gr_unique, unique_tvt) if len(unique_tvt) >= 3 else np.zeros_like(gr_unique)
    return unique_tvt, gr_unique, dgr


def attach_schema10(
    frame: pd.DataFrame,
    *,
    model_path: Path,
    config_path: Path,
    oof_path: Path | None,
    data_dir: Path,
    logger: RunLogger,
) -> pd.DataFrame:
    if model_path.exists() and config_path.exists():
        return attach_schema10_oof(
            frame,
            model_path=model_path,
            config_path=config_path,
            data_dir=data_dir,
            logger=logger,
        )
    if oof_path is None or not oof_path.exists():
        raise FileNotFoundError(
            "No schema10 OOF source found. Expected either model/config "
            f"({model_path}, {config_path}) or schema10_oof_path={oof_path}."
        )
    with logger.step("Attach schema10 OOF artifact", path=oof_path):
        if oof_path.suffix == ".parquet":
            schema = pd.read_parquet(oof_path)
        else:
            schema = pd.read_csv(oof_path)
    if "schema10_oof_raw" not in schema.columns:
        raise ValueError(f"{oof_path} must contain schema10_oof_raw")
    keep = ["id", "schema10_oof_raw"]
    true_col = None
    for candidate in ("tvt_true_schema10", "TVT", "tvt_true"):
        if candidate in schema.columns:
            true_col = candidate
            keep.append(candidate)
            break
    merged = frame.merge(schema[keep], on="id", how="left")
    missing = int(merged["schema10_oof_raw"].isna().sum())
    if missing:
        raise ValueError(f"Schema10 OOF artifact missing {missing} rows")
    if true_col is not None:
        if not np.allclose(
            merged["tvt_true"].to_numpy(dtype=float),
            merged[true_col].to_numpy(dtype=float),
            equal_nan=False,
        ):
            raise ValueError("Schema10 OOF artifact true TVT does not align")
        merged = merged.drop(columns=[true_col])
    return merged


def _candidate_grid(cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    offset_min, offset_max, offset_step = cfg.get("offset_grid", [-80.0, 80.0, 8.0])
    dip_min, dip_max, dip_step = cfg.get("dip_grid", [-24.0, 24.0, 6.0])
    stretch_min, stretch_max, stretch_step = cfg.get("stretch_grid", [0.0, 0.0, 1.0])
    offsets = np.arange(float(offset_min), float(offset_max) + 0.5 * float(offset_step), float(offset_step))
    dips = np.arange(float(dip_min), float(dip_max) + 0.5 * float(dip_step), float(dip_step))
    if float(stretch_step) <= 0:
        stretches = np.array([0.0], dtype=float)
    else:
        stretches = np.arange(
            float(stretch_min),
            float(stretch_max) + 0.5 * float(stretch_step),
            float(stretch_step),
        )
    shift_grid, dip_grid, stretch_grid = np.meshgrid(offsets, dips, stretches, indexing="ij")
    return shift_grid.ravel(), dip_grid.ravel(), stretch_grid.ravel()


def _surface_prior(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, int]:
    columns = [
        column
        for column in [
            "kg_form_mean_tvt",
            "kg_form_ancc_tvt",
            "kg_dense_ancc_tvt",
            *(f"kg_form_{formation}_tvt" for formation in FORMATIONS),
        ]
        if column in frame.columns
    ]
    if not columns:
        base = frame["schema10_oof_raw"].to_numpy(dtype=float)
        return base, np.full(len(frame), 30.0, dtype=float), 0
    matrix = frame[columns].to_numpy(dtype=float)
    valid = np.isfinite(matrix)
    prior = np.nanmedian(matrix, axis=1)
    base = frame["schema10_oof_raw"].to_numpy(dtype=float)
    prior = np.where(np.isfinite(prior), prior, base)
    spread = np.nanstd(matrix, axis=1)
    spread = np.where(np.isfinite(spread), spread, 30.0)
    spread = np.maximum(spread, 5.0)
    return prior, spread, int(valid.any(axis=1).sum())


def _weighted_mean(paths: np.ndarray, costs: np.ndarray, temperature: float) -> tuple[np.ndarray, float]:
    shifted = costs - float(np.nanmin(costs))
    weights = np.exp(-shifted / max(float(temperature), 1e-6))
    weights = weights / max(float(np.nansum(weights)), 1e-12)
    entropy = -float(np.nansum(weights * np.log(np.maximum(weights, 1e-12))))
    entropy /= float(np.log(max(len(weights), 2)))
    return paths @ weights, entropy


def solve_virtual_geologist_well(
    frame: pd.DataFrame,
    *,
    typewell: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    ordered = frame.sort_values("row_index").reset_index(drop=True)
    n = len(ordered)
    base = ordered["schema10_oof_raw"].to_numpy(dtype=float)
    true = ordered["tvt_true"].to_numpy(dtype=float)
    row_index = ordered["row_index"].to_numpy(dtype=int)
    gr = ordered["gr"].to_numpy(dtype=float) if "gr" in ordered.columns else np.full(n, np.nan)
    frac = np.linspace(0.0, 1.0, n, dtype=float) if n > 1 else np.zeros(n, dtype=float)
    centered = frac - 0.5
    curve = centered * centered - float(np.mean(centered * centered))
    last_known = float(np.nanmedian(ordered["last_known_tvt"].to_numpy(dtype=float)))

    shifts, dips, stretches = _candidate_grid(cfg)
    paths = (
        base[:, None]
        + shifts[None, :]
        + dips[None, :] * centered[:, None]
        + stretches[None, :] * curve[:, None]
    )

    surface_prior, surface_spread, surface_coverage = _surface_prior(ordered)
    gr_scale = _safe_std(gr, default=10.0)
    w_gr = float(cfg.get("w_gr", 1.0))
    w_dgr = float(cfg.get("w_dgr", 0.20))
    w_surface = float(cfg.get("w_surface", 0.35))
    w_known = float(cfg.get("w_known", 0.15))
    w_smooth = float(cfg.get("w_smooth", 0.04))
    w_tail = float(cfg.get("w_tail", 0.35))
    huber_clip = float(cfg.get("huber_clip", 4.0))

    if typewell is not None and np.isfinite(gr).any():
        tw_tvt, tw_gr, tw_dgr = typewell
        interp_gr = np.interp(
            paths.ravel(),
            tw_tvt,
            tw_gr,
            left=tw_gr[0],
            right=tw_gr[-1],
        ).reshape(paths.shape)
        gr_resid = np.clip((interp_gr - gr[:, None]) / gr_scale, -huber_clip, huber_clip)
        gr_cost = np.nanmean(gr_resid * gr_resid, axis=0)

        gr_grad = np.gradient(pd.Series(gr).interpolate(limit_direction="both").bfill().ffill().to_numpy(dtype=float))
        interp_dgr = np.interp(
            paths.ravel(),
            tw_tvt,
            tw_dgr,
            left=tw_dgr[0],
            right=tw_dgr[-1],
        ).reshape(paths.shape)
        dgr_scale = _safe_std(gr_grad, default=1.0)
        dgr_resid = np.clip((interp_dgr - gr_grad[:, None]) / dgr_scale, -huber_clip, huber_clip)
        dgr_cost = np.nanmean(dgr_resid * dgr_resid, axis=0)
    else:
        gr_cost = np.zeros(paths.shape[1], dtype=float)
        dgr_cost = np.zeros(paths.shape[1], dtype=float)

    surface_resid = np.clip(
        (paths - surface_prior[:, None]) / np.maximum(surface_spread[:, None], 5.0),
        -huber_clip,
        huber_clip,
    )
    surface_cost = np.nanmean(surface_resid * surface_resid, axis=0)
    known_fit_rmse = np.sqrt(np.nanmean((shifts[None, :] + dips[None, :] * centered[:, None]) ** 2, axis=0))
    tail_cost = np.abs(paths[0, :] - last_known) / 20.0
    smooth_cost = (np.abs(dips) / 30.0) + (np.abs(stretches) / 20.0)
    total_cost = (
        w_gr * gr_cost
        + w_dgr * dgr_cost
        + w_surface * surface_cost
        + w_known * (known_fit_rmse / 15.0)
        + w_smooth * smooth_cost
        + w_tail * tail_cost
    )

    top_k = max(2, int(cfg.get("top_k", 5)))
    top_k = min(top_k, paths.shape[1])
    order = np.argsort(total_cost)[:top_k]
    top_paths = paths[:, order]
    top_costs = total_cost[order]
    best_pos = int(order[0])
    mean_path, entropy = _weighted_mean(
        top_paths,
        top_costs,
        temperature=float(cfg.get("posterior_temperature", 0.35)),
    )
    top2_gap = (
        float((top_costs[1] - top_costs[0]) / (abs(top_costs[0]) + 1e-6))
        if len(top_costs) > 1
        else float("nan")
    )
    output = pd.DataFrame(
        {
            "id": ordered["id"].to_numpy(object),
            "well": ordered["well"].to_numpy(object),
            "row_index": row_index,
            "fold": ordered["fold"].to_numpy(dtype=int),
            "tvt_true": true,
            "schema10_oof_raw": base,
            "vg_best_tvt": paths[:, best_pos],
            "vg_mean_tvt": mean_path,
            "vg_p10_tvt": np.nanquantile(top_paths, 0.10, axis=1),
            "vg_p50_tvt": np.nanquantile(top_paths, 0.50, axis=1),
            "vg_p90_tvt": np.nanquantile(top_paths, 0.90, axis=1),
            "vg_entropy": np.full(n, entropy, dtype=float),
            "vg_top2_gap": np.full(n, top2_gap, dtype=float),
            "vg_min_cost": np.full(n, float(total_cost[best_pos]), dtype=float),
            "vg_shift": np.full(n, float(shifts[best_pos]), dtype=float),
            "vg_dip": np.full(n, float(dips[best_pos]), dtype=float),
            "vg_stretch": np.full(n, float(stretches[best_pos]), dtype=float),
            "vg_known_fit_rmse": np.full(n, float(known_fit_rmse[best_pos]), dtype=float),
            "vg_gr_cost": np.full(n, float(gr_cost[best_pos]), dtype=float),
            "vg_dgr_cost": np.full(n, float(dgr_cost[best_pos]), dtype=float),
            "vg_surface_cost": np.full(n, float(surface_cost[best_pos]), dtype=float),
            "vg_tail_cost": np.full(n, float(tail_cost[best_pos]), dtype=float),
            "vg_tiepoint_count": np.full(n, int(np.isfinite(gr).sum()), dtype=float),
            "vg_tiepoint_conf": np.full(n, 1.0 / (1.0 + float(total_cost[best_pos])), dtype=float),
            "vg_surface_coverage": np.full(n, surface_coverage / max(n, 1), dtype=float),
        }
    )
    for rank, pos in enumerate(order, start=1):
        output[f"vg_top{rank}_tvt"] = paths[:, int(pos)]
        output[f"vg_top{rank}_cost"] = float(total_cost[int(pos)])
    return output


def _topk_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(
        [column for column in frame.columns if column.startswith("vg_top") and column.endswith("_tvt")],
        key=lambda column: int(column.split("_top", 1)[1].split("_", 1)[0]),
    )


def _score_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for column in columns:
        rows.append(
            {
                "candidate": column,
                "coverage": float(np.isfinite(frame[column]).mean()),
                "rmse": _rmse(frame[column], frame["tvt_true"]),
                "mae": _mae(frame[column], frame["tvt_true"]),
                "bias": float(np.nanmean(frame[column].to_numpy(dtype=float) - frame["tvt_true"].to_numpy(dtype=float))),
            }
        )
    return pd.DataFrame(rows).sort_values("rmse")


def _oracle_matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    y = frame["tvt_true"].to_numpy(dtype=float)
    matrix = frame[columns].to_numpy(dtype=float)
    err = np.abs(matrix - y[:, None])
    err[~np.isfinite(err)] = np.inf
    best = np.argmin(err, axis=1)
    return matrix[np.arange(len(frame)), best]


def _whole_oracle(frame: pd.DataFrame, columns: list[str], group_col: str = "well") -> np.ndarray:
    pred = np.full(len(frame), np.nan, dtype=float)
    for _group, idx in frame.groupby(group_col, sort=False).groups.items():
        sub = frame.loc[idx]
        scores = [_rmse(sub[column], sub["tvt_true"]) for column in columns]
        best_col = columns[int(np.nanargmin(scores))]
        pred[sub.index.to_numpy()] = sub[best_col].to_numpy(dtype=float)
    return pred


def _thirds_oracle(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    pred = np.full(len(frame), np.nan, dtype=float)
    for _well, idx in frame.groupby("well", sort=False).groups.items():
        sub = frame.loc[idx].sort_values("row_index")
        groups = np.array_split(sub.index.to_numpy(), 3)
        for group_idx in groups:
            if len(group_idx) == 0:
                continue
            part = frame.loc[group_idx]
            scores = [_rmse(part[column], part["tvt_true"]) for column in columns]
            best_col = columns[int(np.nanargmin(scores))]
            pred[group_idx] = part[best_col].to_numpy(dtype=float)
    return pred


def _smooth_top2_thirds_oracle(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    pred = np.full(len(frame), np.nan, dtype=float)
    for _well, idx in frame.groupby("well", sort=False).groups.items():
        sub = frame.loc[idx].sort_values("row_index")
        groups = np.array_split(sub.index.to_numpy(), 3)
        for group_idx in groups:
            if len(group_idx) == 0:
                continue
            part = frame.loc[group_idx]
            scores = np.array([_rmse(part[column], part["tvt_true"]) for column in columns])
            best = np.argsort(scores)[: min(2, len(columns))]
            pred[group_idx] = np.nanmean(part[[columns[int(i)] for i in best]].to_numpy(dtype=float), axis=1)
    return pred


def oracle_scores(frame: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    return {
        "row_topk_oracle_rmse": _rmse(_oracle_matrix(frame, columns), frame["tvt_true"]),
        "well_topk_oracle_rmse": _rmse(_whole_oracle(frame, columns), frame["tvt_true"]),
        "thirds_topk_oracle_rmse": _rmse(_thirds_oracle(frame, columns), frame["tvt_true"]),
        "smooth_top2_thirds_oracle_rmse": _rmse(
            _smooth_top2_thirds_oracle(frame, columns),
            frame["tvt_true"],
        ),
    }


def run_virtual_geologist(
    *,
    config: dict[str, Any],
    data_dir: Path,
    output_dir: Path,
    max_wells: int | None = None,
) -> dict[str, Any]:
    logger = RunLogger()
    cfg = config.get("virtual_geologist", {})
    seed = int(cfg.get("seed", config.get("seed", 42)))
    prepared_config = _prepare_feature_config(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dir = resolve_train_dir(data_dir, prepared_config)
    train_paths = horizontal_files(train_dir, limit=max_wells)
    logger.log(
        "RUN",
        "VirtualGeologist v0 started",
        wells=len(train_paths),
        output_dir=output_dir,
    )
    with logger.step("Build VirtualGeologist fold-safe feature frame", wells=len(train_paths)):
        frame, fold_rows = build_fold_safe_feature_frame(
            train_paths=train_paths,
            config=prepared_config,
            seed=seed,
            logger=logger,
        )
    schema_model_path = Path(
        cfg.get("schema10_model_path", "artifacts/clearml/ed4d9dc6c7cb479881f087fee1217253/model.pkl")
    )
    schema_config_path = Path(
        cfg.get("schema10_config_path", "artifacts/clearml/ed4d9dc6c7cb479881f087fee1217253/config.yml")
    )
    schema10_oof_path = cfg.get("schema10_oof_path")
    frame = attach_schema10(
        frame,
        model_path=schema_model_path,
        config_path=schema_config_path,
        oof_path=Path(schema10_oof_path) if schema10_oof_path else None,
        data_dir=data_dir,
        logger=logger,
    )

    parts: list[pd.DataFrame] = []
    started = perf_counter()
    by_path = {well_name(path): path for path in train_paths}
    total = int(frame["well"].nunique())
    progress_interval = max(1, int(cfg.get("progress_interval", 25)))
    with logger.step("Solve VirtualGeologist wells", wells=total):
        for current, (well, group) in enumerate(frame.groupby("well", sort=True), start=1):
            tw = _read_typewell(typewell_path(by_path[str(well)]))
            solved = solve_virtual_geologist_well(group, typewell=tw, cfg=cfg)
            parts.append(solved)
            if current == 1 or current % progress_interval == 0 or current == total:
                elapsed = perf_counter() - started
                rows = int(sum(len(part) for part in parts))
                eta = elapsed / max(current, 1) * max(total - current, 0)
                logger.info(
                    "VirtualGeologist progress",
                    current=current,
                    total=total,
                    well=well,
                    rows=rows,
                    elapsed=format_duration(elapsed),
                    eta=format_duration(eta),
                )
    oof = pd.concat(parts, ignore_index=True)
    candidate_cols = [
        "schema10_oof_raw",
        "vg_best_tvt",
        "vg_mean_tvt",
        "vg_p50_tvt",
        *(_topk_columns(oof)),
    ]
    candidate_cols = list(dict.fromkeys([column for column in candidate_cols if column in oof.columns]))
    score_frame = _score_columns(oof, candidate_cols)
    topk_cols = _topk_columns(oof)
    oracle = oracle_scores(oof, topk_cols) if topk_cols else {}
    metrics = {
        "rows": int(len(oof)),
        "wells": int(oof["well"].nunique()),
        "fold_rows": fold_rows,
        "candidate_scores": score_frame.to_dict("records"),
        "oracle": oracle,
        "config": cfg,
        "vg_best_rmse": _rmse(oof["vg_best_tvt"], oof["tvt_true"]),
        "schema10_raw_rmse": _rmse(oof["schema10_oof_raw"], oof["tvt_true"]),
        "vg_topk_oracle_rmse": oracle.get("thirds_topk_oracle_rmse"),
    }
    try:
        oof_path = output_dir / "virtual_geologist_oof.parquet"
        oof.to_parquet(oof_path, index=False)
    except Exception:
        oof_path = output_dir / "virtual_geologist_oof.csv"
        oof.to_csv(oof_path, index=False, float_format="%.6f")
    score_frame.to_csv(output_dir / "virtual_geologist_candidate_scores.csv", index=False)
    (output_dir / "virtual_geologist_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    write_report(output_dir, metrics, score_frame)
    logger.log(
        "DONE",
        "VirtualGeologist v0 complete",
        rows=len(oof),
        schema10_rmse=metrics["schema10_raw_rmse"],
        vg_best_rmse=metrics["vg_best_rmse"],
        topk_oracle=metrics["vg_topk_oracle_rmse"],
    )
    return metrics


def _markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    visible = frame[columns].head(max_rows) if max_rows is not None else frame[columns]
    if visible.empty:
        return "_empty_"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in visible.itertuples(index=False, name=None):
        cells = []
        for value in row:
            if isinstance(value, float):
                cells.append(f"{value:.6f}" if np.isfinite(value) else "nan")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(output_dir: Path, metrics: dict[str, Any], score_frame: pd.DataFrame) -> None:
    oracle = metrics.get("oracle", {})
    lines = [
        "# VirtualGeologist v0 Report",
        "",
        "Constrained stratigraphic path search around schema10 OOF. "
        "Path family is `base + shift + dip * centered + stretch * curvature_term`; "
        "this is intentionally not an arbitrary HMM state path.",
        "",
        "## Summary",
        "",
        f"- Rows: `{metrics['rows']}`",
        f"- Wells: `{metrics['wells']}`",
        f"- Schema10 raw RMSE: `{metrics['schema10_raw_rmse']:.6f}`",
        f"- VG best RMSE: `{metrics['vg_best_rmse']:.6f}`",
        f"- Row top-K oracle RMSE: `{oracle.get('row_topk_oracle_rmse', float('nan')):.6f}`",
        f"- Well top-K oracle RMSE: `{oracle.get('well_topk_oracle_rmse', float('nan')):.6f}`",
        f"- Thirds top-K oracle RMSE: `{oracle.get('thirds_topk_oracle_rmse', float('nan')):.6f}`",
        f"- Smooth top-2 thirds oracle RMSE: `{oracle.get('smooth_top2_thirds_oracle_rmse', float('nan')):.6f}`",
        "",
        "## Candidate Scores",
        "",
        _markdown_table(score_frame, ["candidate", "coverage", "rmse", "mae", "bias"], max_rows=30),
        "",
    ]
    (output_dir / "virtual_geologist_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VirtualGeologist v0 OOF candidate search.")
    parser.add_argument("--config", type=Path, default=Path("configs/virtual_geologist.yml"))
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-wells", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_dir = args.data_dir or resolve_data_dir(config)
    cfg = config.get("virtual_geologist", {})
    output_dir = args.output_dir or Path(cfg.get("output_dir", "artifacts/virtual_geologist"))
    max_wells = args.max_wells
    if max_wells is None and cfg.get("max_wells") is not None:
        max_wells = int(cfg["max_wells"])
    metrics = run_virtual_geologist(
        config=config,
        data_dir=data_dir,
        output_dir=output_dir,
        max_wells=max_wells,
    )
    print(
        "VirtualGeologist complete | "
        f"rows={metrics['rows']} vg_best_rmse={metrics['vg_best_rmse']:.6f} "
        f"thirds_topk_oracle={metrics.get('vg_topk_oracle_rmse')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
