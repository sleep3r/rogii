from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import DataConfig
from .correlation_panel import (
    _compress_nanmean,
    _json_clean,
    _rank_of_true,
    _regular_typewell_grid,
    _topk_indices,
    score_gr_patch_against_typewell,
    score_gr_patch_multiscale,
)
from .heatmap import fill_nan
from .io import discover_wells, load_well


@dataclass(frozen=True)
class TypewellMismatchConfig:
    rows_per_step: int = 32
    vertical_step_ft: float = 5.0
    patch_radius: int = 3
    patch_radii: tuple[int, ...] = ()
    stretch_factors: tuple[float, ...] = (1.0,)
    min_patch_points: int = 3
    mad_weight: float = 0.15
    raw_mad_weight: float = 0.02
    topk: tuple[int, ...] = (1, 3, 10)
    mismatch_topk: int = 10
    mismatch_rate_threshold: float = 0.20
    shuffled_better_rate_threshold: float = 0.50


@dataclass(frozen=True)
class WellTypewellMismatchResult:
    well_id: str
    metrics: dict[str, float | int | str]
    steps: pd.DataFrame


def _shuffle_finite(values: np.ndarray, finite: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.asarray(values, dtype=np.float32).copy()
    finite_idx = np.flatnonzero(np.asarray(finite) > 0.5)
    if finite_idx.size:
        out[finite_idx] = rng.permutation(out[finite_idx])
    return out


def _score_percentile(scores: np.ndarray, score: float) -> float:
    finite = np.isfinite(scores)
    if not finite.any() or not np.isfinite(score):
        return float("nan")
    return float(np.mean(scores[finite] <= score))


def _nearest_sqerr(candidate_tvt: np.ndarray, true_tvt: float) -> float:
    if candidate_tvt.size == 0:
        return float("nan")
    return float(np.min((candidate_tvt.astype(np.float32) - true_tvt) ** 2))


def _collapse_zone(label: object) -> str:
    if label is None:
        return "__unknown__"
    try:
        if isinstance(label, float) and np.isnan(label):
            return "__unknown__"
    except TypeError:
        pass
    value = str(label).strip()
    if value == "" or value.lower() == "nan":
        return "__unknown__"
    if value in {"ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA", "OLMOS"}:
        return value
    upper = value.upper()
    if any(token in upper for token in ("THL", "TGT", "BHL", "MNSS", "LTGT", "LTHL", "LBHL")):
        return "EGFDL_SUB"
    return "OTHER"


def _typewell_grid_zones(typewell: pd.DataFrame, tvt_grid: np.ndarray) -> np.ndarray:
    if "Geology" not in typewell.columns or tvt_grid.size == 0:
        return np.full(tvt_grid.size, "__unknown__", dtype=object)
    tvt = pd.to_numeric(typewell["TVT"], errors="coerce").to_numpy(np.float32)
    labels = typewell["Geology"].map(_collapse_zone).to_numpy(dtype=object)
    finite = np.isfinite(tvt)
    if not finite.any():
        return np.full(tvt_grid.size, "__unknown__", dtype=object)
    order = np.argsort(tvt[finite])
    tvt_sorted = tvt[finite][order]
    labels_sorted = labels[finite][order]
    nearest = np.searchsorted(tvt_sorted, tvt_grid, side="left")
    nearest = np.clip(nearest, 0, len(tvt_sorted) - 1)
    prev = np.clip(nearest - 1, 0, len(tvt_sorted) - 1)
    use_prev = np.abs(tvt_sorted[prev] - tvt_grid) < np.abs(tvt_sorted[nearest] - tvt_grid)
    nearest = np.where(use_prev, prev, nearest)
    return labels_sorted[nearest].astype(object)


def _true_path_best_params(
    horizontal_gr: np.ndarray,
    typewell_gr: np.ndarray,
    *,
    step_index: int,
    true_bin: int,
    patch_radii: tuple[int, ...],
    stretch_factors: tuple[float, ...],
    min_patch_points: int,
    mad_weight: float,
    raw_mad_weight: float,
) -> tuple[float, float, float]:
    best_score = -np.inf
    best_radius = np.nan
    best_stretch = np.nan
    for radius in patch_radii:
        for stretch in stretch_factors:
            scores = score_gr_patch_against_typewell(
                horizontal_gr,
                typewell_gr,
                step_index=step_index,
                patch_radius=int(radius),
                stretch_factor=float(stretch),
                min_patch_points=min_patch_points,
                mad_weight=mad_weight,
                raw_mad_weight=raw_mad_weight,
            )
            if 0 <= true_bin < len(scores) and np.isfinite(scores[true_bin]):
                score = float(scores[true_bin])
                if score > best_score:
                    best_score = score
                    best_radius = float(radius)
                    best_stretch = float(stretch)
    if not np.isfinite(best_score):
        best_score = np.nan
    return float(best_score), float(best_radius), float(best_stretch)


def build_well_typewell_mismatch(
    well_id: str,
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    cfg: TypewellMismatchConfig,
    *,
    shuffle_gr_seed: int = 42,
) -> WellTypewellMismatchResult:
    tvt = pd.to_numeric(horizontal["TVT"], errors="coerce").to_numpy(dtype=np.float32)
    tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(
        dtype=np.float32
    )
    gr_raw = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(dtype=np.float32)
    gr_filled, gr_finite = fill_nan(gr_raw)
    shuffled_gr = _shuffle_finite(gr_filled, gr_finite, shuffle_gr_seed)

    comp_tvt = _compress_nanmean(tvt, cfg.rows_per_step)
    comp_tvt_input = _compress_nanmean(tvt_input, cfg.rows_per_step)
    comp_gr = _compress_nanmean(gr_filled, cfg.rows_per_step)
    comp_shuffled_gr = _compress_nanmean(shuffled_gr, cfg.rows_per_step)
    comp_gr_finite = _compress_nanmean(gr_finite, cfg.rows_per_step)
    tvt_grid, typewell_gr = _regular_typewell_grid(typewell, cfg.vertical_step_ft)
    if tvt_grid.size == 0 or typewell_gr.size == 0 or comp_tvt.size == 0:
        return WellTypewellMismatchResult(well_id, _empty_metrics(well_id), pd.DataFrame())
    grid_zones = _typewell_grid_zones(typewell, tvt_grid)

    hidden_steps = np.flatnonzero(~np.isfinite(comp_tvt_input) & np.isfinite(comp_tvt))
    patch_radii = cfg.patch_radii or (cfg.patch_radius,)
    max_topk = max(cfg.topk)
    rows: list[dict[str, float | int | str]] = []
    for step in hidden_steps:
        if (
            step >= comp_gr.size
            or step >= comp_shuffled_gr.size
            or not np.isfinite(comp_gr[step])
        ):
            continue
        scores, best_params = score_gr_patch_multiscale(
            comp_gr,
            typewell_gr,
            step_index=int(step),
            patch_radii=patch_radii,
            stretch_factors=cfg.stretch_factors,
            min_patch_points=cfg.min_patch_points,
            mad_weight=cfg.mad_weight,
            raw_mad_weight=cfg.raw_mad_weight,
        )
        shuffled_scores, _ = score_gr_patch_multiscale(
            comp_shuffled_gr,
            typewell_gr,
            step_index=int(step),
            patch_radii=patch_radii,
            stretch_factors=cfg.stretch_factors,
            min_patch_points=cfg.min_patch_points,
            mad_weight=cfg.mad_weight,
            raw_mad_weight=cfg.raw_mad_weight,
        )
        true_tvt = float(comp_tvt[step])
        true_bin = int(np.abs(tvt_grid - true_tvt).argmin())
        true_score = float(scores[true_bin]) if np.isfinite(scores[true_bin]) else np.nan
        true_best_score, true_best_radius, true_best_stretch = _true_path_best_params(
            comp_gr,
            typewell_gr,
            step_index=int(step),
            true_bin=true_bin,
            patch_radii=patch_radii,
            stretch_factors=cfg.stretch_factors,
            min_patch_points=cfg.min_patch_points,
            mad_weight=cfg.mad_weight,
            raw_mad_weight=cfg.raw_mad_weight,
        )
        shuffled_true_score = (
            float(shuffled_scores[true_bin])
            if np.isfinite(shuffled_scores[true_bin])
            else np.nan
        )
        top = _topk_indices(scores, max_topk)
        top1 = top[:1]
        top3 = top[: min(3, len(top))]
        top10 = top[: min(10, len(top))]
        top1_score = float(scores[top1[0]]) if top1.size else np.nan
        score_margin = top1_score - true_score if np.isfinite(true_score) else np.nan
        rows.append(
            {
                "well_id": well_id,
                "step": int(step),
                "true_tvt": true_tvt,
                "true_bin": true_bin,
                "true_zone": str(grid_zones[true_bin]) if true_bin < len(grid_zones) else "__unknown__",
                "true_score": true_score,
                "true_best_score": true_best_score,
                "true_best_patch_radius": true_best_radius,
                "true_best_stretch_factor": true_best_stretch,
                "shuffled_true_score": shuffled_true_score,
                "score_gap_vs_shuffled": true_score - shuffled_true_score
                if np.isfinite(true_score) and np.isfinite(shuffled_true_score)
                else np.nan,
                "score_percentile": _score_percentile(scores, true_score),
                "true_rank": _rank_of_true(scores, true_bin),
                "shuffled_true_rank": _rank_of_true(shuffled_scores, true_bin),
                "top1_tvt": float(tvt_grid[top1[0]]) if top1.size else np.nan,
                "top1_score": top1_score,
                "score_margin": score_margin,
                "top3_oracle_sqerr": _nearest_sqerr(tvt_grid[top3], true_tvt),
                "top10_oracle_sqerr": _nearest_sqerr(tvt_grid[top10], true_tvt),
                "gr_finite_frac": float(comp_gr_finite[step])
                if step < comp_gr_finite.size and np.isfinite(comp_gr_finite[step])
                else 0.0,
                "best_patch_radius": best_params[0] if best_params is not None else np.nan,
                "best_stretch_factor": best_params[1] if best_params is not None else np.nan,
            }
        )

    steps = pd.DataFrame(rows)
    metrics = _metrics_from_steps(well_id, steps, cfg)
    return WellTypewellMismatchResult(well_id, metrics, steps)


def _empty_metrics(well_id: str) -> dict[str, float | int | str]:
    return {
        "well_id": well_id,
        "num_steps": 0,
        "true_score_mean": float("nan"),
        "true_rank_mean": float("nan"),
        "true_top1_rate": float("nan"),
        "true_top3_rate": float("nan"),
        "true_top10_rate": float("nan"),
        "true_score_better_than_shuffled_rate": float("nan"),
        "mismatch_flag": 0,
    }


def _metrics_from_steps(
    well_id: str, steps: pd.DataFrame, cfg: TypewellMismatchConfig
) -> dict[str, float | int | str]:
    if steps.empty:
        return _empty_metrics(well_id)
    ranks = steps["true_rank"].to_numpy(np.float32)
    gaps = steps["score_gap_vs_shuffled"].to_numpy(np.float32)
    top1_sqerr = (
        steps["top1_tvt"].to_numpy(np.float32) - steps["true_tvt"].to_numpy(np.float32)
    ) ** 2
    topk_col = f"true_top{cfg.mismatch_topk}_rate"
    metrics: dict[str, float | int | str] = {
        "well_id": well_id,
        "num_steps": int(len(steps)),
        "true_score_mean": float(np.nanmean(steps["true_score"])),
        "true_score_median": float(np.nanmedian(steps["true_score"])),
        "true_score_p10": float(np.nanpercentile(steps["true_score"], 10)),
        "shuffled_true_score_mean": float(np.nanmean(steps["shuffled_true_score"])),
        "score_gap_vs_shuffled_mean": float(np.nanmean(gaps)),
        "true_score_better_than_shuffled_rate": float(np.nanmean(gaps > 0)),
        "score_percentile_mean": float(np.nanmean(steps["score_percentile"])),
        "score_margin_mean": float(np.nanmean(steps["score_margin"])),
        "score_margin_p90": float(np.nanpercentile(steps["score_margin"], 90)),
        "true_rank_mean": float(np.nanmean(ranks)),
        "shuffled_true_rank_mean": float(np.nanmean(steps["shuffled_true_rank"])),
        "true_top1_rate": float(np.nanmean(ranks <= 1)),
        "true_top3_rate": float(np.nanmean(ranks <= 3)),
        "true_top10_rate": float(np.nanmean(ranks <= 10)),
        "corr_top1_rmse_ft": float(np.sqrt(np.nanmean(top1_sqerr))),
        "corr_top3_oracle_rmse_ft": float(np.sqrt(np.nanmean(steps["top3_oracle_sqerr"]))),
        "corr_top10_oracle_rmse_ft": float(np.sqrt(np.nanmean(steps["top10_oracle_sqerr"]))),
        "gr_finite_mean": float(np.nanmean(steps["gr_finite_frac"])),
    }
    topk_rate = float(metrics.get(topk_col, np.nan))
    shuffled_rate = float(metrics["true_score_better_than_shuffled_rate"])
    metrics["mismatch_flag"] = int(
        (np.isfinite(topk_rate) and topk_rate < cfg.mismatch_rate_threshold)
        or (
            np.isfinite(shuffled_rate)
            and shuffled_rate < cfg.shuffled_better_rate_threshold
        )
    )
    return metrics


def _aggregate_metrics(
    results: Iterable[WellTypewellMismatchResult], cfg: TypewellMismatchConfig
) -> dict[str, float | int]:
    results = list(results)
    frames = [result.steps for result in results if not result.steps.empty]
    if not frames:
        return {"num_wells": 0, "num_steps": 0}
    steps = pd.concat(frames, ignore_index=True)
    metrics = _metrics_from_steps("__all__", steps, cfg)
    metrics.pop("well_id", None)
    metrics["num_wells"] = int(steps["well_id"].nunique())
    metrics["mismatch_wells"] = int(
        sum(int(result.metrics.get("mismatch_flag", 0)) for result in results)
    )
    metrics["mismatch_well_rate"] = float(metrics["mismatch_wells"] / max(len(results), 1))
    return metrics


def _write_report(
    output_dir: Path,
    cfg: TypewellMismatchConfig,
    metrics: dict[str, object],
) -> None:
    aggregate = metrics.get("aggregate", {})
    if not isinstance(aggregate, dict):
        aggregate = {}
    lines = [
        "# TYPEWELL_MISMATCH_AUDIT_V0",
        "",
        "## Config",
        "```json",
        json.dumps(_json_clean(asdict(cfg)), indent=2),
        "```",
        "",
        "## Aggregate",
        f"- wells: {aggregate.get('num_wells')}",
        f"- steps: {aggregate.get('num_steps')}",
        f"- true_score_mean: {aggregate.get('true_score_mean')}",
        f"- shuffled_true_score_mean: {aggregate.get('shuffled_true_score_mean')}",
        f"- score_gap_vs_shuffled_mean: {aggregate.get('score_gap_vs_shuffled_mean')}",
        f"- true_score_better_than_shuffled_rate: {aggregate.get('true_score_better_than_shuffled_rate')}",
        f"- true_top3_rate: {aggregate.get('true_top3_rate')}",
        f"- true_top10_rate: {aggregate.get('true_top10_rate')}",
        f"- mismatch_wells: {aggregate.get('mismatch_wells')}",
        "",
        "## Interpretation",
        "This audit scores the provided typewell exactly at the true TVT path. If the true-path score is not reliably better than shuffled GR, the blocker is the transfer of typewell/lateral log matching itself rather than only the downstream model.",
    ]
    (output_dir / "typewell_mismatch_report.md").write_text("\n".join(lines) + "\n")


def run_typewell_mismatch_audit(
    *,
    data_dir: Path,
    output_dir: Path,
    rows_per_step: int = 32,
    vertical_step_ft: float = 5.0,
    patch_radius: int = 3,
    patch_radii: tuple[int, ...] = (),
    stretch_factors: tuple[float, ...] = (1.0,),
    k_wells: int = -1,
    seed: int = 42,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = TypewellMismatchConfig(
        rows_per_step=rows_per_step,
        vertical_step_ft=vertical_step_ft,
        patch_radius=patch_radius,
        patch_radii=patch_radii,
        stretch_factors=stretch_factors,
    )
    wells = discover_wells(DataConfig(data_dir=data_dir, k_wells=k_wells))
    results: list[WellTypewellMismatchResult] = []
    for idx, well in enumerate(wells):
        if idx == 0 or (idx + 1) % 50 == 0 or idx + 1 == len(wells):
            print(f"[typewell-mismatch] well {idx + 1}/{len(wells)}", flush=True)
        horizontal, typewell = load_well(well)
        results.append(
            build_well_typewell_mismatch(
                well.well_id,
                horizontal,
                typewell,
                cfg,
                shuffle_gr_seed=seed + idx,
            )
        )

    by_well = pd.DataFrame([result.metrics for result in results])
    by_well.to_csv(output_dir / "typewell_mismatch_by_well.csv", index=False)
    step_frames = [result.steps for result in results if not result.steps.empty]
    if step_frames:
        pd.concat(step_frames, ignore_index=True).to_parquet(
            output_dir / "typewell_mismatch_steps.parquet", index=False
        )
    else:
        pd.DataFrame().to_parquet(output_dir / "typewell_mismatch_steps.parquet", index=False)

    metrics: dict[str, object] = {
        "aggregate": _aggregate_metrics(results, cfg),
        "config": asdict(cfg),
    }
    (output_dir / "typewell_mismatch_metrics.json").write_text(
        json.dumps(_json_clean(metrics), indent=2) + "\n"
    )
    _write_report(output_dir, cfg, _json_clean(metrics))
    print(json.dumps(_json_clean(metrics), indent=2), flush=True)
    return metrics
