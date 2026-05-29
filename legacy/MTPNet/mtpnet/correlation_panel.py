from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import DataConfig
from .heatmap import fill_nan
from .io import discover_wells, load_well


@dataclass(frozen=True)
class CorrelationPanelConfig:
    rows_per_step: int = 32
    vertical_step_ft: float = 5.0
    patch_radius: int = 3
    patch_radii: tuple[int, ...] = ()
    stretch_factors: tuple[float, ...] = (1.0,)
    min_patch_points: int = 3
    mad_weight: float = 0.15
    raw_mad_weight: float = 0.02
    topk: tuple[int, ...] = (1, 3, 10)
    anchor_source: str = "none"
    search_radius_ft: float | None = None


@dataclass(frozen=True)
class WellCorrelationResult:
    well_id: str
    metrics: dict[str, float | int | str]
    steps: pd.DataFrame


def known_tail_linear_anchor(comp_tvt_input: np.ndarray) -> np.ndarray:
    """Fill hidden compressed steps by extending the last known TVT slope."""
    anchor = np.asarray(comp_tvt_input, dtype=np.float32).copy()
    known = np.flatnonzero(np.isfinite(anchor))
    if known.size == 0:
        return anchor
    last_known = int(known[-1])
    if known.size >= 2:
        prev_known = int(known[-2])
        denom = max(last_known - prev_known, 1)
        slope = float((anchor[last_known] - anchor[prev_known]) / denom)
    else:
        slope = 0.0
    for step in range(last_known + 1, len(anchor)):
        anchor[step] = float(anchor[last_known] + slope * (step - last_known))
    return anchor.astype(np.float32)


def _compress_nanmean(values: np.ndarray, rows_per_step: int) -> np.ndarray:
    usable = (len(values) // rows_per_step) * rows_per_step
    if usable <= 0:
        return np.empty(0, dtype=np.float32)
    arr = np.asarray(values[:usable], dtype=np.float32).reshape(-1, rows_per_step)
    finite = np.isfinite(arr)
    sums = np.where(finite, arr, 0.0).sum(axis=1)
    counts = finite.sum(axis=1)
    out = np.full(arr.shape[0], np.nan, dtype=np.float32)
    np.divide(sums, counts, out=out, where=counts > 0)
    return out.astype(np.float32)


def _regular_typewell_grid(
    typewell: pd.DataFrame, vertical_step_ft: float
) -> tuple[np.ndarray, np.ndarray]:
    tvt = pd.to_numeric(typewell["TVT"], errors="coerce").to_numpy(dtype=np.float32)
    gr_raw = pd.to_numeric(typewell["GR"], errors="coerce").to_numpy(dtype=np.float32)
    gr, _ = fill_nan(gr_raw)
    order = np.argsort(tvt)
    tvt_sorted = tvt[order]
    gr_sorted = gr[order]
    finite = np.isfinite(tvt_sorted) & np.isfinite(gr_sorted)
    if not finite.any():
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
    lo = float(np.nanmin(tvt_sorted[finite]))
    hi = float(np.nanmax(tvt_sorted[finite]))
    if hi < lo:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
    grid = np.arange(lo, hi + vertical_step_ft * 0.5, vertical_step_ft, dtype=np.float32)
    gr_grid = np.interp(grid, tvt_sorted[finite], gr_sorted[finite]).astype(np.float32)
    return grid.astype(np.float32), gr_grid


def _standardize_rows(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    arr = np.where(valid, values, np.nan).astype(np.float32)
    mean = np.nanmean(arr, axis=1, keepdims=True)
    std = np.nanstd(arr, axis=1, keepdims=True)
    std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0)
    out = (arr - mean) / std
    return np.where(np.isfinite(out), out, 0.0).astype(np.float32)


def score_gr_patch_against_typewell(
    horizontal_gr: np.ndarray,
    typewell_gr: np.ndarray,
    *,
    step_index: int,
    patch_radius: int,
    stretch_factor: float = 1.0,
    min_patch_points: int = 3,
    mad_weight: float = 0.15,
    raw_mad_weight: float = 0.02,
) -> np.ndarray:
    """Score one lateral GR patch against every vertical typewell patch."""
    h = np.asarray(horizontal_gr, dtype=np.float32)
    t = np.asarray(typewell_gr, dtype=np.float32)
    scores = np.full(t.shape[0], -np.inf, dtype=np.float32)
    if h.size == 0 or t.size == 0:
        return scores

    offsets_all = np.arange(-patch_radius, patch_radius + 1, dtype=np.int32)
    h_idx = step_index + offsets_all
    h_valid_offsets = offsets_all[(h_idx >= 0) & (h_idx < h.size)]
    if h_valid_offsets.size < min_patch_points:
        return scores

    h_patch = h[step_index + h_valid_offsets].astype(np.float32)
    h_finite = np.isfinite(h_patch)
    if int(h_finite.sum()) < min_patch_points:
        return scores

    candidate_centers = np.arange(t.size, dtype=np.int32)
    t_idx = candidate_centers[:, None].astype(np.float32) + (
        h_valid_offsets[None, :].astype(np.float32) * float(stretch_factor)
    )
    t_valid = (t_idx >= 0) & (t_idx <= t.size - 1)
    grid = np.arange(t.size, dtype=np.float32)
    t_patch = np.interp(t_idx, grid, t, left=np.nan, right=np.nan).astype(np.float32)
    valid = t_valid & np.isfinite(t_patch) & h_finite[None, :]
    valid_counts = valid.sum(axis=1)

    h_matrix = np.broadcast_to(h_patch[None, :], t_patch.shape).astype(np.float32)
    h_z = _standardize_rows(h_matrix, valid)
    t_z = _standardize_rows(t_patch, valid)
    corr = np.divide(
        (h_z * t_z * valid).sum(axis=1),
        valid_counts,
        out=np.full(t.size, -np.inf, dtype=np.float32),
        where=valid_counts >= min_patch_points,
    )
    mad = np.divide(
        (np.abs(h_z - t_z) * valid).sum(axis=1),
        valid_counts,
        out=np.full(t.size, np.inf, dtype=np.float32),
        where=valid_counts >= min_patch_points,
    )
    raw_scale = float(np.nanstd(np.concatenate([h_patch[h_finite], t[np.isfinite(t)]])))
    raw_scale = max(raw_scale, 1.0)
    raw_mad = np.divide(
        (np.abs(h_matrix - t_patch) * valid).sum(axis=1),
        valid_counts,
        out=np.full(t.size, np.inf, dtype=np.float32),
        where=valid_counts >= min_patch_points,
    ) / raw_scale
    scores = (corr - mad_weight * mad - raw_mad_weight * raw_mad).astype(np.float32)
    scores[valid_counts < min_patch_points] = -np.inf
    return scores


def score_gr_patch_multiscale(
    horizontal_gr: np.ndarray,
    typewell_gr: np.ndarray,
    *,
    step_index: int,
    patch_radii: tuple[int, ...],
    stretch_factors: tuple[float, ...] = (1.0,),
    min_patch_points: int = 3,
    mad_weight: float = 0.15,
    raw_mad_weight: float = 0.02,
) -> tuple[np.ndarray, tuple[int, float] | None]:
    """Return the best score per typewell bin across patch radii."""
    best_scores: np.ndarray | None = None
    best_params_by_global: tuple[int, float] | None = None
    best_global_score = -np.inf
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
            if best_scores is None:
                best_scores = scores
            else:
                best_scores = np.maximum(best_scores, scores)
            if np.isfinite(scores).any():
                idx = int(np.nanargmax(scores))
                if float(scores[idx]) > best_global_score:
                    best_global_score = float(scores[idx])
                    best_params_by_global = (int(radius), float(stretch))
    if best_scores is None:
        best_scores = np.empty(0, dtype=np.float32)
    return best_scores.astype(np.float32), best_params_by_global


def _apply_search_band(
    scores: np.ndarray,
    tvt_grid: np.ndarray,
    *,
    anchor_tvt: float | None,
    search_radius_ft: float | None,
) -> np.ndarray:
    if anchor_tvt is None or search_radius_ft is None or not np.isfinite(anchor_tvt):
        return scores
    out = np.asarray(scores, dtype=np.float32).copy()
    keep = np.abs(tvt_grid.astype(np.float32) - float(anchor_tvt)) <= float(search_radius_ft)
    out[~keep] = -np.inf
    return out


def _topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    finite = np.isfinite(scores)
    if not finite.any():
        return np.empty(0, dtype=np.int32)
    k = min(k, int(finite.sum()))
    finite_idx = np.flatnonzero(finite)
    part = finite_idx[np.argpartition(scores[finite_idx], -k)[-k:]]
    return part[np.argsort(scores[part])[::-1]].astype(np.int32)


def _rank_of_true(scores: np.ndarray, true_bin: int) -> int:
    if true_bin < 0 or true_bin >= len(scores) or not np.isfinite(scores[true_bin]):
        return len(scores) + 1
    return int(1 + np.sum(scores > scores[true_bin]))


def _nearest_rmse(candidate_tvt: np.ndarray, true_tvt: float) -> float:
    if candidate_tvt.size == 0:
        return float("nan")
    return float(np.min((candidate_tvt.astype(np.float32) - true_tvt) ** 2))


def build_well_correlation_panel(
    well_id: str,
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    cfg: CorrelationPanelConfig,
    *,
    shuffle_gr_seed: int | None = None,
) -> WellCorrelationResult:
    tvt = pd.to_numeric(horizontal["TVT"], errors="coerce").to_numpy(dtype=np.float32)
    tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(
        dtype=np.float32
    )
    gr_raw = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(dtype=np.float32)
    gr_filled, gr_finite = fill_nan(gr_raw)
    if shuffle_gr_seed is not None:
        rng = np.random.default_rng(shuffle_gr_seed)
        finite_idx = np.flatnonzero(gr_finite > 0.5)
        shuffled = gr_filled.copy()
        shuffled[finite_idx] = rng.permutation(shuffled[finite_idx])
        gr_filled = shuffled

    comp_tvt = _compress_nanmean(tvt, cfg.rows_per_step)
    comp_tvt_input = _compress_nanmean(tvt_input, cfg.rows_per_step)
    comp_gr = _compress_nanmean(gr_filled, cfg.rows_per_step)
    comp_gr_finite = _compress_nanmean(gr_finite, cfg.rows_per_step)

    tvt_grid, typewell_gr = _regular_typewell_grid(typewell, cfg.vertical_step_ft)
    rows: list[dict[str, float | int | str]] = []
    if tvt_grid.size == 0 or comp_tvt.size == 0:
        return WellCorrelationResult(well_id, _empty_metrics(well_id), pd.DataFrame())

    hidden_steps = np.flatnonzero(~np.isfinite(comp_tvt_input) & np.isfinite(comp_tvt))
    max_topk = max(cfg.topk)
    patch_radii = cfg.patch_radii or (cfg.patch_radius,)
    anchor_path: np.ndarray | None = None
    if cfg.anchor_source == "known_tail_linear":
        anchor_path = known_tail_linear_anchor(comp_tvt_input)
    elif cfg.anchor_source != "none":
        raise ValueError(f"Unsupported correlation anchor_source: {cfg.anchor_source}")
    for step in hidden_steps:
        if step >= comp_gr.size or not np.isfinite(comp_gr[step]):
            continue
        scores, best_patch_radius = score_gr_patch_multiscale(
            comp_gr,
            typewell_gr,
            step_index=int(step),
            patch_radii=patch_radii,
            stretch_factors=cfg.stretch_factors,
            min_patch_points=cfg.min_patch_points,
            mad_weight=cfg.mad_weight,
            raw_mad_weight=cfg.raw_mad_weight,
        )
        anchor_tvt = (
            float(anchor_path[step])
            if anchor_path is not None and step < len(anchor_path) and np.isfinite(anchor_path[step])
            else None
        )
        scores = _apply_search_band(
            scores,
            tvt_grid,
            anchor_tvt=anchor_tvt,
            search_radius_ft=cfg.search_radius_ft,
        )
        true_tvt = float(comp_tvt[step])
        true_bin = int(np.abs(tvt_grid - true_tvt).argmin())
        top = _topk_indices(scores, max_topk)
        top1 = top[:1]
        top3 = top[: min(3, len(top))]
        top10 = top[: min(10, len(top))]
        rows.append(
            {
                "well_id": well_id,
                "step": int(step),
                "true_tvt": true_tvt,
                "true_bin": true_bin,
                "true_score": float(scores[true_bin]) if np.isfinite(scores[true_bin]) else np.nan,
                "true_rank": _rank_of_true(scores, true_bin),
                "gr_finite_frac": float(comp_gr_finite[step])
                if step < comp_gr_finite.size and np.isfinite(comp_gr_finite[step])
                else 0.0,
                "top1_tvt": float(tvt_grid[top1[0]]) if top1.size else np.nan,
                "top1_score": float(scores[top1[0]]) if top1.size else np.nan,
                "anchor_tvt": anchor_tvt if anchor_tvt is not None else np.nan,
                "anchor_abs_err": abs(anchor_tvt - true_tvt)
                if anchor_tvt is not None
                else np.nan,
                "best_patch_radius": best_patch_radius[0]
                if best_patch_radius is not None
                else np.nan,
                "best_stretch_factor": best_patch_radius[1]
                if best_patch_radius is not None
                else np.nan,
                "top3_oracle_sqerr": _nearest_rmse(tvt_grid[top3], true_tvt),
                "top10_oracle_sqerr": _nearest_rmse(tvt_grid[top10], true_tvt),
            }
        )
    steps = pd.DataFrame(rows)
    metrics = _metrics_from_steps(well_id, steps)
    return WellCorrelationResult(well_id, metrics, steps)


def _empty_metrics(well_id: str) -> dict[str, float | int | str]:
    return {
        "well_id": well_id,
        "num_steps": 0,
        "corr_top1_rmse_ft": float("nan"),
        "corr_top3_oracle_rmse_ft": float("nan"),
        "corr_top10_oracle_rmse_ft": float("nan"),
        "corr_target_rank_mean": float("nan"),
        "corr_target_top1_rate": float("nan"),
        "corr_target_top3_rate": float("nan"),
        "corr_target_top10_rate": float("nan"),
    }


def _metrics_from_steps(
    well_id: str, steps: pd.DataFrame
) -> dict[str, float | int | str]:
    if steps.empty:
        return _empty_metrics(well_id)
    top1_sqerr = (steps["top1_tvt"].to_numpy(np.float32) - steps["true_tvt"].to_numpy(np.float32)) ** 2
    ranks = steps["true_rank"].to_numpy(np.float32)
    metrics: dict[str, float | int | str] = {
        "well_id": well_id,
        "num_steps": int(len(steps)),
        "corr_top1_rmse_ft": float(np.sqrt(np.nanmean(top1_sqerr))),
        "corr_top3_oracle_rmse_ft": float(np.sqrt(np.nanmean(steps["top3_oracle_sqerr"]))),
        "corr_top10_oracle_rmse_ft": float(np.sqrt(np.nanmean(steps["top10_oracle_sqerr"]))),
        "corr_target_rank_mean": float(np.nanmean(ranks)),
        "corr_target_top1_rate": float(np.mean(ranks <= 1)),
        "corr_target_top3_rate": float(np.mean(ranks <= 3)),
        "corr_target_top10_rate": float(np.mean(ranks <= 10)),
    }
    if "anchor_abs_err" in steps.columns:
        anchor_abs = steps["anchor_abs_err"].to_numpy(np.float32)
        finite_anchor = np.isfinite(anchor_abs)
        if finite_anchor.any():
            metrics["anchor_rmse_ft"] = float(np.sqrt(np.nanmean(anchor_abs[finite_anchor] ** 2)))
            metrics["anchor_median_abs_err_ft"] = float(np.nanmedian(anchor_abs[finite_anchor]))
    return metrics


def _aggregate_metrics(results: Iterable[WellCorrelationResult]) -> dict[str, float | int]:
    frames = [result.steps for result in results if not result.steps.empty]
    if not frames:
        return {"num_wells": 0, "num_steps": 0}
    steps = pd.concat(frames, ignore_index=True)
    metrics = _metrics_from_steps("__all__", steps)
    metrics.pop("well_id", None)
    metrics["num_wells"] = int(steps["well_id"].nunique())
    return metrics


def _json_clean(value):
    if isinstance(value, dict):
        return {k: _json_clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_clean(v) for v in value]
    if isinstance(value, tuple):
        return [_json_clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_report(
    output_dir: Path,
    cfg: CorrelationPanelConfig,
    metrics: dict[str, dict[str, float | int]],
) -> None:
    normal = metrics.get("normal", {})
    shuffled = metrics.get("shuffled_gr", {})
    gap = None
    if normal.get("corr_top1_rmse_ft") is not None and shuffled.get("corr_top1_rmse_ft") is not None:
        gap = float(shuffled["corr_top1_rmse_ft"]) - float(normal["corr_top1_rmse_ft"])
    lines = [
        "# WEBINAR_CORRELATION_PANEL_V0",
        "",
        "## Config",
        "```json",
        json.dumps(_json_clean(asdict(cfg)), indent=2),
        "```",
        "",
        "## Normal",
        f"- wells: {normal.get('num_wells')}",
        f"- steps: {normal.get('num_steps')}",
        f"- corr_top1_rmse_ft: {normal.get('corr_top1_rmse_ft')}",
        f"- corr_top3_oracle_rmse_ft: {normal.get('corr_top3_oracle_rmse_ft')}",
        f"- corr_top10_oracle_rmse_ft: {normal.get('corr_top10_oracle_rmse_ft')}",
        f"- corr_target_top3_rate: {normal.get('corr_target_top3_rate')}",
        f"- corr_target_rank_mean: {normal.get('corr_target_rank_mean')}",
        "",
        "## Shuffled GR",
        f"- corr_top1_rmse_ft: {shuffled.get('corr_top1_rmse_ft')}",
        f"- corr_top3_oracle_rmse_ft: {shuffled.get('corr_top3_oracle_rmse_ft')}",
        f"- corr_target_top3_rate: {shuffled.get('corr_target_top3_rate')}",
        f"- normal_vs_shuffled_top1_gap_ft: {gap}",
        "",
        "## Interpretation",
        "If true TVT is not frequently in top-3/top-10 of this panel, the blocker is log-matching transfer rather than tracker/model selection.",
    ]
    (output_dir / "correlation_panel_report.md").write_text("\n".join(lines) + "\n")


def run_correlation_panel(
    *,
    data_dir: Path,
    output_dir: Path,
    rows_per_step: int = 32,
    vertical_step_ft: float = 5.0,
    patch_radius: int = 3,
    patch_radii: tuple[int, ...] = (),
    stretch_factors: tuple[float, ...] = (1.0,),
    anchor_source: str = "none",
    search_radius_ft: float | None = None,
    k_wells: int = -1,
    include_shuffled: bool = True,
    seed: int = 42,
) -> dict[str, dict[str, float | int]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = CorrelationPanelConfig(
        rows_per_step=rows_per_step,
        vertical_step_ft=vertical_step_ft,
        patch_radius=patch_radius,
        patch_radii=patch_radii,
        stretch_factors=stretch_factors,
        anchor_source=anchor_source,
        search_radius_ft=search_radius_ft,
    )
    wells = discover_wells(DataConfig(data_dir=data_dir, k_wells=k_wells))
    normal_results: list[WellCorrelationResult] = []
    shuffled_results: list[WellCorrelationResult] = []
    for idx, well in enumerate(wells):
        if idx == 0 or (idx + 1) % 50 == 0 or idx + 1 == len(wells):
            print(f"[corr-panel] well {idx + 1}/{len(wells)}", flush=True)
        horizontal, typewell = load_well(well)
        normal_results.append(
            build_well_correlation_panel(well.well_id, horizontal, typewell, cfg)
        )
        if include_shuffled:
            shuffled_results.append(
                build_well_correlation_panel(
                    well.well_id,
                    horizontal,
                    typewell,
                    cfg,
                    shuffle_gr_seed=seed + idx,
                )
            )

    by_well = pd.DataFrame([result.metrics for result in normal_results])
    by_well.to_csv(output_dir / "correlation_panel_by_well.csv", index=False)
    step_frames = [result.steps.assign(variant="normal") for result in normal_results if not result.steps.empty]
    if include_shuffled:
        step_frames.extend(
            result.steps.assign(variant="shuffled_gr")
            for result in shuffled_results
            if not result.steps.empty
        )
    if step_frames:
        pd.concat(step_frames, ignore_index=True).to_parquet(
            output_dir / "correlation_panel_steps.parquet", index=False
        )
    else:
        pd.DataFrame().to_parquet(output_dir / "correlation_panel_steps.parquet", index=False)

    metrics: dict[str, dict[str, float | int]] = {
        "normal": _aggregate_metrics(normal_results),
    }
    if include_shuffled:
        metrics["shuffled_gr"] = _aggregate_metrics(shuffled_results)
        normal_top1 = metrics["normal"].get("corr_top1_rmse_ft")
        shuffled_top1 = metrics["shuffled_gr"].get("corr_top1_rmse_ft")
        if isinstance(normal_top1, float) and isinstance(shuffled_top1, float):
            metrics["normal_vs_shuffled"] = {
                "top1_gap_ft": shuffled_top1 - normal_top1
            }
    (output_dir / "correlation_panel_metrics.json").write_text(
        json.dumps(_json_clean(metrics), indent=2) + "\n"
    )
    _write_report(output_dir, cfg, _json_clean(metrics))
    print(json.dumps(_json_clean(metrics), indent=2), flush=True)
    return metrics
