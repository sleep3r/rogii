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
    known_tail_linear_anchor,
    score_gr_patch_multiscale,
)
from .heatmap import fill_nan
from .io import discover_wells, load_well


FORMATION_SURFACE_ORDER = ("ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA")
EGFDL_TO_BUDA_SUBZONES = ("EGFDL", "LTHL", "LTGT", "LBHL", "MNSS")


@dataclass(frozen=True)
class FormationCorrelationConfig:
    rows_per_step: int = 32
    vertical_step_ft: float = 5.0
    patch_radius: int = 3
    patch_radii: tuple[int, ...] = ()
    stretch_factors: tuple[float, ...] = (1.0,)
    min_patch_points: int = 3
    mad_weight: float = 0.15
    raw_mad_weight: float = 0.02
    topk: tuple[int, ...] = (1, 3, 10)


@dataclass(frozen=True)
class WellFormationCorrelationResult:
    well_id: str
    metrics: dict[str, dict[str, float | int | str]]
    steps: pd.DataFrame


def infer_allowed_geology_from_surfaces(horizontal: pd.DataFrame) -> list[tuple[str, ...]]:
    """Infer deployable coarse typewell geology labels from row Z and formation tops."""
    missing = {"Z", *FORMATION_SURFACE_ORDER}.difference(horizontal.columns)
    if missing:
        return [("__unknown__",)] * len(horizontal)
    z = pd.to_numeric(horizontal["Z"], errors="coerce").to_numpy(dtype=np.float32)
    surfaces = {
        col: pd.to_numeric(horizontal[col], errors="coerce").to_numpy(dtype=np.float32)
        for col in FORMATION_SURFACE_ORDER
    }
    allowed: list[tuple[str, ...]] = []
    for idx, z_value in enumerate(z):
        if not np.isfinite(z_value):
            allowed.append(("__unknown__",))
            continue
        tops = {col: surfaces[col][idx] for col in FORMATION_SURFACE_ORDER}
        if any(not np.isfinite(value) for value in tops.values()):
            allowed.append(("__unknown__",))
        elif z_value > tops["ANCC"]:
            allowed.append(("__unknown__",))
        elif z_value > tops["ASTNU"]:
            allowed.append(("ANCC",))
        elif z_value > tops["ASTNL"]:
            allowed.append(("ASTNU",))
        elif z_value > tops["EGFDU"]:
            allowed.append(("ASTNL",))
        elif z_value > tops["EGFDL"]:
            allowed.append(("EGFDU",))
        elif z_value > tops["BUDA"]:
            allowed.append(EGFDL_TO_BUDA_SUBZONES)
        else:
            allowed.append(("BUDA",))
    return allowed


def _allowed_geology_by_step(
    horizontal: pd.DataFrame, rows_per_step: int
) -> list[tuple[str, ...]]:
    allowed_rows = infer_allowed_geology_from_surfaces(horizontal)
    usable = (len(allowed_rows) // rows_per_step) * rows_per_step
    out: list[tuple[str, ...]] = []
    for start in range(0, usable, rows_per_step):
        chunk = allowed_rows[start : start + rows_per_step]
        counts: dict[tuple[str, ...], int] = {}
        for item in chunk:
            counts[item] = counts.get(item, 0) + 1
        out.append(max(counts, key=counts.get) if counts else ("__unknown__",))
    return out


def _typewell_grid_with_geology(
    typewell: pd.DataFrame, vertical_step_ft: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tvt_grid, gr_grid = _regular_typewell_grid(typewell, vertical_step_ft)
    if tvt_grid.size == 0:
        return tvt_grid, gr_grid, np.empty(0, dtype=object)
    if "Geology" not in typewell.columns:
        return tvt_grid, gr_grid, np.full(tvt_grid.size, "__unknown__", dtype=object)
    tvt = pd.to_numeric(typewell["TVT"], errors="coerce").to_numpy(dtype=np.float32)
    geology = typewell["Geology"].fillna("__unknown__").astype(str).to_numpy(dtype=object)
    finite = np.isfinite(tvt)
    if not finite.any():
        return tvt_grid, gr_grid, np.full(tvt_grid.size, "__unknown__", dtype=object)
    order = np.argsort(tvt[finite])
    tvt_sorted = tvt[finite][order]
    geology_sorted = geology[finite][order]
    nearest = np.searchsorted(tvt_sorted, tvt_grid, side="left")
    nearest = np.clip(nearest, 0, len(tvt_sorted) - 1)
    prev = np.clip(nearest - 1, 0, len(tvt_sorted) - 1)
    choose_prev = np.abs(tvt_grid - tvt_sorted[prev]) <= np.abs(tvt_grid - tvt_sorted[nearest])
    idx = np.where(choose_prev, prev, nearest)
    return tvt_grid, gr_grid, geology_sorted[idx].astype(object)


def _shuffle_finite(values: np.ndarray, finite: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.asarray(values, dtype=np.float32).copy()
    finite_idx = np.flatnonzero(np.asarray(finite) > 0.5)
    if finite_idx.size:
        out[finite_idx] = rng.permutation(out[finite_idx])
    return out


def _mask_scores_by_geology(
    scores: np.ndarray, geology_grid: np.ndarray, target_geology: object
) -> np.ndarray:
    out = np.asarray(scores, dtype=np.float32).copy()
    if target_geology is None:
        return out
    if isinstance(target_geology, (tuple, list, set)):
        allowed = {str(item) for item in target_geology if str(item) != "__unknown__"}
    else:
        allowed = {str(target_geology)} if str(target_geology) != "__unknown__" else set()
    if not allowed:
        return out
    out[~np.isin(geology_grid.astype(str), list(allowed))] = -np.inf
    return out


def _nearest_sqerr(candidate_tvt: np.ndarray, true_tvt: float) -> float:
    if candidate_tvt.size == 0:
        return float("nan")
    return float(np.min((candidate_tvt.astype(np.float32) - true_tvt) ** 2))


def _variant_row(
    *,
    well_id: str,
    step: int,
    variant: str,
    scores: np.ndarray,
    shuffled_scores: np.ndarray,
    tvt_grid: np.ndarray,
    geology_grid: np.ndarray,
    true_tvt: float,
    true_bin: int,
    true_geology: str,
    anchor_geology: str | tuple[str, ...] | None,
    max_topk: int,
) -> dict[str, float | int | str]:
    top = _topk_indices(scores, max_topk)
    shuffled_top = _topk_indices(shuffled_scores, max_topk)
    top1 = top[:1]
    top3 = top[: min(3, len(top))]
    top10 = top[: min(10, len(top))]
    shuffled_top10 = shuffled_top[: min(10, len(shuffled_top))]
    top1_bin = int(top1[0]) if top1.size else -1
    if isinstance(anchor_geology, tuple):
        allowed_geology = tuple(str(item) for item in anchor_geology)
    elif anchor_geology is None:
        allowed_geology = ()
    else:
        allowed_geology = (str(anchor_geology),)
    return {
        "well_id": well_id,
        "step": int(step),
        "variant": variant,
        "true_tvt": float(true_tvt),
        "true_bin": int(true_bin),
        "true_geology": str(true_geology),
        "anchor_geology": "|".join(allowed_geology) if allowed_geology else "__unknown__",
        "geology_match": int(str(true_geology) in allowed_geology)
        if allowed_geology
        else np.nan,
        "true_score": float(scores[true_bin]) if np.isfinite(scores[true_bin]) else np.nan,
        "shuffled_true_score": float(shuffled_scores[true_bin])
        if np.isfinite(shuffled_scores[true_bin])
        else np.nan,
        "true_rank": _rank_of_true(scores, true_bin),
        "shuffled_true_rank": _rank_of_true(shuffled_scores, true_bin),
        "top1_tvt": float(tvt_grid[top1_bin]) if top1_bin >= 0 else np.nan,
        "top1_geology": str(geology_grid[top1_bin]) if top1_bin >= 0 else "__none__",
        "top3_oracle_sqerr": _nearest_sqerr(tvt_grid[top3], true_tvt),
        "top10_oracle_sqerr": _nearest_sqerr(tvt_grid[top10], true_tvt),
        "shuffled_top10_oracle_sqerr": _nearest_sqerr(tvt_grid[shuffled_top10], true_tvt),
        "candidate_bins": int(np.isfinite(scores).sum()),
    }


def build_well_formation_correlation(
    well_id: str,
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    cfg: FormationCorrelationConfig,
    *,
    shuffle_gr_seed: int = 42,
) -> WellFormationCorrelationResult:
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
    anchor_path = known_tail_linear_anchor(comp_tvt_input)
    surface_allowed_by_step = _allowed_geology_by_step(horizontal, cfg.rows_per_step)

    tvt_grid, typewell_gr, geology_grid = _typewell_grid_with_geology(
        typewell, cfg.vertical_step_ft
    )
    if tvt_grid.size == 0 or typewell_gr.size == 0 or comp_tvt.size == 0:
        return WellFormationCorrelationResult(well_id, {}, pd.DataFrame())

    hidden_steps = np.flatnonzero(~np.isfinite(comp_tvt_input) & np.isfinite(comp_tvt))
    patch_radii = cfg.patch_radii or (cfg.patch_radius,)
    max_topk = max(cfg.topk)
    rows: list[dict[str, float | int | str]] = []
    for step in hidden_steps:
        if step >= comp_gr.size or step >= comp_shuffled_gr.size or not np.isfinite(comp_gr[step]):
            continue
        scores, _ = score_gr_patch_multiscale(
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
        true_geology = str(geology_grid[true_bin])
        anchor_tvt = (
            float(anchor_path[step])
            if step < len(anchor_path) and np.isfinite(anchor_path[step])
            else np.nan
        )
        anchor_bin = int(np.abs(tvt_grid - anchor_tvt).argmin()) if np.isfinite(anchor_tvt) else -1
        anchor_geology = str(geology_grid[anchor_bin]) if anchor_bin >= 0 else None
        surface_allowed = (
            surface_allowed_by_step[step]
            if step < len(surface_allowed_by_step)
            else ("__unknown__",)
        )
        variant_scores = {
            "global": (scores, shuffled_scores, None),
            "oracle_geology": (
                _mask_scores_by_geology(scores, geology_grid, true_geology),
                _mask_scores_by_geology(shuffled_scores, geology_grid, true_geology),
                None,
            ),
            "anchor_geology": (
                _mask_scores_by_geology(scores, geology_grid, anchor_geology),
                _mask_scores_by_geology(shuffled_scores, geology_grid, anchor_geology),
                anchor_geology,
            ),
            "surface_geology": (
                _mask_scores_by_geology(scores, geology_grid, surface_allowed),
                _mask_scores_by_geology(shuffled_scores, geology_grid, surface_allowed),
                surface_allowed,
            ),
        }
        for variant, (variant_score, shuffled_variant_score, row_anchor_geology) in variant_scores.items():
            rows.append(
                _variant_row(
                    well_id=well_id,
                    step=int(step),
                    variant=variant,
                    scores=variant_score,
                    shuffled_scores=shuffled_variant_score,
                    tvt_grid=tvt_grid,
                    geology_grid=geology_grid,
                    true_tvt=true_tvt,
                    true_bin=true_bin,
                    true_geology=true_geology,
                    anchor_geology=row_anchor_geology,
                    max_topk=max_topk,
                )
            )

    steps = pd.DataFrame(rows)
    metrics = _metrics_by_variant(well_id, steps)
    return WellFormationCorrelationResult(well_id, metrics, steps)


def _metrics_from_steps(steps: pd.DataFrame) -> dict[str, float | int | str]:
    if steps.empty:
        return {"num_steps": 0}
    ranks = steps["true_rank"].to_numpy(np.float32)
    top1_sqerr = (
        steps["top1_tvt"].to_numpy(np.float32) - steps["true_tvt"].to_numpy(np.float32)
    ) ** 2
    metrics: dict[str, float | int | str] = {
        "num_steps": int(len(steps)),
        "corr_top1_rmse_ft": float(np.sqrt(np.nanmean(top1_sqerr))),
        "corr_top3_oracle_rmse_ft": float(np.sqrt(np.nanmean(steps["top3_oracle_sqerr"]))),
        "corr_top10_oracle_rmse_ft": float(np.sqrt(np.nanmean(steps["top10_oracle_sqerr"]))),
        "shuffled_top10_oracle_rmse_ft": float(
            np.sqrt(np.nanmean(steps["shuffled_top10_oracle_sqerr"]))
        ),
        "corr_target_rank_mean": float(np.nanmean(ranks)),
        "corr_target_top1_rate": float(np.nanmean(ranks <= 1)),
        "corr_target_top3_rate": float(np.nanmean(ranks <= 3)),
        "corr_target_top10_rate": float(np.nanmean(ranks <= 10)),
        "candidate_bins_mean": float(np.nanmean(steps["candidate_bins"])),
    }
    if "geology_match" in steps.columns and steps["geology_match"].notna().any():
        metrics["geology_match_rate"] = float(np.nanmean(steps["geology_match"]))
    return metrics


def _metrics_by_variant(
    well_id: str, steps: pd.DataFrame
) -> dict[str, dict[str, float | int | str]]:
    out: dict[str, dict[str, float | int | str]] = {}
    if steps.empty:
        return out
    for variant, group in steps.groupby("variant"):
        metrics = _metrics_from_steps(group)
        metrics["well_id"] = well_id
        if variant == "anchor_geology" and "geology_match_rate" in metrics:
            metrics["anchor_geology_match_rate"] = metrics["geology_match_rate"]
        if variant == "surface_geology" and "geology_match_rate" in metrics:
            metrics["surface_geology_match_rate"] = metrics["geology_match_rate"]
        out[str(variant)] = metrics
    return out


def _aggregate_metrics(
    results: Iterable[WellFormationCorrelationResult],
) -> dict[str, dict[str, float | int]]:
    frames = [result.steps for result in results if not result.steps.empty]
    if not frames:
        return {}
    steps = pd.concat(frames, ignore_index=True)
    out: dict[str, dict[str, float | int]] = {}
    for variant, group in steps.groupby("variant"):
        metrics = _metrics_from_steps(group)
        metrics["num_wells"] = int(group["well_id"].nunique())
        if variant == "anchor_geology" and "geology_match_rate" in metrics:
            metrics["anchor_geology_match_rate"] = metrics["geology_match_rate"]
        if variant == "surface_geology" and "geology_match_rate" in metrics:
            metrics["surface_geology_match_rate"] = metrics["geology_match_rate"]
        out[str(variant)] = metrics
    return out


def _write_report(
    output_dir: Path,
    cfg: FormationCorrelationConfig,
    metrics: dict[str, dict[str, float | int]],
) -> None:
    lines = [
        "# FORMATION_AWARE_CORRELATION_V0",
        "",
        "## Config",
        "```json",
        json.dumps(_json_clean(asdict(cfg)), indent=2),
        "```",
        "",
        "## Metrics",
        "| variant | top1 RMSE ft | top10 oracle RMSE ft | true top10 % | candidate bins |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant in ["global", "surface_geology", "anchor_geology", "oracle_geology"]:
        src = metrics.get(variant, {})
        lines.append(
            f"| {variant} | {src.get('corr_top1_rmse_ft')} | {src.get('corr_top10_oracle_rmse_ft')} | "
            f"{100 * float(src.get('corr_target_top10_rate', 0)):.2f} | {src.get('candidate_bins_mean')} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "Oracle geology measures whether the GR score becomes useful if the correct stratigraphic zone is known. Anchor geology measures whether a simple known-tail extrapolated zone is enough to constrain the typewell search without target leakage.",
        ]
    )
    (output_dir / "formation_correlation_report.md").write_text("\n".join(lines) + "\n")


def run_formation_correlation_audit(
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
) -> dict[str, dict[str, float | int]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = FormationCorrelationConfig(
        rows_per_step=rows_per_step,
        vertical_step_ft=vertical_step_ft,
        patch_radius=patch_radius,
        patch_radii=patch_radii,
        stretch_factors=stretch_factors,
    )
    wells = discover_wells(DataConfig(data_dir=data_dir, k_wells=k_wells))
    results: list[WellFormationCorrelationResult] = []
    for idx, well in enumerate(wells):
        if idx == 0 or (idx + 1) % 50 == 0 or idx + 1 == len(wells):
            print(f"[formation-corr] well {idx + 1}/{len(wells)}", flush=True)
        horizontal, typewell = load_well(well)
        results.append(
            build_well_formation_correlation(
                well.well_id,
                horizontal,
                typewell,
                cfg,
                shuffle_gr_seed=seed + idx,
            )
        )

    metrics = _aggregate_metrics(results)
    by_well_rows: list[dict[str, float | int | str]] = []
    for result in results:
        for variant, values in result.metrics.items():
            by_well_rows.append({"well_id": result.well_id, "variant": variant, **values})
    pd.DataFrame(by_well_rows).to_csv(
        output_dir / "formation_correlation_by_well.csv", index=False
    )
    step_frames = [result.steps for result in results if not result.steps.empty]
    if step_frames:
        pd.concat(step_frames, ignore_index=True).to_parquet(
            output_dir / "formation_correlation_steps.parquet", index=False
        )
    else:
        pd.DataFrame().to_parquet(output_dir / "formation_correlation_steps.parquet", index=False)

    (output_dir / "formation_correlation_metrics.json").write_text(
        json.dumps(_json_clean(metrics), indent=2) + "\n"
    )
    _write_report(output_dir, cfg, _json_clean(metrics))
    print(json.dumps(_json_clean(metrics), indent=2), flush=True)
    return metrics
