from __future__ import annotations

import argparse
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
    _nearest_rmse,
    _rank_of_true,
    _regular_typewell_grid,
    _topk_indices,
    known_tail_linear_anchor,
    score_gr_patch_multiscale,
)
from .heatmap import fill_nan
from .io import discover_wells, load_well


@dataclass(frozen=True)
class ConditionalCorrConfig:
    rows_per_step: int = 32
    vertical_step_ft: float = 5.0
    patch_radius: int = 3
    patch_radii: tuple[int, ...] = ()
    stretch_factors: tuple[float, ...] = (1.0,)
    min_patch_points: int = 3
    mad_weight: float = 0.15
    raw_mad_weight: float = 0.02
    location_sigma_ft: float = 80.0
    gr_weight: float = 1.0
    location_weight: float = 1.0
    topk: tuple[int, ...] = (1, 3, 10)


@dataclass(frozen=True)
class ConditionalCorrResult:
    well_id: str
    metrics: dict[str, dict[str, float | int | str]]
    steps: pd.DataFrame


VARIANTS: tuple[str, ...] = (
    "gr_only",
    "location_only",
    "gr_location",
    "shuffled_gr_location",
    "zero_gr_location",
)


def location_prior_scores(
    tvt_grid: np.ndarray,
    *,
    anchor_tvt: float | None,
    sigma_ft: float,
) -> np.ndarray:
    """Return centered log-likelihood scores from a test-available TVT location prior."""
    grid = np.asarray(tvt_grid, dtype=np.float32)
    if anchor_tvt is None or not np.isfinite(anchor_tvt) or grid.size == 0:
        return np.zeros(grid.shape, dtype=np.float32)
    sigma = max(float(sigma_ft), 1.0)
    scores = -0.5 * ((grid - float(anchor_tvt)) / sigma) ** 2
    return scores.astype(np.float32)


def _finite_or_zero(scores: np.ndarray) -> np.ndarray:
    out = np.asarray(scores, dtype=np.float32).copy()
    out[~np.isfinite(out)] = 0.0
    return out


def _variant_scores(
    gr_scores: np.ndarray,
    shuffled_scores: np.ndarray,
    loc_scores: np.ndarray,
    cfg: ConditionalCorrConfig,
) -> dict[str, np.ndarray]:
    gr = _finite_or_zero(gr_scores)
    shuffled = _finite_or_zero(shuffled_scores)
    loc = _finite_or_zero(loc_scores)
    return {
        "gr_only": gr,
        "location_only": loc,
        "gr_location": float(cfg.gr_weight) * gr + float(cfg.location_weight) * loc,
        "shuffled_gr_location": float(cfg.gr_weight) * shuffled
        + float(cfg.location_weight) * loc,
        "zero_gr_location": float(cfg.location_weight) * loc,
    }


def _empty_variant_metrics() -> dict[str, float | int]:
    return {
        "num_steps": 0,
        "corr_top1_rmse_ft": float("nan"),
        "corr_top3_oracle_rmse_ft": float("nan"),
        "corr_top10_oracle_rmse_ft": float("nan"),
        "corr_target_rank_mean": float("nan"),
        "corr_target_top1_rate": float("nan"),
        "corr_target_top3_rate": float("nan"),
        "corr_target_top10_rate": float("nan"),
    }


def _metrics_from_steps(steps: pd.DataFrame) -> dict[str, float | int]:
    if steps.empty:
        return _empty_variant_metrics()
    top1_sqerr = (
        steps["top1_tvt"].to_numpy(np.float32)
        - steps["true_tvt"].to_numpy(np.float32)
    ) ** 2
    ranks = steps["true_rank"].to_numpy(np.float32)
    metrics: dict[str, float | int] = {
        "num_steps": int(len(steps)),
        "corr_top1_rmse_ft": float(np.sqrt(np.nanmean(top1_sqerr))),
        "corr_top3_oracle_rmse_ft": float(np.sqrt(np.nanmean(steps["top3_oracle_sqerr"]))),
        "corr_top10_oracle_rmse_ft": float(np.sqrt(np.nanmean(steps["top10_oracle_sqerr"]))),
        "corr_target_rank_mean": float(np.nanmean(ranks)),
        "corr_target_top1_rate": float(np.nanmean(ranks <= 1)),
        "corr_target_top3_rate": float(np.nanmean(ranks <= 3)),
        "corr_target_top10_rate": float(np.nanmean(ranks <= 10)),
    }
    if "anchor_abs_err" in steps.columns:
        anchor_abs = steps["anchor_abs_err"].to_numpy(np.float32)
        finite = np.isfinite(anchor_abs)
        if finite.any():
            metrics["anchor_rmse_ft"] = float(np.sqrt(np.nanmean(anchor_abs[finite] ** 2)))
    return metrics


def _comparison(metrics: dict[str, dict[str, float | int | str]]) -> dict[str, float]:
    def f(variant: str, key: str) -> float:
        value = metrics.get(variant, {}).get(key, np.nan)
        return float(value) if isinstance(value, (int, float, np.integer, np.floating)) else float("nan")

    return {
        "gr_location_vs_shuffled_top10_rate_gap": f("gr_location", "corr_target_top10_rate")
        - f("shuffled_gr_location", "corr_target_top10_rate"),
        "gr_location_vs_location_only_top10_rate_gap": f("gr_location", "corr_target_top10_rate")
        - f("location_only", "corr_target_top10_rate"),
        "gr_location_vs_zero_top10_rate_gap": f("gr_location", "corr_target_top10_rate")
        - f("zero_gr_location", "corr_target_top10_rate"),
        "gr_location_vs_shuffled_top10_rmse_gap_ft": f(
            "shuffled_gr_location", "corr_top10_oracle_rmse_ft"
        )
        - f("gr_location", "corr_top10_oracle_rmse_ft"),
        "gr_location_vs_location_only_top10_rmse_gap_ft": f(
            "location_only", "corr_top10_oracle_rmse_ft"
        )
        - f("gr_location", "corr_top10_oracle_rmse_ft"),
        "gr_location_vs_location_only_top1_gap_ft": f(
            "location_only", "corr_top1_rmse_ft"
        )
        - f("gr_location", "corr_top1_rmse_ft"),
    }


def build_well_conditional_corr_panel(
    well_id: str,
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    cfg: ConditionalCorrConfig,
    *,
    seed: int = 42,
) -> ConditionalCorrResult:
    tvt = pd.to_numeric(horizontal["TVT"], errors="coerce").to_numpy(dtype=np.float32)
    tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(dtype=np.float32)
    gr_raw = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(dtype=np.float32)
    gr_filled, gr_finite = fill_nan(gr_raw)

    rng = np.random.default_rng(seed)
    shuffled_gr = gr_filled.copy()
    finite_idx = np.flatnonzero(gr_finite > 0.5)
    if finite_idx.size:
        shuffled_gr[finite_idx] = rng.permutation(shuffled_gr[finite_idx])

    comp_tvt = _compress_nanmean(tvt, cfg.rows_per_step)
    comp_tvt_input = _compress_nanmean(tvt_input, cfg.rows_per_step)
    comp_gr = _compress_nanmean(gr_filled, cfg.rows_per_step)
    comp_shuffled_gr = _compress_nanmean(shuffled_gr, cfg.rows_per_step)
    comp_gr_finite = _compress_nanmean(gr_finite, cfg.rows_per_step)
    anchor_path = known_tail_linear_anchor(comp_tvt_input)

    tvt_grid, typewell_gr = _regular_typewell_grid(typewell, cfg.vertical_step_ft)
    if tvt_grid.size == 0 or comp_tvt.size == 0:
        return ConditionalCorrResult(
            well_id,
            {variant: _empty_variant_metrics() for variant in VARIANTS},
            pd.DataFrame(),
        )

    hidden_steps = np.flatnonzero(~np.isfinite(comp_tvt_input) & np.isfinite(comp_tvt))
    patch_radii = cfg.patch_radii or (cfg.patch_radius,)
    max_topk = max(cfg.topk)
    rows: list[dict[str, float | int | str]] = []

    for step in hidden_steps:
        if step >= comp_gr.size:
            continue
        gr_scores, best_params = score_gr_patch_multiscale(
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
        anchor_tvt = (
            float(anchor_path[step])
            if step < anchor_path.size and np.isfinite(anchor_path[step])
            else None
        )
        loc_scores = location_prior_scores(
            tvt_grid, anchor_tvt=anchor_tvt, sigma_ft=cfg.location_sigma_ft
        )
        true_tvt = float(comp_tvt[step])
        true_bin = int(np.abs(tvt_grid - true_tvt).argmin())
        for variant, scores in _variant_scores(
            gr_scores, shuffled_scores, loc_scores, cfg
        ).items():
            top = _topk_indices(scores, max_topk)
            top1 = top[:1]
            top3 = top[: min(3, len(top))]
            top10 = top[: min(10, len(top))]
            rows.append(
                {
                    "well_id": well_id,
                    "step": int(step),
                    "variant": variant,
                    "true_tvt": true_tvt,
                    "true_bin": true_bin,
                    "true_score": float(scores[true_bin])
                    if np.isfinite(scores[true_bin])
                    else np.nan,
                    "true_rank": _rank_of_true(scores, true_bin),
                    "top1_tvt": float(tvt_grid[top1[0]]) if top1.size else np.nan,
                    "top1_score": float(scores[top1[0]]) if top1.size else np.nan,
                    "top3_oracle_sqerr": _nearest_rmse(tvt_grid[top3], true_tvt),
                    "top10_oracle_sqerr": _nearest_rmse(tvt_grid[top10], true_tvt),
                    "anchor_tvt": anchor_tvt if anchor_tvt is not None else np.nan,
                    "anchor_abs_err": abs(float(anchor_tvt) - true_tvt)
                    if anchor_tvt is not None
                    else np.nan,
                    "gr_finite_frac": float(comp_gr_finite[step])
                    if step < comp_gr_finite.size and np.isfinite(comp_gr_finite[step])
                    else 0.0,
                    "best_patch_radius": best_params[0] if best_params is not None else np.nan,
                    "best_stretch_factor": best_params[1] if best_params is not None else np.nan,
                }
            )

    steps = pd.DataFrame(rows)
    metrics: dict[str, dict[str, float | int | str]] = {}
    for variant in VARIANTS:
        variant_steps = steps[steps["variant"] == variant] if not steps.empty else pd.DataFrame()
        metrics[variant] = _metrics_from_steps(variant_steps)
        metrics[variant]["well_id"] = well_id
    metrics["comparison"] = _comparison(metrics)
    return ConditionalCorrResult(well_id, metrics, steps)


def _aggregate_variant_metrics(results: Iterable[ConditionalCorrResult], variant: str) -> dict[str, float | int]:
    frames = [
        result.steps[result.steps["variant"] == variant]
        for result in results
        if not result.steps.empty and "variant" in result.steps.columns
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return _empty_variant_metrics()
    steps = pd.concat(frames, ignore_index=True)
    metrics = _metrics_from_steps(steps)
    metrics["num_wells"] = int(steps["well_id"].nunique())
    return metrics


def _write_report(
    output_dir: Path,
    cfg: ConditionalCorrConfig,
    metrics: dict[str, dict[str, float | int]],
) -> None:
    comp = metrics.get("comparison", {})
    lines = [
        "# CONDITIONAL_CORR_PANEL_V0",
        "",
        "## Question",
        "Kaggle suggestion: concatenate GR values with location values, so matching uses both log value and distance.",
        "",
        "This audit separates three effects:",
        "- `gr_only`: raw GR/typewell score.",
        "- `location_only`: known-tail TVT location prior, no GR.",
        "- `gr_location` vs `shuffled_gr_location`: the decisive test for GR signal beyond the same location prior.",
        "",
        "## Config",
        "```json",
        json.dumps(_json_clean(asdict(cfg)), indent=2),
        "```",
        "",
        "## Metrics",
        "| variant | top1 RMSE ft | top10 oracle RMSE ft | true top10 rate | true rank mean |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        src = metrics.get(variant, {})
        lines.append(
            f"| {variant} | {src.get('corr_top1_rmse_ft')} | "
            f"{src.get('corr_top10_oracle_rmse_ft')} | "
            f"{src.get('corr_target_top10_rate')} | "
            f"{src.get('corr_target_rank_mean')} |"
        )
    lines.extend(
        [
            "",
            "## Decisive Gaps",
            f"- gr_location_vs_shuffled_top10_rate_gap: {comp.get('gr_location_vs_shuffled_top10_rate_gap')}",
            f"- gr_location_vs_location_only_top10_rate_gap: {comp.get('gr_location_vs_location_only_top10_rate_gap')}",
            f"- gr_location_vs_shuffled_top10_rmse_gap_ft: {comp.get('gr_location_vs_shuffled_top10_rmse_gap_ft')}",
            f"- gr_location_vs_location_only_top10_rmse_gap_ft: {comp.get('gr_location_vs_location_only_top10_rmse_gap_ft')}",
            "",
            "## Interpretation",
            "If `gr_location` is not clearly better than `shuffled_gr_location`, the apparent improvement comes from location/state restriction rather than GR correlation.",
        ]
    )
    (output_dir / "conditional_corr_report.md").write_text("\n".join(lines) + "\n")


def run_conditional_corr_panel(
    *,
    data_dir: Path,
    output_dir: Path,
    rows_per_step: int = 32,
    vertical_step_ft: float = 5.0,
    patch_radius: int = 3,
    patch_radii: tuple[int, ...] = (),
    stretch_factors: tuple[float, ...] = (1.0,),
    min_patch_points: int = 3,
    location_sigma_ft: float = 80.0,
    gr_weight: float = 1.0,
    location_weight: float = 1.0,
    k_wells: int = -1,
    seed: int = 42,
) -> dict[str, dict[str, float | int]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = ConditionalCorrConfig(
        rows_per_step=rows_per_step,
        vertical_step_ft=vertical_step_ft,
        patch_radius=patch_radius,
        patch_radii=patch_radii,
        stretch_factors=stretch_factors,
        min_patch_points=min_patch_points,
        location_sigma_ft=location_sigma_ft,
        gr_weight=gr_weight,
        location_weight=location_weight,
    )
    wells = discover_wells(DataConfig(data_dir=data_dir, k_wells=k_wells))
    results: list[ConditionalCorrResult] = []
    for idx, well in enumerate(wells):
        if idx == 0 or (idx + 1) % 50 == 0 or idx + 1 == len(wells):
            print(f"[conditional-corr] well {idx + 1}/{len(wells)}", flush=True)
        horizontal, typewell = load_well(well)
        results.append(
            build_well_conditional_corr_panel(
                well.well_id,
                horizontal,
                typewell,
                cfg,
                seed=seed + idx,
            )
        )

    metrics: dict[str, dict[str, float | int]] = {
        variant: _aggregate_variant_metrics(results, variant) for variant in VARIANTS
    }
    metrics["comparison"] = _comparison(metrics)
    by_well_rows: list[dict[str, float | int | str]] = []
    for result in results:
        for variant in VARIANTS:
            row = dict(result.metrics.get(variant, {}))
            row["variant"] = variant
            by_well_rows.append(row)
    pd.DataFrame(by_well_rows).to_csv(output_dir / "conditional_corr_by_well.csv", index=False)
    step_frames = [result.steps for result in results if not result.steps.empty]
    if step_frames:
        pd.concat(step_frames, ignore_index=True).to_parquet(
            output_dir / "conditional_corr_steps.parquet", index=False
        )
    else:
        pd.DataFrame().to_parquet(output_dir / "conditional_corr_steps.parquet", index=False)
    (output_dir / "conditional_corr_metrics.json").write_text(
        json.dumps(_json_clean(metrics), indent=2) + "\n"
    )
    _write_report(output_dir, cfg, _json_clean(metrics))
    print(json.dumps(_json_clean(metrics), indent=2), flush=True)
    return metrics


def _parse_tuple_int(value: str | None) -> tuple[int, ...]:
    if not value:
        return ()
    return tuple(int(part) for part in value.split(",") if part.strip())


def _parse_tuple_float(value: str | None) -> tuple[float, ...]:
    if not value:
        return (1.0,)
    return tuple(float(part) for part in value.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run conditional GR + location correlation audit.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/conditional_corr_panel_v0"))
    parser.add_argument("--rows-per-step", type=int, default=32)
    parser.add_argument("--vertical-step-ft", type=float, default=5.0)
    parser.add_argument("--patch-radius", type=int, default=3)
    parser.add_argument("--patch-radii", type=str, default="")
    parser.add_argument("--stretch-factors", type=str, default="1.0")
    parser.add_argument("--min-patch-points", type=int, default=3)
    parser.add_argument("--location-sigma-ft", type=float, default=80.0)
    parser.add_argument("--gr-weight", type=float, default=1.0)
    parser.add_argument("--location-weight", type=float, default=1.0)
    parser.add_argument("--k-wells", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_conditional_corr_panel(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        rows_per_step=args.rows_per_step,
        vertical_step_ft=args.vertical_step_ft,
        patch_radius=args.patch_radius,
        patch_radii=_parse_tuple_int(args.patch_radii),
        stretch_factors=_parse_tuple_float(args.stretch_factors),
        min_patch_points=args.min_patch_points,
        location_sigma_ft=args.location_sigma_ft,
        gr_weight=args.gr_weight,
        location_weight=args.location_weight,
        k_wells=args.k_wells,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
