"""Experiment 0: Plane-Coordinate Heatmap Audit.

Tests the forum hypothesis that ``TVT + Z`` ("plane coordinate") provides
stronger GR alignment signal than raw TVT alone.

Scoring formulations
--------------------
Raw:
    score_raw[j] = -abs(tw_gr[j] - h_gr[step])
    true_bin     = argmin_j |tw_tvt[j] - tvt[step]|

Plane-coord:
    plane_coord    = tvt[step] + Z[step]
    h_sampled_gr   = interp(plane_coord, tvt_curve, gr_curve)
    score_plane[j] = -abs(tw_gr[j] - h_sampled_gr)
    true_bin       = argmin_j |tw_tvt[j] - plane_coord|

Controls:
    shuffled_raw   – comp_gr shuffled, raw true bin
    shuffled_plane – comp_gr shuffled, interp for h_sampled_gr, plane true bin
    zero_gr        – h_gr = 0, raw true bin
    shuffled_tw    – typewell GR shuffled, raw true bin

GO condition (either):
    normal_vs_shuffled_top10_gap  >= 0.05
    top10_oracle_rmse improvement >= 0.30 ft
"""
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
    _rank_of_true,
    _regular_typewell_grid,
    _topk_indices,
)
from .heatmap import fill_nan
from .io import discover_wells, load_well
from .typewell_mismatch import _collapse_zone, _typewell_grid_zones


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlaneCoordAuditConfig:
    rows_per_step: int = 32
    vertical_step_ft: float = 5.0
    k_wells: int = -1
    seed: int = 42
    n_panel_wells: int = 20
    topk: tuple[int, ...] = (1, 3, 10)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _shuffle_finite_gr(
    gr_filled: np.ndarray,
    gr_finite: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Shuffle finite GR values; return new array of same length."""
    out = np.asarray(gr_filled, dtype=np.float32).copy()
    finite_idx = np.flatnonzero(np.asarray(gr_finite) > 0.5)
    if finite_idx.size > 1:
        out[finite_idx] = rng.permutation(out[finite_idx])
    return out


def _point_scores(typewell_gr: np.ndarray, scalar_gr: float) -> np.ndarray:
    """Score every typewell bin against one scalar GR value.

    Returns ``-abs(tw_gr[j] - scalar_gr)``; higher = better alignment.
    Returns ``-inf`` everywhere when scalar_gr is not finite.
    """
    if not np.isfinite(scalar_gr):
        return np.full(typewell_gr.size, -np.inf, dtype=np.float32)
    return -np.abs(typewell_gr.astype(np.float32) - float(scalar_gr))


def _nearest_sqerr(candidate_tvt: np.ndarray, true_tvt: float) -> float:
    if candidate_tvt.size == 0 or not np.isfinite(true_tvt):
        return float("nan")
    return float(np.min((candidate_tvt.astype(np.float32) - float(true_tvt)) ** 2))


def _safe_mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def _safe_median(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def _safe_rate(mask: np.ndarray | Iterable) -> float:
    arr = np.asarray(list(mask), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def _rmse(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    return float(np.sqrt(np.mean(arr**2))) if arr.size else float("nan")


# ---------------------------------------------------------------------------
# Per-well processing
# ---------------------------------------------------------------------------

_VARIANTS = ["raw", "plane", "shuffled_raw", "shuffled_plane", "zero_gr", "shuffled_tw"]


def _process_well(
    well_id: str,
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    cfg: PlaneCoordAuditConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Return a DataFrame of per-step records for one well, or empty."""
    # ── Extract raw series ──────────────────────────────────────────────────
    tvt = pd.to_numeric(horizontal["TVT"], errors="coerce").to_numpy(dtype=np.float32)
    tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(
        dtype=np.float32
    )
    gr_raw = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(dtype=np.float32)
    gr_filled, gr_finite = fill_nan(gr_raw)

    has_z = "Z" in horizontal.columns
    if has_z:
        z_raw = pd.to_numeric(horizontal["Z"], errors="coerce").to_numpy(dtype=np.float32)
        z_filled, _ = fill_nan(z_raw)
    else:
        z_filled = np.zeros_like(tvt, dtype=np.float32)

    # ── Compress to steps ───────────────────────────────────────────────────
    comp_tvt = _compress_nanmean(tvt, cfg.rows_per_step)
    comp_tvt_input = _compress_nanmean(tvt_input, cfg.rows_per_step)
    comp_gr = _compress_nanmean(gr_filled, cfg.rows_per_step)
    comp_gr_finite = _compress_nanmean(gr_finite, cfg.rows_per_step)
    comp_z = _compress_nanmean(z_filled, cfg.rows_per_step)

    # Shuffled horizontal GR
    shuffled_gr_filled = _shuffle_finite_gr(gr_filled, gr_finite, rng)
    comp_shuffled_gr = _compress_nanmean(shuffled_gr_filled, cfg.rows_per_step)

    # ── Typewell grid ───────────────────────────────────────────────────────
    tvt_grid, tw_gr_grid = _regular_typewell_grid(typewell, cfg.vertical_step_ft)
    if tvt_grid.size == 0 or tw_gr_grid.size == 0 or comp_tvt.size == 0:
        return pd.DataFrame()

    # Shuffled typewell GR (once per well)
    shuffled_tw_gr_grid = rng.permutation(tw_gr_grid).astype(np.float32)

    # Grid zones for zone-level breakdown
    grid_zones = _typewell_grid_zones(typewell, tvt_grid)

    # ── Interpolation basis (sorted by TVT for np.interp) ──────────────────
    n_steps = min(comp_tvt.size, comp_tvt_input.size, comp_gr.size, comp_z.size)
    valid_interp = (
        np.isfinite(comp_tvt[:n_steps]) & np.isfinite(comp_gr[:n_steps])
    )
    if valid_interp.sum() < 2:
        return pd.DataFrame()

    tvt_for_interp = comp_tvt[:n_steps][valid_interp]
    gr_for_interp = comp_gr[:n_steps][valid_interp]
    shuffled_gr_for_interp = comp_shuffled_gr[:n_steps][valid_interp]

    sort_idx = np.argsort(tvt_for_interp)
    tvt_for_interp = tvt_for_interp[sort_idx]
    gr_for_interp = gr_for_interp[sort_idx]
    shuffled_gr_for_interp = shuffled_gr_for_interp[sort_idx]

    # ── Hidden steps ────────────────────────────────────────────────────────
    hidden_mask = (
        ~np.isfinite(comp_tvt_input[:n_steps]) & np.isfinite(comp_tvt[:n_steps])
    )
    hidden_steps = np.flatnonzero(hidden_mask)
    if hidden_steps.size == 0:
        return pd.DataFrame()

    max_topk = max(cfg.topk)
    rows: list[dict] = []

    for step in hidden_steps:
        # Skip steps where horizontal GR is not finite after compression
        h_gr_raw = comp_gr[step] if step < comp_gr.size else np.nan
        if not np.isfinite(h_gr_raw):
            continue

        true_tvt = float(comp_tvt[step])
        cz = float(comp_z[step]) if step < comp_z.size and np.isfinite(comp_z[step]) else 0.0
        plane_coord = true_tvt + cz

        # True bins (raw coordinate vs plane coordinate)
        true_bin_raw = int(np.abs(tvt_grid - true_tvt).argmin())
        true_bin_plane = int(np.abs(tvt_grid - plane_coord).argmin())
        true_zone = (
            str(grid_zones[true_bin_raw])
            if true_bin_raw < len(grid_zones)
            else "__unknown__"
        )

        # Resampled GR at plane coordinate (normal and shuffled)
        h_sampled_gr = float(
            np.interp(plane_coord, tvt_for_interp, gr_for_interp)
        )
        shuffled_h_sampled_gr = float(
            np.interp(plane_coord, tvt_for_interp, shuffled_gr_for_interp)
        )
        h_gr_shuffled = (
            float(comp_shuffled_gr[step])
            if step < comp_shuffled_gr.size and np.isfinite(comp_shuffled_gr[step])
            else np.nan
        )

        # Score vectors per variant: (score_array, true_bin_index, reference_tvt)
        variant_defs: list[tuple[str, np.ndarray, int, float]] = [
            ("raw", _point_scores(tw_gr_grid, h_gr_raw), true_bin_raw, true_tvt),
            ("plane", _point_scores(tw_gr_grid, h_sampled_gr), true_bin_plane, plane_coord),
            (
                "shuffled_raw",
                _point_scores(tw_gr_grid, h_gr_shuffled),
                true_bin_raw,
                true_tvt,
            ),
            (
                "shuffled_plane",
                _point_scores(tw_gr_grid, shuffled_h_sampled_gr),
                true_bin_plane,
                plane_coord,
            ),
            ("zero_gr", _point_scores(tw_gr_grid, 0.0), true_bin_raw, true_tvt),
            (
                "shuffled_tw",
                _point_scores(shuffled_tw_gr_grid, h_gr_raw),
                true_bin_raw,
                true_tvt,
            ),
        ]

        row: dict = {
            "well_id": well_id,
            "step": int(step),
            "true_tvt": true_tvt,
            "plane_coord": plane_coord,
            "true_bin_raw": true_bin_raw,
            "true_bin_plane": true_bin_plane,
            "true_zone": true_zone,
            "h_gr": float(h_gr_raw),
            "h_sampled_gr": h_sampled_gr,
            "gr_finite_frac": float(comp_gr_finite[step])
            if step < comp_gr_finite.size and np.isfinite(comp_gr_finite[step])
            else 0.0,
            "has_z": has_z,
        }

        for vname, scores, true_bin, ref_tvt in variant_defs:
            rank = _rank_of_true(scores, true_bin)
            top10_idx = _topk_indices(scores, max_topk)[:10]
            top10_sqerr = _nearest_sqerr(tvt_grid[top10_idx], ref_tvt)
            score_at_true = (
                float(scores[true_bin])
                if 0 <= true_bin < scores.size and np.isfinite(scores[true_bin])
                else np.nan
            )
            top1_idx = _topk_indices(scores, 1)
            top1_tvt = float(tvt_grid[top1_idx[0]]) if top1_idx.size else np.nan

            row[f"{vname}_rank"] = rank
            row[f"{vname}_score"] = score_at_true
            row[f"{vname}_top10_oracle_sqerr"] = top10_sqerr
            row[f"{vname}_top1_tvt"] = top1_tvt

        rows.append(row)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Metrics aggregation
# ---------------------------------------------------------------------------

def _variant_metrics(
    steps: pd.DataFrame,
    variant: str,
) -> dict[str, float | int]:
    """Compute standard ranking metrics for one score variant from step records."""
    if steps.empty:
        return {"num_steps": 0}
    rank_col = f"{variant}_rank"
    oracle_col = f"{variant}_top10_oracle_sqerr"
    score_col = f"{variant}_score"
    top1_tvt_col = f"{variant}_top1_tvt"

    if rank_col not in steps.columns:
        return {"num_steps": 0}

    ranks = steps[rank_col].to_numpy(np.float32)
    oracle_sqerrs = steps[oracle_col].to_numpy(np.float32) if oracle_col in steps.columns else np.full(len(steps), np.nan)
    scores = steps[score_col].to_numpy(np.float32) if score_col in steps.columns else np.full(len(steps), np.nan)

    return {
        "num_steps": int(len(steps)),
        "num_wells": int(steps["well_id"].nunique()) if "well_id" in steps.columns else 0,
        "true_rank_mean": _safe_mean(ranks),
        "true_rank_median": _safe_median(ranks),
        "true_score_mean": _safe_mean(scores),
        "true_top1_rate": _safe_rate(ranks <= 1),
        "true_top3_rate": _safe_rate(ranks <= 3),
        "true_top10_rate": _safe_rate(ranks <= 10),
        "top10_oracle_rmse_ft": _rmse(np.sqrt(oracle_sqerrs[np.isfinite(oracle_sqerrs)])),
    }


def _compute_all_metrics(
    steps: pd.DataFrame,
    tail_classes: pd.DataFrame,
) -> dict:
    """Build full metrics dict: aggregate + per-variant + per-tail-class breakdowns."""
    if steps.empty:
        return {"error": "no_steps", "aggregate": {}}

    # Merge tail classes
    if not tail_classes.empty and "well_id" in steps.columns:
        steps = steps.merge(tail_classes, on="well_id", how="left")

    aggregate: dict[str, dict] = {}
    for v in _VARIANTS:
        aggregate[v] = _variant_metrics(steps, v)

    # GO / NO-GO evaluation (plane vs shuffled_plane)
    plane_top10 = aggregate["plane"].get("true_top10_rate", float("nan"))
    shuffled_top10 = aggregate["shuffled_plane"].get("true_top10_rate", float("nan"))
    plane_rmse = aggregate["plane"].get("top10_oracle_rmse_ft", float("nan"))
    shuffled_rmse = aggregate["shuffled_plane"].get("top10_oracle_rmse_ft", float("nan"))

    if np.isfinite(plane_top10) and np.isfinite(shuffled_top10):
        gap_top10 = float(plane_top10 - shuffled_top10)
    else:
        gap_top10 = float("nan")

    if np.isfinite(plane_rmse) and np.isfinite(shuffled_rmse):
        rmse_improvement = float(shuffled_rmse - plane_rmse)
    else:
        rmse_improvement = float("nan")

    go_by_top10 = np.isfinite(gap_top10) and gap_top10 >= 0.05
    go_by_rmse = np.isfinite(rmse_improvement) and rmse_improvement >= 0.30
    verdict = "GO" if (go_by_top10 or go_by_rmse) else "NO_GO"

    # Also compare raw vs shuffled_raw
    raw_top10 = aggregate["raw"].get("true_top10_rate", float("nan"))
    shuffled_raw_top10 = aggregate["shuffled_raw"].get("true_top10_rate", float("nan"))
    raw_vs_shuffled_gap = (
        float(raw_top10 - shuffled_raw_top10)
        if np.isfinite(raw_top10) and np.isfinite(shuffled_raw_top10)
        else float("nan")
    )

    summary = {
        "plane_top10_rate": plane_top10,
        "shuffled_plane_top10_rate": shuffled_top10,
        "raw_top10_rate": raw_top10,
        "shuffled_raw_top10_rate": shuffled_raw_top10,
        "normal_vs_shuffled_top10_gap": gap_top10,
        "raw_vs_shuffled_top10_gap": raw_vs_shuffled_gap,
        "plane_vs_raw_top10_gap": float(plane_top10 - raw_top10)
        if np.isfinite(plane_top10) and np.isfinite(raw_top10)
        else float("nan"),
        "plane_top10_oracle_rmse_ft": plane_rmse,
        "shuffled_plane_top10_oracle_rmse_ft": shuffled_rmse,
        "normal_vs_shuffled_oracle_rmse_gap": rmse_improvement,
        "verdict": verdict,
        "go_by_top10": go_by_top10,
        "go_by_rmse": go_by_rmse,
    }

    # Per-zone breakdown (plane variant)
    by_zone: dict = {}
    if "true_zone" in steps.columns:
        for zone, grp in steps.groupby("true_zone", dropna=False):
            by_zone[str(zone)] = _variant_metrics(grp, "plane")

    # Per-tail-class breakdown (plane variant)
    by_tail_class: dict = {}
    if "tail_class" in steps.columns:
        for tc, grp in steps.groupby("tail_class", dropna=False):
            by_tail_class[str(tc)] = _variant_metrics(grp, "plane")

    return {
        "summary": summary,
        "by_variant": aggregate,
        "by_zone": by_zone,
        "by_tail_class": by_tail_class,
        "total_wells": int(steps["well_id"].nunique()) if "well_id" in steps.columns else 0,
        "total_steps": int(len(steps)),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _build_well_heatmaps(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    cfg: PlaneCoordAuditConfig,
    rng: np.random.Generator,
) -> dict | None:
    """Build raw and plane heatmap matrices for one well's hidden steps."""
    tvt = pd.to_numeric(horizontal["TVT"], errors="coerce").to_numpy(dtype=np.float32)
    tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(dtype=np.float32)
    gr_raw = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(dtype=np.float32)
    gr_filled, gr_finite = fill_nan(gr_raw)

    has_z = "Z" in horizontal.columns
    if has_z:
        z_raw = pd.to_numeric(horizontal["Z"], errors="coerce").to_numpy(dtype=np.float32)
        z_filled, _ = fill_nan(z_raw)
    else:
        z_filled = np.zeros_like(tvt, dtype=np.float32)

    comp_tvt = _compress_nanmean(tvt, cfg.rows_per_step)
    comp_tvt_input = _compress_nanmean(tvt_input, cfg.rows_per_step)
    comp_gr = _compress_nanmean(gr_filled, cfg.rows_per_step)
    comp_z = _compress_nanmean(z_filled, cfg.rows_per_step)

    shuffled_gr_filled = _shuffle_finite_gr(gr_filled, gr_finite, rng)
    comp_shuffled_gr = _compress_nanmean(shuffled_gr_filled, cfg.rows_per_step)

    tvt_grid, tw_gr_grid = _regular_typewell_grid(typewell, cfg.vertical_step_ft)
    if tvt_grid.size == 0 or tw_gr_grid.size == 0 or comp_tvt.size == 0:
        return None

    n_steps = min(comp_tvt.size, comp_tvt_input.size, comp_gr.size, comp_z.size)
    hidden_mask = ~np.isfinite(comp_tvt_input[:n_steps]) & np.isfinite(comp_tvt[:n_steps])
    hidden_steps = np.flatnonzero(
        hidden_mask
        & np.isfinite(comp_gr[:n_steps])
        & np.isfinite(comp_tvt[:n_steps])
    )
    if hidden_steps.size < 3:
        return None

    # Interpolation basis
    valid_interp = np.isfinite(comp_tvt[:n_steps]) & np.isfinite(comp_gr[:n_steps])
    if valid_interp.sum() < 2:
        return None

    tvt_for_interp = comp_tvt[:n_steps][valid_interp]
    gr_for_interp = comp_gr[:n_steps][valid_interp]
    shuffled_gr_for_interp = comp_shuffled_gr[:n_steps][valid_interp]
    sort_idx = np.argsort(tvt_for_interp)
    tvt_for_interp = tvt_for_interp[sort_idx]
    gr_for_interp = gr_for_interp[sort_idx]
    shuffled_gr_for_interp = shuffled_gr_for_interp[sort_idx]

    S = len(hidden_steps)
    T = len(tvt_grid)

    # Build heatmaps: shape (T, S)
    h_gr_vec = comp_gr[hidden_steps]  # (S,)
    plane_coords = comp_tvt[hidden_steps] + comp_z[hidden_steps]  # (S,)
    h_sampled_gr_vec = np.interp(plane_coords, tvt_for_interp, gr_for_interp).astype(np.float32)
    shuffled_h_sampled_gr_vec = np.interp(plane_coords, tvt_for_interp, shuffled_gr_for_interp).astype(np.float32)

    # Heatmap: raw  (T, S) = -abs(tw_gr[:, None] - h_gr[None, :])
    hm_raw = -np.abs(tw_gr_grid[:, None] - h_gr_vec[None, :])
    hm_plane = -np.abs(tw_gr_grid[:, None] - h_sampled_gr_vec[None, :])
    hm_shuffled_plane = -np.abs(tw_gr_grid[:, None] - shuffled_h_sampled_gr_vec[None, :])

    # True path bins
    true_tvt_vec = comp_tvt[hidden_steps]
    true_bin_raw = np.array([int(np.abs(tvt_grid - v).argmin()) for v in true_tvt_vec])
    true_bin_plane = np.array([int(np.abs(tvt_grid - v).argmin()) for v in plane_coords])

    # Per-step ranks
    rank_raw = np.array(
        [_rank_of_true(hm_raw[:, i], true_bin_raw[i]) for i in range(S)]
    )
    rank_plane = np.array(
        [_rank_of_true(hm_plane[:, i], true_bin_plane[i]) for i in range(S)]
    )

    return {
        "tvt_grid": tvt_grid,
        "hidden_steps": hidden_steps,
        "hm_raw": hm_raw,
        "hm_plane": hm_plane,
        "hm_shuffled_plane": hm_shuffled_plane,
        "true_bin_raw": true_bin_raw,
        "true_bin_plane": true_bin_plane,
        "rank_raw": rank_raw,
        "rank_plane": rank_plane,
        "plane_coords": plane_coords,
        "true_tvt_vec": true_tvt_vec,
    }


def _write_panel_figures(
    output_dir: Path,
    panel_wells: list[tuple[str, pd.DataFrame, pd.DataFrame]],
    cfg: PlaneCoordAuditConfig,
    seed: int = 0,
) -> None:
    """Save up to n_panel_wells individual heatmap comparison figures."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
    except Exception:
        return

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed + 1000)

    for well_id, horizontal, typewell in panel_wells:
        data = _build_well_heatmaps(horizontal, typewell, cfg, rng)
        if data is None:
            continue

        tvt_grid = data["tvt_grid"]
        S = len(data["hidden_steps"])
        T = len(tvt_grid)
        step_idx = np.arange(S)

        fig, axes = plt.subplots(
            2, 2,
            figsize=(14, 9),
            gridspec_kw={"height_ratios": [3, 1]},
            dpi=120,
        )
        fig.suptitle(
            f"Plane-Coordinate Audit: {well_id}  |  T={T} tw-bins, S={S} hidden steps",
            fontsize=9,
        )

        cmap = "plasma"

        def _plot_hm(ax, heatmap, true_bins, title, color="white"):
            finite = np.isfinite(heatmap)
            vmin = float(np.nanpercentile(heatmap[finite], 5)) if finite.any() else -1
            vmax = float(np.nanpercentile(heatmap[finite], 95)) if finite.any() else 0
            ax.imshow(
                heatmap,
                aspect="auto",
                origin="lower",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
            ax.scatter(
                step_idx,
                true_bins,
                c=color,
                s=4,
                linewidths=0,
                label="true path",
                zorder=3,
            )
            ytick_every = max(1, T // 8)
            ax.set_yticks(np.arange(0, T, ytick_every))
            ax.set_yticklabels(
                [f"{tvt_grid[i]:.0f}" for i in np.arange(0, T, ytick_every)],
                fontsize=5,
            )
            ax.set_title(title, fontsize=8)
            ax.set_xlabel("hidden step", fontsize=7)
            ax.set_ylabel("tw TVT (ft)", fontsize=7)

        _plot_hm(axes[0, 0], data["hm_raw"], data["true_bin_raw"], "Raw heatmap + true path (raw TVT)")
        _plot_hm(axes[0, 1], data["hm_plane"], data["true_bin_plane"], "Plane heatmap + true path (TVT+Z)", color="cyan")
        _plot_hm(axes[1, 0], data["hm_shuffled_plane"], data["true_bin_plane"], "Shuffled-plane heatmap", color="cyan")

        # True-rank curve
        ax_rank = axes[1, 1]
        ax_rank.plot(step_idx, data["rank_raw"], color="#4C72B0", linewidth=0.8, label="raw rank", alpha=0.8)
        ax_rank.plot(step_idx, data["rank_plane"], color="#C44E52", linewidth=0.8, label="plane rank", alpha=0.8)
        ax_rank.axhline(10, color="#888888", linestyle="--", linewidth=0.7, label="rank=10")
        ax_rank.invert_yaxis()
        ax_rank.set_ylim(bottom=max(data["rank_raw"].max(), data["rank_plane"].max()) + 5, top=0)
        ax_rank.set_title("True-rank curve", fontsize=8)
        ax_rank.set_xlabel("hidden step", fontsize=7)
        ax_rank.set_ylabel("rank (lower=better)", fontsize=7)
        ax_rank.legend(fontsize=6)
        for ax in axes.flat:
            ax.tick_params(labelsize=6)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(
            figures_dir / f"panel_{well_id}.png",
            bbox_inches="tight",
            dpi=120,
        )
        plt.close(fig)


def _write_aggregate_figures(output_dir: Path, steps: pd.DataFrame) -> None:
    """Save aggregate summary figures."""
    if steps.empty:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ── True-rank histogram: raw vs plane ───────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=140)
    for ax, col, label, color in [
        (axes[0], "raw_rank", "raw TVT", "#4C72B0"),
        (axes[1], "plane_rank", "plane (TVT+Z)", "#C44E52"),
    ]:
        if col in steps.columns:
            ranks = steps[col].to_numpy(np.float32)
            ranks = ranks[np.isfinite(ranks)]
            ax.hist(
                np.clip(ranks, 1, 100),
                bins=np.arange(1, 102),
                color=color,
                alpha=0.8,
                edgecolor="white",
            )
        ax.set_title(f"True-rank histogram: {label}", fontsize=9)
        ax.set_xlabel("rank (clipped at 100)", fontsize=8)
        ax.set_ylabel("hidden steps", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(figures_dir / "rank_histograms.png", bbox_inches="tight")
    plt.close(fig)

    # ── Top-10 rate: all variants bar chart ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4), dpi=140)
    variant_labels = {
        "raw": "raw",
        "plane": "plane",
        "shuffled_raw": "shuffled raw",
        "shuffled_plane": "shuffled plane",
        "zero_gr": "zero GR",
        "shuffled_tw": "shuffled TW",
    }
    rates = []
    labels = []
    colors = ["#4C72B0", "#C44E52", "#7FA8C9", "#E99395", "#55A868", "#8172B3"]
    for vname, vlabel in variant_labels.items():
        rank_col = f"{vname}_rank"
        if rank_col in steps.columns:
            ranks = steps[rank_col].to_numpy(np.float32)
            rate = float(np.nanmean(ranks <= 10))
        else:
            rate = 0.0
        rates.append(rate)
        labels.append(vlabel)
    x = np.arange(len(rates))
    ax.bar(x, rates, color=colors[: len(rates)], alpha=0.85, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("top-10 rate", fontsize=8)
    ax.set_title("True-path top-10 rate by scoring variant", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(figures_dir / "top10_rate_by_variant.png", bbox_inches="tight")
    plt.close(fig)

    # ── Oracle RMSE comparison ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4), dpi=140)
    rmses = []
    rmse_labels = []
    for vname, vlabel in variant_labels.items():
        ocol = f"{vname}_top10_oracle_sqerr"
        if ocol in steps.columns:
            sqerrs = steps[ocol].to_numpy(np.float32)
            sqerrs = sqerrs[np.isfinite(sqerrs)]
            rmse = float(np.sqrt(np.mean(sqerrs))) if sqerrs.size else float("nan")
        else:
            rmse = float("nan")
        rmses.append(rmse)
        rmse_labels.append(vlabel)
    valid_mask = [np.isfinite(r) for r in rmses]
    ax.bar(
        x[valid_mask],
        [r for r, v in zip(rmses, valid_mask) if v],
        color=[c for c, v in zip(colors[: len(rates)], valid_mask) if v],
        alpha=0.85,
        edgecolor="white",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(rmse_labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("top-10 oracle RMSE (ft)", fontsize=8)
    ax.set_title("Oracle RMSE by scoring variant (top-10 candidates)", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(figures_dir / "oracle_rmse_by_variant.png", bbox_inches="tight")
    plt.close(fig)

    # ── Score-gap distribution: plane vs shuffled_plane ─────────────────────
    if "plane_score" in steps.columns and "shuffled_plane_score" in steps.columns:
        plane_scores = steps["plane_score"].to_numpy(np.float32)
        shuffled_scores = steps["shuffled_plane_score"].to_numpy(np.float32)
        gap = plane_scores - shuffled_scores
        finite_gap = gap[np.isfinite(gap)]
        if finite_gap.size:
            fig, ax = plt.subplots(figsize=(9, 4), dpi=140)
            ax.hist(finite_gap, bins=60, color="#C44E52", alpha=0.85, edgecolor="white")
            ax.axvline(0, color="#222222", linewidth=1.5, linestyle="--")
            ax.set_title("Plane score - shuffled_plane score at true bin", fontsize=9)
            ax.set_xlabel("score gap (positive = plane is better)", fontsize=8)
            ax.set_ylabel("hidden steps", fontsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            fig.tight_layout()
            fig.savefig(figures_dir / "plane_vs_shuffled_gap.png", bbox_inches="tight")
            plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _write_report(
    output_dir: Path,
    metrics: dict,
    cfg: PlaneCoordAuditConfig,
) -> None:
    summary = metrics.get("summary", {})
    by_variant = metrics.get("by_variant", {})

    verdict = summary.get("verdict", "UNKNOWN")
    gap_top10 = summary.get("normal_vs_shuffled_top10_gap", float("nan"))
    rmse_gain = summary.get("normal_vs_shuffled_oracle_rmse_gap", float("nan"))
    plane_top10 = summary.get("plane_top10_rate", float("nan"))
    shuffled_top10 = summary.get("shuffled_plane_top10_rate", float("nan"))
    raw_top10 = summary.get("raw_top10_rate", float("nan"))
    plane_rmse = summary.get("plane_top10_oracle_rmse_ft", float("nan"))
    shuffled_rmse = summary.get("shuffled_plane_top10_oracle_rmse_ft", float("nan"))

    def _fmt(v):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "n/a"
        return f"{v:.4f}"

    lines = [
        "# PLANE_COORDINATE_AUDIT_V0",
        "",
        "## Question",
        "Does replacing `TVT` with `plane_coord = TVT + Z` as the typewell alignment",
        "coordinate provide stronger GR signal than raw TVT?",
        "",
        "## Config",
        "```json",
        json.dumps(_json_clean(asdict(cfg)), indent=2),
        "```",
        "",
        f"## VERDICT: {verdict}",
        "",
        "| Metric | Value | Threshold |",
        "|--------|-------|-----------|",
        f"| plane top-10 rate | {_fmt(plane_top10)} | — |",
        f"| shuffled_plane top-10 rate | {_fmt(shuffled_top10)} | — |",
        f"| **normal_vs_shuffled_top10_gap** | **{_fmt(gap_top10)}** | >= 0.05 (GO) |",
        f"| raw top-10 rate | {_fmt(raw_top10)} | — |",
        f"| plane oracle RMSE (ft) | {_fmt(plane_rmse)} | — |",
        f"| shuffled_plane oracle RMSE (ft) | {_fmt(shuffled_rmse)} | — |",
        f"| **oracle_rmse_improvement (ft)** | **{_fmt(rmse_gain)}** | >= 0.30 ft (GO) |",
        "",
        "## All-variant top-10 rates",
        "",
    ]

    for vname in _VARIANTS:
        vm = by_variant.get(vname, {})
        r10 = vm.get("true_top10_rate", float("nan"))
        rmse = vm.get("top10_oracle_rmse_ft", float("nan"))
        rk_mean = vm.get("true_rank_mean", float("nan"))
        lines.append(
            f"- **{vname}**: top10={_fmt(r10)}, "
            f"oracle_rmse={_fmt(rmse)} ft, "
            f"rank_mean={_fmt(rk_mean)}"
        )

    lines += [
        "",
        f"## Totals: {metrics.get('total_wells', 0)} wells, "
        f"{metrics.get('total_steps', 0)} hidden steps",
        "",
        "## Figures",
        "- [Rank histograms](figures/rank_histograms.png)",
        "- [Top-10 rate by variant](figures/top10_rate_by_variant.png)",
        "- [Oracle RMSE by variant](figures/oracle_rmse_by_variant.png)",
        "- [Plane vs shuffled gap distribution](figures/plane_vs_shuffled_gap.png)",
        "- Per-well panel figures: `figures/panel_<well_id>.png`",
        "",
        "## Interpretation",
        "GO means the plane formulation (TVT+Z) provides a measurably stronger",
        "GR alignment signal than shuffled GR, justifying building downstream models",
        "on top of this coordinate. NO-GO means the signal is too weak to justify DL.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def _load_tail_classes(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=["well_id", "tail_class"])
    frame = pd.read_csv(path)
    if "well_id" not in frame.columns or "tail_class" not in frame.columns:
        return pd.DataFrame(columns=["well_id", "tail_class"])
    return frame[["well_id", "tail_class"]].drop_duplicates("well_id")


def run_plane_coordinate_audit(
    *,
    data_dir: Path,
    output_dir: Path,
    rows_per_step: int = 32,
    vertical_step_ft: float = 5.0,
    k_wells: int = -1,
    seed: int = 42,
    tail_classes_path: Path | None = None,
    n_panel_wells: int = 20,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = PlaneCoordAuditConfig(
        rows_per_step=rows_per_step,
        vertical_step_ft=vertical_step_ft,
        k_wells=k_wells,
        seed=seed,
        n_panel_wells=n_panel_wells,
    )

    wells = discover_wells(DataConfig(data_dir=data_dir, k_wells=k_wells))
    print(f"[plane-coord-audit] Processing {len(wells)} wells ...", flush=True)

    rng = np.random.default_rng(seed)
    all_step_frames: list[pd.DataFrame] = []
    # For panel figures: store (well_id, horizontal_df, typewell_df)
    well_cache: list[tuple[str, pd.DataFrame, pd.DataFrame, int]] = []

    for idx, well in enumerate(wells):
        if idx == 0 or (idx + 1) % 100 == 0 or idx + 1 == len(wells):
            print(f"[plane-coord-audit] well {idx + 1}/{len(wells)}: {well.well_id}", flush=True)
        well_rng = np.random.default_rng(seed + idx * 31337)
        horizontal, typewell = load_well(well)
        steps = _process_well(well.well_id, horizontal, typewell, cfg, well_rng)
        if not steps.empty:
            all_step_frames.append(steps)
            well_cache.append((well.well_id, horizontal, typewell, len(steps)))

    # Combine all steps
    if all_step_frames:
        all_steps = pd.concat(all_step_frames, ignore_index=True)
    else:
        all_steps = pd.DataFrame()

    print(
        f"[plane-coord-audit] Total steps: {len(all_steps)}, "
        f"total wells with steps: {len(all_step_frames)}",
        flush=True,
    )

    # Save step_scores parquet
    if not all_steps.empty:
        all_steps.to_parquet(output_dir / "step_scores.parquet", index=False)
    else:
        pd.DataFrame().to_parquet(output_dir / "step_scores.parquet", index=False)

    # Compute metrics
    tail_classes = _load_tail_classes(tail_classes_path)
    metrics = _compute_all_metrics(all_steps.copy(), tail_classes)

    # Save metrics JSON
    (output_dir / "metrics.json").write_text(
        json.dumps(_json_clean(metrics), indent=2) + "\n"
    )

    # Panel figures: top n_panel_wells by number of hidden steps
    panel_data = sorted(well_cache, key=lambda x: x[3], reverse=True)[:n_panel_wells]
    panel_wells = [(w, h, t) for w, h, t, _ in panel_data]
    _write_panel_figures(output_dir, panel_wells, cfg, seed=seed)

    # Aggregate figures
    _write_aggregate_figures(output_dir, all_steps)

    # Report
    _write_report(output_dir, _json_clean(metrics), cfg)

    # Print summary
    summary = metrics.get("summary", {})
    print("\n=== PLANE-COORDINATE AUDIT RESULTS ===", flush=True)
    print(json.dumps(_json_clean(summary), indent=2), flush=True)
    print(f"\nVERDICT: {summary.get('verdict', 'UNKNOWN')}", flush=True)
    print(f"Reports saved to: {output_dir}", flush=True)

    return metrics


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Experiment 0: Plane-Coordinate Heatmap Audit (TVT+Z hypothesis)"
    )
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/plane_coordinate_audit_v0"),
    )
    p.add_argument("--rows-per-step", type=int, default=32)
    p.add_argument("--vertical-step-ft", type=float, default=5.0)
    p.add_argument("--k-wells", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--tail-classes-path",
        type=Path,
        default=None,
        help="Optional path to well tail audit CSV with well_id and tail_class columns",
    )
    p.add_argument("--n-panel-wells", type=int, default=20)
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    run_plane_coordinate_audit(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        rows_per_step=args.rows_per_step,
        vertical_step_ft=args.vertical_step_ft,
        k_wells=args.k_wells,
        seed=args.seed,
        tail_classes_path=args.tail_classes_path,
        n_panel_wells=args.n_panel_wells,
    )
