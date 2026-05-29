from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DataConfig
from .correlation_panel import _compress_nanmean, _regular_typewell_grid
from .heatmap import fill_nan
from .io import discover_wells, load_well
from .soft_segment import _bridge_tvt, _zscore


@dataclass(frozen=True)
class LocationAwareCorrConfig:
    data_dir: Path = Path("data")
    output_dir: Path = Path("artifacts/location_aware_corr_v0")
    rows_per_step: int = 32
    vertical_step_ft: float = 5.0
    location_sigma_ft: float = 80.0
    gr_weight: float = 1.0
    location_weight: float = 1.0
    k_wells: int = -1
    seed: int = 42
    topk: tuple[int, ...] = (1, 3, 10)
    n_panel_wells: int = 12


_VARIANTS = (
    "value_only",
    "location_only",
    "value_location_raw",
    "value_location_plane",
    "shuffled_value_location_raw",
    "shuffled_value_location_plane",
    "zero_value_location_raw",
)


def location_aware_score(
    *,
    typewell_gr: np.ndarray,
    horizontal_gr: float,
    tvt_grid: np.ndarray,
    query_tvt: float,
    location_sigma_ft: float,
    gr_weight: float = 1.0,
    location_weight: float = 1.0,
) -> np.ndarray:
    """Score candidate typewell bins by GR similarity plus location distance.

    This is the forum comment made explicit: match a vector of
    ``[GR value, location value]`` rather than GR alone.
    """
    tw = np.asarray(typewell_gr, dtype=np.float32)
    grid = np.asarray(tvt_grid, dtype=np.float32)
    if tw.size == 0 or grid.size != tw.size or not np.isfinite(query_tvt):
        return np.full(tw.size, -np.inf, dtype=np.float32)
    if np.isfinite(horizontal_gr):
        gr_term = np.abs(tw - float(horizontal_gr))
    else:
        gr_term = np.zeros_like(tw, dtype=np.float32)
    loc_sigma = max(float(location_sigma_ft), 1.0e-6)
    loc_term = np.abs(grid - float(query_tvt)) / loc_sigma
    score = -(float(gr_weight) * gr_term + float(location_weight) * loc_term)
    return score.astype(np.float32)


def _shuffle_finite(values: np.ndarray, finite_mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.asarray(values, dtype=np.float32).copy()
    idx = np.flatnonzero(np.asarray(finite_mask) > 0.5)
    if idx.size > 1:
        out[idx] = rng.permutation(out[idx])
    return out


def _topk_metrics(records: pd.DataFrame, topk: tuple[int, ...]) -> dict[str, float]:
    metrics: dict[str, float] = {
        "num_steps": float(len(records)),
    }
    if records.empty:
        for k in topk:
            metrics[f"top{k}_rate"] = float("nan")
            metrics[f"top{k}_oracle_rmse_ft"] = float("nan")
        metrics["top1_rmse_ft"] = float("nan")
        return metrics
    top1_sq = (records["top1_tvt"].to_numpy() - records["true_tvt"].to_numpy()) ** 2
    metrics["top1_rmse_ft"] = float(np.sqrt(np.mean(top1_sq)))
    for k in topk:
        metrics[f"top{k}_rate"] = float((records["true_rank"] <= k).mean())
        col = f"top{k}_sqerr"
        metrics[f"top{k}_oracle_rmse_ft"] = float(np.sqrt(records[col].mean()))
    return metrics


def _oracle_sqerr(scores: np.ndarray, tvt_grid: np.ndarray, true_tvt: float, k: int) -> float:
    finite = np.isfinite(scores)
    if not finite.any():
        return float("nan")
    k = min(int(k), int(finite.sum()))
    finite_idx = np.flatnonzero(finite)
    local = scores[finite_idx]
    top_idx = finite_idx[np.argpartition(-local, kth=k - 1)[:k]]
    return float(np.min((tvt_grid[top_idx] - true_tvt) ** 2))


def _rank(scores: np.ndarray, true_bin: int) -> int:
    order = np.argsort(-scores)
    found = np.flatnonzero(order == int(true_bin))
    return int(found[0]) + 1 if found.size else int(scores.size)


def _process_well(
    well_id: str,
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    cfg: LocationAwareCorrConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    tvt = pd.to_numeric(horizontal["TVT"], errors="coerce").to_numpy(dtype=np.float32)
    tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(dtype=np.float32)
    gr_raw = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(dtype=np.float32)
    gr_filled, gr_finite = fill_nan(gr_raw)
    z_raw = (
        pd.to_numeric(horizontal["Z"], errors="coerce").to_numpy(dtype=np.float32)
        if "Z" in horizontal.columns
        else np.zeros_like(tvt, dtype=np.float32)
    )
    z_filled, _ = fill_nan(z_raw)
    md = (
        pd.to_numeric(horizontal["MD"], errors="coerce").to_numpy(dtype=np.float32)
        if "MD" in horizontal.columns
        else np.arange(tvt.size, dtype=np.float32)
    )

    comp_tvt = _compress_nanmean(tvt, cfg.rows_per_step)
    comp_tvt_input = _compress_nanmean(tvt_input, cfg.rows_per_step)
    comp_gr = _compress_nanmean(gr_filled, cfg.rows_per_step)
    comp_gr_finite = _compress_nanmean(gr_finite, cfg.rows_per_step)
    comp_z = _compress_nanmean(z_filled, cfg.rows_per_step)
    comp_md = _compress_nanmean(md, cfg.rows_per_step)
    shuffled_gr = _compress_nanmean(
        _shuffle_finite(gr_filled, gr_finite, rng),
        cfg.rows_per_step,
    )
    n = min(
        comp_tvt.size,
        comp_tvt_input.size,
        comp_gr.size,
        comp_gr_finite.size,
        comp_z.size,
        comp_md.size,
        shuffled_gr.size,
    )
    if n == 0:
        return pd.DataFrame()
    comp_tvt = comp_tvt[:n]
    comp_tvt_input = comp_tvt_input[:n]
    comp_gr = comp_gr[:n]
    comp_z = comp_z[:n]
    comp_md = comp_md[:n]
    shuffled_gr = shuffled_gr[:n]

    tvt_grid, tw_gr = _regular_typewell_grid(typewell, cfg.vertical_step_ft)
    if tvt_grid.size == 0:
        return pd.DataFrame()

    tw_gr_z = _zscore(tw_gr)
    comp_gr_z = _zscore(comp_gr)
    shuffled_gr_z = _zscore(shuffled_gr)
    bridge = _bridge_tvt(comp_md, comp_tvt_input)
    plane_prior = bridge + comp_z

    valid_bridge = np.isfinite(bridge) & np.isfinite(comp_gr)
    if valid_bridge.sum() >= 2:
        order = np.argsort(bridge[valid_bridge])
        plane_gr = np.interp(
            plane_prior,
            bridge[valid_bridge][order],
            comp_gr[valid_bridge][order],
        ).astype(np.float32)
        shuffled_plane_gr = np.interp(
            plane_prior,
            bridge[valid_bridge][order],
            shuffled_gr[valid_bridge][order],
        ).astype(np.float32)
    else:
        plane_gr = comp_gr.copy()
        shuffled_plane_gr = shuffled_gr.copy()
    plane_gr_z = _zscore(plane_gr)
    shuffled_plane_gr_z = _zscore(shuffled_plane_gr)

    hidden_steps = np.flatnonzero(~np.isfinite(comp_tvt_input) & np.isfinite(comp_tvt))
    rows: list[dict] = []
    for step in hidden_steps:
        true_tvt = float(comp_tvt[step])
        true_bin = int(np.abs(tvt_grid - true_tvt).argmin())
        raw_value = location_aware_score(
            typewell_gr=tw_gr_z,
            horizontal_gr=float(comp_gr_z[step]),
            tvt_grid=tvt_grid,
            query_tvt=float(bridge[step]),
            location_sigma_ft=cfg.location_sigma_ft,
            gr_weight=cfg.gr_weight,
            location_weight=0.0,
        )
        location_only = location_aware_score(
            typewell_gr=tw_gr_z,
            horizontal_gr=float(comp_gr_z[step]),
            tvt_grid=tvt_grid,
            query_tvt=float(bridge[step]),
            location_sigma_ft=cfg.location_sigma_ft,
            gr_weight=0.0,
            location_weight=cfg.location_weight,
        )
        scores_by_variant = {
            "value_only": raw_value,
            "location_only": location_only,
            "value_location_raw": location_aware_score(
                typewell_gr=tw_gr_z,
                horizontal_gr=float(comp_gr_z[step]),
                tvt_grid=tvt_grid,
                query_tvt=float(bridge[step]),
                location_sigma_ft=cfg.location_sigma_ft,
                gr_weight=cfg.gr_weight,
                location_weight=cfg.location_weight,
            ),
            "value_location_plane": location_aware_score(
                typewell_gr=tw_gr_z,
                horizontal_gr=float(plane_gr_z[step]),
                tvt_grid=tvt_grid,
                query_tvt=float(plane_prior[step]),
                location_sigma_ft=cfg.location_sigma_ft,
                gr_weight=cfg.gr_weight,
                location_weight=cfg.location_weight,
            ),
            "shuffled_value_location_raw": location_aware_score(
                typewell_gr=tw_gr_z,
                horizontal_gr=float(shuffled_gr_z[step]),
                tvt_grid=tvt_grid,
                query_tvt=float(bridge[step]),
                location_sigma_ft=cfg.location_sigma_ft,
                gr_weight=cfg.gr_weight,
                location_weight=cfg.location_weight,
            ),
            "shuffled_value_location_plane": location_aware_score(
                typewell_gr=tw_gr_z,
                horizontal_gr=float(shuffled_plane_gr_z[step]),
                tvt_grid=tvt_grid,
                query_tvt=float(plane_prior[step]),
                location_sigma_ft=cfg.location_sigma_ft,
                gr_weight=cfg.gr_weight,
                location_weight=cfg.location_weight,
            ),
            "zero_value_location_raw": location_aware_score(
                typewell_gr=tw_gr_z,
                horizontal_gr=0.0,
                tvt_grid=tvt_grid,
                query_tvt=float(bridge[step]),
                location_sigma_ft=cfg.location_sigma_ft,
                gr_weight=cfg.gr_weight,
                location_weight=cfg.location_weight,
            ),
        }
        for variant, scores in scores_by_variant.items():
            top1_bin = int(np.nanargmax(scores))
            row = {
                "well_id": well_id,
                "step": int(step),
                "variant": variant,
                "true_tvt": true_tvt,
                "query_tvt": float(bridge[step]),
                "plane_query_tvt": float(plane_prior[step]),
                "true_bin": true_bin,
                "top1_bin": top1_bin,
                "top1_tvt": float(tvt_grid[top1_bin]),
                "true_rank": _rank(scores, true_bin),
            }
            for k in cfg.topk:
                row[f"top{k}_sqerr"] = _oracle_sqerr(scores, tvt_grid, true_tvt, k)
            rows.append(row)
    return pd.DataFrame(rows)


def _aggregate(steps: pd.DataFrame, cfg: LocationAwareCorrConfig) -> dict:
    reference_k = 10 if 10 in cfg.topk else max(cfg.topk)
    rate_key = f"top{reference_k}_rate"
    rmse_key = f"top{reference_k}_oracle_rmse_ft"
    metrics: dict[str, object] = {
        "num_wells": int(steps["well_id"].nunique()) if not steps.empty else 0,
        "num_steps": int(steps[["well_id", "step"]].drop_duplicates().shape[0])
        if not steps.empty
        else 0,
    }
    for variant in _VARIANTS:
        metrics[variant] = _topk_metrics(
            steps.loc[steps["variant"] == variant],
            cfg.topk,
        )
    if not steps.empty:
        metrics[f"normal_vs_shuffled_raw_top{reference_k}_rate_gap"] = (
            metrics["value_location_raw"][rate_key]
            - metrics["shuffled_value_location_raw"][rate_key]
            if rate_key in metrics["value_location_raw"]
            else float("nan")
        )
        metrics[f"normal_vs_shuffled_plane_top{reference_k}_rate_gap"] = (
            metrics["value_location_plane"][rate_key]
            - metrics["shuffled_value_location_plane"][rate_key]
            if rate_key in metrics["value_location_plane"]
            else float("nan")
        )
        metrics[f"raw_gain_over_location_only_top{reference_k}_rmse_ft"] = (
            metrics["location_only"][rmse_key]
            - metrics["value_location_raw"][rmse_key]
        )
        metrics[f"plane_gain_over_location_only_top{reference_k}_rmse_ft"] = (
            metrics["location_only"][rmse_key]
            - metrics["value_location_plane"][rmse_key]
        )
    return metrics


def _write_report(output_dir: Path, cfg: LocationAwareCorrConfig, metrics: dict) -> None:
    lines = [
        "# Location-Aware GR Correlation Audit",
        "",
        "Tests the forum suggestion: match GR value plus location/distance, not GR alone.",
        "",
        "Important control: `location_only` is reported separately so any gain from the bridge/coordinate prior is not mistaken for GR signal.",
        "",
        "## Config",
        "",
        f"- rows_per_step: `{cfg.rows_per_step}`",
        f"- vertical_step_ft: `{cfg.vertical_step_ft}`",
        f"- location_sigma_ft: `{cfg.location_sigma_ft}`",
        f"- gr_weight: `{cfg.gr_weight}`",
        f"- location_weight: `{cfg.location_weight}`",
        "",
        "## Summary",
        "",
        f"- num_wells: `{metrics.get('num_wells')}`",
        f"- num_steps: `{metrics.get('num_steps')}`",
        f"- raw normal-vs-shuffled top10 gap: `{metrics.get('normal_vs_shuffled_raw_top10_rate_gap', float('nan')):.6g}`",
        f"- plane normal-vs-shuffled top10 gap: `{metrics.get('normal_vs_shuffled_plane_top10_rate_gap', float('nan')):.6g}`",
        f"- raw gain over location-only top10 RMSE ft: `{metrics.get('raw_gain_over_location_only_top10_rmse_ft', float('nan')):.6g}`",
        f"- plane gain over location-only top10 RMSE ft: `{metrics.get('plane_gain_over_location_only_top10_rmse_ft', float('nan')):.6g}`",
        "",
        "## Variants",
        "",
    ]
    for variant in _VARIANTS:
        m = metrics.get(variant, {})
        lines.append(
            f"- `{variant}`: top1_rmse=`{m.get('top1_rmse_ft', float('nan')):.6g}`, "
            f"top10_rate=`{m.get('top10_rate', float('nan')):.6g}`, "
            f"top10_oracle_rmse=`{m.get('top10_oracle_rmse_ft', float('nan')):.6g}`"
        )
    (output_dir / "location_aware_corr_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _write_example_figure(steps: pd.DataFrame, output_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    if steps.empty:
        return
    well_id = str(steps["well_id"].iloc[0])
    panel = steps[
        (steps["well_id"] == well_id) & (steps["variant"] == "value_location_plane")
    ].copy()
    if panel.empty:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    axes[0].plot(panel["step"], panel["true_tvt"], color="black", linewidth=2, label="true TVT")
    axes[0].plot(panel["step"], panel["top1_tvt"], color="tab:orange", linewidth=1.5, label="location-aware top1")
    axes[0].plot(panel["step"], panel["plane_query_tvt"], color="tab:blue", linewidth=1.0, label="plane query prior")
    axes[0].invert_yaxis()
    axes[0].set_title(f"{well_id}: location-aware GR correlation")
    axes[0].set_ylabel("TVT ft")
    axes[0].legend(loc="best")
    axes[1].plot(panel["step"], panel["true_rank"], color="tab:green", linewidth=1.5)
    axes[1].axhline(10, color="black", linestyle="--", linewidth=1)
    axes[1].set_ylabel("true rank")
    axes[1].set_xlabel("compressed hidden step")
    axes[1].set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_location_aware_corr(cfg: LocationAwareCorrConfig) -> dict:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    wells = discover_wells(DataConfig(data_dir=cfg.data_dir, k_wells=cfg.k_wells))
    rng = np.random.default_rng(cfg.seed)
    frames: list[pd.DataFrame] = []
    for well in wells:
        horizontal, typewell = load_well(well)
        frame = _process_well(well.well_id, horizontal, typewell, cfg, rng)
        if not frame.empty:
            frames.append(frame)
    steps = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    metrics = _aggregate(steps, cfg)
    (cfg.output_dir / "location_aware_corr_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    steps.to_parquet(cfg.output_dir / "location_aware_corr_steps.parquet", index=False)
    _write_report(cfg.output_dir, cfg, metrics)
    _write_example_figure(
        steps,
        cfg.output_dir / "figures" / "example_location_aware_heatmap.png",
    )
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run location-aware GR correlation audit")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/location_aware_corr_v0"))
    parser.add_argument("--rows-per-step", type=int, default=32)
    parser.add_argument("--vertical-step-ft", type=float, default=5.0)
    parser.add_argument("--location-sigma-ft", type=float, default=80.0)
    parser.add_argument("--gr-weight", type=float, default=1.0)
    parser.add_argument("--location-weight", type=float, default=1.0)
    parser.add_argument("--k-wells", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    run_location_aware_corr(
        LocationAwareCorrConfig(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            rows_per_step=args.rows_per_step,
            vertical_step_ft=args.vertical_step_ft,
            location_sigma_ft=args.location_sigma_ft,
            gr_weight=args.gr_weight,
            location_weight=args.location_weight,
            k_wells=args.k_wells,
            seed=args.seed,
        )
    )


if __name__ == "__main__":
    main()
