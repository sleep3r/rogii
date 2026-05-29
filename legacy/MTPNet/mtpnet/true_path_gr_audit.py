from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import DataConfig
from .correlation_panel import _json_clean
from .io import discover_wells, load_well
from .typewell_mismatch import (
    TypewellMismatchConfig,
    WellTypewellMismatchResult,
    build_well_typewell_mismatch,
)


def _safe_mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def _safe_rate(mask: Iterable[bool]) -> float:
    arr = np.asarray(list(mask), dtype=bool)
    return float(arr.mean()) if arr.size else float("nan")


def _rmse(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    return float(np.sqrt(np.mean(arr**2))) if arr.size else float("nan")


def _metrics_from_steps(steps: pd.DataFrame) -> dict[str, float | int]:
    if steps.empty:
        return {"num_steps": 0, "num_wells": 0}
    ranks = steps["true_rank"].to_numpy(np.float32)
    gaps = steps["score_gap_vs_shuffled"].to_numpy(np.float32)
    top1_err = steps["top1_tvt"].to_numpy(np.float32) - steps["true_tvt"].to_numpy(np.float32)
    return {
        "num_steps": int(len(steps)),
        "num_wells": int(steps["well_id"].nunique()),
        "true_score_mean": _safe_mean(steps["true_score"]),
        "shuffled_true_score_mean": _safe_mean(steps["shuffled_true_score"]),
        "score_gap_vs_shuffled_mean": _safe_mean(gaps),
        "true_score_better_than_shuffled_rate": _safe_rate(gaps > 0),
        "score_percentile_mean": _safe_mean(steps["score_percentile"]),
        "true_rank_mean": _safe_mean(ranks),
        "true_top1_rate": _safe_rate(ranks <= 1),
        "true_top3_rate": _safe_rate(ranks <= 3),
        "true_top10_rate": _safe_rate(ranks <= 10),
        "corr_top1_rmse_ft": _rmse(top1_err),
        "corr_top10_oracle_rmse_ft": _rmse(np.sqrt(steps["top10_oracle_sqerr"].to_numpy(np.float32))),
        "gr_finite_mean": _safe_mean(steps["gr_finite_frac"]),
    }


def _load_tail_classes(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=["well_id", "tail_class"])
    frame = pd.read_csv(path)
    if "well_id" not in frame.columns or "tail_class" not in frame.columns:
        return pd.DataFrame(columns=["well_id", "tail_class"])
    return frame[["well_id", "tail_class"]].drop_duplicates("well_id")


def _write_figures(output_dir: Path, steps: pd.DataFrame) -> None:
    if steps.empty:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    gaps = steps["score_gap_vs_shuffled"].to_numpy(np.float32)
    gaps = gaps[np.isfinite(gaps)]
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    ax.hist(gaps, bins=60, color="#4C72B0", edgecolor="white", alpha=0.85)
    ax.axvline(0.0, color="#C44E52", linewidth=1.5, linestyle="--")
    ax.set_title("True-path GR score minus shuffled-GR score")
    ax.set_xlabel("score gap at true TVT bin")
    ax.set_ylabel("hidden compressed steps")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(figures_dir / "true_path_gap_distribution.png", bbox_inches="tight")
    plt.close(fig)

    ranks = steps["true_rank"].to_numpy(np.float32)
    ranks = ranks[np.isfinite(ranks)]
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    ax.hist(np.clip(ranks, 1, 100), bins=np.arange(1, 102), color="#55A868", alpha=0.85)
    ax.set_title("Rank of the true TVT bin under GR/typewell score")
    ax.set_xlabel("rank, clipped at 100")
    ax.set_ylabel("hidden compressed steps")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(figures_dir / "true_rank_histogram.png", bbox_inches="tight")
    plt.close(fig)

    zone = (
        steps.groupby("true_zone", dropna=False)
        .agg(
            score_percentile_mean=("score_percentile", "mean"),
            top10_rate=("true_rank", lambda values: float(np.mean(np.asarray(values) <= 10))),
            n=("well_id", "size"),
        )
        .reset_index()
        .sort_values("score_percentile_mean")
    )
    if not zone.empty:
        fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(zone))), dpi=150)
        ax.barh(zone["true_zone"].astype(str), zone["score_percentile_mean"], color="#8172B3")
        ax.set_xlim(0, 1)
        ax.set_title("True-path score percentile by geology zone")
        ax.set_xlabel("mean percentile of true TVT bin score")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(figures_dir / "true_path_percentile_by_zone.png", bbox_inches="tight")
        plt.close(fig)

    example_well = (
        steps.assign(abs_gap=np.abs(steps["score_gap_vs_shuffled"]))
        .sort_values("abs_gap", ascending=False)
        .iloc[0]["well_id"]
    )
    example = steps[steps["well_id"] == example_well].sort_values("step")
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, dpi=150)
    axes[0].plot(example["step"], example["true_score"], label="true path score", color="#4C72B0")
    axes[0].plot(
        example["step"],
        example["shuffled_true_score"],
        label="shuffled score at true path",
        color="#C44E52",
        alpha=0.8,
    )
    axes[0].set_ylabel("score")
    axes[0].legend(loc="best")
    axes[0].set_title(f"True-path score trace: {example_well}")
    axes[1].plot(example["step"], example["true_rank"], color="#55A868")
    axes[1].axhline(10, color="#222222", linestyle="--", linewidth=1)
    axes[1].invert_yaxis()
    axes[1].set_ylabel("true bin rank")
    axes[1].set_xlabel("compressed hidden step")
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(figures_dir / "example_true_path_trace.png", bbox_inches="tight")
    plt.close(fig)


def _write_report(output_dir: Path, metrics: dict[str, object], cfg: TypewellMismatchConfig) -> None:
    agg = metrics.get("aggregate", {})
    if not isinstance(agg, dict):
        agg = {}
    lines = [
        "# TRUE_PATH_GR_MATCH_REPORT",
        "",
        "## Question",
        "At the known true TVT path on train wells, does lateral GR match the provided typewell GR better than shuffled lateral GR?",
        "",
        "## Config",
        "```json",
        json.dumps(_json_clean(asdict(cfg)), indent=2),
        "```",
        "",
        "## Aggregate",
        f"- wells: {agg.get('num_wells')}",
        f"- hidden steps: {agg.get('num_steps')}",
        f"- true_score_mean: {agg.get('true_score_mean')}",
        f"- shuffled_true_score_mean: {agg.get('shuffled_true_score_mean')}",
        f"- score_gap_vs_shuffled_mean: {agg.get('score_gap_vs_shuffled_mean')}",
        f"- true_score_better_than_shuffled_rate: {agg.get('true_score_better_than_shuffled_rate')}",
        f"- score_percentile_mean: {agg.get('score_percentile_mean')}",
        f"- true_top10_rate: {agg.get('true_top10_rate')}",
        f"- corr_top10_oracle_rmse_ft: {agg.get('corr_top10_oracle_rmse_ft')}",
        "",
        "## Figures",
        "- [True path gap distribution](figures/true_path_gap_distribution.png)",
        "- [True rank histogram](figures/true_rank_histogram.png)",
        "- [True path percentile by zone](figures/true_path_percentile_by_zone.png)",
        "- [Example true path trace](figures/example_true_path_trace.png)",
        "",
        "## Interpretation",
        "If the true path is not consistently above shuffled GR, then raw GR/typewell log matching is not a reliable standalone signal on these real wells.",
    ]
    (output_dir / "true_path_gr_report.md").write_text("\n".join(lines) + "\n")


def run_true_path_gr_audit(
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
    tail_classes_path: Path | None = None,
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
            print(f"[true-path-gr] well {idx + 1}/{len(wells)}", flush=True)
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
    steps = (
        pd.concat([result.steps for result in results if not result.steps.empty], ignore_index=True)
        if any(not result.steps.empty for result in results)
        else pd.DataFrame()
    )
    tail_classes = _load_tail_classes(tail_classes_path)
    if not tail_classes.empty:
        by_well = by_well.merge(tail_classes, on="well_id", how="left")
        steps = steps.merge(tail_classes, on="well_id", how="left") if not steps.empty else steps

    by_well.to_csv(output_dir / "true_path_gr_by_well.csv", index=False)
    steps.to_parquet(output_dir / "true_path_gr_steps.parquet", index=False)

    metrics: dict[str, object] = {
        "aggregate": _metrics_from_steps(steps),
        "by_zone": {
            str(zone): _metrics_from_steps(group)
            for zone, group in steps.groupby("true_zone", dropna=False)
        }
        if not steps.empty and "true_zone" in steps.columns
        else {},
        "by_tail_class": {
            str(tail_class): _metrics_from_steps(group)
            for tail_class, group in steps.groupby("tail_class", dropna=False)
        }
        if not steps.empty and "tail_class" in steps.columns
        else {},
        "config": asdict(cfg),
    }
    (output_dir / "true_path_gr_metrics.json").write_text(
        json.dumps(_json_clean(metrics), indent=2) + "\n"
    )
    _write_figures(output_dir, steps)
    _write_report(output_dir, _json_clean(metrics), cfg)
    print(json.dumps(_json_clean(metrics), indent=2), flush=True)
    return metrics
