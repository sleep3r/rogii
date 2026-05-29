from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUNS = {
    "v1 prior-conditioned": "mtp_v1_prior_conditioned",
    "v2 anchor dropout": "mtp_v2_anchor_dropout",
    "v2 selector loss": "mtp_v2_train_time_selection",
    "v3 GR-forced": "mtp_v3_gr_forced",
    "v4 sim2real corr": "mtp_v4_sim2real",
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "n/a"
        return f"{float(value):.{digits}f}"
    return str(value)


def _metric(data: dict[str, Any], *keys: str) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _ensure_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )


def _bar_label(ax, bars, fmt: str = "{:.2f}") -> None:
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width,
            bar.get_y() + bar.get_height() / 2,
            " " + fmt.format(width),
            va="center",
            ha="left",
            fontsize=8,
        )


def _save_window_metrics(artifacts_dir: Path, figures_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, run_dir in RUNS.items():
        metrics = _read_json(artifacts_dir / run_dir / "metrics.json")
        valid = metrics.get("valid_base_center_all_hidden") or metrics.get("valid") or {}
        if not valid:
            continue
        rows.append(
            {
                "run": label,
                "top1": valid.get("top1_rmse_ft"),
                "weighted": valid.get("weighted_mean_rmse_ft"),
                "oracle_topK": valid.get("oracle_topk_rmse_ft"),
                "best_mode_top3": valid.get("best_mode_top3_rate"),
                "spearman": valid.get("logit_error_spearman"),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df))
    width = 0.26
    ax.bar(x - width, df["top1"], width, label="top1", color="#4C72B0")
    ax.bar(x, df["weighted"], width, label="weighted", color="#55A868")
    ax.bar(x + width, df["oracle_topK"], width, label="oracle top-K", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(df["run"], rotation=18, ha="right")
    ax.set_ylabel("Window RMSE, ft")
    ax.set_title("Window-Level MTP: Mode Space Is Strong, Deployable Path Is Modest")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(figures_dir / "window_metrics.png", bbox_inches="tight")
    plt.close(fig)
    return df


def _save_sanity_gaps(artifacts_dir: Path, figures_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, run_dir in {
        "v3 GR-forced": "mtp_v3_gr_forced",
        "v4 sim2real corr": "mtp_v4_sim2real",
    }.items():
        metrics = _read_json(artifacts_dir / run_dir / "metrics.json")
        normal = _metric(metrics, "valid_base_center_all_hidden", "top1_rmse_ft")
        sanity = metrics.get("sanity", {})
        for variant in ["no_gr", "shuffled_gr", "no_all_priors", "no_history"]:
            value = _metric(sanity, variant, "top1_rmse_ft")
            if normal is not None and value is not None:
                rows.append({"run": label, "variant": variant, "gap_ft": value - normal})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    pivot = df.pivot(index="variant", columns="run", values="gap_ft").fillna(0.0)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    pivot.plot(kind="barh", ax=ax, color=["#4C72B0", "#DD8452"])
    ax.axvline(0.0, color="#333333", linewidth=1)
    ax.set_xlabel("RMSE increase vs normal, ft")
    ax.set_title("Sanity Gaps: GR Ablations Barely Hurt")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(figures_dir / "sanity_gaps.png", bbox_inches="tight")
    plt.close(fig)
    return df


def _best_candidate_from_track(metrics: dict[str, Any]) -> dict[str, Any] | None:
    candidates = metrics.get("candidates")
    if isinstance(candidates, list) and candidates:
        return min(candidates, key=lambda row: row.get("rmse", float("inf")))
    tracker = metrics.get("tracker", {})
    nn_best = tracker.get("nn_best")
    beta = tracker.get("beta_summaries", {})
    bests = [v.get("best") for v in beta.values() if isinstance(v, dict)]
    if isinstance(nn_best, dict):
        bests.append(nn_best)
    bests = [b for b in bests if isinstance(b, dict)]
    if bests:
        return min(bests, key=lambda row: row.get("rmse", float("inf")))
    return None


def _save_row_progress(artifacts_dir: Path, figures_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    track_runs = {
        "v1 tracker/ranker": "mtp_v1_prior_conditioned",
        "v2 anchor dropout": "mtp_v2_anchor_dropout",
        "v2 selector loss": "mtp_v2_train_time_selection",
        "v3 GR-forced": "mtp_v3_gr_forced",
        "v4 NN/corr": "mtp_v4_sim2real",
    }
    for label, run_dir in track_runs.items():
        metrics = _read_json(artifacts_dir / run_dir / "track_metrics.json")
        best = _best_candidate_from_track(metrics)
        baseline = _metric(metrics, "baselines", "b2_guarded_submit", "rmse")
        if best:
            rows.append(
                {
                    "run": label,
                    "rmse": best.get("rmse"),
                    "gain_vs_legacy_anchor": (
                        baseline - best.get("rmse") if baseline is not None else np.nan
                    ),
                    "candidate": best.get("candidate"),
                }
            )
    for label, run_dir in {
        "v2 OOF2": "mtp_v2_train_time_selection_oof2",
        "v3 OOF2": "mtp_v3_gr_forced_oof2",
    }.items():
        metrics = _read_json(artifacts_dir / run_dir / "oof_metrics.json")
        best = _metric(metrics, "aggregate", "best_candidate")
        gain = _metric(metrics, "aggregate", "gain_vs_b2")
        if best:
            rows.append(
                {
                    "run": label,
                    "rmse": best.get("rmse"),
                    "gain_vs_legacy_anchor": gain,
                    "candidate": best.get("candidate"),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    plot = df.dropna(subset=["gain_vs_legacy_anchor"]).copy()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.barh(plot["run"], plot["gain_vs_legacy_anchor"], color="#4C72B0")
    _bar_label(ax, bars, "{:.3f}")
    ax.axvline(0.10, color="#C44E52", linestyle="--", linewidth=1, label="+0.10 ft target")
    ax.set_xlabel("Row-level gain vs legacy anchor, ft")
    ax.set_title("Row-Level Gains Stay Small After Leakage Is Removed")
    ax.legend(loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(figures_dir / "row_level_progress.png", bbox_inches="tight")
    plt.close(fig)
    return df


def _save_corr_panel(artifacts_dir: Path, figures_dir: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    metrics = _read_json(artifacts_dir / "corr_panel_v0" / "correlation_panel_metrics.json")
    localized_metrics = _read_json(
        artifacts_dir / "corr_panel_localized_v0" / "correlation_panel_metrics.json"
    )
    stretch_metrics = _read_json(
        artifacts_dir / "corr_panel_stretch_v0" / "correlation_panel_metrics.json"
    )
    by_well_path = artifacts_dir / "corr_panel_v0" / "correlation_panel_by_well.csv"
    by_well = pd.read_csv(by_well_path) if by_well_path.exists() else pd.DataFrame()
    if metrics:
        normal = metrics.get("normal", {})
        shuffled = metrics.get("shuffled_gr", {})
        hit_rows = []
        for variant, src in [("normal", normal), ("shuffled", shuffled)]:
            hit_rows.extend(
                [
                    {"variant": variant, "k": "top1", "rate": src.get("corr_target_top1_rate", 0) * 100},
                    {"variant": variant, "k": "top3", "rate": src.get("corr_target_top3_rate", 0) * 100},
                    {"variant": variant, "k": "top10", "rate": src.get("corr_target_top10_rate", 0) * 100},
                ]
            )
        hit_df = pd.DataFrame(hit_rows)
        fig, ax = plt.subplots(figsize=(7, 4))
        pivot = hit_df.pivot(index="k", columns="variant", values="rate").loc[["top1", "top3", "top10"]]
        pivot.plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452"])
        ax.set_ylabel("True TVT in top-K, %")
        ax.set_title("Naive GR/Typewell Correlation Panel Does Not Recover True Path")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(figures_dir / "corr_panel_hit_rates.png", bbox_inches="tight")
        plt.close(fig)
    if metrics and localized_metrics:
        rows = []
        panel_series = [
            ("global", metrics),
            ("localized", localized_metrics),
        ]
        if stretch_metrics:
            panel_series.append(("localized+stretch", stretch_metrics))
        for panel_name, panel_metrics in panel_series:
            for variant in ["normal", "shuffled_gr"]:
                src = panel_metrics.get(variant, {})
                rows.append(
                    {
                        "panel": panel_name,
                        "variant": "shuffled" if variant == "shuffled_gr" else variant,
                        "top10_oracle": src.get("corr_top10_oracle_rmse_ft", np.nan),
                        "top10_hit": src.get("corr_target_top10_rate", np.nan) * 100,
                    }
                )
        compare = pd.DataFrame(rows)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        pivot_rmse = compare.pivot(index="panel", columns="variant", values="top10_oracle")
        pivot_hit = compare.pivot(index="panel", columns="variant", values="top10_hit")
        pivot_rmse.plot(kind="bar", ax=axes[0], color=["#4C72B0", "#DD8452"])
        axes[0].set_ylabel("Top-10 oracle RMSE, ft")
        axes[0].set_title("Localization Improves Geometry")
        pivot_hit.plot(kind="bar", ax=axes[1], color=["#4C72B0", "#DD8452"])
        axes[1].set_ylabel("True TVT in top-10, %")
        axes[1].set_title("But GR Score Still Does Not Beat Shuffled")
        for ax in axes:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(figures_dir / "corr_panel_global_vs_localized.png", bbox_inches="tight")
        plt.close(fig)
    if not by_well.empty:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist(by_well["corr_top10_oracle_rmse_ft"], bins=35, color="#4C72B0", edgecolor="white")
        axes[0].set_xlabel("Top-10 oracle RMSE, ft")
        axes[0].set_ylabel("Wells")
        axes[0].set_title("Top-10 Correlation Oracle By Well")
        axes[1].hist(by_well["corr_target_top10_rate"] * 100, bins=30, color="#DD8452", edgecolor="white")
        axes[1].set_xlabel("True path in top-10, % of hidden steps")
        axes[1].set_title("True Ridge Rarely Appears")
        for ax in axes:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(figures_dir / "corr_panel_distribution.png", bbox_inches="tight")
        plt.close(fig)
    if localized_metrics:
        metrics = {**metrics, "localized": localized_metrics}
    if stretch_metrics:
        metrics = {**metrics, "localized_stretch": stretch_metrics}
    return metrics, by_well


def _save_typewell_mismatch(
    artifacts_dir: Path, figures_dir: Path
) -> tuple[dict[str, Any], pd.DataFrame]:
    metrics = _read_json(
        artifacts_dir / "typewell_mismatch_v0" / "typewell_mismatch_metrics.json"
    )
    by_well_path = artifacts_dir / "typewell_mismatch_v0" / "typewell_mismatch_by_well.csv"
    by_well = pd.read_csv(by_well_path) if by_well_path.exists() else pd.DataFrame()
    if not by_well.empty:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist(
            by_well["true_score_better_than_shuffled_rate"] * 100,
            bins=30,
            color="#4C72B0",
            edgecolor="white",
        )
        axes[0].axvline(50, color="#C44E52", linestyle="--", linewidth=1)
        axes[0].set_xlabel("True-score > shuffled, % of hidden steps")
        axes[0].set_ylabel("Wells")
        axes[0].set_title("True Path Often Does Not Beat Shuffled GR")
        axes[1].hist(
            by_well["true_top10_rate"] * 100,
            bins=30,
            color="#DD8452",
            edgecolor="white",
        )
        axes[1].axvline(20, color="#C44E52", linestyle="--", linewidth=1)
        axes[1].set_xlabel("True path in top-10, % of hidden steps")
        axes[1].set_title("True Ridge Is Rarely High-Ranked")
        for ax in axes:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(figures_dir / "typewell_mismatch_distribution.png", bbox_inches="tight")
        plt.close(fig)

        plot = by_well.copy()
        fig, ax = plt.subplots(figsize=(7, 5))
        colors = np.where(plot["mismatch_flag"].to_numpy(np.float32) > 0.5, "#C44E52", "#55A868")
        ax.scatter(
            plot["score_gap_vs_shuffled_mean"],
            plot["true_top10_rate"] * 100,
            s=18,
            c=colors,
            alpha=0.65,
            linewidth=0,
        )
        ax.axvline(0, color="#333333", linewidth=1)
        ax.axhline(20, color="#C44E52", linestyle="--", linewidth=1)
        ax.set_xlabel("Mean true-score minus shuffled-score")
        ax.set_ylabel("True path in top-10, %")
        ax.set_title("Typewell Mismatch By Well")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(figures_dir / "typewell_mismatch_scatter.png", bbox_inches="tight")
        plt.close(fig)
    return metrics, by_well


def _save_formation_correlation(
    artifacts_dir: Path, figures_dir: Path
) -> tuple[dict[str, Any], pd.DataFrame]:
    metrics = _read_json(
        artifacts_dir / "formation_correlation_v0" / "formation_correlation_metrics.json"
    )
    by_well_path = artifacts_dir / "formation_correlation_v0" / "formation_correlation_by_well.csv"
    by_well = pd.read_csv(by_well_path) if by_well_path.exists() else pd.DataFrame()
    if metrics:
        rows = []
        labels = {
            "global": "global",
            "anchor_geology": "anchor geology",
            "oracle_geology": "oracle geology",
        }
        for variant in ["global", "anchor_geology", "oracle_geology"]:
            src = metrics.get(variant, {})
            rows.append(
                {
                    "variant": labels[variant],
                    "top1_rmse": src.get("corr_top1_rmse_ft", np.nan),
                    "top10_oracle": src.get("corr_top10_oracle_rmse_ft", np.nan),
                    "top10_hit": src.get("corr_target_top10_rate", np.nan) * 100,
                    "candidate_bins": src.get("candidate_bins_mean", np.nan),
                }
            )
        df = pd.DataFrame(rows)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        colors = ["#4C72B0", "#DD8452", "#55A868"]
        for ax, col, title, ylabel in [
            (axes[0], "top10_oracle", "Top-10 Oracle RMSE", "RMSE, ft"),
            (axes[1], "top10_hit", "True Path In Top-10", "% hidden steps"),
            (axes[2], "candidate_bins", "Candidate Bins After Mask", "bins"),
        ]:
            bars = ax.bar(df["variant"], df[col], color=colors)
            _bar_label(ax, bars, "{:.1f}")
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.tick_params(axis="x", rotation=15)
            for label in ax.get_xticklabels():
                label.set_ha("right")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(figures_dir / "formation_correlation_comparison.png", bbox_inches="tight")
        plt.close(fig)
    return metrics, by_well


def _save_pseudo_zone(artifacts_dir: Path, figures_dir: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    metrics = _read_json(artifacts_dir / "pseudo_zone_v0" / "pseudo_zone_metrics.json")
    by_well_path = artifacts_dir / "pseudo_zone_v0" / "pseudo_zone_by_well.csv"
    by_well = pd.read_csv(by_well_path) if by_well_path.exists() else pd.DataFrame()
    if not by_well.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(
            by_well["hidden_zone_match_rate"] * 100,
            bins=30,
            color="#4C72B0",
            edgecolor="white",
        )
        ax.axvline(50, color="#333333", linewidth=1)
        ax.axvline(
            by_well["hidden_zone_match_rate"].mean() * 100,
            color="#C44E52",
            linestyle="--",
            linewidth=1.5,
            label="mean",
        )
        ax.set_xlabel("Hidden zone match, %")
        ax.set_ylabel("Wells")
        ax.set_title("Test-Schema-Safe Pseudo-Zone Template")
        ax.legend()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(figures_dir / "pseudo_zone_distribution.png", bbox_inches="tight")
        plt.close(fig)
    return metrics, by_well


def _save_tail_audit(artifacts_dir: Path, figures_dir: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    metrics = _read_json(artifacts_dir / "tail_audit_v1" / "tail_audit_metrics.json")
    well_path = artifacts_dir / "tail_audit_v1" / "well_tail_audit.csv"
    wells = pd.read_csv(well_path) if well_path.exists() else pd.DataFrame()
    if not wells.empty and "tail_class" in wells.columns:
        counts = wells["tail_class"].value_counts().sort_values()
        fig, ax = plt.subplots(figsize=(8, 4.5))
        bars = ax.barh(counts.index, counts.values, color="#4C72B0")
        _bar_label(ax, bars, "{:.0f}")
        ax.set_xlabel("Wells")
        ax.set_title("Tail Classes: Candidate-Space Failures Are Common")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(figures_dir / "tail_class_counts.png", bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        col = "rmse_b2" if "rmse_b2" in wells.columns else "rmse_base_schema10"
        ordered = wells.groupby("tail_class")[col].median().sort_values()
        bars = ax.barh(ordered.index, ordered.values, color="#DD8452")
        _bar_label(ax, bars, "{:.1f}")
        ax.set_xlabel("Median well RMSE, ft")
        ax.set_title("Bad Wells Are Tail-Class Concentrated")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(figures_dir / "tail_class_rmse.png", bbox_inches="tight")
        plt.close(fig)
    return metrics, wells


def _save_tail_well_panels(
    artifacts_dir: Path,
    figures_dir: Path,
    tail_wells: pd.DataFrame,
    *,
    max_wells: int = 6,
) -> list[str]:
    panel_dir = figures_dir / "tail_well_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    steps_path = artifacts_dir / "corr_panel_localized_v0" / "correlation_panel_steps.parquet"
    if tail_wells.empty or not steps_path.exists():
        return []
    steps = pd.read_parquet(steps_path)
    if "variant" in steps.columns:
        steps = steps[steps["variant"] == "normal"].copy()
    if steps.empty:
        return []
    if "top_worst_flag" in tail_wells.columns:
        selected = tail_wells[tail_wells["top_worst_flag"].astype(bool)].copy()
    else:
        selected = tail_wells.copy()
    score_col = "rmse_b2" if "rmse_b2" in selected.columns else "rmse_base_schema10"
    selected = selected.sort_values(score_col, ascending=False).head(max_wells)
    outputs: list[str] = []
    for _, well in selected.iterrows():
        well_id = str(well["well_id"])
        frame = steps[steps["well_id"] == well_id].sort_values("step")
        if frame.empty:
            continue
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, height_ratios=[2, 1])
        axes[0].plot(frame["step"], frame["true_tvt"], label="true TVT", color="#222222", linewidth=2)
        axes[0].plot(frame["step"], frame["top1_tvt"], label="corr top1", color="#C44E52", linewidth=1.4)
        if "anchor_tvt" in frame.columns and frame["anchor_tvt"].notna().any():
            axes[0].plot(frame["step"], frame["anchor_tvt"], label="known-tail anchor", color="#4C72B0", linestyle="--")
        axes[0].invert_yaxis()
        axes[0].set_ylabel("TVT, ft")
        axes[0].set_title(
            f"{well_id}: localized correlation vs true path "
            f"({well.get('tail_class', 'unknown')})"
        )
        axes[0].legend(loc="best")
        rank = frame["true_rank"].clip(upper=120)
        axes[1].plot(frame["step"], rank, color="#DD8452", linewidth=1.2)
        axes[1].axhline(10, color="#55A868", linestyle="--", linewidth=1)
        axes[1].set_ylabel("true rank")
        axes[1].set_xlabel("compressed hidden step")
        axes[1].set_ylim(bottom=0)
        for ax in axes:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        fig.tight_layout()
        out = panel_dir / f"{well_id}.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        outputs.append(str(out.relative_to(figures_dir.parent)))
    return outputs


def _save_ranker_leakage(artifacts_dir: Path, figures_dir: Path) -> dict[str, Any]:
    audit = _read_json(artifacts_dir / "mtp_v1_prior_conditioned" / "track_split_audit.json")
    subsets = audit.get("subsets", {})
    rows: list[dict[str, Any]] = []
    for name, data in subsets.items():
        if isinstance(data, dict):
            ranker = data.get("ranker") or data.get("ranker_tracker") or {}
            nn = data.get("nn") or data.get("nn_tracker") or {}
            for label, src in [("NN logits", nn), ("ranker logits", ranker)]:
                gain = src.get("gain_vs_b2") or src.get("gain")
                if gain is not None:
                    rows.append({"subset": name, "source": label, "gain": gain})
    if rows:
        df = pd.DataFrame(rows)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        pivot = df.pivot(index="subset", columns="source", values="gain").fillna(0.0)
        pivot.plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452"])
        ax.set_ylabel("Gain vs legacy anchor, ft")
        ax.set_title("Ranker Gain Was Mostly In-Sample")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(figures_dir / "ranker_leakage.png", bbox_inches="tight")
        plt.close(fig)
    return audit


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "_No data._"
    table = df[columns].copy()
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for _, row in table.iterrows():
        rows.append("| " + " | ".join(_fmt(row[col]) for col in columns) + " |")
    return "\n".join([header, sep, *rows])


def write_diagnostic_report(
    *,
    artifacts_dir: Path = Path("artifacts"),
    output_dir: Path = Path("artifacts/diagnostic_report_v1"),
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    _ensure_style()

    window_df = _save_window_metrics(artifacts_dir, figures_dir)
    sanity_df = _save_sanity_gaps(artifacts_dir, figures_dir)
    row_df = _save_row_progress(artifacts_dir, figures_dir)
    corr_metrics, corr_by_well = _save_corr_panel(artifacts_dir, figures_dir)
    mismatch_metrics, mismatch_by_well = _save_typewell_mismatch(artifacts_dir, figures_dir)
    formation_metrics, formation_by_well = _save_formation_correlation(artifacts_dir, figures_dir)
    pseudo_zone_metrics, pseudo_zone_by_well = _save_pseudo_zone(artifacts_dir, figures_dir)
    tail_metrics, tail_wells = _save_tail_audit(artifacts_dir, figures_dir)
    tail_panel_paths = _save_tail_well_panels(artifacts_dir, figures_dir, tail_wells)
    split_audit = _save_ranker_leakage(artifacts_dir, figures_dir)
    v4 = _read_json(artifacts_dir / "mtp_v4_sim2real" / "metrics.json")
    v3_oof = _read_json(artifacts_dir / "mtp_v3_gr_forced_oof2" / "oof_metrics.json")

    normal_corr = corr_metrics.get("normal", {})
    shuffled_corr = corr_metrics.get("shuffled_gr", {})
    localized_corr = corr_metrics.get("localized", {})
    localized_normal = localized_corr.get("normal", {}) if isinstance(localized_corr, dict) else {}
    localized_shuffled = localized_corr.get("shuffled_gr", {}) if isinstance(localized_corr, dict) else {}
    stretch_corr = corr_metrics.get("localized_stretch", {})
    stretch_normal = stretch_corr.get("normal", {}) if isinstance(stretch_corr, dict) else {}
    stretch_shuffled = stretch_corr.get("shuffled_gr", {}) if isinstance(stretch_corr, dict) else {}
    mismatch_agg = mismatch_metrics.get("aggregate", {})
    if not isinstance(mismatch_agg, dict):
        mismatch_agg = {}
    formation_global = formation_metrics.get("global", {})
    formation_anchor = formation_metrics.get("anchor_geology", {})
    formation_oracle = formation_metrics.get("oracle_geology", {})
    pseudo_zone_agg = pseudo_zone_metrics.get("aggregate", {})
    if not isinstance(pseudo_zone_agg, dict):
        pseudo_zone_agg = {}
    tail_diag = tail_metrics.get("diagnostics", {})
    b2_tail = tail_metrics.get("b2", {})
    sanity_gaps = v4.get("sanity_gaps", {})
    split_decision = split_audit.get("decision", {})

    lines = [
        "# GEOMTP_DIAGNOSTIC_REPORT",
        "",
        "## Executive Summary",
        "",
        "We implemented the research-inspired stack: MTP heatmap windows, K trajectory modes, mode logits, prior/SDF channels, sequential particle tracking, CatBoost rankers, synthetic/correlation head experiments, and a direct PathFormer branch. The main failure is now specific: the system can generate plausible local trajectories, but the real GR/typewell signal does not become a reliable selector or standalone correlation ridge on Kaggle hidden wells.",
        "",
        "**Root Diagnosis:**",
        "",
        "1. Window-level MTP works as a candidate generator: oracle top-K is much better than top1.",
        "2. Row-level deployable gains stay small once leakage is removed.",
        "3. Simple GR/typewell correlation does not recover true TVT: shuffled GR is comparable or better on top-K hit rate.",
        "4. Tail audit says many bad wells are candidate-space failures, not selector failures.",
        "5. Direct full-path learning initially collapsed to known-tail/base-like behavior, which confirms the model needs stronger physical/correlation constraints.",
        "",
        "## Research Expectation vs What We Tested",
        "",
        "Research/webinar expectation:",
        "",
        "- Build a heatmap or spectrum panel from lateral GR vs typewell GR.",
        "- Keep multiple interpretations alive because log inversion is non-unique.",
        "- Apply interpretations sequentially, merging/pruning likely realizations.",
        "- Use synthetic corruptions of logs to teach robust correlation under noise, scale changes, stretch/squeeze, and missing data.",
        "",
        "Our implementation:",
        "",
        "- `MTPNet`: CNN heatmap model with K paths + logits, MTP loss, SDF/prior channels.",
        "- `track.py`: sequential multi-realization tracker with merge/prune.",
        "- `ranker.py`: learned post-hoc mode selector.",
        "- `synthetic.py` + corr head: sim2real attempt to teach dense vertical correlation.",
        "- `correlation_panel.py`: direct webinar-style GR/typewell panel diagnostic.",
        "- `pathformer/`: direct full-well sequence model to create new candidate paths.",
        "",
        "```mermaid",
        "flowchart LR",
        '  A["Lateral GR + known TVT"] --> B["GR/typewell heatmap"]',
        '  C["Typewell GR"] --> B',
        '  D["Prior paths / formation context"] --> B',
        '  B --> E["MTP CNN: K trajectories + logits"]',
        '  E --> F["Sequential tracker: carry, merge, prune"]',
        '  F --> G["Row-level path candidate"]',
        '  B --> H["Correlation panel diagnostic"]',
        '  H --> I["Does true path lie on GR ridge?"]',
        "```",
        "",
        "## 1. Window-Level MTP Evidence",
        "",
        "![Window metrics](figures/window_metrics.png)",
        "",
        _markdown_table(window_df, ["run", "top1", "weighted", "oracle_topK", "best_mode_top3", "spearman"]),
        "",
        "Interpretation: local mode space exists. In v1/v4, oracle top-K is around a few feet while deployable top1/weighted stays around 7-8 ft. This means generation is not the core bottleneck; selection/conditioning/row assembly is.",
        "",
        "## 2. GR Sanity Evidence",
        "",
        "![Sanity gaps](figures/sanity_gaps.png)",
        "",
        f"v4 no-GR top1 gap: `{_fmt(sanity_gaps.get('no_gr_top1_gap_ft'))}` ft.",
        f"v4 shuffled-GR top1 gap: `{_fmt(sanity_gaps.get('shuffled_gr_top1_gap_ft'))}` ft.",
        f"v4 no-all-priors top1 gap: `{_fmt(sanity_gaps.get('no_all_priors_top1_gap_ft'))}` ft.",
        "",
        "Interpretation: removing/shuffling GR barely hurts compared with removing priors. The network is not using GR/typewell as an independent causal alignment signal strongly enough.",
        "",
        "## 3. Row-Level Tracker Evidence",
        "",
        "![Row progress](figures/row_level_progress.png)",
        "",
        _markdown_table(row_df, ["run", "rmse", "gain_vs_legacy_anchor", "candidate"]),
        "",
        f"2-fold v3 OOF gain: `{_fmt(_metric(v3_oof, 'aggregate', 'gain_vs_b2'))}` ft, fold min/max `{_fmt(_metric(v3_oof, 'aggregate', 'fold_gain_min'))}` / `{_fmt(_metric(v3_oof, 'aggregate', 'fold_gain_max'))}`.",
        "",
        "Interpretation: sequential tracking is real infrastructure, but the clean OOF gain is far below the scale needed for a decisive solution.",
        "",
        "## 4. Ranker Leakage / Selector Evidence",
        "",
        "![Ranker leakage](figures/ranker_leakage.png)",
        "",
        f"Leakage-risk flag: `{split_decision.get('leakage_risk')}`. Train-valid gain gap: `{_fmt(split_decision.get('train_valid_gain_gap'))}` ft.",
        "",
        "Interpretation: the big ranker/tracker gain was mostly in-sample. Cross-fit rankers improve window metrics but do not reliably translate into row-level path gains.",
        "",
        "## 5. Direct Webinar-Style Correlation Panel",
        "",
        "![Correlation hit rates](figures/corr_panel_hit_rates.png)",
        "",
        "![Correlation distribution](figures/corr_panel_distribution.png)",
        "",
        "![Global vs localized correlation](figures/corr_panel_global_vs_localized.png)",
        "",
        "| variant | top1 RMSE ft | top3 oracle RMSE ft | top10 oracle RMSE ft | top3 hit % | top10 hit % |",
        "|---|---:|---:|---:|---:|---:|",
        f"| normal | {_fmt(normal_corr.get('corr_top1_rmse_ft'))} | {_fmt(normal_corr.get('corr_top3_oracle_rmse_ft'))} | {_fmt(normal_corr.get('corr_top10_oracle_rmse_ft'))} | {_fmt(100 * normal_corr.get('corr_target_top3_rate', 0), 2)} | {_fmt(100 * normal_corr.get('corr_target_top10_rate', 0), 2)} |",
        f"| shuffled GR | {_fmt(shuffled_corr.get('corr_top1_rmse_ft'))} | {_fmt(shuffled_corr.get('corr_top3_oracle_rmse_ft'))} | {_fmt(shuffled_corr.get('corr_top10_oracle_rmse_ft'))} | {_fmt(100 * shuffled_corr.get('corr_target_top3_rate', 0), 2)} | {_fmt(100 * shuffled_corr.get('corr_target_top10_rate', 0), 2)} |",
        "",
        "Localized/multiscale panel (`known_tail_linear`, ±120 ft, patch radii 2/4/8/16):",
        "",
        "| variant | top1 RMSE ft | top3 oracle RMSE ft | top10 oracle RMSE ft | top3 hit % | top10 hit % | anchor RMSE ft |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| normal | {_fmt(localized_normal.get('corr_top1_rmse_ft'))} | {_fmt(localized_normal.get('corr_top3_oracle_rmse_ft'))} | {_fmt(localized_normal.get('corr_top10_oracle_rmse_ft'))} | {_fmt(100 * localized_normal.get('corr_target_top3_rate', 0), 2)} | {_fmt(100 * localized_normal.get('corr_target_top10_rate', 0), 2)} | {_fmt(localized_normal.get('anchor_rmse_ft'))} |",
        f"| shuffled GR | {_fmt(localized_shuffled.get('corr_top1_rmse_ft'))} | {_fmt(localized_shuffled.get('corr_top3_oracle_rmse_ft'))} | {_fmt(localized_shuffled.get('corr_top10_oracle_rmse_ft'))} | {_fmt(100 * localized_shuffled.get('corr_target_top3_rate', 0), 2)} | {_fmt(100 * localized_shuffled.get('corr_target_top10_rate', 0), 2)} | {_fmt(localized_shuffled.get('anchor_rmse_ft'))} |",
        "",
        "Localized + multiscale + stretch/squeeze panel (`known_tail_linear`, ±120 ft, patch radii 2/4/8/16, stretch factors 0.5/0.75/1.0/1.25/1.5):",
        "",
        "| variant | top1 RMSE ft | top3 oracle RMSE ft | top10 oracle RMSE ft | top3 hit % | top10 hit % | anchor RMSE ft |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| normal | {_fmt(stretch_normal.get('corr_top1_rmse_ft'))} | {_fmt(stretch_normal.get('corr_top3_oracle_rmse_ft'))} | {_fmt(stretch_normal.get('corr_top10_oracle_rmse_ft'))} | {_fmt(100 * stretch_normal.get('corr_target_top3_rate', 0), 2)} | {_fmt(100 * stretch_normal.get('corr_target_top10_rate', 0), 2)} | {_fmt(stretch_normal.get('anchor_rmse_ft'))} |",
        f"| shuffled GR | {_fmt(stretch_shuffled.get('corr_top1_rmse_ft'))} | {_fmt(stretch_shuffled.get('corr_top3_oracle_rmse_ft'))} | {_fmt(stretch_shuffled.get('corr_top10_oracle_rmse_ft'))} | {_fmt(100 * stretch_shuffled.get('corr_target_top3_rate', 0), 2)} | {_fmt(100 * stretch_shuffled.get('corr_target_top10_rate', 0), 2)} | {_fmt(stretch_shuffled.get('anchor_rmse_ft'))} |",
        "",
        "Interpretation: naive global GR/typewell matching fails badly. Localizing the search band improves geometry a lot, so unrestricted typewell search was indeed part of the problem. Stretch/squeeze improves top1 and top-K hit rate further, but normal GR still does not beat shuffled GR inside the localized band. That is the key negative result: the current log-matching score is better as a broad proposal generator than as a reliable likelihood for choosing the true ridge.",
        "",
        "## 6. True-Path Typewell Mismatch Audit",
        "",
        "This audit asks a stricter question: if we score the provided typewell exactly at the true hidden TVT, does the true path itself look like a good GR/typewell match?",
        "",
        "![Typewell mismatch distribution](figures/typewell_mismatch_distribution.png)",
        "",
        "![Typewell mismatch scatter](figures/typewell_mismatch_scatter.png)",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| wells | {mismatch_agg.get('num_wells')} |",
        f"| hidden steps | {mismatch_agg.get('num_steps')} |",
        f"| true score mean | {_fmt(mismatch_agg.get('true_score_mean'))} |",
        f"| shuffled true-score mean | {_fmt(mismatch_agg.get('shuffled_true_score_mean'))} |",
        f"| score gap vs shuffled | {_fmt(mismatch_agg.get('score_gap_vs_shuffled_mean'))} |",
        f"| true-score better than shuffled rate | {_fmt(100 * mismatch_agg.get('true_score_better_than_shuffled_rate', 0), 2)}% |",
        f"| true top-3 rate | {_fmt(100 * mismatch_agg.get('true_top3_rate', 0), 2)}% |",
        f"| true top-10 rate | {_fmt(100 * mismatch_agg.get('true_top10_rate', 0), 2)}% |",
        f"| mismatch wells | {mismatch_agg.get('mismatch_wells')} / {mismatch_agg.get('num_wells')} |",
        "",
        "Interpretation: even when evaluated at the true TVT, the typewell/log score is only weakly better than shuffled and the true ridge is rarely top-ranked. This directly explains why CNNs, rankers, and trackers fail to turn GR channels into a strong selector: the supervised target path often does not have a clean standalone typewell-GR likelihood under this scoring model.",
        "",
        "## 7. Formation-Aware Correlation",
        "",
        "This diagnostic constrains the typewell search by the `Geology` label on the typewell grid. Important schema caveat: Kaggle test typewells do not include `Geology`, and test horizontals do not include formation surface columns such as `ANCC/ASTNU/...`. Therefore this section is a train-only upper-bound / root-cause diagnostic, not a directly deployable feature.",
        "",
        "![Formation-aware correlation](figures/formation_correlation_comparison.png)",
        "",
        "| variant | top1 RMSE ft | top10 oracle RMSE ft | true top10 % | candidate bins | shuffled top10 RMSE ft |",
        "|---|---:|---:|---:|---:|---:|",
        f"| global | {_fmt(formation_global.get('corr_top1_rmse_ft'))} | {_fmt(formation_global.get('corr_top10_oracle_rmse_ft'))} | {_fmt(100 * formation_global.get('corr_target_top10_rate', 0), 2)} | {_fmt(formation_global.get('candidate_bins_mean'))} | {_fmt(formation_global.get('shuffled_top10_oracle_rmse_ft'))} |",
        f"| anchor geology | {_fmt(formation_anchor.get('corr_top1_rmse_ft'))} | {_fmt(formation_anchor.get('corr_top10_oracle_rmse_ft'))} | {_fmt(100 * formation_anchor.get('corr_target_top10_rate', 0), 2)} | {_fmt(formation_anchor.get('candidate_bins_mean'))} | {_fmt(formation_anchor.get('shuffled_top10_oracle_rmse_ft'))} |",
        f"| oracle geology | {_fmt(formation_oracle.get('corr_top1_rmse_ft'))} | {_fmt(formation_oracle.get('corr_top10_oracle_rmse_ft'))} | {_fmt(100 * formation_oracle.get('corr_target_top10_rate', 0), 2)} | {_fmt(formation_oracle.get('candidate_bins_mean'))} | {_fmt(formation_oracle.get('shuffled_top10_oracle_rmse_ft'))} |",
        "",
        f"Anchor geology match rate on train labels: `{_fmt(100 * formation_anchor.get('anchor_geology_match_rate', 0), 2)}%`.",
        "",
        "Interpretation: formation/zone conditioning is the first diagnostic that materially changes the search problem. Oracle geology collapses the top-10 oracle from roughly 60 ft to roughly 12 ft and raises true top-10 coverage to about half of hidden steps. But shuffled GR remains nearly identical inside the same zone, so the main gain is stratigraphic search-space restriction, not GR discrimination. The deployable problem is now sharper: infer pseudo-formation zones from test-available signals only (`TVT_input`, `Z`, `MD`, lateral GR, typewell TVT/GR), then use correlation as a proposal feature inside those inferred zones.",
        "",
        "## 8. Test-Schema-Safe Pseudo-Zone Baseline",
        "",
        "Because test files lack both `Geology` labels and train-only formation surface columns, this baseline hides labels by well and predicts typewell zones from normalized typewell TVT position only. It is intentionally simple: if this weak baseline is already useful, a proper GR/sequence zone model is worth building.",
        "",
        "![Pseudo-zone distribution](figures/pseudo_zone_distribution.png)",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| wells | {pseudo_zone_agg.get('num_wells')} |",
        f"| hidden steps | {pseudo_zone_agg.get('num_steps')} |",
        f"| OOF hidden zone match | {_fmt(100 * pseudo_zone_agg.get('hidden_zone_match_rate', 0), 2)}% |",
        f"| predicted unknown rate | {_fmt(100 * pseudo_zone_agg.get('pred_unknown_rate', 0), 2)}% |",
        "",
        "Interpretation: a test-schema-safe relative-TVT template recovers about 72.5% of hidden zones OOF. That is close to the train-only anchor-geology match rate, but still far from the oracle-geology upper bound. This is now the clean engineering target: improve pseudo-zone prediction, then rerun formation-constrained correlation using predicted zones.",
        "",
        "## 9. Tail / Candidate-Space Audit",
        "",
        "![Tail class counts](figures/tail_class_counts.png)",
        "",
        "![Tail class RMSE](figures/tail_class_rmse.png)",
        "",
        f"Wells: `{tail_metrics.get('wells')}`. Rows: `{tail_metrics.get('rows')}`. Candidates audited: `{tail_metrics.get('candidates')}`.",
        "",
        "| diagnostic | wells |",
        "|---|---:|",
        f"| all candidates bad | {tail_diag.get('all_candidates_bad_wells')} |",
        f"| selector fail | {tail_diag.get('selector_fail_wells')} |",
        f"| anchor bad and oracle good | {tail_diag.get('b2_bad_and_oracle_good_wells')} |",
        f"| MTP improves | {tail_diag.get('mtp_improves_wells')} |",
        f"| MTP worsens | {tail_diag.get('mtp_worsens_wells')} |",
        "",
        f"Legacy tail p95/worst well RMSE: `{_fmt(b2_tail.get('p95_well_rmse'))}` / `{_fmt(b2_tail.get('worst_well_rmse'))}` ft.",
        "",
        "Interpretation: the main tail problem is not choosing among existing candidates. For many bad wells, the candidate set itself lacks a good path.",
        "",
        "## 10. What Is Actually Not Working",
        "",
        "The failed transfer is not one single bug. It is this chain:",
        "",
        "1. Research assumes a robust correlation between lateral log pieces and offset/typewell log pieces.",
        "2. Our direct correlation-panel diagnostic says this is not globally true on the Kaggle train wells.",
        "3. MTP learns useful local trajectory priors, especially when SDF/prior paths are present.",
        "4. GR channels do not strongly disambiguate modes: no-GR/shuffled-GR metrics remain close to normal.",
        "5. Sequential tracking gives small clean gains, but cannot invent correct paths for candidate-space failure wells.",
        "6. PathFormer can overfit and can use base-like skips, but raw full-well sequence learning does not yet create a new robust path family.",
        "",
        "## 11. Tail Well Visual Panels",
        "",
        "These panels show the localized/multiscale correlation top1 path, true TVT, known-tail linear anchor, and the rank of the true bin. A good correlation panel would keep the true rank near top-10 for long stretches.",
        "",
        *[
            f"![{Path(path).stem}]({path})"
            for path in tail_panel_paths[:6]
        ],
        "",
        "## 12. Concrete Next Diagnostics",
        "",
        "1. Replace the relative-TVT pseudo-zone template with a real typewell GR/TVT sequence labeler, evaluated OOF by wells.",
        "2. Use predicted pseudo-zones to run formation-constrained correlation and measure top10 oracle/hit vs oracle-geology upper bound.",
        "3. Typewell mismatch root-cause slices: compare mismatch against GR missingness, tail classes, well geometry, and typewell score percentiles.",
        "4. Candidate-space generator: use inferred-zone corr ridges as broad path proposals only for `all_candidates_bad` wells, not as final selector.",
        "",
        "## Kaggle Discussion Draft",
        "",
        "We implemented a research-inspired multi-modal geosteering stack: heatmap CNN-MTP with K trajectory hypotheses and logits, sequential merge/prune tracking, post-hoc mode rankers, synthetic log-correlation pretraining, and a direct webinar-style correlation panel. Window-level MTP works as a candidate generator: oracle top-K is much better than deployable top1/weighted paths. But row-level OOF gains remain small once leakage is removed.",
        "",
        "The surprising diagnostic is that a direct GR/typewell correlation panel does not recover the true TVT path globally on train wells: true TVT is in top-10 only ~5% of hidden compressed steps, and shuffled GR is comparable. Localizing the search band and adding multi-scale/stretch matching improves the proposal geometry substantially, but shuffled GR remains comparable to normal GR. A stricter true-path audit says the same thing: even at the known target TVT, the typewell score beats shuffled GR on less than half of hidden steps. Formation-zone conditioning is the first intervention that materially changes the search problem: train-only oracle geology raises true top-10 coverage to roughly 50%, but shuffled remains comparable inside the zone. Because those labels are absent in test, this is an upper bound and points to pseudo-zone inference rather than direct feature usage.",
        "",
        "Question for others: did you find that raw GR/typewell correlation is only useful after constraining the TVT search band, using multi-scale/stretch matching, or adding formation/geologic context? Are there known pitfalls with using the provided typewell as a global correlation reference for the hidden lateral interval?",
        "",
        "## Artifact Index",
        "",
        "- `artifacts/diagnostic_report_v1/diagnostic_report.md`",
        "- `artifacts/diagnostic_report_v1/figures/*.png`",
        "- `artifacts/corr_panel_v0/correlation_panel_metrics.json`",
        "- `artifacts/corr_panel_localized_v0/correlation_panel_metrics.json`",
        "- `artifacts/corr_panel_stretch_v0/correlation_panel_metrics.json`",
        "- `artifacts/typewell_mismatch_v0/typewell_mismatch_metrics.json`",
        "- `artifacts/formation_correlation_v0/formation_correlation_metrics.json`",
        "- `artifacts/pseudo_zone_v0/pseudo_zone_metrics.json`",
        "- `artifacts/tail_audit_v1/tail_audit_metrics.json`",
        "- `artifacts/mtp_v3_gr_forced_oof2/oof_metrics.json`",
    ]

    report_path = output_dir / "diagnostic_report.md"
    report_path.write_text("\n".join(lines) + "\n")
    return report_path


def main() -> None:
    report = write_diagnostic_report()
    print(f"Wrote diagnostic report to {report}", flush=True)


if __name__ == "__main__":
    main()
