from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .formation_b_lite import _prediction_from_candidate_map
from .formation_plane_knn import CANDIDATE_COLUMNS, json_safe, markdown_table, write_frame
from .formation_selector import attach_schema10, make_score_context, score_prediction


FILTER_MODES: dict[str, dict[str, float]] = {
    "loose": {
        "median_diff": 60.0,
        "p95_diff": 120.0,
        "endpoint_diff": 80.0,
        "anchor_q": 0.80,
        "roughness_q": 0.90,
        "b_std_q": 0.80,
    },
    "medium": {
        "median_diff": 40.0,
        "p95_diff": 80.0,
        "endpoint_diff": 60.0,
        "anchor_q": 0.70,
        "roughness_q": 0.80,
        "b_std_q": 0.70,
    },
    "strict": {
        "median_diff": 25.0,
        "p95_diff": 50.0,
        "endpoint_diff": 40.0,
        "anchor_q": 0.50,
        "roughness_q": 0.70,
        "b_std_q": 0.50,
    },
}
SAFE_ALPHAS: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30, 0.40)
SAFE_CLIPS: tuple[float, ...] = (10.0, 15.0, 20.0, 30.0)
FORMATIONS: tuple[str, ...] = ("ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA")


def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _numeric(frame: pd.DataFrame, column: str, default: float = np.nan) -> np.ndarray:
    if column not in frame.columns:
        return np.full(len(frame), default, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def _first(group: pd.DataFrame, column: str) -> float:
    if column not in group.columns:
        return float("nan")
    arr = _numeric(group, column)
    finite = arr[np.isfinite(arr)]
    return float(finite[0]) if len(finite) else float("nan")


def _safe_median(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.nanmedian(finite)) if len(finite) else float("nan")


def _safe_percentile(values: np.ndarray, q: float) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.nanpercentile(finite, q)) if len(finite) else float("nan")


def _formation_for_candidate(candidate: str) -> str | None:
    if candidate.startswith("tvtF_"):
        parts = candidate.split("_")
        if len(parts) >= 2 and parts[1] in FORMATIONS:
            return parts[1]
    if candidate in {"row_ancc_tvt", "dense_ancc_tvt"}:
        return "ANCC"
    return None


def _mean_b_std(group: pd.DataFrame) -> float:
    cols = [f"b_{formation}_std" for formation in FORMATIONS if f"b_{formation}_std" in group.columns]
    if not cols:
        return float("nan")
    values = group[cols].iloc[0].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    return float(np.nanmean(finite)) if len(finite) else float("nan")


def _candidate_b_std(group: pd.DataFrame, candidate: str) -> float:
    formation = _formation_for_candidate(candidate)
    if formation is not None:
        return _first(group, f"b_{formation}_std")
    return _mean_b_std(group)


def _candidate_level_stats(
    group: pd.DataFrame,
    candidate: str,
    *,
    baseline_available: bool,
) -> dict[str, float]:
    values = _numeric(group, candidate)
    schema = _numeric(group, "schema10_oof_raw")
    diff = np.abs(values - schema) if baseline_available else np.full(len(values), np.nan)
    anchor = _first(group, f"anchor_fit_rmse__{candidate}")
    late_anchor = _first(group, f"late_anchor_fit_rmse__{candidate}")
    roughness = _first(group, f"roughness__{candidate}")
    finite_frac = _first(group, f"finite_frac__{candidate}")
    if not np.isfinite(finite_frac):
        finite_frac = float(np.isfinite(values).mean())
    if not np.isfinite(roughness):
        finite_values = values[np.isfinite(values)]
        if len(finite_values) >= 3:
            roughness = float(np.sqrt(np.mean(np.diff(finite_values, n=2) ** 2)))
    endpoint_diff = float("nan")
    mask = np.isfinite(values) & np.isfinite(schema)
    if baseline_available and mask.any():
        endpoint_diff = float(abs(values[mask][-1] - schema[mask][-1]))
    return {
        "anchor_fit_rmse": anchor,
        "late_anchor_rmse": late_anchor,
        "roughness": roughness,
        "finite_frac": finite_frac,
        "b_std": _candidate_b_std(group, candidate),
        "median_abs_diff_vs_schema10": _safe_median(diff),
        "p95_abs_diff_vs_schema10": _safe_percentile(diff, 95),
        "endpoint_diff_vs_schema10": endpoint_diff,
    }


def build_b2_metadata(
    frame: pd.DataFrame,
    b_scores: pd.DataFrame,
    *,
    baseline_available: bool,
    progress_interval: int = 50,
) -> pd.DataFrame:
    b_candidate_names = set(b_scores["candidate_name"].astype(str))
    candidates = [
        column
        for column in CANDIDATE_COLUMNS
        if column in frame.columns and column in b_candidate_names
    ]
    b_key = b_scores.set_index(["well_id", "candidate_name"], drop=False)
    rows: list[dict[str, Any]] = []
    grouped = list(frame.groupby("well_id", sort=False))
    for idx, (well, group) in enumerate(grouped, start=1):
        if progress_interval > 0 and (idx == 1 or idx % progress_interval == 0 or idx == len(grouped)):
            print(f"b2 metadata well={idx}/{len(grouped)} name={well}", flush=True)
        for candidate in candidates:
            if (well, candidate) not in b_key.index:
                continue
            b_row = b_key.loc[(well, candidate)]
            if isinstance(b_row, pd.DataFrame):
                b_row = b_row.iloc[0]
            rows.append(
                {
                    "well_id": well,
                    "candidate_name": candidate,
                    "hidden_rmse": float(b_row["hidden_rmse"]),
                    "b_combined_score": float(b_row["b_combined_score"]),
                    "b_combined_without_surface_terms": float(
                        b_row["b_combined_without_surface_terms"]
                    ),
                    "b_combined_gr_only": float(b_row["b_combined_gr_only"]),
                    "b_path_corr": float(b_row["b_path_corr"]),
                    "b_dgr_corr": float(b_row["b_dgr_corr"]),
                    "b_ncc15_mean": float(b_row["b_ncc15_mean"]),
                    "b_ncc_multiscale_mean": float(b_row["b_ncc_multiscale_mean"]),
                    "surface_std": float(b_row["surface_std"]),
                    **_candidate_level_stats(
                        group,
                        candidate,
                        baseline_available=baseline_available,
                    ),
                }
            )
    meta = pd.DataFrame(rows)
    return add_ranks(meta)


def add_ranks(meta: pd.DataFrame) -> pd.DataFrame:
    out = meta.copy()
    out["a_rank_late_anchor"] = out.groupby("well_id")["late_anchor_rmse"].rank(
        method="first", ascending=True, na_option="bottom"
    )
    out["a_rank_anchor"] = out.groupby("well_id")["anchor_fit_rmse"].rank(
        method="first", ascending=True, na_option="bottom"
    )
    out["a_rank"] = out["a_rank_late_anchor"]
    out["b_rank"] = out.groupby("well_id")["b_combined_score"].rank(
        method="first", ascending=True, na_option="bottom"
    )
    out["b_gr_rank"] = out.groupby("well_id")["b_combined_gr_only"].rank(
        method="first", ascending=True, na_option="bottom"
    )
    out["oracle_rank"] = out.groupby("well_id")["hidden_rmse"].rank(
        method="first", ascending=True, na_option="bottom"
    )
    return out


def _quantile_by_well(meta: pd.DataFrame, column: str, q: float) -> pd.Series:
    return meta.groupby("well_id")[column].transform(lambda values: values.quantile(q))


def apply_filter_mode(meta: pd.DataFrame, mode: str, *, baseline_available: bool) -> pd.Series:
    cfg = FILTER_MODES[mode]
    finite_gate = meta["finite_frac"].fillna(0.0) >= 0.999
    anchor_gate = meta["late_anchor_rmse"] <= _quantile_by_well(meta, "late_anchor_rmse", cfg["anchor_q"])
    rough_gate = meta["roughness"] <= _quantile_by_well(meta, "roughness", cfg["roughness_q"])
    bstd_gate = meta["b_std"] <= _quantile_by_well(meta, "b_std", cfg["b_std_q"])
    baseline_gate = pd.Series(True, index=meta.index)
    if baseline_available:
        baseline_gate = (
            (meta["median_abs_diff_vs_schema10"] <= cfg["median_diff"])
            & (meta["p95_abs_diff_vs_schema10"] <= cfg["p95_diff"])
            & (meta["endpoint_diff_vs_schema10"] <= cfg["endpoint_diff"])
        )
    return finite_gate & anchor_gate.fillna(False) & rough_gate.fillna(False) & bstd_gate.fillna(False) & baseline_gate


def _choice_by_min(group: pd.DataFrame, column: str) -> pd.Series:
    ordered = group.sort_values([column, "candidate_name"], ascending=[True, True])
    return ordered.iloc[0]


def _candidate_map_from_choice(choice: pd.DataFrame) -> dict[Any, str]:
    return dict(zip(choice["well_id"], choice["candidate_name"], strict=False))


def _score_candidate_map(
    frame: pd.DataFrame,
    candidate_map: dict[Any, str],
    name: str,
    context: Any,
) -> dict[str, Any]:
    pred = _prediction_from_candidate_map(frame, candidate_map)
    return score_prediction(frame, pred, name, context)


def _hit_rates(choice: pd.DataFrame, meta: pd.DataFrame) -> dict[str, float]:
    rank_lookup = meta.set_index(["well_id", "candidate_name"])["oracle_rank"].to_dict()
    ranks = [
        float(rank_lookup.get((row.well_id, row.candidate_name), np.nan))
        for row in choice.itertuples(index=False)
    ]
    ranks_arr = np.asarray(ranks, dtype=float)
    return {
        "top1_hit_rate": float(np.nanmean(ranks_arr <= 1)),
        "top3_hit_rate": float(np.nanmean(ranks_arr <= 3)),
        "top5_hit_rate": float(np.nanmean(ranks_arr <= 5)),
        "top10_hit_rate": float(np.nanmean(ranks_arr <= 10)),
    }


def _oracle_choice(meta: pd.DataFrame, subset: pd.Series | None = None) -> pd.DataFrame:
    work = meta[subset] if subset is not None else meta
    return (
        work.sort_values(["well_id", "hidden_rmse", "candidate_name"])
        .groupby("well_id", sort=False)
        .head(1)
        .reset_index(drop=True)
    )


def filter_report(
    frame: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    baseline_available: bool,
) -> pd.DataFrame:
    context = make_score_context(frame)
    rows = []
    all_oracle = _oracle_choice(meta)
    all_oracle_candidates = set(zip(all_oracle["well_id"], all_oracle["candidate_name"], strict=False))
    for mode in FILTER_MODES:
        mask = apply_filter_mode(meta, mode, baseline_available=baseline_available)
        # Keep at least the best late-anchor candidate per well if gates are too strict.
        fallback = (
            meta.sort_values(["well_id", "late_anchor_rmse", "candidate_name"])
            .groupby("well_id", sort=False)
            .head(1)
        )
        fallback_index = set(zip(fallback["well_id"], fallback["candidate_name"], strict=False))
        mask = mask | pd.Series(
            [
                (row.well_id, row.candidate_name) in fallback_index
                for row in meta.itertuples(index=False)
            ],
            index=meta.index,
        )
        kept = meta[mask].copy()
        kept_counts = kept.groupby("well_id")["candidate_name"].count()
        kept_oracle = _oracle_choice(meta, mask)
        b_choice = (
            kept.sort_values(["well_id", "b_combined_score", "candidate_name"])
            .groupby("well_id", sort=False)
            .head(1)
            .reset_index(drop=True)
        )
        kept_oracle_map = _candidate_map_from_choice(kept_oracle)
        b_map = _candidate_map_from_choice(b_choice)
        kept_oracle_score = _score_candidate_map(
            frame, kept_oracle_map, f"{mode}_kept_oracle", context
        )
        b_score = _score_candidate_map(frame, b_map, f"{mode}_B_selected", context)
        kept_keys = set(zip(kept["well_id"], kept["candidate_name"], strict=False))
        oracle_kept_rate = float(
            np.mean([key in kept_keys for key in all_oracle_candidates])
        )
        rows.append(
            {
                "mode": mode,
                "avg_candidates_kept": float(np.nanmean(kept_counts)),
                "min_candidates_kept": int(kept_counts.min()),
                "oracle_kept_rate": oracle_kept_rate,
                "kept_oracle_rmse": kept_oracle_score["rmse"],
                "B_selected_rmse": b_score["rmse"],
                "B_selected_p95": b_score["p95_well_rmse"],
                "B_selected_worst": b_score["worst_well_rmse"],
                **_hit_rates(b_choice, meta),
            }
        )
    return pd.DataFrame(rows)


def intersection_selectors(frame: pd.DataFrame, meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    context = make_score_context(frame)
    selector_frames: list[pd.DataFrame] = []

    for top_k in (10, 20, 50):
        choice = (
            meta[meta["a_rank"] <= top_k]
            .sort_values(["well_id", "b_combined_score", "candidate_name"])
            .groupby("well_id", sort=False)
            .head(1)
            .reset_index(drop=True)
        )
        choice["selector_name"] = f"B_among_A_top{top_k}"
        selector_frames.append(choice)

    for top_k in (10, 20):
        choice = (
            meta[meta["b_rank"] <= top_k]
            .sort_values(["well_id", "a_rank", "candidate_name"])
            .groupby("well_id", sort=False)
            .head(1)
            .reset_index(drop=True)
        )
        choice["selector_name"] = f"A_among_B_top{top_k}"
        selector_frames.append(choice)

    rank_specs = {
        "rank_A_plus_B": (1.0, 1.0),
        "rank_0_3A_0_7B": (0.3, 0.7),
        "rank_0_7A_0_3B": (0.7, 0.3),
    }
    for name, (a_weight, b_weight) in rank_specs.items():
        work = meta.copy()
        work["_combined_rank"] = a_weight * work["a_rank"] + b_weight * work["b_rank"]
        choice = (
            work.sort_values(["well_id", "_combined_rank", "candidate_name"])
            .groupby("well_id", sort=False)
            .head(1)
            .reset_index(drop=True)
        )
        choice["selector_name"] = name
        selector_frames.append(choice)

    choices = pd.concat(selector_frames, ignore_index=True)
    rows = []
    for selector, choice in choices.groupby("selector_name", sort=False):
        candidate_map = _candidate_map_from_choice(choice)
        row = _score_candidate_map(frame, candidate_map, selector, context)
        row.update(_hit_rates(choice, meta))
        rows.append(row)
    return choices, pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)


def catastrophe_analysis(meta: pd.DataFrame) -> pd.DataFrame:
    b_choice = (
        meta.sort_values(["well_id", "b_combined_score", "candidate_name"])
        .groupby("well_id", sort=False)
        .head(1)
        .set_index("well_id")
    )
    oracle = (
        meta.sort_values(["well_id", "hidden_rmse", "candidate_name"])
        .groupby("well_id", sort=False)
        .head(1)
        .set_index("well_id")
    )
    rows = []
    for well in b_choice.index:
        selected = b_choice.loc[well]
        best = oracle.loc[well]
        rows.append(
            {
                "well_id": well,
                "B_selected_candidate": selected["candidate_name"],
                "B_selected_rmse": selected["hidden_rmse"],
                "oracle_A_candidate": best["candidate_name"],
                "oracle_A_rmse": best["hidden_rmse"],
                "B_score_selected": selected["b_combined_score"],
                "B_score_oracle": best["b_combined_score"],
                "median_diff_selected_vs_schema10": selected["median_abs_diff_vs_schema10"],
                "median_diff_oracle_vs_schema10": best["median_abs_diff_vs_schema10"],
                "endpoint_diff_selected_vs_schema10": selected["endpoint_diff_vs_schema10"],
                "endpoint_diff_oracle_vs_schema10": best["endpoint_diff_vs_schema10"],
                "late_anchor_rmse_selected": selected["late_anchor_rmse"],
                "late_anchor_rmse_oracle": best["late_anchor_rmse"],
                "roughness_selected": selected["roughness"],
                "roughness_oracle": best["roughness"],
                "oracle_rank_by_B": best["b_rank"],
            }
        )
    return pd.DataFrame(rows).sort_values("B_selected_rmse", ascending=False).reset_index(drop=True)


def safe_blends(
    frame: pd.DataFrame,
    choices: pd.DataFrame,
    *,
    baseline_available: bool,
) -> pd.DataFrame:
    if not baseline_available:
        return pd.DataFrame()
    context = make_score_context(frame)
    schema = _numeric(frame, "schema10_oof_raw")
    rows = []
    for selector, choice in choices.groupby("selector_name", sort=False):
        candidate_map = _candidate_map_from_choice(choice)
        selected = _prediction_from_candidate_map(frame, candidate_map)
        delta_raw = selected - schema
        for alpha in SAFE_ALPHAS:
            for clip in SAFE_CLIPS:
                delta = np.where(
                    np.isfinite(delta_raw),
                    np.clip(delta_raw, -float(clip), float(clip)),
                    0.0,
                )
                pred = schema + float(alpha) * delta
                row = score_prediction(
                    frame,
                    pred,
                    f"safe__{selector}__a{alpha:g}_clip{clip:g}",
                    context,
                )
                shift = np.abs(pred - schema)
                row.update(
                    {
                        "base_selector": selector,
                        "alpha": float(alpha),
                        "clip": float(clip),
                        "median_shift": _safe_median(shift),
                        "p95_shift": _safe_percentile(shift, 95),
                    }
                )
                rows.append(row)
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)


def write_report(
    *,
    output_dir: Path,
    baseline_available: bool,
    catastrophe: pd.DataFrame,
    filter_scores: pd.DataFrame,
    intersection_scores: pd.DataFrame,
    safe_scores: pd.DataFrame,
) -> None:
    lines = [
        "# B2 Constrained Report",
        f"Schema10 available: `{'yes' if baseline_available else 'no'}`",
        "## Catastrophe Analysis",
        markdown_table(catastrophe.head(30)),
        "## Plausibility Filters",
        markdown_table(filter_scores),
        "## Intersection Selectors",
        markdown_table(intersection_scores),
        "## Safe Blends",
        markdown_table(safe_scores) if not safe_scores.empty else "_schema10 unavailable_",
        "## Decision",
        "_Use B2 gates: preserved oracle <= 9.5, selected <= 13, or safe blend gain._",
    ]
    (output_dir / "B2_CONSTRAINED_REPORT.md").write_text("\n\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = _read_frame(Path(args.input))
    frame, baseline_available = attach_schema10(
        frame,
        Path(args.schema10_oof) if args.schema10_oof else None,
        schema10_column=args.schema10_column,
    )
    b_scores = _read_frame(Path(args.b_scores))
    print("b2 build metadata", flush=True)
    meta = build_b2_metadata(
        frame,
        b_scores,
        baseline_available=baseline_available,
        progress_interval=int(args.progress_interval),
    )
    print("b2 filters", flush=True)
    filter_scores = filter_report(frame, meta, baseline_available=baseline_available)
    print("b2 intersections", flush=True)
    choices, intersection_scores = intersection_selectors(frame, meta)
    safe_scores = safe_blends(frame, choices, baseline_available=baseline_available)
    catastrophe = catastrophe_analysis(meta)

    write_frame(meta, output_dir / "b2_candidate_metadata.parquet")
    choices.to_csv(output_dir / "b2_selector_choices.csv", index=False)
    filter_scores.to_csv(output_dir / "b2_filter_report.csv", index=False)
    intersection_scores.to_csv(output_dir / "b2_intersection_selectors.csv", index=False)
    safe_scores.to_csv(output_dir / "b2_safe_blends.csv", index=False)
    catastrophe.to_csv(output_dir / "b2_catastrophe_analysis.csv", index=False)
    write_report(
        output_dir=output_dir,
        baseline_available=baseline_available,
        catastrophe=catastrophe,
        filter_scores=filter_scores,
        intersection_scores=intersection_scores,
        safe_scores=safe_scores,
    )
    metrics = {
        "input": str(args.input),
        "b_scores": str(args.b_scores),
        "rows": int(len(frame)),
        "wells": int(frame["well_id"].nunique()),
        "schema10_available": bool(baseline_available),
        "best_filter": filter_scores.sort_values("B_selected_rmse").head(1).to_dict("records"),
        "best_intersection": intersection_scores.head(1).to_dict("records"),
        "best_safe_blend": safe_scores.head(1).to_dict("records") if not safe_scores.empty else [],
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(json_safe(metrics), file, indent=2)
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B2 constrained scorer over A/B-lite candidate metadata.")
    parser.add_argument("--input", type=Path, default=Path("artifacts/formation_plane_knn/oof_candidates.parquet"))
    parser.add_argument("--b-scores", type=Path, default=Path("artifacts/formation_b_lite/b_candidate_scores.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/formation_b2_constrained"))
    parser.add_argument("--schema10-oof", type=Path, default=None)
    parser.add_argument("--schema10-column", type=str, default=None)
    parser.add_argument("--progress-interval", type=int, default=50)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    metrics = run(parse_args(argv))
    print(json.dumps(json_safe(metrics), indent=2), flush=True)


if __name__ == "__main__":
    main()
