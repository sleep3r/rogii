from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import TemplateEnvelopeConfig, VALID_COORD_SOURCES, parse_float_grid
from .data import (
    FORBIDDEN_INFERENCE_COLUMNS,
    assert_schema_safe_columns,
    compress_horizontal,
    discover_wells,
    hidden_rows_with_steps,
    load_well_pair,
)
from .envelope import build_envelope_from_paths, select_template_indices
from .scoring import gr_variant, score_template_match
from .templates import build_template_path, sample_typewell_gr


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _rmse(pred: np.ndarray, true: np.ndarray) -> float:
    pred_arr = np.asarray(pred, dtype=np.float64)
    true_arr = np.asarray(true, dtype=np.float64)
    mask = np.isfinite(pred_arr) & np.isfinite(true_arr)
    if not mask.any():
        return float("nan")
    return float(np.sqrt(np.mean(np.square(pred_arr[mask] - true_arr[mask]))))


def _candidate_rows_for_variant(
    horizontal: pd.DataFrame,
    step_frame: pd.DataFrame,
    *,
    rows_per_step: int,
    variant: str,
) -> pd.DataFrame:
    hidden = hidden_rows_with_steps(horizontal, rows_per_step=rows_per_step)
    if hidden.empty:
        return pd.DataFrame(columns=["id", "well_id", "row_idx", "step", "candidate", "pred_tvt"])
    pred_cols = ["well_id", "step", "envelope_low", "envelope_high", "envelope_mid"]
    merged = hidden.merge(step_frame[pred_cols], on=["well_id", "step"], how="left")
    parts: list[pd.DataFrame] = []
    for source, candidate in (
        ("envelope_mid", "template_envelope_mid"),
        ("envelope_low", "template_envelope_low"),
        ("envelope_high", "template_envelope_high"),
    ):
        valid = pd.to_numeric(merged[source], errors="coerce").notna()
        if not valid.any():
            continue
        out = merged.loc[valid, ["id", "well_id", "row_idx", "step"]].copy()
        out["candidate"] = candidate if variant == "normal_GR" else f"{candidate}_{variant}"
        out["pred_tvt"] = pd.to_numeric(merged.loc[valid, source], errors="coerce").to_numpy(dtype=np.float64)
        parts.append(out)
    if not parts:
        return pd.DataFrame(columns=["id", "well_id", "row_idx", "step", "candidate", "pred_tvt"])
    return pd.concat(parts, ignore_index=True)


def _score_one_well(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    *,
    config: TemplateEnvelopeConfig,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    comp = compress_horizontal(
        horizontal,
        rows_per_step=config.rows_per_step,
        coord_source=config.coord_source,
    )
    coord = pd.to_numeric(comp["coord"], errors="coerce").to_numpy(dtype=np.float64)
    base_gr = pd.to_numeric(comp["GR_filled"], errors="coerce").to_numpy(dtype=np.float64)
    hidden_mask = comp["hidden_mask"].to_numpy(dtype=bool)
    true_steps = pd.to_numeric(comp.get("true_tvt", pd.Series(np.nan, index=comp.index)), errors="coerce").to_numpy(dtype=np.float64)
    weights = hidden_mask.astype(np.float64)
    if weights.sum() < 3:
        weights = np.ones_like(weights, dtype=np.float64)

    well_id = str(comp["well_id"].iloc[0]) if len(comp) else "?"

    # ------------------------------------------------------------------
    # Coord sanity gate. The legacy `bridge` proxy collapses to a constant
    # on the hidden region in heel-only ROGII wells. A constant coord makes
    # `path_tvt = a*coord + offset` constant per (a, offset), which makes
    # `template_gr = TW_GR(const)` constant, which makes the z-scored score
    # collapse to zero for every variant. Skip the well early and record it.
    # ------------------------------------------------------------------
    coord_hidden = coord[hidden_mask] if hidden_mask.any() else coord
    coord_hidden = coord_hidden[np.isfinite(coord_hidden)]
    coord_std_hidden = float(np.std(coord_hidden)) if coord_hidden.size >= 2 else 0.0
    coord_finite = coord[np.isfinite(coord)]
    coord_std_all = float(np.std(coord_finite)) if coord_finite.size >= 2 else 0.0
    if coord_std_hidden < 1e-6 or coord_std_all < 1e-6:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            {
                "skipped": {
                    "reason": "constant_coord_on_hidden",
                    "coord_source": config.coord_source,
                    "well_id": well_id,
                    "coord_std_hidden": coord_std_hidden,
                    "coord_std_all": coord_std_all,
                }
            },
        )

    grid_records: list[dict[str, Any]] = []
    step_records: list[pd.DataFrame] = []
    candidate_parts: list[pd.DataFrame] = []
    variant_metrics: dict[str, Any] = {}

    template_paths: list[np.ndarray] = []
    template_meta: list[tuple[float, float]] = []
    template_grs: list[np.ndarray] = []
    for scale in config.scale_grid:
        for offset in config.offset_grid:
            path = build_template_path(coord, scale=float(scale), offset=float(offset))
            template_paths.append(path)
            template_grs.append(sample_typewell_gr(typewell, path))
            template_meta.append((float(scale), float(offset)))
    paths_arr = np.vstack(template_paths) if template_paths else np.empty((0, len(coord)))

    # Pre-compute shuffled-baseline score per template (used both for the
    # shuffled_GR variant and, optionally, as a per-template gating threshold
    # for the normal variant).
    shuffled_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
    shuffled_gr_pre = gr_variant(
        base_gr,
        hidden_mask=hidden_mask,
        variant="shuffled_hidden_GR",
        rng=shuffled_rng,
    )
    shuffled_scores = np.array(
        [score_template_match(tg, shuffled_gr_pre, weights=weights) for tg in template_grs],
        dtype=np.float64,
    )
    shuffled_median = (
        float(np.nanmedian(shuffled_scores[np.isfinite(shuffled_scores)]))
        if np.isfinite(shuffled_scores).any()
        else 0.0
    )

    for variant in config.variants:
        variant_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        gr = gr_variant(base_gr, hidden_mask=hidden_mask, variant=variant, rng=variant_rng)
        scores = np.array(
            [score_template_match(template_gr, gr, weights=weights) for template_gr in template_grs],
            dtype=np.float64,
        )
        # Threshold: positive correlation by default; optionally also beat
        # the per-well shuffled median (only for the normal variant -
        # otherwise the shuffled/zero diagnostic variants would self-gate
        # and never produce an envelope, which kills the sanity comparison).
        threshold = float(config.min_template_score)
        if config.require_beats_shuffled and variant == "normal_GR":
            threshold = max(threshold, shuffled_median)
        selected = select_template_indices(
            scores,
            top_n=config.top_n_templates,
            min_score=threshold,
        )
        envelope: Any = None
        if selected.size > 0:
            envelope = build_envelope_from_paths(
                paths_arr,
                selected_indices=selected,
                low_quantile=config.envelope_low_quantile,
                high_quantile=config.envelope_high_quantile,
            )
        finite_scores_mask = np.isfinite(scores)
        for rank, idx in enumerate(np.argsort(scores)[::-1]):
            if not finite_scores_mask[idx]:
                continue
            scale, offset = template_meta[int(idx)]
            grid_records.append(
                {
                    "well_id": well_id,
                    "variant": variant,
                    "scale": scale,
                    "offset": offset,
                    "score": float(scores[idx]),
                    "rank": int(rank + 1),
                    "selected": bool(idx in set(selected.tolist())),
                }
            )
        step_frame = comp[["well_id", "step", "row_start", "row_end"]].copy()
        step_frame["variant"] = variant
        if envelope is not None:
            step_frame["envelope_low"] = envelope.low
            step_frame["envelope_high"] = envelope.high
            step_frame["envelope_mid"] = envelope.mid
            step_frame["envelope_width"] = envelope.high - envelope.low
        else:
            step_frame["envelope_low"] = np.nan
            step_frame["envelope_high"] = np.nan
            step_frame["envelope_mid"] = np.nan
            step_frame["envelope_width"] = np.nan
        step_frame["top_template_score"] = (
            float(np.nanmax(scores[finite_scores_mask])) if finite_scores_mask.any() else float("nan")
        )
        step_frame["shuffled_median_score"] = shuffled_median
        step_records.append(step_frame)
        if envelope is None:
            variant_metrics[variant] = {
                "step_rmse": float("nan"),
                "row_rmse": float("nan"),
                "candidate_rows": 0,
                "top_template_score": float(np.nanmax(scores)) if finite_scores_mask.any() else float("nan"),
                "shuffled_median_score": shuffled_median,
                "selected_templates": 0,
            }
            continue
        candidates = _candidate_rows_for_variant(
            horizontal,
            step_frame,
            rows_per_step=config.rows_per_step,
            variant=variant,
        )
        candidate_parts.append(candidates)
        step_rmse = _rmse(envelope.mid[hidden_mask], true_steps[hidden_mask])
        row_rmse = float("nan")
        if not candidates.empty and "TVT" in horizontal.columns:
            mid_candidates = candidates[candidates["candidate"].astype(str).str.startswith("template_envelope_mid")]
            truth = horizontal.set_index("id")["TVT"]
            row_rmse = _rmse(
                pd.to_numeric(mid_candidates["pred_tvt"], errors="coerce").to_numpy(dtype=np.float64),
                pd.to_numeric(truth.reindex(mid_candidates["id"]), errors="coerce").to_numpy(dtype=np.float64),
            )
        variant_metrics[variant] = {
            "step_rmse": step_rmse,
            "row_rmse": row_rmse,
            "candidate_rows": int(len(candidates)),
            "top_template_score": float(np.nanmax(scores[finite_scores_mask])) if finite_scores_mask.any() else float("nan"),
            "shuffled_median_score": shuffled_median,
            "selected_templates": int(selected.size),
        }

    return (
        pd.DataFrame(grid_records),
        pd.concat(step_records, ignore_index=True) if step_records else pd.DataFrame(),
        pd.concat(candidate_parts, ignore_index=True) if candidate_parts else pd.DataFrame(),
        variant_metrics,
    )


def _aggregate_variant_metrics(per_well: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-well variant metrics; ignore skip-only well entries."""
    variants: dict[str, dict[str, list[float]]] = {}
    for well_metrics in per_well:
        if not well_metrics:
            continue
        if set(well_metrics.keys()) == {"skipped"}:
            continue
        for variant, values in well_metrics.items():
            if variant == "skipped":
                continue
            bucket = variants.setdefault(variant, {})
            for key, value in values.items():
                bucket.setdefault(key, []).append(float(value))
    out: dict[str, Any] = {}
    for variant, values in variants.items():
        out[variant] = {}
        for key, series in values.items():
            arr = np.asarray(series, dtype=np.float64)
            if key.endswith("rows") or key == "selected_templates":
                out[variant][key] = int(np.nansum(arr))
                continue
            finite = arr[np.isfinite(arr)]
            out[variant][key] = float(finite.mean()) if finite.size else float("nan")
    return out


def _aggregate_skip_metrics(per_well: list[dict[str, Any]]) -> dict[str, Any]:
    skipped: list[dict[str, Any]] = []
    for well_metrics in per_well:
        if not well_metrics:
            continue
        info = well_metrics.get("skipped")
        if info:
            skipped.append(info)
    if not skipped:
        return {"count": 0}
    reasons: dict[str, int] = {}
    for info in skipped:
        reasons[str(info.get("reason", "unknown"))] = reasons.get(str(info.get("reason", "unknown")), 0) + 1
    return {
        "count": len(skipped),
        "by_reason": reasons,
        "example_well_ids": [str(s.get("well_id", "?")) for s in skipped[:10]],
    }


def _per_well_summary(per_well: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize per-well outcomes for the normal_GR variant.

    Returns counts of wells where (1) an envelope was produced, (2) the
    envelope_mid beat the shuffled-variant envelope_mid (RMSE comparison),
    plus min/max/median step_rmse buckets. Used as a quick health signal
    that the verdict line cannot show.
    """
    n_total = len(per_well)
    n_skipped = 0
    n_normal_env = 0
    n_shuffled_env = 0
    n_normal_better_than_shuffled = 0
    normal_rmses: list[float] = []
    shuffled_rmses: list[float] = []
    normal_top_scores: list[float] = []
    selected_counts: list[int] = []
    for well in per_well:
        if not well:
            continue
        if "skipped" in well:
            n_skipped += 1
            continue
        normal = well.get("normal_GR")
        shuffled = well.get("shuffled_hidden_GR")
        if normal and normal.get("candidate_rows", 0) > 0:
            n_normal_env += 1
            r = float(normal.get("step_rmse", float("nan")))
            if np.isfinite(r):
                normal_rmses.append(r)
            s = float(normal.get("top_template_score", float("nan")))
            if np.isfinite(s):
                normal_top_scores.append(s)
            selected_counts.append(int(normal.get("selected_templates", 0)))
        if shuffled and shuffled.get("candidate_rows", 0) > 0:
            n_shuffled_env += 1
            r = float(shuffled.get("step_rmse", float("nan")))
            if np.isfinite(r):
                shuffled_rmses.append(r)
        if normal and shuffled:
            nr = float(normal.get("step_rmse", float("nan")))
            sr = float(shuffled.get("step_rmse", float("nan")))
            if np.isfinite(nr) and np.isfinite(sr) and nr < sr:
                n_normal_better_than_shuffled += 1

    def _stat(values: list[float]) -> dict[str, float]:
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0:
            return {"n": 0}
        return {
            "n": int(arr.size),
            "min": float(arr.min()),
            "median": float(np.median(arr)),
            "mean": float(arr.mean()),
            "max": float(arr.max()),
        }

    return {
        "wells_total": int(n_total),
        "wells_skipped": int(n_skipped),
        "wells_with_normal_envelope": int(n_normal_env),
        "wells_with_shuffled_envelope": int(n_shuffled_env),
        "wells_normal_rmse_lt_shuffled_rmse": int(n_normal_better_than_shuffled),
        "normal_step_rmse_stats": _stat(normal_rmses),
        "shuffled_step_rmse_stats": _stat(shuffled_rmses),
        "normal_top_score_stats": _stat(normal_top_scores),
        "selected_templates_per_well_stats": _stat([float(x) for x in selected_counts]),
    }


def _sanity_summary(variants: dict[str, Any]) -> dict[str, Any]:
    """Compute a one-line normal-vs-shuffled-vs-zero verdict.

    Returns a small dict with each variant's top_template_score and step_rmse
    plus boolean gates that the next-step decision tree depends on.
    """
    if not variants:
        return {"status": "no_variants"}
    normal = variants.get("normal_GR", {})
    shuffled = variants.get("shuffled_hidden_GR", {})
    zero = variants.get("zero_hidden_GR", {})

    def _get(d: dict[str, Any], key: str) -> float:
        try:
            return float(d.get(key, float("nan")))
        except Exception:
            return float("nan")

    normal_top = _get(normal, "top_template_score")
    shuffled_top = _get(shuffled, "top_template_score")
    zero_top = _get(zero, "top_template_score")
    normal_rmse = _get(normal, "step_rmse")
    shuffled_rmse = _get(shuffled, "step_rmse")

    def _gt(a: float, b: float) -> bool:
        return bool(np.isfinite(a) and np.isfinite(b) and a > b)

    def _lt(a: float, b: float) -> bool:
        return bool(np.isfinite(a) and np.isfinite(b) and a < b)

    return {
        "normal_top_score": normal_top,
        "shuffled_top_score": shuffled_top,
        "zero_top_score": zero_top,
        "normal_step_rmse": normal_rmse,
        "shuffled_step_rmse": shuffled_rmse,
        "normal_beats_shuffled_score": _gt(normal_top, shuffled_top),
        "normal_beats_zero_score": _gt(normal_top, zero_top),
        "normal_beats_shuffled_rmse": _lt(normal_rmse, shuffled_rmse),
        "verdict": (
            "GO_to_full_run"
            if _gt(normal_top, shuffled_top) and _lt(normal_rmse, shuffled_rmse)
            else "NO_GO_no_GR_signal"
        ),
    }


def _write_report(metrics: dict[str, Any], output_dir: Path) -> None:
    lines = [
        "# TEMPLATE_ENVELOPE_V0_REPORT",
        "",
        "summary:",
        "```json",
        json.dumps(_json_safe(metrics), indent=2, ensure_ascii=False),
        "```",
        "",
        "interpretation:",
        "",
        "- This is a corridor/candidate generator, not a standalone trajectory model.",
        "- `envelope_candidates.parquet` is intentionally schema-safe: no hidden target columns are stored.",
        "- Trust only variants where normal GR beats shuffled/zero GR on the same metric.",
        "- See `sanity` for the normal-vs-shuffled GO/NO-GO verdict.",
        "- See `skipped_wells` for wells with constant coord proxies; revisit `coord_source` if many wells skip.",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run_template_envelope(config: TemplateEnvelopeConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wells = discover_wells(Path(config.data_dir), k_wells=config.k_wells)
    rng = np.random.default_rng(config.seed)

    all_scores: list[pd.DataFrame] = []
    all_steps: list[pd.DataFrame] = []
    normal_candidates: list[pd.DataFrame] = []
    all_candidate_variants: list[pd.DataFrame] = []
    per_well_metrics: list[dict[str, Any]] = []
    processed = 0

    for processed, well in enumerate(wells, start=1):
        horizontal, typewell = load_well_pair(well)
        scores, steps, candidates, well_metrics = _score_one_well(
            horizontal,
            typewell,
            config=config,
            rng=rng,
        )
        if not scores.empty:
            all_scores.append(scores)
        if not steps.empty:
            all_steps.append(steps)
        if not candidates.empty:
            all_candidate_variants.append(candidates)
            normal = candidates[~candidates["candidate"].astype(str).str.endswith(("_shuffled_hidden_GR", "_zero_hidden_GR"))]
            normal_candidates.append(normal)
        per_well_metrics.append(well_metrics)
        if config.progress_every > 0 and processed % config.progress_every == 0:
            print(f"[template-envelope] processed wells={processed}/{len(wells)}", file=sys.stderr, flush=True)

    scores_frame = pd.concat(all_scores, ignore_index=True) if all_scores else pd.DataFrame()
    steps_frame = pd.concat(all_steps, ignore_index=True) if all_steps else pd.DataFrame()
    candidates_frame = pd.concat(normal_candidates, ignore_index=True) if normal_candidates else pd.DataFrame()
    candidate_variants_frame = (
        pd.concat(all_candidate_variants, ignore_index=True) if all_candidate_variants else pd.DataFrame()
    )

    # Schema-safety guard: deployable envelope_candidates.parquet must never
    # carry hidden inference columns. Diagnostic variants are guarded too.
    if not candidates_frame.empty:
        assert_schema_safe_columns(candidates_frame.columns)
    if not candidate_variants_frame.empty:
        assert_schema_safe_columns(candidate_variants_frame.columns)

    scores_frame.to_parquet(output_dir / "step_scores.parquet", index=False)
    steps_frame.to_parquet(output_dir / "envelope_steps.parquet", index=False)
    candidates_frame.to_parquet(output_dir / "envelope_candidates.parquet", index=False)
    candidate_variants_frame.to_parquet(output_dir / "envelope_candidate_variants.parquet", index=False)

    sanity = _sanity_summary(_aggregate_variant_metrics(per_well_metrics))
    per_well_summary = _per_well_summary(per_well_metrics)

    metrics = {
        "wells": int(len(wells)),
        "processed_wells": int(processed),
        "rows_per_step": int(config.rows_per_step),
        "coord_source": config.coord_source,
        "min_template_score": float(config.min_template_score),
        "require_beats_shuffled": bool(config.require_beats_shuffled),
        "scale_grid": list(config.scale_grid),
        "offset_grid": list(config.offset_grid),
        "top_n_templates": int(config.top_n_templates),
        "variants": _aggregate_variant_metrics(per_well_metrics),
        "per_well_summary": per_well_summary,
        "skipped_wells": _aggregate_skip_metrics(per_well_metrics),
        "sanity": sanity,
        "artifacts": {
            "step_scores": str(output_dir / "step_scores.parquet"),
            "envelope_steps": str(output_dir / "envelope_steps.parquet"),
            "envelope_candidates": str(output_dir / "envelope_candidates.parquet"),
            "envelope_candidate_variants": str(output_dir / "envelope_candidate_variants.parquet"),
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(_json_safe(metrics), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_report(metrics, output_dir)
    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run template-envelope v0 corridor generator")
    parser.add_argument("--data-dir", type=Path, default=TemplateEnvelopeConfig.data_dir)
    parser.add_argument("--output-dir", type=Path, default=TemplateEnvelopeConfig.output_dir)
    parser.add_argument("--rows-per-step", type=int, default=TemplateEnvelopeConfig.rows_per_step)
    parser.add_argument("--k-wells", type=int, default=TemplateEnvelopeConfig.k_wells)
    parser.add_argument("--seed", type=int, default=TemplateEnvelopeConfig.seed)
    parser.add_argument("--scale-grid", type=str, default=",".join(str(x) for x in TemplateEnvelopeConfig.scale_grid))
    parser.add_argument("--offset-grid", type=str, default=",".join(str(x) for x in TemplateEnvelopeConfig.offset_grid))
    parser.add_argument("--top-n-templates", type=int, default=TemplateEnvelopeConfig.top_n_templates)
    parser.add_argument("--progress-every", type=int, default=TemplateEnvelopeConfig.progress_every)
    parser.add_argument(
        "--coord-source",
        type=str,
        default=TemplateEnvelopeConfig.coord_source,
        choices=list(VALID_COORD_SOURCES),
        help="Coordinate proxy used as `tvt` in a*tvt+offset template paths.",
    )
    parser.add_argument(
        "--min-template-score",
        type=float,
        default=TemplateEnvelopeConfig.min_template_score,
        help="Minimum scoring threshold for a template to be eligible for selection.",
    )
    parser.add_argument(
        "--require-beats-shuffled",
        action="store_true",
        default=TemplateEnvelopeConfig.require_beats_shuffled,
        help="Additionally require normal-variant templates to beat the shuffled-variant median score.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    config = TemplateEnvelopeConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        rows_per_step=args.rows_per_step,
        k_wells=args.k_wells,
        seed=args.seed,
        scale_grid=parse_float_grid(args.scale_grid),
        offset_grid=parse_float_grid(args.offset_grid),
        top_n_templates=args.top_n_templates,
        progress_every=args.progress_every,
        coord_source=args.coord_source,
        min_template_score=args.min_template_score,
        require_beats_shuffled=args.require_beats_shuffled,
    )
    metrics = run_template_envelope(config)
    print(json.dumps(_json_safe(metrics), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

