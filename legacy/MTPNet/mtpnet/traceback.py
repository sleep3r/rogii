from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .residual_stack import ResidualStackConfig, load_training_frame
from .traceback_candidates import (
    build_traceback_bands,
    build_traceback_candidates_with_offsets,
    evaluate_traceback_candidate_oracle,
    matches_to_anchors,
)
from .traceback_dictionary import (
    build_fold_safe_train_dictionary,
    build_same_well_dictionary,
    build_typewell_dictionary,
)
from .traceback_events import TracebackConfig, compress_well_rows, extract_traceback_events
from .traceback_match import (
    evaluate_event_matches,
    make_sanity_events,
    score_event_against_dictionary,
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _first_typewell_from_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if {"TVT", "GR"}.issubset(frame.columns):
        valid = frame[pd.to_numeric(frame["TVT"], errors="coerce").notna()].copy()
        return valid[["TVT", "GR"]].dropna(subset=["TVT"]).sort_values("TVT")
    return pd.DataFrame({"TVT": [], "GR": []})


def _write_parquet_safe(frame: pd.DataFrame, path: Path) -> None:
    out = frame.copy()
    for column in out.columns:
        if len(out) and isinstance(out[column].iloc[0], np.ndarray):
            out[column] = out[column].map(lambda arr: np.asarray(arr).tolist())
    out.to_parquet(path, index=False)


def _log_progress(message: str) -> None:
    print(f"[traceback] {message}", flush=True)


def add_known_tvt_bridge(comp: pd.DataFrame) -> pd.DataFrame:
    out = comp.copy()
    bridge_values = []
    for _, well in out.groupby("well_id", sort=False):
        well = well.sort_values("step")
        steps = pd.to_numeric(well["step"], errors="coerce").to_numpy(dtype=np.float64)
        tvt_input = pd.to_numeric(well["TVT_input"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        finite = np.isfinite(steps) & np.isfinite(tvt_input)
        if finite.any():
            bridge = np.interp(steps, steps[finite], tvt_input[finite])
        else:
            bridge = np.full(len(well), np.nan, dtype=np.float64)
        bridge_values.append(pd.Series(bridge, index=well.index))
    if bridge_values:
        out["bridge_TVT"] = pd.concat(bridge_values).sort_index()
    else:
        out["bridge_TVT"] = np.nan
    return out


def run_traceback_from_frames(
    frame: pd.DataFrame,
    *,
    typewell: pd.DataFrame,
    output_dir: Path,
    k_wells: int = -1,
    seed: int = 42,
    cfg: TracebackConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or TracebackConfig(seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    wells = sorted(frame["well_id"].astype(str).unique())
    if k_wells > 0:
        wells = wells[:k_wells]
    frame = frame[frame["well_id"].astype(str).isin(wells)].copy()
    _log_progress(f"start wells={len(wells)} rows={len(frame)}")

    comp_parts = []
    event_parts = []
    for idx, (well_id, well) in enumerate(frame.groupby("well_id", sort=False), start=1):
        comp = compress_well_rows(well, rows_per_step=cfg.rows_per_step)
        comp = add_known_tvt_bridge(comp)
        comp_parts.append(comp)
        event_parts.append(extract_traceback_events(comp, cfg))
        if idx == 1 or idx % 25 == 0 or idx == len(wells):
            _log_progress(f"compressed/events wells={idx}/{len(wells)}")
    comp_all = pd.concat(comp_parts, ignore_index=True) if comp_parts else pd.DataFrame()
    events_all = pd.concat(event_parts, ignore_index=True) if event_parts else pd.DataFrame()
    _log_progress(f"compressed_steps={len(comp_all)} events={len(events_all)}")

    same_dict = build_same_well_dictionary(events_all)
    train_dict = (
        build_fold_safe_train_dictionary(events_all, train_wells=wells)
        if cfg.use_train_dictionary
        else pd.DataFrame()
    )
    typewell_dict = build_typewell_dictionary(
        typewell,
        vertical_step_ft=cfg.vertical_step_ft,
        patch_radii=cfg.patch_radii,
        min_prominence_z=cfg.min_prominence_z,
    )
    dictionary = pd.concat([same_dict, typewell_dict, train_dict], ignore_index=True)
    _log_progress(
        "dictionary "
        f"same_known={len(same_dict)} typewell={len(typewell_dict)} "
        f"train={len(train_dict)} total={len(dictionary)}"
    )

    match_events = (
        events_all[~events_all["known_mask"].astype(bool)].copy()
        if not events_all.empty and "known_mask" in events_all.columns
        else events_all
    )
    if not match_events.empty and "bridge_TVT" not in match_events.columns:
        bridge_lookup = comp_all[["well_id", "step", "bridge_TVT"]].drop_duplicates(
            ["well_id", "step"]
        )
        match_events = match_events.merge(
            bridge_lookup, on=["well_id", "step"], how="left"
        )
    _log_progress(f"match_events_hidden={len(match_events)}")
    variants = make_sanity_events(match_events, seed=seed) if not match_events.empty else {}
    variant_metrics: dict[str, Any] = {
        name: evaluate_event_matches(pd.DataFrame())
        for name in ("normal_GR", "shuffled_hidden_GR", "zero_hidden_GR")
    }
    match_parts = []
    for variant_name, variant_events in variants.items():
        _log_progress(f"match variant={variant_name} events={len(variant_events)}")
        rows = []
        for event_idx, (_, event) in enumerate(variant_events.iterrows(), start=1):
            scored = score_event_against_dictionary(
                event,
                dictionary,
                location_weight=cfg.location_weight,
                bridge_tvt=(
                    float(event["bridge_TVT"])
                    if "bridge_TVT" in event and np.isfinite(event["bridge_TVT"])
                    else None
                ),
                location_sigma_ft=cfg.location_sigma_ft,
                top_k=cfg.top_k_matches,
                max_candidates=cfg.max_match_candidates,
            )
            if scored.empty:
                continue
            scored["variant"] = variant_name
            scored["event_bridge_tvt"] = (
                float(event["bridge_TVT"])
                if "bridge_TVT" in event and np.isfinite(event["bridge_TVT"])
                else np.nan
            )
            scored["true_TVT"] = (
                float(event["true_TVT"])
                if "true_TVT" in event and np.isfinite(event["true_TVT"])
                else np.nan
            )
            rows.append(scored)
            if cfg.progress_every > 0 and event_idx % cfg.progress_every == 0:
                _log_progress(
                    f"matched variant={variant_name} events={event_idx}/{len(variant_events)}"
                )
        matches = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        variant_metrics[variant_name] = evaluate_event_matches(matches)
        _log_progress(
            f"done variant={variant_name} matches={len(matches)} "
            f"metrics={variant_metrics[variant_name]}"
        )
        if not matches.empty:
            match_parts.append(matches)
    matches_all = pd.concat(match_parts, ignore_index=True) if match_parts else pd.DataFrame()

    if not matches_all.empty:
        normal_matches = matches_all[matches_all["variant"] == "normal_GR"].copy()
    else:
        normal_matches = pd.DataFrame()
    anchors = (
        matches_to_anchors(
            normal_matches,
            min_score_quantile=cfg.anchor_min_score_quantile,
            min_top1_gap=cfg.anchor_min_top1_gap,
            max_bridge_delta_ft=cfg.anchor_max_bridge_delta_ft,
        )
        if not normal_matches.empty
        else pd.DataFrame()
    )
    bands = build_traceback_bands(comp_all, anchors) if not comp_all.empty else pd.DataFrame()
    hidden = frame[pd.to_numeric(frame["TVT_input"], errors="coerce").isna()].copy()
    if not hidden.empty and "bridge_TVT" not in hidden.columns:
        bridge_rows = []
        for _, comp in comp_all.groupby("well_id", sort=False):
            step_bridge = comp[["well_id", "step", "bridge_TVT"]].copy()
            bridge_rows.append(step_bridge)
        if bridge_rows:
            bridge_frame = pd.concat(bridge_rows, ignore_index=True).drop_duplicates(
                ["well_id", "step"]
            )
            if "step" not in hidden:
                hidden["step"] = hidden["row_idx"] // cfg.rows_per_step
            hidden = hidden.merge(bridge_frame, on=["well_id", "step"], how="left")
    candidates = (
        build_traceback_candidates_with_offsets(
            hidden,
            bands,
            offset_fracs=(-1.0, -0.5, 0.0, 0.5, 1.0),
        )
        if not hidden.empty
        else pd.DataFrame()
    )
    _log_progress(
        f"anchors={len(anchors)} bands={len(bands)} candidates={len(candidates)}"
    )
    oracle_metrics = (
        evaluate_traceback_candidate_oracle(hidden, candidates)
        if {"TVT", "b2_tvt"}.issubset(hidden.columns)
        else {}
    )

    _write_parquet_safe(events_all, output_dir / "traceback_events.parquet")
    _write_parquet_safe(dictionary, output_dir / "traceback_dictionary.parquet")
    _write_parquet_safe(matches_all, output_dir / "traceback_event_matches.parquet")
    bands.to_parquet(output_dir / "traceback_bands.parquet", index=False)
    candidates.to_parquet(output_dir / "traceback_candidates.parquet", index=False)
    pd.DataFrame([oracle_metrics]).to_csv(
        output_dir / "traceback_candidate_oracle.csv", index=False
    )

    metrics = {
        "wells": int(len(wells)),
        "compressed_steps": int(len(comp_all)),
        "events": int(len(events_all)),
        "dictionary_entries": int(len(dictionary)),
        "matches": int(len(matches_all)),
        "candidate_rows": int(len(candidates)),
        "variants": variant_metrics,
        "candidate_oracle": oracle_metrics,
    }
    (output_dir / "traceback_metrics.json").write_text(
        json.dumps(_json_safe(metrics), indent=2), encoding="utf-8"
    )
    report = [
        "TRACEBACK_V0_REPORT",
        "",
        "summary:",
        "```json",
        json.dumps(_json_safe(metrics), indent=2),
        "```",
    ]
    (output_dir / "traceback_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/traceback_v0"))
    parser.add_argument("--rows-per-step", type=int, default=32)
    parser.add_argument("--vertical-step-ft", type=float, default=5.0)
    parser.add_argument("--patch-radii", default="3,5,9,15")
    parser.add_argument("--k-wells", type=int, default=-1)
    parser.add_argument("--max-match-candidates", type=int, default=2000)
    parser.add_argument("--use-train-dictionary", action="store_true")
    parser.add_argument("--location-weight", type=float, default=1.0)
    parser.add_argument("--location-sigma-ft", type=float, default=120.0)
    parser.add_argument("--anchor-min-score-quantile", type=float, default=0.8)
    parser.add_argument("--anchor-min-top1-gap", type=float, default=0.05)
    parser.add_argument("--anchor-max-bridge-delta-ft", type=float, default=160.0)
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frame = load_training_frame(
        ResidualStackConfig(data_dir=args.data_dir, k_wells=args.k_wells)
    )
    typewell = _first_typewell_from_frame(frame)
    cfg = TracebackConfig(
        rows_per_step=args.rows_per_step,
        vertical_step_ft=args.vertical_step_ft,
        patch_radii=tuple(int(v) for v in args.patch_radii.split(",") if v),
        max_match_candidates=args.max_match_candidates,
        use_train_dictionary=args.use_train_dictionary,
        location_weight=args.location_weight,
        location_sigma_ft=args.location_sigma_ft,
        anchor_min_score_quantile=args.anchor_min_score_quantile,
        anchor_min_top1_gap=args.anchor_min_top1_gap,
        anchor_max_bridge_delta_ft=args.anchor_max_bridge_delta_ft,
        progress_every=args.progress_every,
        seed=args.seed,
    )
    metrics = run_traceback_from_frames(
        frame,
        typewell=typewell,
        output_dir=args.output_dir,
        k_wells=args.k_wells,
        seed=args.seed,
        cfg=cfg,
    )
    print(json.dumps(_json_safe(metrics), indent=2))


if __name__ == "__main__":
    main()
