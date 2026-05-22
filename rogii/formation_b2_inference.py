from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .formation_b_lite import _prediction_from_candidate_map
from .formation_b_lite import add_b_scores, compute_raw_scores
from .formation_b2_constrained import _read_frame, _safe_median, _safe_percentile
from .formation_b2_constrained import build_b2_metadata
from .formation_b2_guarded import (
    _alpha_summary,
    _apply_policy,
    _candidate_map,
    _choice_for_selector,
    _make_context,
    _score,
    selector_diagnostics,
)
from .formation_plane_knn import (
    FormationPlaneConfig,
    FormationPlaneKNN,
    NearbyPathLibrary,
    build_well_candidates,
)
from .formation_plane_knn import json_safe, markdown_table, write_frame
from .formation_selector import attach_schema10
from .io import horizontal_files


@dataclass(frozen=True)
class B2InferenceConfig:
    policy_name: str = "B_among_A_top10__danger_current_kill_ge4"
    selector: str = "B_among_A_top10"
    mode: str = "danger_current"
    action: str = "kill_ge4"
    alpha: float = 0.30
    clip_high: float = 30.0
    clip_low: float = 15.0
    delta_p95_downgrade: float = 20.0
    endpoint_downgrade: float = 25.0
    kill_threshold: float = 4.0
    boost_max_danger: float = 1.0


def load_b2_config(path: Path | None) -> B2InferenceConfig:
    if path is None:
        return B2InferenceConfig()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    selector_cfg = data.get("b_selector", {}) or {}
    safe_cfg = data.get("safe_blend", {}) or {}
    guard_cfg = data.get("danger_guard", {}) or {}
    policy_name = str(data.get("policy_name", B2InferenceConfig.policy_name))
    selector = str(selector_cfg.get("selector", B2InferenceConfig.selector))
    action = str(guard_cfg.get("current_policy", guard_cfg.get("action", "kill_ge4")))
    mode = str(guard_cfg.get("mode", "danger_current"))
    clip = safe_cfg.get("clip", B2InferenceConfig.clip_high)
    return B2InferenceConfig(
        policy_name=policy_name,
        selector=selector,
        mode=mode,
        action=action,
        alpha=float(safe_cfg.get("alpha", B2InferenceConfig.alpha)),
        clip_high=float(guard_cfg.get("clip_high", clip)),
        clip_low=float(guard_cfg.get("clip_low", B2InferenceConfig.clip_low)),
        delta_p95_downgrade=float(
            guard_cfg.get("delta_p95_downgrade", B2InferenceConfig.delta_p95_downgrade)
        ),
        endpoint_downgrade=float(
            guard_cfg.get("endpoint_downgrade", B2InferenceConfig.endpoint_downgrade)
        ),
        kill_threshold=float(guard_cfg.get("kill_threshold", B2InferenceConfig.kill_threshold)),
        boost_max_danger=float(
            guard_cfg.get("boost_max_danger", B2InferenceConfig.boost_max_danger)
        ),
    )


def load_formation_config(path: Path | None) -> FormationPlaneConfig:
    default = FormationPlaneConfig()
    if path is None:
        return default
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg = data.get("formation_a", {}) or {}
    return FormationPlaneConfig(
        k_wells=int(cfg.get("k_wells", default.k_wells)),
        sample_rows_per_well=int(
            cfg.get("sample_rows_per_well", default.sample_rows_per_well)
        ),
        min_points=int(cfg.get("min_points", default.min_points)),
        dense_k=int(cfg.get("dense_k", default.dense_k)),
        weight_power=float(cfg.get("weight_power", default.weight_power)),
        eps=float(cfg.get("eps", default.eps)),
        bootstrap_samples=int(cfg.get("bootstrap_samples", default.bootstrap_samples)),
        bootstrap_fraction=float(
            cfg.get("bootstrap_fraction", default.bootstrap_fraction)
        ),
        query_chunk=int(cfg.get("query_chunk", default.query_chunk)),
        seed=int(cfg.get("seed", default.seed)),
    )


def _policy_from_config(config: B2InferenceConfig) -> dict[str, Any]:
    return {
        "policy_name": config.policy_name,
        "base_selector": config.selector,
        "mode": config.mode,
        "action": config.action,
        "alpha": config.alpha,
        "clip": config.clip_high,
        "clip_high": config.clip_high,
        "clip_low": config.clip_low,
        "delta_p95_downgrade": config.delta_p95_downgrade,
        "endpoint_downgrade": config.endpoint_downgrade,
        "kill_threshold": config.kill_threshold,
        "boost_max_danger": config.boost_max_danger,
    }


def _metric_summary(frame: pd.DataFrame, pred: np.ndarray, base: np.ndarray) -> dict[str, Any]:
    ctx = _make_context(frame)
    score = _score(pred, ctx)
    base_score = _score(base, ctx)
    shift = np.abs(pred - base)
    score.update(
        {
            "base_rmse": base_score["rmse"],
            "gain": base_score["rmse"] - score["rmse"],
            "median_shift": _safe_median(shift),
            "p95_shift": _safe_percentile(shift, 95),
            "max_shift": _safe_percentile(shift, 100),
        }
    )
    return score


def apply_b2_guarded_correction(
    frame: pd.DataFrame,
    choices: pd.DataFrame,
    metadata: pd.DataFrame,
    config: B2InferenceConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    cfg = config or B2InferenceConfig()
    if "schema10_oof_raw" not in frame.columns:
        raise ValueError("Frame must contain schema10_oof_raw before B2 correction.")
    ctx = _make_context(frame)
    choice = _choice_for_selector(choices, cfg.selector)
    candidate_map = _candidate_map(choice)
    selected = _prediction_from_candidate_map(frame, candidate_map)
    raw_delta = selected - ctx.base
    diagnostics = selector_diagnostics(frame, metadata, choices, cfg.selector, selected, ctx)
    pred, shift, alpha, clip = _apply_policy(
        ctx.base,
        raw_delta,
        ctx,
        diagnostics,
        _policy_from_config(cfg),
    )
    diag_by_well = diagnostics.set_index("well_id")
    out = pd.DataFrame(
        {
            "id": frame["id"].astype(str).to_numpy(),
            "well_id": frame["well_id"].astype(str).to_numpy(),
            "b2_base_tvt": ctx.base,
            "b2_selected_A_path": selected,
            "b2_selected_delta": raw_delta,
            "b2_guarded_submit": pred,
            "b2_submit_guarded_shift": shift,
            "b2_submit_blend_delta": pred - ctx.base,
            "b2_submit_alpha_final": alpha,
            "b2_submit_clip_final": clip,
            "b2_submit_policy_name": np.full(len(frame), cfg.policy_name, dtype=object),
            "b2_submit_danger_score": frame["well_id"].map(diag_by_well["danger_score"]).to_numpy(
                dtype=float
            ),
            "b2_submit_shift_p95": frame["well_id"].map(diag_by_well["delta_abs_p95"]).to_numpy(
                dtype=float
            ),
            "b2_submit_endpoint_shift": frame["well_id"]
            .map(diag_by_well["delta_endpoint"])
            .to_numpy(dtype=float),
        }
    )
    if "TVT" in frame.columns:
        out["TVT"] = pd.to_numeric(frame["TVT"], errors="coerce").to_numpy(dtype=float)
    summary = _metric_summary(frame, pred, ctx.base)
    summary.update(_alpha_summary(alpha, ctx))
    summary["policy_name"] = cfg.policy_name
    summary["selector"] = cfg.selector
    return out, diagnostics, summary


def _compare_reference(
    predictions: pd.DataFrame,
    reference_path: Path | None,
    *,
    reference_column: str,
    tolerance: float,
) -> dict[str, Any]:
    if reference_path is None:
        return {"available": False}
    reference = _read_frame(reference_path)
    if reference_column not in reference.columns:
        raise ValueError(f"Reference column not found: {reference_column}")
    ref = reference[["id", reference_column]].rename(columns={reference_column: "_reference"})
    merged = predictions[["id", "b2_guarded_submit"]].merge(ref, on="id", how="left")
    diff = np.abs(
        pd.to_numeric(merged["b2_guarded_submit"], errors="coerce").to_numpy(dtype=float)
        - pd.to_numeric(merged["_reference"], errors="coerce").to_numpy(dtype=float)
    )
    finite = diff[np.isfinite(diff)]
    max_abs = float(np.nanmax(finite)) if len(finite) else float("nan")
    return {
        "available": True,
        "reference": str(reference_path),
        "reference_column": reference_column,
        "rows_compared": int(len(merged)),
        "missing_reference_rows": int(merged["_reference"].isna().sum()),
        "max_abs_diff": max_abs,
        "mean_abs_diff": float(np.nanmean(finite)) if len(finite) else float("nan"),
        "p99_abs_diff": _safe_percentile(finite, 99),
        "passed": bool(np.isfinite(max_abs) and max_abs <= float(tolerance)),
        "tolerance": float(tolerance),
    }


def _alpha_distribution(predictions: pd.DataFrame) -> dict[str, Any]:
    alpha = pd.to_numeric(predictions["b2_submit_alpha_final"], errors="coerce")
    return {
        "alpha_min": float(alpha.min()),
        "alpha_p50": float(alpha.quantile(0.50)),
        "alpha_p95": float(alpha.quantile(0.95)),
        "alpha_max": float(alpha.max()),
        "disabled_rows": int((alpha <= 1e-9).sum()),
    }


def write_report(
    output_dir: Path,
    *,
    config: B2InferenceConfig,
    summary: dict[str, Any],
    parity: dict[str, Any],
    diagnostics: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    focus = diagnostics[diagnostics["well_id"] == "389ae58f"]
    lines = [
        "# B2 Inference Parity Report",
        "## Config",
        "```json\n" + json.dumps(json_safe(config.__dict__), indent=2) + "\n```",
        "## OOF Replay",
        markdown_table(pd.DataFrame([summary])),
        "## Parity",
        markdown_table(pd.DataFrame([parity])),
        "## Alpha Distribution",
        markdown_table(pd.DataFrame([_alpha_distribution(predictions)])),
        "## 389ae58f",
        markdown_table(focus),
        "## Decision",
        "`submit yes` only if parity passes and the reference column is `b2_guarded_submit`.",
    ]
    (output_dir / "B2_INFERENCE_PARITY_REPORT.md").write_text(
        "\n\n".join(lines),
        encoding="utf-8",
    )


def build_test_a_candidates(
    *,
    data_dir: Path,
    train_dir: Path,
    test_dir: Path,
    config: FormationPlaneConfig,
    progress_interval: int = 1,
) -> pd.DataFrame:
    train_paths = horizontal_files(train_dir, None)
    test_paths = horizontal_files(test_dir, None)
    if not train_paths:
        raise FileNotFoundError(f"No train horizontal wells found under {train_dir}")
    if not test_paths:
        raise FileNotFoundError(f"No test horizontal wells found under {test_dir}")
    print(
        f"b2 test build A context train_wells={len(train_paths)} test_wells={len(test_paths)}",
        flush=True,
    )
    solver = FormationPlaneKNN.from_paths(train_paths, config)
    nearby = NearbyPathLibrary.from_paths(train_paths, weight_power=config.weight_power)
    rows: list[pd.DataFrame] = []
    for idx, path in enumerate(test_paths, start=1):
        if progress_interval > 0 and (
            idx == 1 or idx % int(progress_interval) == 0 or idx == len(test_paths)
        ):
            print(f"b2 test A candidates well={idx}/{len(test_paths)} path={path.name}", flush=True)
        frame = build_well_candidates(
            path,
            solver,
            nearby,
            fold_id=0,
            seed=int(config.seed),
        )
        if not frame.empty:
            rows.append(frame)
    if not rows:
        raise RuntimeError("B2 test A candidate generation produced no hidden rows.")
    out = pd.concat(rows, ignore_index=True)
    out["TVT"] = np.nan
    return out


def build_inference_choices(metadata: pd.DataFrame) -> pd.DataFrame:
    selector_frames: list[pd.DataFrame] = []
    for top_k in (10, 20, 50):
        choice = (
            metadata[metadata["a_rank"] <= top_k]
            .sort_values(["well_id", "b_combined_score", "candidate_name"])
            .groupby("well_id", sort=False)
            .head(1)
            .reset_index(drop=True)
        )
        choice["selector_name"] = f"B_among_A_top{top_k}"
        selector_frames.append(choice)
    for top_k in (10, 20):
        choice = (
            metadata[metadata["b_rank"] <= top_k]
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
        work = metadata.copy()
        work["_combined_rank"] = a_weight * work["a_rank"] + b_weight * work["b_rank"]
        choice = (
            work.sort_values(["well_id", "_combined_rank", "candidate_name"])
            .groupby("well_id", sort=False)
            .head(1)
            .reset_index(drop=True)
        )
        choice["selector_name"] = name
        selector_frames.append(choice)
    return pd.concat(selector_frames, ignore_index=True)


def _write_submission(predictions: pd.DataFrame, output_dir: Path) -> Path:
    submission = predictions[["id", "b2_guarded_submit"]].rename(
        columns={"b2_guarded_submit": "tvt"}
    )
    path = output_dir / "submission.csv"
    submission.to_csv(path, index=False)
    return path


def _shift_sanity(predictions: pd.DataFrame) -> dict[str, Any]:
    base = pd.to_numeric(predictions["b2_base_tvt"], errors="coerce").to_numpy(dtype=float)
    pred = pd.to_numeric(predictions["b2_guarded_submit"], errors="coerce").to_numpy(dtype=float)
    shift = np.abs(pred - base)
    return {
        "nan_predictions": int((~np.isfinite(pred)).sum()),
        "prediction_min": float(np.nanmin(pred)),
        "prediction_max": float(np.nanmax(pred)),
        "base_min": float(np.nanmin(base)),
        "base_max": float(np.nanmax(base)),
        "median_shift": _safe_median(shift),
        "p95_shift": _safe_percentile(shift, 95),
        "max_shift": _safe_percentile(shift, 100),
    }


def run_test_inference(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_b2_config(Path(args.config) if args.config else None)
    formation_config = load_formation_config(Path(args.config) if args.config else None)
    data_dir = Path(args.data_dir)
    train_dir = Path(args.train_dir) if args.train_dir else data_dir / "train"
    test_dir = Path(args.test_dir) if args.test_dir else data_dir / "test"
    frame = build_test_a_candidates(
        data_dir=data_dir,
        train_dir=train_dir,
        test_dir=test_dir,
        config=formation_config,
        progress_interval=int(args.progress_interval),
    )
    frame, baseline_available = attach_schema10(
        frame,
        Path(args.base_submission),
        schema10_column=args.base_column,
    )
    if not baseline_available:
        raise ValueError("B2 test inference requires base submission predictions.")
    print("b2 test B scores", flush=True)
    raw_scores, diagnostic_scores, info = compute_raw_scores(
        frame,
        data_dir=data_dir,
        train_dir=test_dir,
        progress_interval=int(args.progress_interval),
    )
    b_scores = add_b_scores(raw_scores)
    print("b2 test metadata", flush=True)
    metadata = build_b2_metadata(
        frame,
        b_scores,
        baseline_available=True,
        progress_interval=int(args.progress_interval),
    )
    choices = build_inference_choices(metadata)
    predictions, diagnostics, summary = apply_b2_guarded_correction(
        frame,
        choices,
        metadata,
        config,
    )
    submission_path = _write_submission(predictions, output_dir)
    write_frame(frame, output_dir / "test_a_candidates.parquet")
    write_frame(b_scores, output_dir / "test_b_candidate_scores.parquet")
    write_frame(metadata, output_dir / "test_b2_candidate_metadata.parquet")
    choices.to_csv(output_dir / "test_b2_selector_choices.csv", index=False)
    diagnostics.to_csv(output_dir / "test_b2_inference_diagnostics.csv", index=False)
    write_frame(predictions, output_dir / "test_b2_predictions.parquet")
    b_finite = pd.to_numeric(b_scores["b_finite_frac"], errors="coerce").to_numpy(dtype=float)
    metrics = {
        "rows": int(len(frame)),
        "wells": int(frame["well_id"].nunique()),
        "config": config.__dict__,
        "formation_config": formation_config.__dict__,
        "b_score_info": info,
        "candidate_typewell_finite_fraction": float(np.nanmean(b_finite)),
        "summary": summary,
        "shift_sanity": _shift_sanity(predictions),
        "submission": str(submission_path),
        "diagnostic_scores_rows": int(len(diagnostic_scores)),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(json_safe(metrics), indent=2),
        encoding="utf-8",
    )
    return metrics


def run_oof_replay(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_b2_config(Path(args.config) if args.config else None)
    frame = _read_frame(Path(args.input))
    frame, baseline_available = attach_schema10(
        frame,
        Path(args.schema10_oof) if args.schema10_oof else None,
        schema10_column=args.schema10_column,
    )
    if not baseline_available:
        raise ValueError("B2 inference replay requires a schema10/base OOF file.")
    choices = _read_frame(Path(args.choices))
    metadata = _read_frame(Path(args.metadata))
    predictions, diagnostics, summary = apply_b2_guarded_correction(
        frame,
        choices,
        metadata,
        config,
    )
    parity = _compare_reference(
        predictions,
        Path(args.reference) if args.reference else None,
        reference_column=str(args.reference_column),
        tolerance=float(args.tolerance),
    )
    write_frame(predictions, output_dir / "b2_inference_predictions.parquet")
    diagnostics.to_csv(output_dir / "b2_inference_diagnostics.csv", index=False)
    metrics = {
        "input": str(args.input),
        "rows": int(len(frame)),
        "wells": int(frame["well_id"].nunique()),
        "config": config.__dict__,
        "summary": summary,
        "parity": parity,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(json_safe(metrics), indent=2),
        encoding="utf-8",
    )
    write_report(
        output_dir,
        config=config,
        summary=summary,
        parity=parity,
        diagnostics=diagnostics,
        predictions=predictions,
    )
    if parity.get("available") and not parity.get("passed"):
        raise RuntimeError(
            f"B2 inference parity failed: max_abs_diff={parity.get('max_abs_diff')}"
        )
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B2 guarded inference/parity utilities.")
    sub = parser.add_subparsers(dest="command")
    replay = sub.add_parser("oof-replay", help="Replay fixed B2 guarded policy on OOF artifacts.")
    replay.add_argument("--input", type=Path, required=True)
    replay.add_argument("--choices", type=Path, required=True)
    replay.add_argument("--metadata", type=Path, required=True)
    replay.add_argument("--config", type=Path, default=Path("configs/formation_b2_guarded_submit.yml"))
    replay.add_argument("--output-dir", type=Path, default=Path("artifacts/formation_b2_infer_oof_replay"))
    replay.add_argument("--schema10-oof", type=Path, required=True)
    replay.add_argument("--schema10-column", type=str, default=None)
    replay.add_argument("--reference", type=Path, default=None)
    replay.add_argument("--reference-column", type=str, default="b2_guarded_submit")
    replay.add_argument("--tolerance", type=float, default=1e-5)
    test = sub.add_parser("test", help="Run fixed B2 guarded correction on hidden test candidates.")
    test.add_argument("--data-dir", type=Path, default=Path("data"))
    test.add_argument("--train-dir", type=Path, default=None)
    test.add_argument("--test-dir", type=Path, default=None)
    test.add_argument("--base-submission", type=Path, required=True)
    test.add_argument("--base-column", type=str, default="tvt")
    test.add_argument("--config", type=Path, default=Path("configs/formation_b2_guarded_submit.yml"))
    test.add_argument("--output-dir", type=Path, default=Path("artifacts/formation_b2_test_submit"))
    test.add_argument("--progress-interval", type=int, default=1)
    args = parser.parse_args(argv)
    if args.command is None:
        parser.error("Choose a command, e.g. oof-replay.")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "oof-replay":
        metrics = run_oof_replay(args)
    elif args.command == "test":
        metrics = run_test_inference(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")
    print(json.dumps(json_safe(metrics), indent=2), flush=True)


if __name__ == "__main__":
    main()
