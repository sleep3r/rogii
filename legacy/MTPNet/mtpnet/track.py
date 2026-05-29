from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import MTPConfig
from .ranker import apply_ranker_logits, normalize_mode_windows
from .stitch import (
    _apply_step_predictions_to_rows,
    _baseline_metrics,
    _blend_with_anchor,
    _load_hidden_rows,
    _load_run_config,
    _softmax_np,
    evaluate_with_b2_fallback,
)

MERGED_COUNT_CAP = 2_147_483_647


@dataclass(frozen=True)
class TrackConfig:
    n_realizations: int = 32
    keep_top: int = 32
    merge_tolerance_ft: float = 3.0
    overlap_penalty: float = 0.10
    max_modes_per_window: int = 8


@dataclass(frozen=True)
class TrackParticle:
    well_id: str
    steps: tuple[int, ...]
    tvt: tuple[float, ...]
    score: float
    lineage: tuple[str, ...]
    merged_count: int = 1

    @property
    def last_step(self) -> int:
        return int(self.steps[-1]) if self.steps else -1

    @property
    def endpoint(self) -> float:
        return float(self.tvt[-1]) if self.tvt else float("nan")


def _log_probs(logits: np.ndarray) -> np.ndarray:
    probs = _softmax_np(np.asarray(logits, dtype=np.float32))
    return np.log(np.clip(probs, 1e-8, 1.0)).astype(np.float32)


def _particle_map(particle: TrackParticle) -> dict[int, float]:
    return dict(zip(particle.steps, particle.tvt, strict=True))


def _overlap_penalty(
    particle: TrackParticle,
    steps: np.ndarray,
    path: np.ndarray,
    *,
    overlap_penalty: float,
) -> float:
    if not particle.steps:
        return 0.0
    lookup = _particle_map(particle)
    diffs = [
        abs(float(lookup[int(step)]) - float(value))
        for step, value in zip(steps, path, strict=True)
        if int(step) in lookup and np.isfinite(value)
    ]
    if not diffs:
        return 0.0
    return float(overlap_penalty * np.mean(diffs))


def _extend_particle(
    particle: TrackParticle | None,
    *,
    well_id: str,
    window_id: str,
    mode_index: int,
    steps: np.ndarray,
    path: np.ndarray,
    log_prob: float,
    overlap_penalty: float,
) -> TrackParticle:
    if particle is None:
        new_steps = tuple(int(step) for step in steps)
        new_tvt = tuple(float(value) for value in path)
        return TrackParticle(
            well_id=well_id,
            steps=new_steps,
            tvt=new_tvt,
            score=float(log_prob),
            lineage=(f"{window_id}:m{mode_index}",),
        )

    lookup = _particle_map(particle)
    appended_steps: list[int] = []
    appended_tvt: list[float] = []
    for step, value in zip(steps, path, strict=True):
        step_int = int(step)
        if step_int not in lookup and step_int > particle.last_step and np.isfinite(value):
            appended_steps.append(step_int)
            appended_tvt.append(float(value))
    penalty = _overlap_penalty(
        particle, steps, path, overlap_penalty=overlap_penalty
    )
    return TrackParticle(
        well_id=particle.well_id,
        steps=(*particle.steps, *appended_steps),
        tvt=(*particle.tvt, *appended_tvt),
        score=float(particle.score + log_prob - penalty),
        lineage=(*particle.lineage, f"{window_id}:m{mode_index}"),
        merged_count=particle.merged_count,
    )


def _tail_distance(a: TrackParticle, b: TrackParticle, tail: int = 8) -> float:
    common = sorted(set(a.steps[-tail:]).intersection(b.steps[-tail:]))
    if not common:
        return abs(a.endpoint - b.endpoint)
    a_lookup = _particle_map(a)
    b_lookup = _particle_map(b)
    return float(np.mean([abs(a_lookup[step] - b_lookup[step]) for step in common]))


def merge_and_prune_particles(
    particles: list[TrackParticle],
    *,
    merge_tolerance_ft: float,
    keep_top: int,
    n_realizations: int,
) -> list[TrackParticle]:
    ordered = sorted(particles, key=lambda item: item.score, reverse=True)
    merged: list[TrackParticle] = []
    limit = max(1, min(int(keep_top), int(n_realizations)))
    for particle in ordered:
        matched_index: int | None = None
        for index, existing in enumerate(merged):
            if existing.last_step != particle.last_step:
                continue
            if abs(existing.endpoint - particle.endpoint) > merge_tolerance_ft:
                continue
            if _tail_distance(existing, particle) > merge_tolerance_ft:
                continue
            matched_index = index
            break
        if matched_index is None:
            merged.append(particle)
        else:
            current = merged[matched_index]
            merged_count = min(
                current.merged_count + particle.merged_count, MERGED_COUNT_CAP
            )
            merged[matched_index] = TrackParticle(
                well_id=current.well_id,
                steps=current.steps,
                tvt=current.tvt,
                score=float(np.logaddexp(current.score, particle.score)),
                lineage=current.lineage,
                merged_count=merged_count,
            )
        if len(merged) >= limit:
            continue
    return sorted(merged, key=lambda item: item.score, reverse=True)[:limit]


def _window_ids(windows: pd.DataFrame) -> list[str]:
    if "window_id" in windows.columns:
        return windows["window_id"].astype(str).tolist()
    return [
        f"{row.well_id}:{int(row.start_step)}:{index}"
        for index, row in enumerate(windows.itertuples(index=False))
    ]


def _ensure_window_ids(windows: pd.DataFrame) -> pd.DataFrame:
    if "window_id" in windows.columns:
        return windows.copy()
    out = windows.copy()
    out["window_id"] = _window_ids(out)
    return out


def _expand_for_window(
    particles: list[TrackParticle],
    *,
    row: Any,
    window_id: str,
    history_steps: int,
    future_steps: int,
    cfg: TrackConfig,
) -> list[TrackParticle]:
    well_id = str(row.well_id)
    paths = np.asarray(row.path_tvt, dtype=np.float32)
    logits = np.asarray(row.logits, dtype=np.float32)
    log_probs = _log_probs(logits)
    order = np.argsort(-logits)[: min(cfg.max_modes_per_window, len(logits))]
    steps = np.arange(
        int(row.start_step) + history_steps,
        int(row.start_step) + history_steps + future_steps,
        dtype=np.int32,
    )
    base_particles: list[TrackParticle | None] = particles if particles else [None]
    expanded: list[TrackParticle] = []
    for particle in base_particles:
        for mode_index in order:
            expanded.append(
                _extend_particle(
                    particle,
                    well_id=well_id,
                    window_id=window_id,
                    mode_index=int(mode_index),
                    steps=steps,
                    path=paths[int(mode_index)],
                    log_prob=float(log_probs[int(mode_index)]),
                    overlap_penalty=cfg.overlap_penalty,
                )
            )
    return expanded


def track_mode_windows(
    mode_windows: pd.DataFrame,
    *,
    history_steps: int,
    future_steps: int,
    cfg: TrackConfig,
) -> dict[str, list[TrackParticle]]:
    windows = _ensure_window_ids(normalize_mode_windows(mode_windows))
    result: dict[str, list[TrackParticle]] = {}
    for well_id, group in windows.sort_values(["well_id", "start_step"]).groupby(
        "well_id", sort=True
    ):
        particles: list[TrackParticle] = []
        for row in group.itertuples(index=False):
            expanded = _expand_for_window(
                particles,
                row=row,
                window_id=str(row.window_id),
                history_steps=history_steps,
                future_steps=future_steps,
                cfg=cfg,
            )
            particles = merge_and_prune_particles(
                expanded,
                merge_tolerance_ft=cfg.merge_tolerance_ft,
                keep_top=cfg.keep_top,
                n_realizations=cfg.n_realizations,
            )
        result[str(well_id)] = particles[: cfg.n_realizations]
    return result


def particles_to_step_predictions(
    particles_by_well: dict[str, list[TrackParticle]],
    *,
    strategy: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for well_id, particles in particles_by_well.items():
        if not particles:
            continue
        if strategy == "top1":
            selected = [particles[0]]
            weights = np.ones(1, dtype=np.float32)
        elif strategy == "weighted":
            selected = particles
            scores = np.asarray([particle.score for particle in selected], dtype=np.float64)
            scores = scores - np.nanmax(scores)
            weights = np.exp(scores)
            weights = weights / max(float(weights.sum()), 1e-12)
        else:
            raise ValueError(f"Unsupported particle prediction strategy: {strategy}")
        accum: dict[int, list[float]] = {}
        for particle, weight in zip(selected, weights, strict=True):
            for step, value in zip(particle.steps, particle.tvt, strict=True):
                item = accum.setdefault(int(step), [0.0, 0.0])
                item[0] += float(value) * float(weight)
                item[1] += float(weight)
        for step, (total, weight) in accum.items():
            if weight > 0.0:
                rows.append(
                    {
                        "well_id": well_id,
                        "step": int(step),
                        "pred_tvt": float(total / weight),
                    }
                )
    if not rows:
        return pd.DataFrame(columns=["well_id", "step", "pred_tvt"])
    return pd.DataFrame(rows).sort_values(["well_id", "step"]).reset_index(drop=True)


def particles_to_frame(particles_by_well: dict[str, list[TrackParticle]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for well_id, particles in particles_by_well.items():
        for particle_index, particle in enumerate(particles):
            rows.append(
                {
                    "well_id": well_id,
                    "particle_id": particle_index,
                    "score": particle.score,
                    "merged_count": min(int(particle.merged_count), MERGED_COUNT_CAP),
                    "steps": list(particle.steps),
                    "tvt": list(particle.tvt),
                    "lineage": list(particle.lineage),
                }
            )
    return pd.DataFrame(rows)


def _candidate_table(metrics: list[dict[str, Any]]) -> str:
    columns = [
        "candidate",
        "rmse",
        "covered_rmse",
        "mean_well_rmse",
        "p95_well_rmse",
        "worst_well_rmse",
        "p95_abs_shift_vs_b2",
    ]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for item in sorted(metrics, key=lambda row: row.get("rmse", float("inf"))):
        values = []
        for column in columns:
            value = item.get(column, "n/a")
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_track_report(
    run_dir: Path,
    *,
    summary: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> Path:
    best = min(candidates, key=lambda item: item.get("rmse", float("inf")))
    b2 = summary["baselines"]["b2_guarded_submit"]["rmse"]
    lines = [
        "MTP_TRACK_V0_REPORT",
        "",
        "tracker:",
        json.dumps(summary["tracker"], indent=2),
        "",
        "coverage:",
        json.dumps(summary["coverage"], indent=2),
        "",
        "baselines:",
        f"  base_schema10_pp: {summary['baselines']['base_schema10_pp']['rmse']}",
        f"  b2_guarded_submit: {b2}",
        "",
        "best candidate:",
        f"  candidate: {best['candidate']}",
        f"  rmse: {best['rmse']}",
        f"  gain_vs_b2: {b2 - best['rmse']}",
        f"  covered_rmse: {best.get('covered_rmse', 'n/a')}",
        f"  p95_abs_shift_vs_b2: {best.get('p95_abs_shift_vs_b2', 'n/a')}",
        f"  worst_well_rmse: {best.get('worst_well_rmse', 'n/a')}",
        "",
        "row-level:",
        _candidate_table(candidates),
        "",
        "decision:",
        f"  beats_b2_by_0_10: {(b2 - best['rmse']) >= 0.10}",
        f"  strong_go_le_9_80: {best['rmse'] <= 9.80}",
    ]
    path = run_dir / "track_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_tracker_from_frames(
    *,
    run_dir: str | Path,
    cfg: MTPConfig,
    mode_windows: pd.DataFrame,
    hidden_rows_all: pd.DataFrame,
    track_config: TrackConfig,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    mode_windows = _ensure_window_ids(normalize_mode_windows(mode_windows))
    particles = track_mode_windows(
        mode_windows,
        history_steps=cfg.window.history_steps,
        future_steps=cfg.window.future_steps,
        cfg=track_config,
    )
    particle_frame = particles_to_frame(particles)
    particle_frame.to_parquet(run_path / "track_particles.parquet", index=False)

    candidate_steps = {
        "mtp_track_top1": particles_to_step_predictions(particles, strategy="top1"),
        "mtp_track_weighted": particles_to_step_predictions(
            particles, strategy="weighted"
        ),
    }
    covered_keys = pd.concat(candidate_steps.values(), ignore_index=True)[
        ["well_id", "step"]
    ].drop_duplicates()
    hidden_covered = hidden_rows_all.merge(
        covered_keys, on=["well_id", "step"], how="inner"
    )
    candidate_metrics: list[dict[str, Any]] = []
    row_predictions: list[pd.DataFrame] = []

    def add_candidate(name: str, rows: pd.DataFrame) -> None:
        item = rows.copy()
        item["candidate"] = name
        row_predictions.append(item)
        candidate_metrics.append(evaluate_with_b2_fallback(hidden_rows_all, item, name))

    for name, steps in candidate_steps.items():
        rows = _apply_step_predictions_to_rows(
            hidden_covered, steps, anchor_column="base_tvt"
        )
        add_candidate(name, rows)
        if name == "mtp_track_weighted":
            for alpha in (0.1, 0.2, 0.3):
                for clip in (20.0, 30.0):
                    add_candidate(
                        _anchored_candidate_name(name, alpha, clip),
                        _blend_with_anchor(
                            rows,
                            anchor_column="b2_tvt",
                            alpha=alpha,
                            clip=clip,
                        ),
                    )
    pd.DataFrame(candidate_metrics).sort_values("rmse").to_csv(
        run_path / "track_candidates.csv", index=False
    )
    if row_predictions:
        pd.concat(row_predictions, ignore_index=True).to_parquet(
            run_path / "track_row_predictions.parquet", index=False
        )
    base_metrics = _baseline_metrics(hidden_rows_all, "base_tvt", "base_schema10_pp")
    b2_metrics = _baseline_metrics(hidden_rows_all, "b2_tvt", "b2_guarded_submit")
    summary: dict[str, Any] = {
        "tracker": {
            "n_realizations": track_config.n_realizations,
            "keep_top": track_config.keep_top,
            "merge_tolerance_ft": track_config.merge_tolerance_ft,
            "overlap_penalty": track_config.overlap_penalty,
            "max_modes_per_window": track_config.max_modes_per_window,
            "wells": int(len(particles)),
            "particles": int(sum(len(items) for items in particles.values())),
        },
        "coverage": {
            "covered_hidden_rows": int(len(hidden_covered)),
            "total_hidden_rows": int(len(hidden_rows_all)),
            "coverage_frac": float(len(hidden_covered) / max(1, len(hidden_rows_all))),
        },
        "baselines": {
            "base_schema10_pp": base_metrics,
            "b2_guarded_submit": b2_metrics,
        },
        "candidates": candidate_metrics,
    }
    (run_path / "track_metrics.json").write_text(
        json.dumps(_json_safe(summary), indent=2), encoding="utf-8"
    )
    _write_track_report(run_path, summary=summary, candidates=candidate_metrics)
    return summary


def _metrics_by_name(metrics: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(item["candidate"]): item for item in metrics}


def _best_metric(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    return min(metrics, key=lambda item: item.get("rmse", float("inf")))


def _anchored_candidate_name(name: str, alpha: float, clip: float) -> str:
    local = str(name).removeprefix("mtp_track_")
    return f"mtp_track_anchored_{local}_a{alpha:g}_clip{int(clip)}"


def _beta_suffix(beta: float) -> str:
    text = f"{float(beta):.1f}" if float(beta).is_integer() else f"{float(beta):g}"
    return "b" + text.replace("0.", "0").replace(".", "")


def _run_tracker_metrics_only(
    *,
    cfg: MTPConfig,
    mode_windows: pd.DataFrame,
    hidden_rows_all: pd.DataFrame,
    track_config: TrackConfig,
    prefix: str,
) -> dict[str, Any]:
    particles = track_mode_windows(
        mode_windows,
        history_steps=cfg.window.history_steps,
        future_steps=cfg.window.future_steps,
        cfg=track_config,
    )
    candidate_steps = {
        f"{prefix}_top1": particles_to_step_predictions(particles, strategy="top1"),
        f"{prefix}_weighted": particles_to_step_predictions(particles, strategy="weighted"),
    }
    covered_keys = pd.concat(candidate_steps.values(), ignore_index=True)[
        ["well_id", "step"]
    ].drop_duplicates()
    hidden_covered = hidden_rows_all.merge(
        covered_keys, on=["well_id", "step"], how="inner"
    )
    metrics: list[dict[str, Any]] = []
    for name, steps in candidate_steps.items():
        rows = _apply_step_predictions_to_rows(
            hidden_covered, steps, anchor_column="base_tvt"
        )
        metrics.append(evaluate_with_b2_fallback(hidden_rows_all, rows, name))
        if name.endswith("_weighted"):
            for alpha in (0.1, 0.2, 0.3):
                for clip in (20.0, 30.0):
                    blended = _blend_with_anchor(
                        rows,
                        anchor_column="b2_tvt",
                        alpha=alpha,
                        clip=clip,
                    )
                    metrics.append(
                        evaluate_with_b2_fallback(
                            hidden_rows_all,
                            blended,
                            _anchored_candidate_name(name, alpha, clip),
                        )
                    )
    return {
        "particles": int(sum(len(items) for items in particles.values())),
        "coverage": {
            "covered_hidden_rows": int(len(hidden_covered)),
            "total_hidden_rows": int(len(hidden_rows_all)),
            "coverage_frac": float(len(hidden_covered) / max(1, len(hidden_rows_all))),
        },
        "candidates": metrics,
        "best": _best_metric(metrics),
    }


def _subset_summary(
    *,
    cfg: MTPConfig,
    mode_windows: pd.DataFrame,
    hidden_rows_all: pd.DataFrame,
    ranker_predictions: pd.DataFrame,
    wells: set[str],
    track_config: TrackConfig,
    tau_ft: float,
    ranker_beta: float,
) -> dict[str, Any]:
    subset_windows = mode_windows[mode_windows["well_id"].astype(str).isin(wells)].copy()
    subset_hidden = hidden_rows_all[hidden_rows_all["well_id"].astype(str).isin(wells)].copy()
    nn = _run_tracker_metrics_only(
        cfg=cfg,
        mode_windows=subset_windows,
        hidden_rows_all=subset_hidden,
        track_config=track_config,
        prefix="mtp_track_nn",
    )
    ranker_windows = apply_ranker_logits(
        subset_windows, ranker_predictions, tau_ft=tau_ft, beta=ranker_beta
    )
    ranker = _run_tracker_metrics_only(
        cfg=cfg,
        mode_windows=ranker_windows,
        hidden_rows_all=subset_hidden,
        track_config=track_config,
        prefix="mtp_track_ranker",
    )
    base = _baseline_metrics(subset_hidden, "base_tvt", "base_schema10_pp")
    b2 = _baseline_metrics(subset_hidden, "b2_tvt", "b2_guarded_submit")
    return {
        "wells": int(len(wells)),
        "rows": int(len(subset_hidden)),
        "base_schema10_pp": base,
        "b2_guarded_submit": b2,
        "nn": nn,
        "ranker": ranker,
        "ranker_gain_vs_b2": float(b2["rmse"] - ranker["best"]["rmse"]),
        "nn_gain_vs_b2": float(b2["rmse"] - nn["best"]["rmse"]),
    }


def _write_split_audit_report(run_dir: Path, summary: dict[str, Any]) -> Path:
    lines = [
        "MTPTRACK_SPLIT_AUDIT",
        "",
        "tracker:",
        json.dumps(summary["tracker"], indent=2),
        "",
    ]
    for name in ("all_valid", "ranker_train", "ranker_valid"):
        item = summary["subsets"][name]
        b2 = item["b2_guarded_submit"]["rmse"]
        nn_best = item["nn"]["best"]
        ranker_best = item["ranker"]["best"]
        lines.extend(
            [
                f"{name}:",
                f"  wells: {item['wells']}",
                f"  rows: {item['rows']}",
                f"  B2: {b2}",
                f"  tracker_NN_best: {nn_best['candidate']}",
                f"  tracker_NN_RMSE: {nn_best['rmse']}",
                f"  tracker_NN_gain: {item['nn_gain_vs_b2']}",
                f"  tracker_ranker_best: {ranker_best['candidate']}",
                f"  tracker_ranker_RMSE: {ranker_best['rmse']}",
                f"  tracker_ranker_gain: {item['ranker_gain_vs_b2']}",
                f"  tracker_ranker_P95_shift: {ranker_best.get('p95_abs_shift_vs_b2')}",
                f"  tracker_ranker_worst: {ranker_best.get('worst_well_rmse')}",
                "",
            ]
        )
    decision = summary["decision"]
    lines.extend(
        [
            "decision:",
            f"  clean_go: {decision['clean_go']}",
            f"  partial_go: {decision['partial_go']}",
            f"  leakage_risk: {decision['leakage_risk']}",
            f"  train_valid_gain_gap: {decision['train_valid_gain_gap']}",
        ]
    )
    path = run_dir / "track_split_audit.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_track_split_audit_from_frames(
    *,
    run_dir: str | Path,
    cfg: MTPConfig,
    mode_windows: pd.DataFrame,
    hidden_rows_all: pd.DataFrame,
    ranker_predictions: pd.DataFrame,
    ranker_train_wells: set[str],
    ranker_valid_wells: set[str],
    track_config: TrackConfig,
    tau_ft: float = 5.0,
    ranker_beta: float = 0.5,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    mode_windows = _ensure_window_ids(normalize_mode_windows(mode_windows))
    ranker_predictions = ranker_predictions.copy()
    all_wells = set(mode_windows["well_id"].astype(str))
    subsets = {
        "all_valid": _subset_summary(
            cfg=cfg,
            mode_windows=mode_windows,
            hidden_rows_all=hidden_rows_all,
            ranker_predictions=ranker_predictions,
            wells=all_wells,
            track_config=track_config,
            tau_ft=tau_ft,
            ranker_beta=ranker_beta,
        ),
        "ranker_train": _subset_summary(
            cfg=cfg,
            mode_windows=mode_windows,
            hidden_rows_all=hidden_rows_all,
            ranker_predictions=ranker_predictions,
            wells=set(ranker_train_wells),
            track_config=track_config,
            tau_ft=tau_ft,
            ranker_beta=ranker_beta,
        ),
        "ranker_valid": _subset_summary(
            cfg=cfg,
            mode_windows=mode_windows,
            hidden_rows_all=hidden_rows_all,
            ranker_predictions=ranker_predictions,
            wells=set(ranker_valid_wells),
            track_config=track_config,
            tau_ft=tau_ft,
            ranker_beta=ranker_beta,
        ),
    }
    train_gain = subsets["ranker_train"]["ranker_gain_vs_b2"]
    valid_gain = subsets["ranker_valid"]["ranker_gain_vs_b2"]
    decision = {
        "clean_go": bool(
            valid_gain >= 0.10
            and subsets["ranker_valid"]["ranker"]["best"].get("p95_abs_shift_vs_b2", 999.0)
            <= 2.5
        ),
        "partial_go": bool(0.05 <= valid_gain < 0.10),
        "leakage_risk": bool((train_gain - valid_gain) > 0.10),
        "train_valid_gain_gap": float(train_gain - valid_gain),
    }
    summary: dict[str, Any] = {
        "tracker": {
            "n_realizations": track_config.n_realizations,
            "keep_top": track_config.keep_top,
            "merge_tolerance_ft": track_config.merge_tolerance_ft,
            "overlap_penalty": track_config.overlap_penalty,
            "max_modes_per_window": track_config.max_modes_per_window,
            "tau_ft": tau_ft,
            "ranker_beta": ranker_beta,
        },
        "subsets": subsets,
        "decision": decision,
    }
    (run_path / "track_split_audit.json").write_text(
        json.dumps(_json_safe(summary), indent=2), encoding="utf-8"
    )
    _write_split_audit_report(run_path, summary)
    return summary


def run_track_split_audit(
    run_dir: str | Path,
    *,
    n_realizations: int = 32,
    keep_top: int = 32,
    merge_tolerance_ft: float = 3.0,
    overlap_penalty: float = 0.10,
    max_modes_per_window: int = 8,
    tau_ft: float = 5.0,
    ranker_beta: float = 0.5,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    cfg = _load_run_config(run_path)
    mode_windows = pd.read_parquet(run_path / "stitch_window_modes.parquet")
    ranker_predictions = pd.read_parquet(run_path / "ranker_predictions.parquet")
    ranker_metrics_path = run_path / "ranker_metrics.json"
    ranker_metrics = json.loads(ranker_metrics_path.read_text(encoding="utf-8"))
    train_wells = set(ranker_metrics["ranker_split"]["train_wells"])
    valid_wells = set(ranker_metrics["ranker_split"]["valid_wells"])
    hidden_rows = _load_hidden_rows(cfg, set(mode_windows["well_id"].astype(str)))
    summary = run_track_split_audit_from_frames(
        run_dir=run_path,
        cfg=cfg,
        mode_windows=mode_windows,
        hidden_rows_all=hidden_rows,
        ranker_predictions=ranker_predictions,
        ranker_train_wells=train_wells,
        ranker_valid_wells=valid_wells,
        track_config=TrackConfig(
            n_realizations=n_realizations,
            keep_top=keep_top,
            merge_tolerance_ft=merge_tolerance_ft,
            overlap_penalty=overlap_penalty,
            max_modes_per_window=max_modes_per_window,
        ),
        tau_ft=tau_ft,
        ranker_beta=ranker_beta,
    )
    print(json.dumps(_json_safe(summary["decision"]), indent=2), flush=True)
    return summary


def apply_tracker_logit_source(
    mode_windows: pd.DataFrame,
    *,
    logit_source: str,
    run_dir: str | Path,
    ranker_logits: str | Path | None,
    tau_ft: float,
    ranker_beta: float,
    corr_beta: float = 0.5,
    corr_score_normalization: str = "centered",
) -> tuple[pd.DataFrame, str]:
    run_path = Path(run_dir)
    windows = _ensure_window_ids(normalize_mode_windows(mode_windows))
    if logit_source == "nn":
        return windows, "nn"
    if logit_source == "corr":
        if "corr_scores" not in windows.columns:
            raise ValueError("logit_source='corr' requires corr_scores in mode windows")
        out = windows.copy()

        def combine(row: Any) -> np.ndarray:
            logits = np.asarray(row.logits, dtype=np.float32)
            corr = np.asarray(row.corr_scores, dtype=np.float32)
            centered = (corr - float(corr.mean())).astype(np.float32)
            if corr_score_normalization == "centered":
                corr_signal = centered
            elif corr_score_normalization == "zscore":
                scale = float(corr.std())
                corr_signal = (
                    np.zeros_like(corr, dtype=np.float32)
                    if scale <= 1e-8
                    else (centered / scale).astype(np.float32)
                )
            else:
                raise ValueError(
                    "corr_score_normalization must be 'centered' or 'zscore'"
                )
            return (logits + float(corr_beta) * corr_signal).astype(np.float32)

        out["logits"] = out.apply(combine, axis=1)
        out["probs"] = out["logits"].map(_softmax_np)
        return out, "corr"
    if logit_source == "ranker":
        ranker_path = run_path / "ranker_predictions.parquet"
        if not ranker_path.exists():
            return windows, "nn"
        return (
            apply_ranker_logits(
                windows, pd.read_parquet(ranker_path), tau_ft=tau_ft, beta=ranker_beta
            ),
            "ranker",
        )
    if logit_source == "ranker_oof":
        ranker_path = Path(ranker_logits) if ranker_logits is not None else (
            run_path / "oof_ranker_logits.parquet"
        )
        if not ranker_path.exists():
            raise FileNotFoundError(f"Missing ranker_oof logits: {ranker_path}")
        return (
            apply_ranker_logits(
                windows, pd.read_parquet(ranker_path), tau_ft=tau_ft, beta=ranker_beta
            ),
            "ranker_oof",
        )
    raise ValueError("logit_source must be 'corr', 'ranker', 'ranker_oof', or 'nn'")


def run_tracker_corr_beta_grid_from_frames(
    *,
    run_dir: str | Path,
    cfg: MTPConfig,
    mode_windows: pd.DataFrame,
    hidden_rows_all: pd.DataFrame,
    track_config: TrackConfig,
    beta_grid: tuple[float, ...],
    score_normalization: str = "centered",
) -> dict[str, Any]:
    from .progress import ProgressLogger

    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    windows = _ensure_window_ids(normalize_mode_windows(mode_windows))
    candidate_metrics: list[dict[str, Any]] = []
    beta_summaries: dict[str, Any] = {}
    beta_candidates: dict[str, list[dict[str, Any]]] = {}
    logger = ProgressLogger(run=cfg.run.name)
    logger.log(
        "tracker_corr_beta_grid_start",
        run_dir=str(run_path),
        beta_grid=[float(beta) for beta in beta_grid],
        score_normalization=score_normalization,
        windows=int(len(windows)),
    )
    with logger.stage("tracker_nn_baseline"):
        nn_summary = _run_tracker_metrics_only(
            cfg=cfg,
            mode_windows=windows,
            hidden_rows_all=hidden_rows_all,
            track_config=track_config,
            prefix="mtp_track_nn",
        )
    for beta_index, beta in enumerate(beta_grid):
        suffix = _beta_suffix(float(beta))
        with logger.stage(
            "tracker_corr_beta",
            beta=float(beta),
            suffix=suffix,
            beta_index=int(beta_index),
            betas_total=int(len(beta_grid)),
        ) as stage:
            corr_windows, _ = apply_tracker_logit_source(
                windows,
                logit_source="corr",
                run_dir=run_path,
                ranker_logits=None,
                tau_ft=5.0,
                ranker_beta=0.5,
                corr_beta=float(beta),
                corr_score_normalization=score_normalization,
            )
            beta_summary = _run_tracker_metrics_only(
                cfg=cfg,
                mode_windows=corr_windows,
                hidden_rows_all=hidden_rows_all,
                track_config=track_config,
                prefix=f"mtp_track_corr_{suffix}",
            )
            stage["best_candidate"] = str(beta_summary["best"]["candidate"])
            stage["best_rmse"] = float(beta_summary["best"]["rmse"])
        beta_summaries[suffix] = {
            "beta": float(beta),
            "best": beta_summary["best"],
            "coverage": beta_summary["coverage"],
        }
        beta_candidates[suffix] = beta_summary["candidates"]
        candidate_metrics.extend(beta_summary["candidates"])
    base_metrics = _baseline_metrics(hidden_rows_all, "base_tvt", "base_schema10_pp")
    b2_metrics = _baseline_metrics(hidden_rows_all, "b2_tvt", "b2_guarded_submit")
    best_corr = min(candidate_metrics, key=lambda item: item.get("rmse", float("inf")))
    best_beta = next(
        (
            value["beta"]
            for value in beta_summaries.values()
            if value["best"]["candidate"] == best_corr["candidate"]
        ),
        None,
    )
    summary: dict[str, Any] = {
        "tracker": {
            "n_realizations": track_config.n_realizations,
            "keep_top": track_config.keep_top,
            "merge_tolerance_ft": track_config.merge_tolerance_ft,
            "overlap_penalty": track_config.overlap_penalty,
            "max_modes_per_window": track_config.max_modes_per_window,
            "logit_source": "corr",
            "corr_beta_grid": [float(beta) for beta in beta_grid],
            "corr_score_normalization": score_normalization,
            "beta_summaries": beta_summaries,
            "nn_best": nn_summary["best"],
            "corr_beta_best": best_beta,
            "corr_tracker_gain_vs_nn": float(
                nn_summary["best"]["rmse"] - best_corr["rmse"]
            ),
        },
        "coverage": next(iter(beta_summaries.values()))["coverage"] if beta_summaries else {},
        "baselines": {
            "base_schema10_pp": base_metrics,
            "b2_guarded_submit": b2_metrics,
        },
        "candidates": candidate_metrics,
    }
    pd.DataFrame(candidate_metrics).sort_values("rmse").to_csv(
        run_path / "track_candidates.csv", index=False
    )
    (run_path / "track_metrics.json").write_text(
        json.dumps(_json_safe(summary), indent=2), encoding="utf-8"
    )
    for suffix, candidates in beta_candidates.items():
        beta_summary = {
            **summary,
            "coverage": beta_summaries[suffix]["coverage"],
            "candidates": candidates,
            "tracker": {
                **summary["tracker"],
                "corr_beta": beta_summaries[suffix]["beta"],
            },
        }
        (run_path / f"track_metrics_corr_{suffix}.json").write_text(
            json.dumps(_json_safe(beta_summary), indent=2), encoding="utf-8"
        )
        pd.DataFrame(candidates).sort_values("rmse").to_csv(
            run_path / f"track_candidates_corr_{suffix}.csv", index=False
        )
        report_path = _write_track_report(
            run_path, summary=beta_summary, candidates=candidates
        )
        (run_path / f"track_report_corr_{suffix}.md").write_text(
            report_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
    _write_track_report(run_path, summary=summary, candidates=candidate_metrics)
    logger.log(
        "tracker_corr_beta_grid_done",
        best_candidate=str(best_corr["candidate"]),
        best_rmse=float(best_corr["rmse"]),
        corr_beta_best=best_beta,
        nn_best_candidate=str(nn_summary["best"]["candidate"]),
        nn_best_rmse=float(nn_summary["best"]["rmse"]),
        corr_tracker_gain_vs_nn=float(nn_summary["best"]["rmse"] - best_corr["rmse"]),
        b2_rmse=float(b2_metrics["rmse"]),
        gain_vs_b2=float(b2_metrics["rmse"] - best_corr["rmse"]),
    )
    return summary


def run_tracker(
    run_dir: str | Path,
    *,
    n_realizations: int = 32,
    keep_top: int = 32,
    merge_tolerance_ft: float = 3.0,
    overlap_penalty: float = 0.10,
    max_modes_per_window: int = 8,
    logit_source: str = "ranker",
    ranker_logits: str | Path | None = None,
    tau_ft: float = 5.0,
    ranker_beta: float = 0.5,
    corr_beta: float | None = None,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    cfg = _load_run_config(run_path)
    windows_path = run_path / "stitch_window_modes.parquet"
    if not windows_path.exists():
        raise FileNotFoundError(
            f"Missing {windows_path}; run `make stitch RUN_DIR={run_path}` first"
        )
    raw_mode_windows = pd.read_parquet(windows_path)
    if logit_source == "corr" and corr_beta is None:
        summary = run_tracker_corr_beta_grid_from_frames(
            run_dir=run_path,
            cfg=cfg,
            mode_windows=raw_mode_windows,
            hidden_rows_all=_load_hidden_rows(
                cfg, set(raw_mode_windows["well_id"].astype(str))
            ),
            track_config=TrackConfig(
                n_realizations=n_realizations,
                keep_top=keep_top,
                merge_tolerance_ft=merge_tolerance_ft,
                overlap_penalty=overlap_penalty,
                max_modes_per_window=max_modes_per_window,
            ),
            beta_grid=cfg.corr_head.tracker_beta_grid,
            score_normalization=cfg.corr_head.score_normalization,
        )
        best = min(summary["candidates"], key=lambda item: item.get("rmse", float("inf")))
        print(json.dumps(_json_safe(best), indent=2), flush=True)
        return summary
    mode_windows, logit_source = apply_tracker_logit_source(
        raw_mode_windows,
        logit_source=logit_source,
        run_dir=run_path,
        ranker_logits=ranker_logits,
        tau_ft=tau_ft,
        ranker_beta=ranker_beta,
        corr_beta=0.5 if corr_beta is None else corr_beta,
        corr_score_normalization=cfg.corr_head.score_normalization,
    )
    well_ids = set(mode_windows["well_id"].astype(str))
    hidden_rows = _load_hidden_rows(cfg, well_ids)
    summary = run_tracker_from_frames(
        run_dir=run_path,
        cfg=cfg,
        mode_windows=mode_windows,
        hidden_rows_all=hidden_rows,
        track_config=TrackConfig(
            n_realizations=n_realizations,
            keep_top=keep_top,
            merge_tolerance_ft=merge_tolerance_ft,
            overlap_penalty=overlap_penalty,
            max_modes_per_window=max_modes_per_window,
        ),
    )
    summary["tracker"]["logit_source"] = logit_source
    summary["tracker"]["tau_ft"] = tau_ft
    summary["tracker"]["ranker_beta"] = ranker_beta
    summary["tracker"]["corr_beta"] = corr_beta
    summary["tracker"]["corr_score_normalization"] = cfg.corr_head.score_normalization
    (run_path / "track_metrics.json").write_text(
        json.dumps(_json_safe(summary), indent=2), encoding="utf-8"
    )
    _write_track_report(run_path, summary=summary, candidates=summary["candidates"])
    if logit_source == "corr":
        suffix = _beta_suffix(corr_beta)
        (run_path / f"track_metrics_corr_{suffix}.json").write_text(
            json.dumps(_json_safe(summary), indent=2), encoding="utf-8"
        )
        pd.DataFrame(summary["candidates"]).sort_values("rmse").to_csv(
            run_path / f"track_candidates_corr_{suffix}.csv", index=False
        )
        report_path = _write_track_report(
            run_path, summary=summary, candidates=summary["candidates"]
        )
        (run_path / f"track_report_corr_{suffix}.md").write_text(
            report_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
    best = min(summary["candidates"], key=lambda item: item.get("rmse", float("inf")))
    print(json.dumps(_json_safe(best), indent=2), flush=True)
    return summary
