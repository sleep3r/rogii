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
                        f"b2_plus_{name}_a{alpha:g}_clip{int(clip)}",
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


def run_tracker(
    run_dir: str | Path,
    *,
    n_realizations: int = 32,
    keep_top: int = 32,
    merge_tolerance_ft: float = 3.0,
    overlap_penalty: float = 0.10,
    max_modes_per_window: int = 8,
    logit_source: str = "ranker",
    tau_ft: float = 5.0,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    cfg = _load_run_config(run_path)
    windows_path = run_path / "stitch_window_modes.parquet"
    if not windows_path.exists():
        raise FileNotFoundError(
            f"Missing {windows_path}; run `make stitch RUN_DIR={run_path}` first"
        )
    mode_windows = _ensure_window_ids(normalize_mode_windows(pd.read_parquet(windows_path)))
    if logit_source == "ranker":
        ranker_path = run_path / "ranker_predictions.parquet"
        if ranker_path.exists():
            mode_windows = apply_ranker_logits(
                mode_windows, pd.read_parquet(ranker_path), tau_ft=tau_ft
            )
        else:
            logit_source = "nn"
    elif logit_source != "nn":
        raise ValueError("logit_source must be 'ranker' or 'nn'")
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
    (run_path / "track_metrics.json").write_text(
        json.dumps(_json_safe(summary), indent=2), encoding="utf-8"
    )
    _write_track_report(run_path, summary=summary, candidates=summary["candidates"])
    best = min(summary["candidates"], key=lambda item: item.get("rmse", float("inf")))
    print(json.dumps(_json_safe(best), indent=2), flush=True)
    return summary
