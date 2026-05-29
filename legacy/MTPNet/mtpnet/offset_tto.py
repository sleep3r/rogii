"""Gradient refinement for the discrete offset path.

This tests the Kaggle comment idea:

    start from TVT = last_TVT - dZ_cumsum + offset * row_delta
    sample typewell GR at the predicted TVT path
    differentiate a small offset correction to better match observed GR

The experiment is deliberately guarded:

* it starts only from a posterior shortlist, usually ``offset_mdn_v0`` top-k;
* it optimizes a scalar offset correction per candidate, not an unconstrained
  per-row TVT curve;
* it runs normal / shuffled / zero GR variants with the same optimizer.

If normal GR does not beat the null variants, the branch is not a deployable
GR signal even if the optimized loss looks pretty.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .discrete_offset import (
    _WellArrays,
    _candidate_path,
    _json_safe,
    _load_wells,
    _metrics_from_errors,
    parse_offset_grid,
)
from .residual_stack import make_group_folds


@dataclass(frozen=True)
class OffsetTTOConfig:
    data_dir: Path = Path("data/train")
    output_dir: Path = Path("artifacts/offset_tto_v0")
    posterior_scores_path: Path | None = Path(
        "artifacts/offset_mdn_v0/offset_mdn_oof_candidate_scores.parquet"
    )
    offset_grid: str = "-0.16:0.16:0.001"
    score_column: str = "pred_log_mse"
    score_lower_is_better: bool = True
    top_k: int = 10
    n_folds: int = 5
    seed: int = 42
    k_wells: int = -1
    steps: int = 60
    learning_rate: float = 0.05
    max_delta: float = 0.015
    offset_l2: float = 0.05
    oob_penalty: float = 1.0
    normalize_gr: bool = True
    include_nulls: bool = True
    progress_every: int = 50


@dataclass
class OffsetTTOResult:
    selected_offset: float
    selected_initial_offset: float
    selected_loss: float
    initial_loss: float
    selected_rank: int
    pred_tvt: np.ndarray
    candidate_count: int


def torch_interp_1d(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """Piecewise-linear interpolation with gradient w.r.t. ``x``."""
    if xp.ndim != 1 or fp.ndim != 1:
        raise ValueError("xp/fp must be 1D tensors")
    if xp.numel() != fp.numel() or xp.numel() < 2:
        raise ValueError("xp/fp must have the same length >= 2")
    idx = torch.searchsorted(xp, x.contiguous(), right=True)
    idx0 = torch.clamp(idx - 1, 0, xp.numel() - 2)
    idx1 = idx0 + 1
    x0 = xp[idx0]
    x1 = xp[idx1]
    y0 = fp[idx0]
    y1 = fp[idx1]
    denom = torch.clamp(x1 - x0, min=1e-12)
    t = torch.clamp((x - x0) / denom, 0.0, 1.0)
    return y0 + t * (y1 - y0)


def _as_torch(values: np.ndarray, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.as_tensor(np.asarray(values, dtype=np.float64), dtype=dtype)


def _normalise_tensor(values: torch.Tensor) -> torch.Tensor:
    finite = torch.isfinite(values)
    if not bool(finite.any()):
        return torch.zeros_like(values)
    subset = values[finite]
    mean = subset.mean()
    std = subset.std(unbiased=False)
    if float(std.detach().cpu()) < 1e-6:
        out = values - mean
    else:
        out = (values - mean) / std
    return torch.where(finite, out, torch.zeros_like(out))


def _hidden_gr_variant(well: _WellArrays, *, variant: str, seed: int) -> np.ndarray:
    values = well.gr[well.hidden_idx].astype(np.float64).copy()
    finite = np.flatnonzero(np.isfinite(values))
    if variant == "normal":
        return values
    if variant == "shuffled_gr":
        if finite.size > 1:
            rng = np.random.default_rng(seed + abs(hash(well.well_id)) % 1_000_000)
            values[finite] = values[rng.permutation(finite)]
        return values
    if variant == "zero_gr":
        values[finite] = 0.0
        return values
    raise ValueError(f"unknown TTO variant: {variant}")


def _score_offsets_no_grad(
    *,
    well: _WellArrays,
    initial_offsets: np.ndarray,
    hidden_gr: np.ndarray,
    normalize_gr: bool,
    offset_l2: float,
    max_delta: float,
    oob_penalty: float,
    raw_delta: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    idx = well.hidden_idx
    mask_np = np.isfinite(hidden_gr)
    if mask_np.sum() < 3:
        nan = torch.full((len(initial_offsets),), float("inf"), dtype=torch.float32)
        return nan, nan, torch.empty((len(initial_offsets), 0), dtype=torch.float32)
    offsets0 = _as_torch(initial_offsets)
    if raw_delta is None:
        delta = torch.zeros_like(offsets0)
    else:
        delta = float(max_delta) * torch.tanh(raw_delta)
    offsets = offsets0 + delta
    base = _as_torch(float(well.tvt_input[well.anchor_row]) - (well.z[idx] - float(well.z[well.anchor_row])))
    row_delta = _as_torch(idx - well.anchor_row)
    path = base[None, :] + offsets[:, None] * row_delta[None, :]

    tw_tvt = _as_torch(well.typewell_tvt)
    tw_gr = _as_torch(well.typewell_gr)
    obs = _as_torch(hidden_gr[mask_np])
    path_masked = path[:, mask_np]
    if normalize_gr:
        tw_gr = _normalise_tensor(tw_gr)
        obs = _normalise_tensor(obs)
    sampled = torch_interp_1d(path_masked, tw_tvt, tw_gr)
    gr_loss = torch.mean((sampled - obs[None, :]) ** 2, dim=1)
    prior = float(offset_l2) * (delta / max(float(max_delta), 1e-9)) ** 2
    lo = torch.min(tw_tvt)
    hi = torch.max(tw_tvt)
    oob = torch.mean((torch.relu(lo - path) / 100.0) ** 2 + (torch.relu(path - hi) / 100.0) ** 2, dim=1)
    total = gr_loss + prior + float(oob_penalty) * oob
    return total, gr_loss, path


def refine_well_offsets(
    well: _WellArrays,
    *,
    initial_offsets: np.ndarray,
    steps: int,
    learning_rate: float,
    max_delta: float,
    offset_l2: float,
    normalize_gr: bool,
    variant: str,
    seed: int,
    oob_penalty: float = 1.0,
) -> OffsetTTOResult:
    """Refine a shortlist of scalar offsets by differentiable GR matching."""
    initial_offsets = np.asarray(initial_offsets, dtype=np.float64)
    initial_offsets = initial_offsets[np.isfinite(initial_offsets)]
    if initial_offsets.size == 0:
        raise ValueError("initial_offsets is empty")
    if well.typewell_tvt.size < 3 or well.typewell_gr.size < 3:
        offset = float(initial_offsets[0])
        return OffsetTTOResult(
            selected_offset=offset,
            selected_initial_offset=offset,
            selected_loss=float("inf"),
            initial_loss=float("inf"),
            selected_rank=0,
            pred_tvt=_candidate_path(well, offset),
            candidate_count=int(initial_offsets.size),
        )

    hidden_gr = _hidden_gr_variant(well, variant=variant, seed=seed)
    raw_delta = torch.zeros(initial_offsets.size, dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
        initial_total, _, _ = _score_offsets_no_grad(
            well=well,
            initial_offsets=initial_offsets,
            hidden_gr=hidden_gr,
            normalize_gr=normalize_gr,
            offset_l2=offset_l2,
            max_delta=max_delta,
            oob_penalty=oob_penalty,
            raw_delta=None,
        )
        if not bool(torch.isfinite(initial_total).any()):
            offset = float(initial_offsets[0])
            return OffsetTTOResult(
                selected_offset=offset,
                selected_initial_offset=offset,
                selected_loss=float("inf"),
                initial_loss=float("inf"),
                selected_rank=0,
                pred_tvt=_candidate_path(well, offset),
                candidate_count=int(initial_offsets.size),
            )
        initial_best_loss = float(torch.min(initial_total).detach().cpu())

    optimizer = torch.optim.Adam([raw_delta], lr=float(learning_rate))
    for _ in range(int(max(steps, 0))):
        optimizer.zero_grad()
        total, _, _ = _score_offsets_no_grad(
            well=well,
            initial_offsets=initial_offsets,
            hidden_gr=hidden_gr,
            normalize_gr=normalize_gr,
            offset_l2=offset_l2,
            max_delta=max_delta,
            oob_penalty=oob_penalty,
            raw_delta=raw_delta,
        )
        loss = torch.mean(total[torch.isfinite(total)])
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        total, _, path = _score_offsets_no_grad(
            well=well,
            initial_offsets=initial_offsets,
            hidden_gr=hidden_gr,
            normalize_gr=normalize_gr,
            offset_l2=offset_l2,
            max_delta=max_delta,
            oob_penalty=oob_penalty,
            raw_delta=raw_delta,
        )
        rank = int(torch.argmin(total).detach().cpu())
        delta = float(max_delta) * torch.tanh(raw_delta).detach().cpu().numpy()
        selected_offset = float(initial_offsets[rank] + delta[rank])
        selected_path = path[rank].detach().cpu().numpy().astype(np.float64)
        selected_loss = float(total[rank].detach().cpu())

    return OffsetTTOResult(
        selected_offset=selected_offset,
        selected_initial_offset=float(initial_offsets[rank]),
        selected_loss=selected_loss,
        initial_loss=initial_best_loss,
        selected_rank=rank,
        pred_tvt=selected_path,
        candidate_count=int(initial_offsets.size),
    )


def _load_initial_offsets(
    *,
    wells: list[_WellArrays],
    scores_path: Path | None,
    offset_grid: np.ndarray,
    score_column: str,
    lower_is_better: bool,
    top_k: int,
) -> dict[str, np.ndarray]:
    if scores_path is None or not Path(scores_path).exists():
        grid = np.asarray(offset_grid, dtype=np.float64)
        if grid.size > top_k > 0:
            # No posterior is available. Use a small centered grid for smoke
            # tests instead of pretending these are ranked.
            center_order = np.argsort(np.abs(grid))
            grid = grid[center_order[:top_k]]
        return {well.well_id: np.asarray(grid, dtype=np.float64) for well in wells}
    scores = pd.read_parquet(scores_path)
    if score_column not in scores.columns:
        raise ValueError(f"score column not found in posterior scores: {score_column}")
    out: dict[str, np.ndarray] = {}
    for well in wells:
        group = scores[scores["well_id"].astype(str) == str(well.well_id)]
        if group.empty:
            out[well.well_id] = np.asarray(offset_grid[: max(top_k, 1)], dtype=np.float64)
            continue
        ordered = group.sort_values(score_column, ascending=lower_is_better)
        offsets = pd.to_numeric(ordered["offset"], errors="coerce").dropna().to_numpy(dtype=np.float64)
        out[well.well_id] = offsets[: max(int(top_k), 1)]
    return out


def _metrics_from_variant_records(records: list[dict[str, Any]], *, candidate: str) -> dict[str, Any]:
    errors_by_well = [np.asarray(row["err"], dtype=np.float64) for row in records]
    metrics = _metrics_from_errors(errors_by_well)
    metrics["candidate"] = candidate
    metrics["selected_offset_mean"] = float(np.mean([row["selected_offset"] for row in records])) if records else float("nan")
    metrics["selected_offset_std"] = float(np.std([row["selected_offset"] for row in records])) if records else float("nan")
    metrics["initial_loss_mean"] = float(np.mean([row["initial_loss"] for row in records])) if records else float("nan")
    metrics["selected_loss_mean"] = float(np.mean([row["selected_loss"] for row in records])) if records else float("nan")
    return metrics


def run_offset_tto(config: OffsetTTOConfig) -> dict[str, Any]:
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(config.data_dir)
    paths = sorted(data_dir.glob("*__horizontal_well.csv"))
    if config.k_wells > 0:
        paths = paths[: config.k_wells]
    well_ids = [path.name.replace("__horizontal_well.csv", "") for path in paths]
    folds = make_group_folds(well_ids, n_folds=config.n_folds, seed=config.seed)
    fold_of_well = {str(w): idx for idx, (_, valid) in enumerate(folds) for w in valid}
    wells = _load_wells(data_dir, k_wells=config.k_wells, fold_of_well=fold_of_well)
    offset_grid = parse_offset_grid(config.offset_grid)
    initial_by_well = _load_initial_offsets(
        wells=wells,
        scores_path=config.posterior_scores_path,
        offset_grid=offset_grid,
        score_column=config.score_column,
        lower_is_better=config.score_lower_is_better,
        top_k=config.top_k,
    )
    variants = ["normal"]
    if config.include_nulls:
        variants.extend(["shuffled_gr", "zero_gr"])
    print(
        f"[offset-tto] wells={len(wells)} variants={variants} top_k={config.top_k} "
        f"steps={config.steps}",
        file=sys.stderr,
        flush=True,
    )
    variant_records: dict[str, list[dict[str, Any]]] = {variant: [] for variant in variants}
    pred_parts: list[pd.DataFrame] = []
    score_rows: list[dict[str, Any]] = []

    for v_idx, variant in enumerate(variants):
        for w_idx, well in enumerate(wells):
            if config.progress_every > 0 and (w_idx == 0 or (w_idx + 1) % config.progress_every == 0):
                print(
                    f"[offset-tto] variant={variant} well={w_idx + 1}/{len(wells)}",
                    file=sys.stderr,
                    flush=True,
                )
            result = refine_well_offsets(
                well,
                initial_offsets=initial_by_well[well.well_id],
                steps=config.steps,
                learning_rate=config.learning_rate,
                max_delta=config.max_delta,
                offset_l2=config.offset_l2,
                normalize_gr=config.normalize_gr,
                variant=variant,
                seed=config.seed + 10_000 * v_idx,
                oob_penalty=config.oob_penalty,
            )
            truth = well.tvt[well.hidden_idx]
            err = result.pred_tvt - truth
            variant_records[variant].append(
                {
                    "well_id": well.well_id,
                    "err": err,
                    "selected_offset": result.selected_offset,
                    "selected_initial_offset": result.selected_initial_offset,
                    "selected_loss": result.selected_loss,
                    "initial_loss": result.initial_loss,
                    "selected_rank": result.selected_rank,
                    "candidate_count": result.candidate_count,
                }
            )
            pred_parts.append(
                pd.DataFrame(
                    {
                        "id": well.ids[well.hidden_idx],
                        "well_id": well.well_id,
                        "row_idx": well.hidden_idx.astype(np.int64),
                        "pred_tvt": result.pred_tvt,
                        "candidate": f"tto_{variant}",
                        "selected_offset": result.selected_offset,
                        "selected_initial_offset": result.selected_initial_offset,
                    }
                )
            )
            score_rows.append(
                {
                    "well_id": well.well_id,
                    "variant": variant,
                    "selected_offset": result.selected_offset,
                    "selected_initial_offset": result.selected_initial_offset,
                    "selected_loss": result.selected_loss,
                    "initial_loss": result.initial_loss,
                    "selected_rank": result.selected_rank,
                    "candidate_count": result.candidate_count,
                    "rmse": float(np.sqrt(np.nanmean(err * err))),
                }
            )

    predictions = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
    predictions.to_parquet(out_dir / "offset_tto_predictions.parquet", index=False)
    pd.DataFrame(score_rows).to_parquet(out_dir / "offset_tto_scores.parquet", index=False)
    candidate_metrics = [
        _metrics_from_variant_records(records, candidate=f"tto_{variant}")
        for variant, records in variant_records.items()
    ]
    normal = next((item for item in candidate_metrics if item["candidate"] == "tto_normal"), None)
    nulls = [item for item in candidate_metrics if item["candidate"] != "tto_normal"]
    metrics: dict[str, Any] = {
        "experiment": "offset_tto_v0",
        "config": asdict(config),
        "wells": int(len(wells)),
        "candidates": candidate_metrics,
        "best_candidate": min(candidate_metrics, key=lambda item: item.get("row_rmse", float("inf"))),
        "normal_minus_best_null_rmse": (
            float(normal["row_rmse"] - min(item["row_rmse"] for item in nulls))
            if normal is not None and nulls
            else float("nan")
        ),
    }
    (out_dir / "offset_tto_metrics.json").write_text(
        json.dumps(_json_safe(metrics), indent=2), encoding="utf-8"
    )
    _write_report(out_dir, metrics)
    return metrics


def _write_report(output_dir: Path, metrics: dict[str, Any]) -> None:
    lines = [
        "# OFFSET_TTO_V0",
        "",
        "Differentiable local refinement of the `TVT = last_TVT - dZ + offset` path.",
        "The optimizer starts from a posterior top-k offset shortlist and adjusts",
        "only a bounded scalar offset correction per candidate.",
        "",
        "## metrics",
        "",
        "| candidate | row RMSE | mean well | p95 | worst | selected loss |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in metrics["candidates"]:
        lines.append(
            f"| {item['candidate']} | {item.get('row_rmse', float('nan')):.4f} | "
            f"{item.get('mean_well_rmse', float('nan')):.4f} | "
            f"{item.get('p95_well_rmse', float('nan')):.4f} | "
            f"{item.get('worst_well_rmse', float('nan')):.4f} | "
            f"{item.get('selected_loss_mean', float('nan')):.4f} |"
        )
    lines.extend(
        [
            "",
            f"`normal_minus_best_null_rmse`: {metrics.get('normal_minus_best_null_rmse', float('nan')):.4f}",
            "",
            "Decision rule: normal GR must beat shuffled/zero GR on row RMSE.",
            "If it does not, the differentiable optimizer is not extracting a",
            "deployable GR signal; it is only finding local matches that do not",
            "transfer to true TVT.",
        ]
    )
    (output_dir / "offset_tto_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run offset test-time optimization diagnostic")
    parser.add_argument("--data-dir", type=Path, default=OffsetTTOConfig.data_dir)
    parser.add_argument("--output-dir", type=Path, default=OffsetTTOConfig.output_dir)
    parser.add_argument("--posterior-scores-path", type=Path, default=OffsetTTOConfig.posterior_scores_path)
    parser.add_argument("--no-posterior-scores", action="store_true")
    parser.add_argument("--offset-grid", type=str, default=OffsetTTOConfig.offset_grid)
    parser.add_argument("--score-column", type=str, default=OffsetTTOConfig.score_column)
    parser.add_argument("--score-higher-is-better", action="store_true")
    parser.add_argument("--top-k", type=int, default=OffsetTTOConfig.top_k)
    parser.add_argument("--n-folds", type=int, default=OffsetTTOConfig.n_folds)
    parser.add_argument("--seed", type=int, default=OffsetTTOConfig.seed)
    parser.add_argument("--k-wells", type=int, default=OffsetTTOConfig.k_wells)
    parser.add_argument("--steps", type=int, default=OffsetTTOConfig.steps)
    parser.add_argument("--learning-rate", type=float, default=OffsetTTOConfig.learning_rate)
    parser.add_argument("--max-delta", type=float, default=OffsetTTOConfig.max_delta)
    parser.add_argument("--offset-l2", type=float, default=OffsetTTOConfig.offset_l2)
    parser.add_argument("--oob-penalty", type=float, default=OffsetTTOConfig.oob_penalty)
    parser.add_argument("--no-normalize-gr", action="store_true")
    parser.add_argument("--no-nulls", action="store_true")
    parser.add_argument("--progress-every", type=int, default=OffsetTTOConfig.progress_every)
    args = parser.parse_args(argv)
    metrics = run_offset_tto(
        OffsetTTOConfig(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            posterior_scores_path=None if args.no_posterior_scores else args.posterior_scores_path,
            offset_grid=args.offset_grid,
            score_column=args.score_column,
            score_lower_is_better=not args.score_higher_is_better,
            top_k=args.top_k,
            n_folds=args.n_folds,
            seed=args.seed,
            k_wells=args.k_wells,
            steps=args.steps,
            learning_rate=args.learning_rate,
            max_delta=args.max_delta,
            offset_l2=args.offset_l2,
            oob_penalty=args.oob_penalty,
            normalize_gr=not args.no_normalize_gr,
            include_nulls=not args.no_nulls,
            progress_every=args.progress_every,
        )
    )
    print(
        json.dumps(
            {
                "experiment": metrics["experiment"],
                "best_candidate": metrics.get("best_candidate", {}),
                "normal_minus_best_null_rmse": metrics.get("normal_minus_best_null_rmse"),
                "report": str(Path(args.output_dir) / "offset_tto_report.md"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
