from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from .config import MTPConfig, load_config
from .heatmap import fill_nan
from .io import discover_wells, load_well
from .model import MTPNet
from .priors import load_prior_tables
from .train import _bins_to_tvt, _loader, prepare_sample_splits, resolve_device


def triangular_weights(length: int) -> np.ndarray:
    if length <= 0:
        return np.empty(0, dtype=np.float32)
    if length == 1:
        return np.ones(1, dtype=np.float32)
    center = (length - 1) / 2.0
    return np.asarray(
        [1.0 - abs(index - center) / (center + 1.0) for index in range(length)],
        dtype=np.float32,
    )


def _softmax_np(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr - np.nanmax(arr)
    exp = np.exp(arr)
    total = exp.sum()
    if total <= 0.0 or not np.isfinite(total):
        return np.full(len(arr), 1.0 / max(1, len(arr)), dtype=np.float32)
    return (exp / total).astype(np.float32)


def _mode_weights(
    logits: np.ndarray,
    strategy: str,
    *,
    top_n: int = 3,
    confidence_gap: float = 1.0,
) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    prob = _softmax_np(logits)
    weights = np.zeros_like(prob)
    order = np.argsort(-logits)
    if strategy == "top1":
        weights[order[0]] = 1.0
    elif strategy == "weighted":
        weights = prob
    elif strategy == "top3":
        selected = order[: min(top_n, len(order))]
        selected_prob = prob[selected]
        weights[selected] = selected_prob / selected_prob.sum()
    elif strategy == "confident_top1_else_weighted":
        gap = float(logits[order[0]] - logits[order[1]]) if len(order) > 1 else np.inf
        if gap >= confidence_gap:
            weights[order[0]] = 1.0
        else:
            selected = order[: min(top_n, len(order))]
            selected_prob = prob[selected]
            weights[selected] = selected_prob / selected_prob.sum()
    else:
        raise ValueError(f"Unsupported stitch strategy: {strategy}")
    return weights.astype(np.float32)


def aggregate_mode_windows(
    windows: pd.DataFrame,
    *,
    history_steps: int,
    future_steps: int,
    strategy: str,
    top_n: int = 3,
    confidence_gap: float = 1.0,
) -> pd.DataFrame:
    accum: dict[tuple[str, int], list[float]] = {}
    time_weights = triangular_weights(future_steps)
    for row in windows.itertuples(index=False):
        logits = np.asarray(row.logits, dtype=np.float32)
        paths = np.asarray(row.path_tvt, dtype=np.float32)
        mode_weights = _mode_weights(
            logits, strategy, top_n=top_n, confidence_gap=confidence_gap
        )
        for future_index in range(future_steps):
            step = int(row.start_step) + history_steps + future_index
            for mode_index, mode_weight in enumerate(mode_weights):
                if mode_weight <= 0.0:
                    continue
                pred = float(paths[mode_index, future_index])
                if not np.isfinite(pred):
                    continue
                weight = float(mode_weight * time_weights[future_index])
                key = (str(row.well_id), step)
                item = accum.setdefault(key, [0.0, 0.0])
                item[0] += pred * weight
                item[1] += weight
    rows = [
        {"well_id": well_id, "step": step, "pred_tvt": total / weight}
        for (well_id, step), (total, weight) in accum.items()
        if weight > 0.0
    ]
    return pd.DataFrame(rows).sort_values(["well_id", "step"]).reset_index(drop=True)


def aggregate_window_oracle(
    windows: pd.DataFrame,
    *,
    history_steps: int,
    future_steps: int,
) -> pd.DataFrame:
    oracle_rows: list[dict[str, Any]] = []
    for row in windows.itertuples(index=False):
        paths = np.asarray(row.path_tvt, dtype=np.float32)
        target = np.asarray(row.target_tvt, dtype=np.float32)
        err = np.sqrt(np.mean(np.square(paths - target[None, :]), axis=1))
        best = int(np.nanargmin(err))
        oracle_rows.append(
            {
                "well_id": row.well_id,
                "start_step": int(row.start_step),
                "logits": np.array([1.0], dtype=np.float32),
                "path_tvt": paths[best : best + 1],
            }
        )
    return aggregate_mode_windows(
        pd.DataFrame(oracle_rows),
        history_steps=history_steps,
        future_steps=future_steps,
        strategy="top1",
    )


def _mode_step_candidates(
    windows: pd.DataFrame,
    *,
    history_steps: int,
    future_steps: int,
    top_n: int | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in windows.itertuples(index=False):
        logits = np.asarray(row.logits, dtype=np.float32)
        paths = np.asarray(row.path_tvt, dtype=np.float32)
        if top_n is None:
            mode_indices = np.arange(len(logits))
        else:
            mode_indices = np.argsort(-logits)[: min(top_n, len(logits))]
        for future_index in range(future_steps):
            step = int(row.start_step) + history_steps + future_index
            for mode_index in mode_indices:
                pred = float(paths[mode_index, future_index])
                if np.isfinite(pred):
                    rows.append(
                        {
                            "well_id": str(row.well_id),
                            "step": int(step),
                            "pred_tvt": pred,
                            "window_start_step": int(row.start_step),
                            "mode_index": int(mode_index),
                        }
                    )
    return pd.DataFrame(rows)


def row_oracle_predictions(
    windows: pd.DataFrame,
    hidden_rows: pd.DataFrame,
    *,
    history_steps: int,
    future_steps: int,
    top_n: int | None = None,
    anchor_column: str = "base_tvt",
) -> pd.DataFrame:
    candidates = _mode_step_candidates(
        windows, history_steps=history_steps, future_steps=future_steps, top_n=top_n
    )
    if candidates.empty:
        return candidates
    step_mean = _step_base_means(hidden_rows, anchor_column)
    merged = hidden_rows.merge(candidates, on=["well_id", "step"], how="inner")
    merged = merged.merge(step_mean, on=["well_id", "step"], how="left")
    anchor_mean = merged[f"{anchor_column}_step_mean"]
    adjusted = np.where(
        np.isfinite(merged[anchor_column]) & np.isfinite(anchor_mean),
        merged[anchor_column] + (merged["pred_tvt"] - anchor_mean),
        merged["pred_tvt"],
    )
    merged = merged.assign(pred_tvt=adjusted.astype(np.float32))
    merged["abs_err"] = (merged["pred_tvt"] - merged["TVT"]).abs()
    best = (
        merged.sort_values(["id", "abs_err", "window_start_step", "mode_index"])
        .groupby("id", as_index=False)
        .head(1)
    )
    return best[
        ["id", "well_id", "row_idx", "step", "TVT", "GR", "base_tvt", "b2_tvt", "pred_tvt"]
    ].reset_index(drop=True)


def dp_decode_mode_windows(
    windows: pd.DataFrame,
    *,
    history_steps: int,
    future_steps: int,
    top_n: int = 5,
    lambda_step: float = 0.05,
    bin_size_ft: float = 2.0,
) -> pd.DataFrame:
    time_weights = triangular_weights(future_steps)
    posterior: dict[str, dict[int, dict[float, float]]] = {}
    for row in windows.itertuples(index=False):
        logits = np.asarray(row.logits, dtype=np.float32)
        paths = np.asarray(row.path_tvt, dtype=np.float32)
        prob = _softmax_np(logits)
        order = np.argsort(-logits)[: min(top_n, len(logits))]
        well_steps = posterior.setdefault(str(row.well_id), {})
        for future_index in range(future_steps):
            step = int(row.start_step) + history_steps + future_index
            step_mass = well_steps.setdefault(step, {})
            for mode_index in order:
                value = float(paths[mode_index, future_index])
                if not np.isfinite(value):
                    continue
                center = round(value / bin_size_ft) * bin_size_ft
                mass = float(prob[mode_index] * time_weights[future_index])
                step_mass[center] = step_mass.get(center, 0.0) + mass

    out_rows: list[dict[str, Any]] = []
    for well_id, by_step in posterior.items():
        steps = sorted(by_step)
        if not steps:
            continue
        states: list[np.ndarray] = []
        node_costs: list[np.ndarray] = []
        for step in steps:
            items = sorted(by_step[step].items())
            values = np.asarray([item[0] for item in items], dtype=np.float32)
            mass = np.asarray([item[1] for item in items], dtype=np.float32)
            mass = mass / max(float(mass.sum()), 1e-8)
            states.append(values)
            node_costs.append(-np.log(mass.clip(1e-8)))

        costs = [node_costs[0]]
        parents: list[np.ndarray] = []
        for index in range(1, len(steps)):
            prev = states[index - 1]
            cur = states[index]
            transition = float(lambda_step) * np.abs(cur[:, None] - prev[None, :])
            total = costs[-1][None, :] + transition + node_costs[index][:, None]
            parent = total.argmin(axis=1)
            costs.append(total[np.arange(len(cur)), parent])
            parents.append(parent.astype(np.int32))

        selected = [int(costs[-1].argmin())]
        for parent in reversed(parents):
            selected.append(int(parent[selected[-1]]))
        selected = list(reversed(selected))
        for step, values, state_index in zip(steps, states, selected, strict=True):
            out_rows.append(
                {"well_id": well_id, "step": int(step), "pred_tvt": float(values[state_index])}
            )
    return pd.DataFrame(out_rows).sort_values(["well_id", "step"]).reset_index(drop=True)


def _load_run_config(run_dir: Path) -> MTPConfig:
    config_path = run_dir / "config_resolved.yml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing resolved config: {config_path}")
    return load_config(config_path)


def _predict_valid_modes(run_dir: Path, cfg: MTPConfig) -> pd.DataFrame:
    device = resolve_device(cfg.train.device)
    splits = prepare_sample_splits(cfg)
    samples = splits.valid_sets.get(splits.primary_valid_name, splits.valid_samples)
    first = samples[0]
    model = MTPNet(
        in_channels=first.x.shape[0],
        height=first.x.shape[1],
        width=first.x.shape[2],
        future_steps=cfg.window.future_steps,
        cfg=cfg.model,
    ).to(device)
    checkpoint_path = run_dir / "checkpoints" / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in _loader(samples, cfg, shuffle=False):
            x = batch["x"].to(device)
            crop_tvt = batch["crop_tvt"].to(device)
            raw_paths, logits = model.forward_raw(x)
            paths = model.bound_paths(raw_paths)
            probs = F.softmax(logits, dim=1)
            path_tvt = _bins_to_tvt(paths.cpu().numpy(), crop_tvt.cpu().numpy())
            for index in range(paths.shape[0]):
                rows.append(
                    {
                        "well_id": batch["well_id"][index],
                        "start_step": int(batch["start_step"][index]),
                        "sample_type": batch["sample_type"][index],
                        "logits": logits[index].cpu().numpy().astype(np.float32),
                        "probs": probs[index].cpu().numpy().astype(np.float32),
                        "path_tvt": path_tvt[index].astype(np.float32),
                        "target_tvt": batch["target_tvt"][index].cpu().numpy().astype(
                            np.float32
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _serializable_mode_windows(mode_windows: pd.DataFrame) -> pd.DataFrame:
    out = mode_windows.copy()
    for column in ("logits", "probs", "path_tvt", "target_tvt", "gr_scores"):
        if column in out.columns:
            out[column] = out[column].map(
                lambda value: np.asarray(value, dtype=np.float32).tolist()
            )
    return out


def _row_id(well_id: str, row_idx: int) -> str:
    return f"{well_id}_{row_idx}"


def _load_hidden_rows(cfg: MTPConfig, well_ids: set[str]) -> pd.DataFrame:
    prior_tables = load_prior_tables(cfg.priors)
    prior_frame = prior_tables.frame if prior_tables is not None else pd.DataFrame()
    rows: list[pd.DataFrame] = []
    by_id = {well.well_id: well for well in discover_wells(cfg.data)}
    for well_id in sorted(well_ids):
        horizontal, _ = load_well(by_id[well_id])
        frame = horizontal.reset_index(drop=True).copy()
        frame["well_id"] = well_id
        frame["row_idx"] = np.arange(len(frame), dtype=np.int32)
        if "id" not in frame.columns:
            frame["id"] = [_row_id(well_id, int(idx)) for idx in frame["row_idx"]]
        hidden = frame["TVT_input"].isna() & frame["TVT"].notna()
        frame = frame.loc[hidden, ["id", "well_id", "row_idx", "TVT", "GR"]].copy()
        frame["step"] = (frame["row_idx"] // cfg.window.rows_per_step).astype(int)
        if not prior_frame.empty:
            aligned = prior_frame.reindex(frame["id"].astype(str))
            for column in ("base_tvt", "b2_tvt"):
                if column in aligned.columns:
                    frame[column] = pd.to_numeric(aligned[column], errors="coerce").to_numpy(
                        dtype=np.float32
                    )
        rows.append(frame)
    if not rows:
        raise RuntimeError("No hidden rows found for stitch validation wells")
    out = pd.concat(rows, ignore_index=True)
    if "base_tvt" not in out.columns:
        out["base_tvt"] = np.nan
    if "b2_tvt" not in out.columns:
        out["b2_tvt"] = np.nan
    return out


def _step_base_means(rows: pd.DataFrame, column: str) -> pd.DataFrame:
    return (
        rows.groupby(["well_id", "step"], as_index=False)[column]
        .mean()
        .rename(columns={column: f"{column}_step_mean"})
    )


def _apply_step_predictions_to_rows(
    hidden_rows: pd.DataFrame,
    step_predictions: pd.DataFrame,
    *,
    anchor_column: str = "base_tvt",
) -> pd.DataFrame:
    step_mean = _step_base_means(hidden_rows, anchor_column)
    merged = hidden_rows.merge(step_predictions, on=["well_id", "step"], how="inner")
    merged = merged.merge(step_mean, on=["well_id", "step"], how="left")
    anchor_mean = merged[f"{anchor_column}_step_mean"]
    pred = merged["pred_tvt"]
    anchor = merged[anchor_column]
    adjusted = np.where(
        np.isfinite(anchor) & np.isfinite(anchor_mean),
        anchor + (pred - anchor_mean),
        pred,
    )
    out = merged[["id", "well_id", "row_idx", "step", "TVT", "GR", "base_tvt", "b2_tvt"]].copy()
    out["pred_tvt"] = adjusted.astype(np.float32)
    return out


def _blend_with_anchor(
    rows: pd.DataFrame,
    *,
    anchor_column: str,
    alpha: float,
    clip: float,
) -> pd.DataFrame:
    out = rows.copy()
    delta = (out["pred_tvt"] - out[anchor_column]).clip(-clip, clip)
    out["pred_tvt"] = out[anchor_column] + float(alpha) * delta
    return out[np.isfinite(out["pred_tvt"])]


def evaluate_with_b2_fallback(
    hidden_rows: pd.DataFrame, predictions: pd.DataFrame, candidate_name: str
) -> dict[str, Any]:
    covered = predictions[["id", "pred_tvt"]].copy()
    merged = hidden_rows.merge(covered, on="id", how="left")
    covered_mask = np.isfinite(merged["pred_tvt"])
    merged["pred_tvt"] = merged["pred_tvt"].where(covered_mask, merged["b2_tvt"])
    metrics = evaluate_row_predictions(hidden_rows, merged, candidate_name)
    covered_metrics = evaluate_row_predictions(
        hidden_rows,
        merged.loc[covered_mask].copy(),
        f"{candidate_name}_covered",
    )
    uncovered_metrics = evaluate_row_predictions(
        hidden_rows,
        merged.loc[~covered_mask].copy(),
        f"{candidate_name}_uncovered_b2",
    )
    metrics.update(
        {
            "covered_rows": int(covered_mask.sum()),
            "uncovered_rows": int((~covered_mask).sum()),
            "coverage_frac": float(covered_mask.mean()),
            "covered_rmse": float(covered_metrics.get("rmse", float("nan"))),
            "uncovered_fallback_rmse": float(
                uncovered_metrics.get("rmse", float("nan"))
            ),
        }
    )
    return metrics


def evaluate_row_predictions(
    hidden_rows: pd.DataFrame, predictions: pd.DataFrame, candidate_name: str
) -> dict[str, Any]:
    if {"TVT", "pred_tvt"}.issubset(predictions.columns):
        merged = predictions.copy()
    else:
        merged = hidden_rows.merge(predictions, on=["well_id", "step"], how="inner")
    merged = merged[np.isfinite(merged["TVT"]) & np.isfinite(merged["pred_tvt"])]
    if merged.empty:
        return {"candidate": candidate_name, "rows": 0, "rmse": float("nan")}
    err = pd.to_numeric(merged["pred_tvt"], errors="coerce") - pd.to_numeric(
        merged["TVT"], errors="coerce"
    )
    well_rmse = (
        merged.assign(sq_err=np.square(err))
        .groupby("well_id")["sq_err"]
        .mean()
        .pow(0.5)
    )
    hidden_counts = merged.groupby("well_id").size()
    median_rows = float(hidden_counts.median())
    long_wells = set(hidden_counts[hidden_counts >= median_rows].index)
    short_wells = set(hidden_counts[hidden_counts < median_rows].index)
    shift = (
        pd.to_numeric(merged["pred_tvt"], errors="coerce")
        - pd.to_numeric(merged["b2_tvt"], errors="coerce")
    ).abs()
    endpoint_shift = (
        merged.sort_values("row_idx")
        .groupby("well_id")
        .tail(1)
        .assign(endpoint_shift=lambda item: (item["pred_tvt"] - item["b2_tvt"]).abs())
    )

    def _rmse_for(mask: pd.Series) -> float:
        if not mask.any():
            return float("nan")
        local = err[mask]
        return float(np.sqrt(np.mean(np.square(local))))

    return {
        "candidate": candidate_name,
        "rows": int(len(merged)),
        "wells": int(merged["well_id"].nunique()),
        "rmse": float(np.sqrt(np.mean(np.square(err)))),
        "mean_well_rmse": float(well_rmse.mean()),
        "p50_well_rmse": float(well_rmse.quantile(0.50)),
        "p90_well_rmse": float(well_rmse.quantile(0.90)),
        "p95_well_rmse": float(well_rmse.quantile(0.95)),
        "worst_well_rmse": float(well_rmse.max()),
        "long_well_rmse": _rmse_for(merged["well_id"].isin(long_wells)),
        "short_well_rmse": _rmse_for(merged["well_id"].isin(short_wells)),
        "high_gr_nan_rmse": _rmse_for(merged["GR"].isna()),
        "p95_abs_shift_vs_b2": float(shift.quantile(0.95)),
        "endpoint_shift_p95_vs_b2": float(endpoint_shift["endpoint_shift"].quantile(0.95)),
    }


def _robust_z(values: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=np.float32)
    med = float(np.nanmedian(arr[finite]))
    q25, q75 = np.nanpercentile(arr[finite], [25.0, 75.0])
    scale = max(float(q75 - q25), eps)
    return ((arr - med) / scale).astype(np.float32)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a_arr = np.asarray(a, dtype=np.float32)
    b_arr = np.asarray(b, dtype=np.float32)
    mask = np.isfinite(a_arr) & np.isfinite(b_arr)
    if mask.sum() < 3:
        return 0.0
    x = a_arr[mask]
    y = b_arr[mask]
    if float(np.std(x)) < 1e-8 or float(np.std(y)) < 1e-8:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _gr_mode_score(horizontal_gr: np.ndarray, typewell_gr: np.ndarray) -> float:
    h_z = _robust_z(horizontal_gr)
    tw_z = _robust_z(typewell_gr)
    path_corr = _corr(h_z, tw_z)
    dgr_corr = _corr(np.gradient(h_z), np.gradient(tw_z))
    mad = float(np.nanmedian(np.abs(h_z - tw_z))) if np.isfinite(h_z - tw_z).any() else 0.0
    return float(path_corr + 0.8 * dgr_corr - 0.1 * mad)


def _compress_mean(values: np.ndarray, rows_per_step: int) -> np.ndarray:
    usable = (len(values) // rows_per_step) * rows_per_step
    if usable <= 0:
        return np.empty(0, dtype=np.float32)
    arr = np.asarray(values[:usable], dtype=np.float32).reshape(-1, rows_per_step)
    return np.nanmean(arr, axis=1).astype(np.float32)


def _load_gr_context(cfg: MTPConfig, well_ids: set[str]) -> dict[str, dict[str, np.ndarray]]:
    context: dict[str, dict[str, np.ndarray]] = {}
    by_id = {well.well_id: well for well in discover_wells(cfg.data)}
    for well_id in sorted(well_ids):
        horizontal, typewell = load_well(by_id[well_id])
        gr_raw = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(dtype=np.float32)
        gr, _ = fill_nan(gr_raw)
        typewell_tvt = pd.to_numeric(typewell["TVT"], errors="coerce").to_numpy(dtype=np.float32)
        typewell_gr_raw = pd.to_numeric(typewell["GR"], errors="coerce").to_numpy(dtype=np.float32)
        typewell_gr, _ = fill_nan(typewell_gr_raw)
        order = np.argsort(typewell_tvt)
        finite = np.isfinite(typewell_tvt[order]) & np.isfinite(typewell_gr[order])
        context[well_id] = {
            "horizontal_gr": _compress_mean(gr, cfg.window.rows_per_step),
            "typewell_tvt": typewell_tvt[order][finite].astype(np.float32),
            "typewell_gr": typewell_gr[order][finite].astype(np.float32),
        }
    return context


def attach_gr_rerank_scores(
    windows: pd.DataFrame,
    context: dict[str, dict[str, np.ndarray]],
    *,
    history_steps: int,
    future_steps: int,
    beta: float,
) -> pd.DataFrame:
    out = windows.copy()
    new_logits: list[np.ndarray] = []
    gr_scores: list[np.ndarray] = []
    for row in out.itertuples(index=False):
        ctx = context[str(row.well_id)]
        start = int(row.start_step) + history_steps
        h_gr = ctx["horizontal_gr"][start : start + future_steps]
        paths = np.asarray(row.path_tvt, dtype=np.float32)
        scores = []
        for mode_index in range(paths.shape[0]):
            tw_gr = np.interp(
                paths[mode_index],
                ctx["typewell_tvt"],
                ctx["typewell_gr"],
                left=ctx["typewell_gr"][0],
                right=ctx["typewell_gr"][-1],
            )
            scores.append(_gr_mode_score(h_gr, tw_gr))
        scores_arr = np.asarray(scores, dtype=np.float32)
        finite = np.isfinite(scores_arr)
        z = np.zeros_like(scores_arr)
        if finite.any():
            mean = float(scores_arr[finite].mean())
            std = max(float(scores_arr[finite].std()), 1e-6)
            z[finite] = (scores_arr[finite] - mean) / std
        gr_scores.append(scores_arr)
        new_logits.append(np.asarray(row.logits, dtype=np.float32) + float(beta) * z)
    out["gr_scores"] = gr_scores
    out["logits"] = new_logits
    return out


def _window_metrics_from_mode_windows(windows: pd.DataFrame) -> dict[str, float]:
    top1_errors: list[float] = []
    weighted_errors: list[float] = []
    top3_errors: list[float] = []
    best_top3: list[float] = []
    for row in windows.itertuples(index=False):
        paths = np.asarray(row.path_tvt, dtype=np.float32)
        target = np.asarray(row.target_tvt, dtype=np.float32)
        logits = np.asarray(row.logits, dtype=np.float32)
        prob = _softmax_np(logits)
        err = np.sqrt(np.mean(np.square(paths - target[None, :]), axis=1))
        order = np.argsort(-logits)
        top1_errors.append(float(err[order[0]]))
        weighted = (paths * prob[:, None]).sum(axis=0)
        weighted_errors.append(float(np.sqrt(np.mean(np.square(weighted - target)))))
        top3 = order[: min(3, len(order))]
        top3_errors.append(float(err[top3].min()))
        best = int(np.argmin(err))
        best_top3.append(float(best in set(top3.tolist())))
    return {
        "window_top1_ft": float(np.mean(top1_errors)),
        "window_weighted_ft": float(np.mean(weighted_errors)),
        "window_oracle_top3_by_logit_ft": float(np.mean(top3_errors)),
        "window_best_mode_top3_rate": float(np.mean(best_top3)),
    }


def _baseline_metrics(hidden_rows: pd.DataFrame, column: str, name: str) -> dict[str, Any]:
    pred = hidden_rows[
        ["id", "well_id", "row_idx", "step", "TVT", "GR", "base_tvt", "b2_tvt"]
    ].copy()
    pred["pred_tvt"] = pred[column]
    return evaluate_row_predictions(hidden_rows, pred, name)


def _candidate_table(metrics: list[dict[str, Any]]) -> str:
    columns = [
        "candidate",
        "rmse",
        "covered_rmse",
        "mean_well_rmse",
        "p50_well_rmse",
        "p90_well_rmse",
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


def _is_oracle_candidate(name: str) -> bool:
    return "oracle" in str(name)


def write_stitch_report(
    *,
    run_dir: Path,
    summary: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> Path:
    deployable = [
        item for item in candidates if not _is_oracle_candidate(str(item["candidate"]))
    ]
    best_deployable = min(deployable, key=lambda item: item.get("rmse", float("inf")))
    best_oracle = min(candidates, key=lambda item: item.get("rmse", float("inf")))
    lines = [
        "MTP_V1_2_ORACLE_RERANK_REPORT",
        "",
        "base:",
        f"  base_schema10_pp_full_rmse: {summary['base_schema10_pp']['rmse']}",
        f"  b2_guarded_submit_full_rmse: {summary['b2_guarded_submit']['rmse']}",
        f"  base_schema10_pp_covered_rmse: {summary['base_schema10_pp_covered']['rmse']}",
        f"  b2_guarded_submit_covered_rmse: {summary['b2_guarded_submit_covered']['rmse']}",
        "",
        "window metrics:",
        f"  top1_ft: {summary['window']['top1_rmse_ft']}",
        f"  weighted_ft: {summary['window']['weighted_mean_rmse_ft']}",
        f"  oracle_topK_ft: {summary['window']['oracle_topk_rmse_ft']}",
        f"  best_mode_top3_rate: {summary['window']['best_mode_top3_rate']}",
        "",
        "coverage:",
        f"  covered_hidden_rows: {summary['coverage']['covered_hidden_rows']}",
        f"  total_hidden_rows: {summary['coverage']['total_hidden_rows']}",
        f"  coverage_frac: {summary['coverage']['coverage_frac']}",
        f"  uncovered_fallback_rmse: {summary['coverage']['uncovered_fallback_rmse']}",
        "",
        "best deployable stitched:",
        f"  candidate: {best_deployable['candidate']}",
        f"  rmse: {best_deployable['rmse']}",
        f"  covered_rmse: {best_deployable.get('covered_rmse', 'n/a')}",
        f"  p95_abs_shift_vs_b2: {best_deployable.get('p95_abs_shift_vs_b2', 'n/a')}",
        f"  worst_well_rmse: {best_deployable.get('worst_well_rmse', 'n/a')}",
        "",
        "best oracle diagnostic:",
        f"  candidate: {best_oracle['candidate']}",
        f"  rmse: {best_oracle['rmse']}",
        f"  covered_rmse: {best_oracle.get('covered_rmse', 'n/a')}",
        "",
        "row-level MTP oracle:",
        f"  mtp_window_oracle_overlap: {summary['oracle']['mtp_window_oracle_overlap']['rmse']}",
        f"  mtp_top3_logit_row_oracle: {summary['oracle']['mtp_top3_logit_row_oracle']['rmse']}",
        f"  mtp_row_oracle: {summary['oracle']['mtp_row_oracle']['rmse']}",
        f"  b2_plus_mtp_oracle_a0.3_clip20: {summary['oracle']['b2_plus_mtp_oracle_a0.3_clip20']['rmse']}",
        "",
        "B/NCC rerank:",
        json.dumps(summary.get("rerank_window_metrics", {}), indent=2),
        "",
        "stitched row-level:",
        _candidate_table(candidates),
        "",
        "decision:",
        "  deployable_beats_b2: "
        f"{best_deployable['rmse'] < summary['b2_guarded_submit']['rmse']}",
        f"  oracle_headroom_strong: {best_oracle['rmse'] < 8.5}",
    ]
    path = run_dir / "stitch_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_stitch(run_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir)
    cfg = _load_run_config(run_path)
    mode_windows = _predict_valid_modes(run_path, cfg)
    _serializable_mode_windows(mode_windows).to_parquet(
        run_path / "stitch_window_modes.parquet", index=False
    )

    metrics_path = run_path / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    well_ids = set(mode_windows["well_id"].astype(str))
    hidden_rows_all = _load_hidden_rows(cfg, well_ids)

    candidate_steps: dict[str, pd.DataFrame] = {
        "mtp_top1_overlap": aggregate_mode_windows(
            mode_windows,
            history_steps=cfg.window.history_steps,
            future_steps=cfg.window.future_steps,
            strategy="top1",
        ),
        "mtp_weighted_overlap": aggregate_mode_windows(
            mode_windows,
            history_steps=cfg.window.history_steps,
            future_steps=cfg.window.future_steps,
            strategy="weighted",
        ),
        "mtp_top3_logit_overlap": aggregate_mode_windows(
            mode_windows,
            history_steps=cfg.window.history_steps,
            future_steps=cfg.window.future_steps,
            strategy="top3",
        ),
        "mtp_confident_top1_else_weighted": aggregate_mode_windows(
            mode_windows,
            history_steps=cfg.window.history_steps,
            future_steps=cfg.window.future_steps,
            strategy="confident_top1_else_weighted",
        ),
        "mtp_window_oracle_overlap": aggregate_window_oracle(
            mode_windows,
            history_steps=cfg.window.history_steps,
            future_steps=cfg.window.future_steps,
        ),
    }
    for lambda_step in (0.03, 0.05, 0.10):
        candidate_steps[f"mtp_dp_decode_l1_{lambda_step:g}"] = dp_decode_mode_windows(
            mode_windows,
            history_steps=cfg.window.history_steps,
            future_steps=cfg.window.future_steps,
            lambda_step=lambda_step,
        )

    gr_context = _load_gr_context(cfg, well_ids)
    rerank_window_metrics: dict[str, Any] = {}
    for beta in (0.25, 0.5, 1.0, 2.0):
        reranked = attach_gr_rerank_scores(
            mode_windows,
            gr_context,
            history_steps=cfg.window.history_steps,
            future_steps=cfg.window.future_steps,
            beta=beta,
        )
        beta_name = f"beta_{beta:g}"
        rerank_window_metrics[beta_name] = _window_metrics_from_mode_windows(reranked)
        candidate_steps[f"mtp_gr_rerank_weighted_b{beta:g}"] = aggregate_mode_windows(
            reranked,
            history_steps=cfg.window.history_steps,
            future_steps=cfg.window.future_steps,
            strategy="weighted",
        )
        candidate_steps[f"mtp_gr_rerank_top1_b{beta:g}"] = aggregate_mode_windows(
            reranked,
            history_steps=cfg.window.history_steps,
            future_steps=cfg.window.future_steps,
            strategy="top1",
        )

    covered_keys = pd.concat(candidate_steps.values(), ignore_index=True)[
        ["well_id", "step"]
    ].drop_duplicates()
    hidden_rows = hidden_rows_all.merge(covered_keys, on=["well_id", "step"], how="inner")
    total_hidden_rows = int(len(hidden_rows_all))
    covered_hidden_rows = int(len(hidden_rows))

    candidate_metrics: list[dict[str, Any]] = []
    row_predictions: list[pd.DataFrame] = []

    def add_candidate(name: str, rows: pd.DataFrame) -> None:
        rows = rows.copy()
        rows["candidate"] = name
        row_predictions.append(rows)
        candidate_metrics.append(evaluate_with_b2_fallback(hidden_rows_all, rows, name))

    for name, steps in candidate_steps.items():
        rows = _apply_step_predictions_to_rows(hidden_rows, steps, anchor_column="base_tvt")
        add_candidate(name, rows)
        if name in {"mtp_weighted_overlap", "mtp_top3_logit_overlap"} or name.startswith(
            "mtp_gr_rerank_weighted"
        ):
            for anchor_name, anchor_col in (("b2", "b2_tvt"), ("base", "base_tvt")):
                for alpha in (0.1, 0.2, 0.3, 0.5):
                    for clip in (10.0, 20.0, 30.0):
                        blend_name = f"{anchor_name}_plus_{name}_a{alpha:g}_clip{int(clip)}"
                        blended = _blend_with_anchor(
                            rows,
                            anchor_column=anchor_col,
                            alpha=alpha,
                            clip=clip,
                        )
                        add_candidate(blend_name, blended)

    row_oracle = row_oracle_predictions(
        mode_windows,
        hidden_rows,
        history_steps=cfg.window.history_steps,
        future_steps=cfg.window.future_steps,
    )
    top3_row_oracle = row_oracle_predictions(
        mode_windows,
        hidden_rows,
        history_steps=cfg.window.history_steps,
        future_steps=cfg.window.future_steps,
        top_n=3,
    )
    add_candidate("mtp_row_oracle", row_oracle)
    add_candidate("mtp_top3_logit_row_oracle", top3_row_oracle)
    oracle_blend = _blend_with_anchor(
        row_oracle,
        anchor_column="b2_tvt",
        alpha=0.3,
        clip=20.0,
    )
    add_candidate("b2_plus_mtp_oracle_a0.3_clip20", oracle_blend)

    base_metrics = _baseline_metrics(hidden_rows_all, "base_tvt", "base_schema10_pp")
    b2_metrics = _baseline_metrics(hidden_rows_all, "b2_tvt", "b2_guarded_submit")
    base_metrics_covered = _baseline_metrics(
        hidden_rows, "base_tvt", "base_schema10_pp_covered"
    )
    b2_metrics_covered = _baseline_metrics(
        hidden_rows, "b2_tvt", "b2_guarded_submit_covered"
    )
    uncovered = hidden_rows_all.merge(
        covered_keys.assign(_covered=1), on=["well_id", "step"], how="left"
    )
    uncovered = uncovered.loc[uncovered["_covered"].isna()].copy()
    uncovered_b2 = _baseline_metrics(uncovered, "b2_tvt", "b2_uncovered_fallback")
    candidate_frame = pd.DataFrame(candidate_metrics).sort_values("rmse")
    candidate_frame.to_csv(run_path / "stitch_candidates.csv", index=False)
    pd.concat(row_predictions, ignore_index=True).to_parquet(
        run_path / "stitch_row_predictions.parquet", index=False
    )

    summary = {
        "run_name": metrics.get("run_name", run_path.name),
        "window": metrics["valid"],
        "base_schema10_pp": base_metrics,
        "b2_guarded_submit": b2_metrics,
        "base_schema10_pp_covered": base_metrics_covered,
        "b2_guarded_submit_covered": b2_metrics_covered,
        "coverage": {
            "covered_hidden_rows": covered_hidden_rows,
            "total_hidden_rows": total_hidden_rows,
            "coverage_frac": covered_hidden_rows / max(1, total_hidden_rows),
            "uncovered_fallback_rmse": uncovered_b2["rmse"],
        },
        "oracle": {
            item["candidate"]: item
            for item in candidate_metrics
            if item["candidate"]
            in {
                "mtp_window_oracle_overlap",
                "mtp_row_oracle",
                "mtp_top3_logit_row_oracle",
                "b2_plus_mtp_oracle_a0.3_clip20",
            }
        },
        "rerank_window_metrics": rerank_window_metrics,
        "candidates": candidate_metrics,
    }
    (run_path / "stitch_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_stitch_report(run_dir=run_path, summary=summary, candidates=candidate_metrics)
    print(json.dumps(candidate_metrics[0], indent=2), flush=True)
    return summary
