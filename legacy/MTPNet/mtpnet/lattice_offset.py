"""LDT-like candidate-lattice scorer for the dZ/offset formulation.

This is *not* a giant sequence transformer.  The LDT-relevant adaptation is:

* represent the current state as a set of still-alive candidates;
* run a small transformer over candidate features;
* learn a soft deduction step that ranks/removes impossible candidates;
* optionally roll forward by committing the selected candidate.

For ROGII, a candidate is ``(offset, lookahead_span)`` from the current
position ``s0``:

    tvt[s0:s1] = last_tvt + cumsum(-dZ + offset)

The first version is a diagnostic scorer.  It compares normal GR features to
shuffled-GR features and writes both teacher-forced state metrics and deployable
rollout metrics.
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
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .discrete_offset import (
    _WellArrays,
    _json_safe,
    _load_wells,
    _metrics_from_errors,
    _safe_corr,
    _safe_mad,
    _safe_stats,
    parse_offset_grid,
)
from .residual_stack import make_group_folds
from .schema_safe import assert_schema_safe_columns


@dataclass(frozen=True)
class LatticeOffsetConfig:
    data_dir: Path = Path("data/train")
    output_dir: Path = Path("artifacts/lattice_offset_v0")
    offset_grid: str = "-0.16:0.16:0.008"
    spans: str = "64,128,256"
    state_stride: int = 128
    n_folds: int = 5
    seed: int = 42
    k_wells: int = 100
    epochs: int = 4
    batch_size: int = 64
    d_model: int = 96
    n_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.05
    learning_rate: float = 1.0e-3
    tau_ft: float = 4.0
    beam_size: int = 8
    branch_top_k: int = 4
    on_policy_rounds: int = 0
    on_policy_state_stride: int = 0
    on_policy_max_wells: int = 0
    include_shuffled: bool = True
    progress_every: int = 1


@dataclass
class LatticeSample:
    well_id: str
    fold: int
    s0: int
    features: np.ndarray
    cost_rmse: np.ndarray
    offsets: np.ndarray
    spans: np.ndarray
    source: str = "teacher"
    last_tvt: float | None = None


class LatticeTransformerScorer(nn.Module):
    def __init__(
        self,
        *,
        feature_dim: int,
        d_model: int = 96,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, d_model),
            nn.GELU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.score = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        # src_key_padding_mask expects True for padded positions.
        encoded = self.encoder(h, src_key_padding_mask=~mask.bool())
        logits = self.score(encoded).squeeze(-1)
        return logits.masked_fill(~mask.bool(), -1.0e30)


def parse_int_list(text: str) -> np.ndarray:
    values = [int(part) for part in str(text).split(",") if part.strip()]
    if not values:
        raise ValueError("integer list is empty")
    return np.asarray(sorted(set(values)), dtype=np.int64)


def soft_target_from_cost(cost: torch.Tensor, mask: torch.Tensor, *, tau: float) -> torch.Tensor:
    if tau <= 0:
        raise ValueError("tau must be positive")
    safe_cost = cost.masked_fill(~mask.bool(), 1.0e9)
    best = torch.min(safe_cost, dim=1, keepdim=True).values
    logits = -(safe_cost - best) / float(tau)
    logits = logits.masked_fill(~mask.bool(), -1.0e30)
    return torch.softmax(logits, dim=1)


def _fill_nan(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if np.isfinite(arr).all():
        return arr
    s = pd.Series(arr)
    return s.interpolate(limit_direction="both").bfill().ffill().fillna(0.0).to_numpy(dtype=np.float64)


def _smooth(values: np.ndarray, window: int = 25) -> np.ndarray:
    arr = _fill_nan(values)
    if arr.size < 3:
        return arr
    w = int(max(3, min(window, arr.size)))
    if w % 2 == 0:
        w -= 1
    kernel = np.ones(w, dtype=np.float64) / float(w)
    pad = w // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _hidden_gr_variant(well: _WellArrays, *, variant: str, seed: int) -> np.ndarray:
    gr = _fill_nan(well.gr).copy()
    hidden = well.hidden_idx
    if variant == "normal":
        return gr
    if variant == "shuffled_gr":
        finite = hidden[np.isfinite(gr[hidden])]
        if finite.size > 1:
            rng = np.random.default_rng(seed + abs(hash(well.well_id)) % 1_000_000)
            gr[finite] = gr[rng.permutation(finite)]
        return gr
    raise ValueError(f"unknown lattice GR variant: {variant}")


def _sample_typewell(well: _WellArrays, tvt_path: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if well.typewell_tvt.size < 3:
        return np.full_like(tvt_path, np.nan, dtype=np.float64), np.zeros_like(tvt_path, dtype=bool)
    lo = float(np.min(well.typewell_tvt))
    hi = float(np.max(well.typewell_tvt))
    inside = np.isfinite(tvt_path) & (tvt_path >= lo) & (tvt_path <= hi)
    sampled = np.full_like(tvt_path, np.nan, dtype=np.float64)
    if inside.any():
        sampled[inside] = np.interp(tvt_path[inside], well.typewell_tvt, well.typewell_gr)
    return sampled, inside


def _candidate_segment(
    well: _WellArrays,
    *,
    s0: int,
    span: int,
    offset: float,
    last_tvt: float,
    dz_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    s1 = int(min(s0 + int(span), len(well.z)))
    rows = np.arange(s0, s1, dtype=np.int64)
    dz = dz_values if dz_values is not None else np.gradient(well.z) if len(well.z) >= 2 else np.zeros_like(well.z)
    pred = float(last_tvt) + np.cumsum(-dz[rows] + float(offset))
    return rows, pred


def _candidate_features(
    well: _WellArrays,
    *,
    s0: int,
    span: int,
    offset: float,
    last_tvt: float,
    gr_values: np.ndarray,
    dz_values: np.ndarray | None = None,
) -> tuple[np.ndarray, float, int]:
    rows, pred = _candidate_segment(
        well,
        s0=s0,
        span=span,
        offset=offset,
        last_tvt=last_tvt,
        dz_values=dz_values,
    )
    truth = well.tvt[rows]
    rmse = float(np.sqrt(np.nanmean((pred - truth) ** 2)))
    obs = _smooth(gr_values[rows])
    sampled, inside = _sample_typewell(well, pred)
    obs_stats = _safe_stats(obs)
    dz = dz_values if dz_values is not None else np.gradient(well.z) if len(well.z) >= 2 else np.zeros_like(well.z)
    dz_stats = _safe_stats(dz[rows])
    sampled_stats = _safe_stats(sampled)
    gr_rmse = float(np.sqrt(np.nanmean((obs - sampled) ** 2))) / 100.0 if np.isfinite(sampled).any() else 0.0
    features = np.asarray(
        [
            float(offset),
            abs(float(offset)),
            float(span) / 512.0,
            float((s0 - well.anchor_row) / max(len(well.z) - well.anchor_row, 1)),
            float(len(rows)) / 512.0,
            float(pred[0]) / 10000.0,
            float(pred[-1]) / 10000.0,
            float(pred[-1] - pred[0]) / 100.0,
            dz_stats["mean"],
            dz_stats["std"],
            obs_stats["mean"] / 100.0,
            obs_stats["std"] / 50.0,
            sampled_stats["mean"] / 100.0,
            sampled_stats["std"] / 50.0,
            _safe_corr(obs, sampled),
            _safe_corr(np.diff(obs), np.diff(sampled)),
            _safe_mad(obs, sampled) / 100.0,
            gr_rmse,
            float(np.mean(inside)) if inside.size else 0.0,
        ],
        dtype=np.float32,
    )
    return features, rmse, int(rows.size)


FEATURE_NAMES: list[str] = [
    "feat_offset",
    "feat_abs_offset",
    "feat_span",
    "feat_hidden_progress",
    "feat_n_rows",
    "feat_pred_start",
    "feat_pred_end",
    "feat_pred_delta",
    "feat_dz_mean",
    "feat_dz_std",
    "feat_obs_gr_mean",
    "feat_obs_gr_std",
    "feat_sampled_gr_mean",
    "feat_sampled_gr_std",
    "feat_gr_corr",
    "feat_dgr_corr",
    "feat_gr_mad",
    "feat_gr_rmse",
    "feat_tvt_inside_frac",
]


def _build_state_sample(
    well: _WellArrays,
    *,
    s0: int,
    last_tvt: float,
    offsets: np.ndarray,
    spans: np.ndarray,
    gr_values: np.ndarray,
    dz_values: np.ndarray,
    source: str,
) -> LatticeSample | None:
    end = int(well.hidden_idx[-1]) + 1
    feats: list[np.ndarray] = []
    costs: list[float] = []
    cand_offsets: list[float] = []
    cand_spans: list[int] = []
    for span in spans:
        if s0 >= end:
            continue
        if s0 + int(span) <= s0:
            continue
        for offset in offsets:
            f, rmse, n_rows = _candidate_features(
                well,
                s0=s0,
                span=int(span),
                offset=float(offset),
                last_tvt=last_tvt,
                gr_values=gr_values,
                dz_values=dz_values,
            )
            if n_rows <= 0:
                continue
            feats.append(f)
            costs.append(float(rmse))
            cand_offsets.append(float(offset))
            cand_spans.append(int(span))
    if not feats:
        return None
    return LatticeSample(
        well_id=well.well_id,
        fold=well.fold,
        s0=int(s0),
        features=np.vstack(feats).astype(np.float32),
        cost_rmse=np.asarray(costs, dtype=np.float32),
        offsets=np.asarray(cand_offsets, dtype=np.float32),
        spans=np.asarray(cand_spans, dtype=np.int64),
        source=str(source),
        last_tvt=float(last_tvt),
    )


def build_lattice_samples(
    wells: list[_WellArrays],
    *,
    offsets: np.ndarray,
    spans: np.ndarray,
    state_stride: int,
    variant: str,
    seed: int,
) -> tuple[list[LatticeSample], list[str]]:
    assert_schema_safe_columns(FEATURE_NAMES, context="LatticeOffset features")
    samples: list[LatticeSample] = []
    for well in wells:
        gr_values = _hidden_gr_variant(well, variant=variant, seed=seed)
        dz_values = np.gradient(well.z) if len(well.z) >= 2 else np.zeros_like(well.z)
        start = int(well.hidden_idx[0])
        end = int(well.hidden_idx[-1]) + 1
        for s0 in range(start, end, max(int(state_stride), 1)):
            if s0 <= 0:
                continue
            # Teacher-forced current state for training/eval diagnostics.
            last_tvt = float(well.tvt[s0 - 1])
            sample = _build_state_sample(
                well,
                s0=int(s0),
                last_tvt=last_tvt,
                offsets=offsets,
                spans=spans,
                gr_values=gr_values,
                dz_values=dz_values,
                source="teacher",
            )
            if sample is not None:
                samples.append(sample)
    return samples, FEATURE_NAMES.copy()


class _LatticeDataset(Dataset):
    def __init__(self, samples: list[LatticeSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> LatticeSample:
        return self.samples[idx]


def _collate(samples: list[LatticeSample]) -> dict[str, Any]:
    max_c = max(sample.features.shape[0] for sample in samples)
    feat_dim = samples[0].features.shape[1]
    x = np.zeros((len(samples), max_c, feat_dim), dtype=np.float32)
    cost = np.full((len(samples), max_c), 1.0e6, dtype=np.float32)
    mask = np.zeros((len(samples), max_c), dtype=bool)
    for i, sample in enumerate(samples):
        c = sample.features.shape[0]
        x[i, :c] = sample.features
        cost[i, :c] = sample.cost_rmse
        mask[i, :c] = True
    return {
        "x": torch.from_numpy(x),
        "cost": torch.from_numpy(cost),
        "mask": torch.from_numpy(mask),
        "samples": samples,
    }


def _train_model(
    train_samples: list[LatticeSample],
    valid_samples: list[LatticeSample],
    *,
    config: LatticeOffsetConfig,
    feature_dim: int,
) -> LatticeTransformerScorer:
    model = LatticeTransformerScorer(
        feature_dim=feature_dim,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        dropout=config.dropout,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    loader = DataLoader(
        _LatticeDataset(train_samples),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=_collate,
    )
    model.train()
    for epoch in range(int(config.epochs)):
        losses: list[float] = []
        for batch in loader:
            opt.zero_grad()
            logits = model(batch["x"], batch["mask"])
            target = soft_target_from_cost(batch["cost"], batch["mask"], tau=config.tau_ft)
            logp = torch.log_softmax(logits, dim=1)
            loss = -(target * logp).sum(dim=1).mean()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        if config.progress_every > 0:
            print(
                f"[lattice-offset] epoch {epoch + 1}/{config.epochs} loss={np.mean(losses):.4f}",
                file=sys.stderr,
                flush=True,
            )
    return model


def _score_samples(model: LatticeTransformerScorer, samples: list[LatticeSample]) -> dict[str, Any]:
    if not samples:
        return {"state_count": 0}
    loader = DataLoader(_LatticeDataset(samples), batch_size=128, shuffle=False, collate_fn=_collate)
    top1_hit: list[float] = []
    top3_hit: list[float] = []
    selected_rmse: list[float] = []
    oracle_rmse: list[float] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["x"], batch["mask"]).cpu().numpy()
            for i, sample in enumerate(batch["samples"]):
                c = sample.features.shape[0]
                order = np.argsort(-logits[i, :c])
                best = int(np.argmin(sample.cost_rmse))
                selected = int(order[0])
                top1_hit.append(float(selected == best))
                top3_hit.append(float(best in set(order[:3].tolist())))
                selected_rmse.append(float(sample.cost_rmse[selected]))
                oracle_rmse.append(float(sample.cost_rmse[best]))
    return {
        "state_count": int(len(samples)),
        "top1_oracle_rate": float(np.mean(top1_hit)),
        "top3_oracle_rate": float(np.mean(top3_hit)),
        "selected_state_rmse_mean": float(np.mean(selected_rmse)),
        "oracle_state_rmse_mean": float(np.mean(oracle_rmse)),
    }


@dataclass
class _BeamState:
    s0: int
    last_tvt: float
    score_sum: float
    n_commits: int
    rows_parts: list[np.ndarray]
    pred_parts: list[np.ndarray]
    branches: int

    @property
    def rows_done(self) -> int:
        return int(sum(part.size for part in self.rows_parts))

    @property
    def rank_score(self) -> float:
        # Average over commits so short branches do not win only by having
        # fewer logits; the tiny progress reward breaks exact ties.
        return float(self.score_sum / max(self.n_commits, 1) + 1.0e-6 * self.rows_done)


def _state_candidate_options(
    well: _WellArrays,
    *,
    s0: int,
    end: int,
    last_tvt: float,
    offsets: np.ndarray,
    spans: np.ndarray,
    gr_values: np.ndarray,
    dz_values: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    feats: list[np.ndarray] = []
    rows_list: list[np.ndarray] = []
    pred_list: list[np.ndarray] = []
    for span in spans:
        for offset in offsets:
            f, _, n_rows = _candidate_features(
                well,
                s0=s0,
                span=int(span),
                offset=float(offset),
                last_tvt=last_tvt,
                gr_values=gr_values,
                dz_values=dz_values,
            )
            rows, pred = _candidate_segment(
                well,
                s0=s0,
                span=int(span),
                offset=float(offset),
                last_tvt=last_tvt,
                dz_values=dz_values,
            )
            keep = rows < int(end)
            rows = rows[keep]
            pred = pred[keep]
            if n_rows <= 0 or rows.size == 0:
                continue
            feats.append(f)
            rows_list.append(rows)
            pred_list.append(pred)
    if not feats:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32), rows_list, pred_list
    return np.vstack(feats).astype(np.float32), rows_list, pred_list


def beam_rollout_well(
    model: LatticeTransformerScorer,
    well: _WellArrays,
    *,
    offsets: np.ndarray,
    spans: np.ndarray,
    variant: str,
    seed: int,
    beam_size: int = 8,
    branch_top_k: int = 4,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Decode a well by keeping several alive offset/span trajectories."""
    beam_size = max(1, int(beam_size))
    branch_top_k = max(1, int(branch_top_k))
    gr_values = _hidden_gr_variant(well, variant=variant, seed=seed)
    dz_values = np.gradient(well.z) if len(well.z) >= 2 else np.zeros_like(well.z)
    start = int(well.hidden_idx[0])
    end = int(well.hidden_idx[-1]) + 1
    beam: list[_BeamState] = [
        _BeamState(
            s0=start,
            last_tvt=float(well.tvt_input[well.anchor_row]),
            score_sum=0.0,
            n_commits=0,
            rows_parts=[],
            pred_parts=[],
            branches=0,
        )
    ]
    model.eval()
    with torch.no_grad():
        for _ in range(max(len(well.hidden_idx) + 2, 2)):
            if all(state.s0 >= end for state in beam):
                break
            next_beam: list[_BeamState] = []
            active: list[tuple[_BeamState, np.ndarray, list[np.ndarray], list[np.ndarray]]] = []
            for state in beam:
                if state.s0 >= end:
                    next_beam.append(state)
                    continue
                feats, rows_list, pred_list = _state_candidate_options(
                    well,
                    s0=state.s0,
                    end=end,
                    last_tvt=state.last_tvt,
                    offsets=offsets,
                    spans=spans,
                    gr_values=gr_values,
                    dz_values=dz_values,
                )
                if feats.size == 0:
                    next_beam.append(state)
                    continue
                active.append((state, feats, rows_list, pred_list))
            if active:
                max_c = max(item[1].shape[0] for item in active)
                feat_dim = active[0][1].shape[1]
                x_np = np.zeros((len(active), max_c, feat_dim), dtype=np.float32)
                mask_np = np.zeros((len(active), max_c), dtype=bool)
                for i, (_state, feats, _rows, _pred) in enumerate(active):
                    c = feats.shape[0]
                    x_np[i, :c] = feats
                    mask_np[i, :c] = True
                x = torch.from_numpy(x_np)
                mask = torch.from_numpy(mask_np)
                logits_batch = model(x, mask).cpu().numpy()
                for b, (state, feats, rows_list, pred_list) in enumerate(active):
                    logits = logits_batch[b, : feats.shape[0]]
                    if logits.size == 0:
                        next_beam.append(state)
                        continue
                    top = np.argsort(-logits)[: min(branch_top_k, logits.size)]
                    for idx in top:
                        rows = rows_list[int(idx)]
                        pred = pred_list[int(idx)]
                        next_beam.append(
                            _BeamState(
                                s0=int(rows[-1]) + 1,
                                last_tvt=float(pred[-1]),
                                score_sum=float(state.score_sum + logits[int(idx)]),
                                n_commits=int(state.n_commits + 1),
                                rows_parts=state.rows_parts + [rows],
                                pred_parts=state.pred_parts + [pred],
                                branches=int(state.branches + 1),
                            )
                        )
            beam = sorted(next_beam, key=lambda state: state.rank_score, reverse=True)[:beam_size]
            if not beam:
                break
    finished = [state for state in beam if state.rows_parts]
    if not finished:
        info = {"beam_size": int(beam_size), "branch_top_k": int(branch_top_k), "branches": 0}
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64), info
    best = max(finished, key=lambda state: state.rank_score)
    rows = np.concatenate(best.rows_parts).astype(np.int64)
    pred = np.concatenate(best.pred_parts).astype(np.float64)
    order = np.argsort(rows)
    rows = rows[order]
    pred = pred[order]
    keep = (rows >= start) & (rows < end)
    rows = rows[keep]
    pred = pred[keep]
    info = {
        "beam_size": int(beam_size),
        "branch_top_k": int(branch_top_k),
        "branches": int(best.branches),
        "commits": int(best.n_commits),
        "rank_score": float(best.rank_score),
    }
    return rows, pred, info


def collect_on_policy_samples(
    model: LatticeTransformerScorer,
    wells: list[_WellArrays],
    *,
    offsets: np.ndarray,
    spans: np.ndarray,
    variant: str,
    seed: int,
    beam_size: int = 8,
    branch_top_k: int = 4,
    state_stride: int = 128,
    max_wells: int = 0,
) -> list[LatticeSample]:
    """Collect training states visited by the model's own beam rollout."""
    assert_schema_safe_columns(FEATURE_NAMES, context="LatticeOffset on-policy features")
    out: list[LatticeSample] = []
    selected_wells = wells[: int(max_wells)] if max_wells and max_wells > 0 else wells
    model.eval()
    with torch.no_grad():
        for well in selected_wells:
            gr_values = _hidden_gr_variant(well, variant=variant, seed=seed)
            dz_values = np.gradient(well.z) if len(well.z) >= 2 else np.zeros_like(well.z)
            start = int(well.hidden_idx[0])
            end = int(well.hidden_idx[-1]) + 1
            beam: list[_BeamState] = [
                _BeamState(
                    s0=start,
                    last_tvt=float(well.tvt_input[well.anchor_row]),
                    score_sum=0.0,
                    n_commits=0,
                    rows_parts=[],
                    pred_parts=[],
                    branches=0,
                )
            ]
            next_record = start
            for _ in range(max(len(well.hidden_idx) + 2, 2)):
                active_state = max(beam, key=lambda state: state.rank_score)
                if active_state.s0 >= end:
                    break
                if active_state.s0 >= next_record:
                    sample = _build_state_sample(
                        well,
                        s0=int(active_state.s0),
                        last_tvt=float(active_state.last_tvt),
                        offsets=offsets,
                        spans=spans,
                        gr_values=gr_values,
                        dz_values=dz_values,
                        source="on_policy",
                    )
                    if sample is not None:
                        out.append(sample)
                    next_record = int(active_state.s0) + max(int(state_stride), 1)

                next_beam: list[_BeamState] = []
                active: list[tuple[_BeamState, np.ndarray, list[np.ndarray], list[np.ndarray]]] = []
                for state in beam:
                    if state.s0 >= end:
                        next_beam.append(state)
                        continue
                    feats, rows_list, pred_list = _state_candidate_options(
                        well,
                        s0=state.s0,
                        end=end,
                        last_tvt=state.last_tvt,
                        offsets=offsets,
                        spans=spans,
                        gr_values=gr_values,
                        dz_values=dz_values,
                    )
                    if feats.size == 0:
                        next_beam.append(state)
                        continue
                    active.append((state, feats, rows_list, pred_list))
                if active:
                    max_c = max(item[1].shape[0] for item in active)
                    feat_dim = active[0][1].shape[1]
                    x_np = np.zeros((len(active), max_c, feat_dim), dtype=np.float32)
                    mask_np = np.zeros((len(active), max_c), dtype=bool)
                    for i, (_state, feats, _rows, _pred) in enumerate(active):
                        c = feats.shape[0]
                        x_np[i, :c] = feats
                        mask_np[i, :c] = True
                    logits_batch = model(torch.from_numpy(x_np), torch.from_numpy(mask_np)).cpu().numpy()
                    for b, (state, feats, rows_list, pred_list) in enumerate(active):
                        logits = logits_batch[b, : feats.shape[0]]
                        top = np.argsort(-logits)[: min(max(int(branch_top_k), 1), logits.size)]
                        for idx in top:
                            rows = rows_list[int(idx)]
                            pred = pred_list[int(idx)]
                            next_beam.append(
                                _BeamState(
                                    s0=int(rows[-1]) + 1,
                                    last_tvt=float(pred[-1]),
                                    score_sum=float(state.score_sum + logits[int(idx)]),
                                    n_commits=int(state.n_commits + 1),
                                    rows_parts=state.rows_parts + [rows],
                                    pred_parts=state.pred_parts + [pred],
                                    branches=int(state.branches + 1),
                                )
                            )
                beam = sorted(next_beam, key=lambda state: state.rank_score, reverse=True)[: max(int(beam_size), 1)]
                if not beam:
                    break
    return out


def _rollout_metrics(
    model_by_fold: dict[int, LatticeTransformerScorer],
    wells: list[_WellArrays],
    *,
    offsets: np.ndarray,
    spans: np.ndarray,
    variant: str,
    seed: int,
    candidate: str,
    beam_size: int,
    branch_top_k: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    errors: list[np.ndarray] = []
    pred_parts: list[pd.DataFrame] = []
    for idx, well in enumerate(wells):
        if idx == 0 or (idx + 1) % 25 == 0 or idx + 1 == len(wells):
            print(
                f"[lattice-offset] rollout {candidate} well {idx + 1}/{len(wells)}",
                file=sys.stderr,
                flush=True,
            )
        model = model_by_fold.get(int(well.fold))
        if model is None:
            continue
        rows, pred, _info = beam_rollout_well(
            model,
            well,
            offsets=offsets,
            spans=spans,
            variant=variant,
            seed=seed,
            beam_size=beam_size,
            branch_top_k=branch_top_k,
        )
        if rows.size == 0:
            continue
        truth = well.tvt[rows]
        errors.append(pred - truth)
        pred_parts.append(
            pd.DataFrame(
                {
                    "id": well.ids[rows],
                    "well_id": well.well_id,
                    "row_idx": rows,
                    "pred_tvt": pred,
                    "candidate": candidate,
                }
            )
        )
    metrics = _metrics_from_errors(errors)
    metrics["candidate"] = candidate
    metrics["beam_size"] = int(beam_size)
    metrics["branch_top_k"] = int(branch_top_k)
    preds = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
    return metrics, preds


def run_lattice_offset(config: LatticeOffsetConfig) -> dict[str, Any]:
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
    offsets = parse_offset_grid(config.offset_grid)
    spans = parse_int_list(config.spans)
    normal_samples, feature_names = build_lattice_samples(
        wells,
        offsets=offsets,
        spans=spans,
        state_stride=config.state_stride,
        variant="normal",
        seed=config.seed,
    )
    shuffled_samples: list[LatticeSample] = []
    if config.include_shuffled:
        shuffled_samples, _ = build_lattice_samples(
            wells,
            offsets=offsets,
            spans=spans,
            state_stride=config.state_stride,
            variant="shuffled_gr",
            seed=config.seed,
        )
    print(
        f"[lattice-offset] wells={len(wells)} states={len(normal_samples)} "
        f"candidates/state~{len(offsets) * len(spans)} features={len(feature_names)} "
        f"beam={config.beam_size}x{config.branch_top_k}",
        file=sys.stderr,
        flush=True,
    )
    model_by_fold: dict[int, LatticeTransformerScorer] = {}
    teacher_metrics: list[dict[str, Any]] = []
    on_policy_state_count = 0
    on_policy_by_fold: list[dict[str, Any]] = []
    folds_present = sorted({sample.fold for sample in normal_samples})
    for fold in folds_present:
        train = [sample for sample in normal_samples if sample.fold != fold]
        valid = [sample for sample in normal_samples if sample.fold == fold]
        if not train or not valid:
            continue
        print(
            f"[lattice-offset] fold {fold + 1}/{len(folds_present)} train_states={len(train)} valid_states={len(valid)}",
            file=sys.stderr,
            flush=True,
        )
        model = _train_model(train, valid, config=config, feature_dim=len(feature_names))
        if config.on_policy_rounds > 0:
            train_wells = [well for well in wells if int(well.fold) != int(fold)]
            for round_idx in range(int(config.on_policy_rounds)):
                op_samples = collect_on_policy_samples(
                    model,
                    train_wells,
                    offsets=offsets,
                    spans=spans,
                    variant="normal",
                    seed=config.seed + 1000 * (round_idx + 1) + int(fold),
                    beam_size=config.beam_size,
                    branch_top_k=config.branch_top_k,
                    state_stride=config.on_policy_state_stride or config.state_stride,
                    max_wells=config.on_policy_max_wells,
                )
                on_policy_state_count += len(op_samples)
                on_policy_by_fold.append(
                    {
                        "fold": int(fold),
                        "round": int(round_idx),
                        "states": int(len(op_samples)),
                    }
                )
                print(
                    f"[lattice-offset] fold {fold + 1}/{len(folds_present)} "
                    f"on_policy_round={round_idx + 1}/{config.on_policy_rounds} "
                    f"states={len(op_samples)}",
                    file=sys.stderr,
                    flush=True,
                )
                if op_samples:
                    train = train + op_samples
                    model = _train_model(train, valid, config=config, feature_dim=len(feature_names))
        model_by_fold[int(fold)] = model
        tm = _score_samples(model, valid)
        tm["fold"] = int(fold)
        tm["variant"] = "normal"
        teacher_metrics.append(tm)
        if shuffled_samples:
            shuf_valid = [sample for sample in shuffled_samples if sample.fold == fold]
            sm = _score_samples(model, shuf_valid)
            sm["fold"] = int(fold)
            sm["variant"] = "shuffled_gr"
            teacher_metrics.append(sm)

    normal_rollout, normal_preds = _rollout_metrics(
        model_by_fold,
        wells,
        offsets=offsets,
        spans=spans,
        variant="normal",
        seed=config.seed,
        candidate=f"lattice_beam_b{config.beam_size}_k{config.branch_top_k}_normal",
        beam_size=config.beam_size,
        branch_top_k=config.branch_top_k,
    )
    candidate_metrics = [normal_rollout]
    pred_frames = [normal_preds]
    if config.include_shuffled:
        shuffled_rollout, shuffled_preds = _rollout_metrics(
            model_by_fold,
            wells,
            offsets=offsets,
            spans=spans,
            variant="shuffled_gr",
            seed=config.seed,
            candidate=f"lattice_beam_b{config.beam_size}_k{config.branch_top_k}_shuffled_gr",
            beam_size=config.beam_size,
            branch_top_k=config.branch_top_k,
        )
        candidate_metrics.append(shuffled_rollout)
        pred_frames.append(shuffled_preds)
    predictions = pd.concat([p for p in pred_frames if p is not None and not p.empty], ignore_index=True)
    predictions.to_parquet(out_dir / "lattice_offset_predictions.parquet", index=False)
    metrics: dict[str, Any] = {
        "experiment": "lattice_offset_v0",
        "config": asdict(config),
        "wells": int(len(wells)),
        "states": int(len(normal_samples)),
        "candidate_count_per_state": int(len(offsets) * len(spans)),
        "feature_names": feature_names,
        "on_policy_state_count": int(on_policy_state_count),
        "on_policy_by_fold": on_policy_by_fold,
        "teacher_forced_metrics": teacher_metrics,
        "candidates": candidate_metrics,
    }
    if len(candidate_metrics) > 1:
        metrics["normal_minus_shuffled_row_rmse"] = float(
            candidate_metrics[0].get("row_rmse", float("nan"))
            - candidate_metrics[1].get("row_rmse", float("nan"))
        )
    (out_dir / "lattice_offset_metrics.json").write_text(
        json.dumps(_json_safe(metrics), indent=2), encoding="utf-8"
    )
    _write_report(out_dir, metrics)
    return metrics


def _write_report(output_dir: Path, metrics: dict[str, Any]) -> None:
    lines = [
        "# LATTICE_OFFSET_V0",
        "",
        "LDT-like candidate lattice scorer for `(offset, lookahead_span)` states.",
        "This is a small recurrent-candidate-set diagnostic, not a full raw-sequence transformer.",
        "",
        "## rollout metrics",
        "",
        "| candidate | row RMSE | mean well | p95 | worst |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in metrics["candidates"]:
        lines.append(
            f"| {item['candidate']} | {item.get('row_rmse', float('nan')):.4f} | "
            f"{item.get('mean_well_rmse', float('nan')):.4f} | "
            f"{item.get('p95_well_rmse', float('nan')):.4f} | "
            f"{item.get('worst_well_rmse', float('nan')):.4f} |"
        )
    lines.extend(
        [
            "",
            f"`beam_size`: {metrics.get('config', {}).get('beam_size')}",
            f"`branch_top_k`: {metrics.get('config', {}).get('branch_top_k')}",
            f"`on_policy_state_count`: {metrics.get('on_policy_state_count', 0)}",
            f"`normal_minus_shuffled_row_rmse`: {metrics.get('normal_minus_shuffled_row_rmse', float('nan')):.4f}",
            "",
            "## teacher-forced state metrics",
            "",
            "| fold | variant | states | top1 oracle | top3 oracle | selected RMSE | oracle RMSE |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in metrics["teacher_forced_metrics"]:
        lines.append(
            f"| {item['fold']} | {item['variant']} | {item.get('state_count', 0)} | "
            f"{item.get('top1_oracle_rate', float('nan')):.3f} | "
            f"{item.get('top3_oracle_rate', float('nan')):.3f} | "
            f"{item.get('selected_state_rmse_mean', float('nan')):.4f} | "
            f"{item.get('oracle_state_rmse_mean', float('nan')):.4f} |"
        )
    (output_dir / "lattice_offset_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate LDT-like offset lattice scorer")
    parser.add_argument("--data-dir", type=Path, default=LatticeOffsetConfig.data_dir)
    parser.add_argument("--output-dir", type=Path, default=LatticeOffsetConfig.output_dir)
    parser.add_argument("--offset-grid", type=str, default=LatticeOffsetConfig.offset_grid)
    parser.add_argument("--spans", type=str, default=LatticeOffsetConfig.spans)
    parser.add_argument("--state-stride", type=int, default=LatticeOffsetConfig.state_stride)
    parser.add_argument("--n-folds", type=int, default=LatticeOffsetConfig.n_folds)
    parser.add_argument("--seed", type=int, default=LatticeOffsetConfig.seed)
    parser.add_argument("--k-wells", type=int, default=LatticeOffsetConfig.k_wells)
    parser.add_argument("--epochs", type=int, default=LatticeOffsetConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=LatticeOffsetConfig.batch_size)
    parser.add_argument("--d-model", type=int, default=LatticeOffsetConfig.d_model)
    parser.add_argument("--n-layers", type=int, default=LatticeOffsetConfig.n_layers)
    parser.add_argument("--n-heads", type=int, default=LatticeOffsetConfig.n_heads)
    parser.add_argument("--dropout", type=float, default=LatticeOffsetConfig.dropout)
    parser.add_argument("--learning-rate", type=float, default=LatticeOffsetConfig.learning_rate)
    parser.add_argument("--tau-ft", type=float, default=LatticeOffsetConfig.tau_ft)
    parser.add_argument("--beam-size", type=int, default=LatticeOffsetConfig.beam_size)
    parser.add_argument("--branch-top-k", type=int, default=LatticeOffsetConfig.branch_top_k)
    parser.add_argument("--on-policy-rounds", type=int, default=LatticeOffsetConfig.on_policy_rounds)
    parser.add_argument("--on-policy-state-stride", type=int, default=LatticeOffsetConfig.on_policy_state_stride)
    parser.add_argument("--on-policy-max-wells", type=int, default=LatticeOffsetConfig.on_policy_max_wells)
    parser.add_argument("--no-shuffled", action="store_true")
    parser.add_argument("--progress-every", type=int, default=LatticeOffsetConfig.progress_every)
    args = parser.parse_args(argv)
    metrics = run_lattice_offset(
        LatticeOffsetConfig(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            offset_grid=args.offset_grid,
            spans=args.spans,
            state_stride=args.state_stride,
            n_folds=args.n_folds,
            seed=args.seed,
            k_wells=args.k_wells,
            epochs=args.epochs,
            batch_size=args.batch_size,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            dropout=args.dropout,
            learning_rate=args.learning_rate,
            tau_ft=args.tau_ft,
            beam_size=args.beam_size,
            branch_top_k=args.branch_top_k,
            on_policy_rounds=args.on_policy_rounds,
            on_policy_state_stride=args.on_policy_state_stride,
            on_policy_max_wells=args.on_policy_max_wells,
            include_shuffled=not args.no_shuffled,
            progress_every=args.progress_every,
        )
    )
    print(
        json.dumps(
            {
                "experiment": metrics["experiment"],
                "candidates": metrics.get("candidates", []),
                "normal_minus_shuffled_row_rmse": metrics.get("normal_minus_shuffled_row_rmse"),
                "report": str(Path(args.output_dir) / "lattice_offset_report.md"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
