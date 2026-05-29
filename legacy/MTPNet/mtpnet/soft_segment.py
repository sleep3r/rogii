from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from .config import DataConfig
from .correlation_panel import _compress_nanmean, score_gr_patch_against_typewell
from .heatmap import fill_nan
from .io import discover_wells, load_well
from .schema_safe import assert_schema_safe_columns


@dataclass(frozen=True)
class SoftSegmentConfig:
    data_dir: Path = Path("data")
    output_dir: Path = Path("artifacts/soft_segment_v0")
    rows_per_step: int = 32
    vertical_bins: int = 96
    tau_bins: float = 1.5
    n_folds: int = 5
    train_fold: int = 0
    seed: int = 42
    k_wells: int = -1
    epochs: int = 8
    batch_size: int = 2
    hidden_channels: int = 32
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    use_pointwise_gr_diff: bool = True
    use_sdf_channels: bool = True
    use_typewell_gr_channel: bool = True
    xcorr_radii: tuple[int, ...] = ()
    xcorr_min_patch_points: int = 3
    xcorr_mad_weight: float = 0.15
    xcorr_raw_mad_weight: float = 0.02
    jump_penalty: float = 0.15
    max_jump_bins: int | None = None
    normalize_dp_scores: bool = True
    curvature_penalty: float = 0.0
    device: str = "auto"


@dataclass(frozen=True)
class SoftSegmentSample:
    well_id: str
    image: np.ndarray
    target: np.ndarray
    target_bins: np.ndarray
    tvt_grid: np.ndarray
    true_tvt: np.ndarray
    step_index: np.ndarray
    channel_names: tuple[str, ...]


def _zscore(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=np.float32)
    mean = float(arr[finite].mean())
    std = float(arr[finite].std())
    if std < 1.0e-6:
        std = 1.0
    out = (arr - mean) / std
    out[~np.isfinite(out)] = 0.0
    return out.astype(np.float32)


def _safe_clip(values: np.ndarray, lo: float = -6.0, hi: float = 6.0) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    arr[~np.isfinite(arr)] = 0.0
    return np.clip(arr, lo, hi).astype(np.float32)


def _linear_extrap_interp(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """1-D interpolation with linear extrapolation at both ends."""
    x = np.asarray(x, dtype=np.float32)
    xp = np.asarray(xp, dtype=np.float32)
    fp = np.asarray(fp, dtype=np.float32)
    order = np.argsort(xp)
    xp = xp[order]
    fp = fp[order]
    out = np.interp(x, xp, fp).astype(np.float32)
    if xp.size >= 2:
        left_slope = (fp[1] - fp[0]) / max(float(xp[1] - xp[0]), 1.0e-6)
        right_slope = (fp[-1] - fp[-2]) / max(float(xp[-1] - xp[-2]), 1.0e-6)
        left = x < xp[0]
        right = x > xp[-1]
        out[left] = fp[0] + left_slope * (x[left] - xp[0])
        out[right] = fp[-1] + right_slope * (x[right] - xp[-1])
    return out.astype(np.float32)


def _bridge_tvt(comp_md: np.ndarray, comp_tvt_input: np.ndarray) -> np.ndarray:
    finite = np.isfinite(comp_md) & np.isfinite(comp_tvt_input)
    if finite.sum() >= 2:
        return _linear_extrap_interp(comp_md, comp_md[finite], comp_tvt_input[finite])
    if finite.sum() == 1:
        return np.full_like(comp_md, float(comp_tvt_input[finite][0]), dtype=np.float32)
    return np.zeros_like(comp_md, dtype=np.float32)


def _fixed_typewell_grid(
    typewell: pd.DataFrame,
    vertical_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    tw_tvt = pd.to_numeric(typewell["TVT"], errors="coerce").to_numpy(dtype=np.float32)
    tw_gr = pd.to_numeric(typewell["GR"], errors="coerce").to_numpy(dtype=np.float32)
    finite = np.isfinite(tw_tvt) & np.isfinite(tw_gr)
    if finite.sum() < 2:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    tw_tvt = tw_tvt[finite]
    tw_gr = tw_gr[finite]
    order = np.argsort(tw_tvt)
    tw_tvt = tw_tvt[order]
    tw_gr = tw_gr[order]
    tvt_grid = np.linspace(float(tw_tvt[0]), float(tw_tvt[-1]), vertical_bins).astype(
        np.float32
    )
    gr_grid = np.interp(tvt_grid, tw_tvt, tw_gr).astype(np.float32)
    return tvt_grid, gr_grid


def _xcorr_channel(
    *,
    comp_gr: np.ndarray,
    tw_gr_grid: np.ndarray,
    hidden_steps: np.ndarray,
    radius: int,
    cfg: SoftSegmentConfig,
) -> np.ndarray:
    h = tw_gr_grid.size
    w = hidden_steps.size
    channel = np.zeros((h, w), dtype=np.float32)
    min_points = min(max(1, int(cfg.xcorr_min_patch_points)), 2 * int(radius) + 1)
    for col, step in enumerate(hidden_steps):
        scores = score_gr_patch_against_typewell(
            comp_gr,
            tw_gr_grid,
            step_index=int(step),
            patch_radius=int(radius),
            min_patch_points=min_points,
            mad_weight=float(cfg.xcorr_mad_weight),
            raw_mad_weight=float(cfg.xcorr_raw_mad_weight),
        )
        scores = np.where(np.isfinite(scores), scores, np.nan)
        if np.isfinite(scores).any():
            finite = np.isfinite(scores)
            mean = float(np.nanmean(scores[finite]))
            std = float(np.nanstd(scores[finite]))
            if std < 1.0e-6:
                std = 1.0
            channel[:, col] = np.where(finite, (scores - mean) / std, -6.0)
        else:
            channel[:, col] = 0.0
    return _safe_clip(channel, -6.0, 6.0)


def vertical_soft_target(
    target_bins: np.ndarray,
    vertical_bins: int,
    *,
    tau_bins: float = 1.5,
) -> np.ndarray:
    target = np.asarray(target_bins, dtype=np.float32)
    bins = np.arange(vertical_bins, dtype=np.float32)[:, None]
    tau = max(float(tau_bins), 1.0e-6)
    logits = -np.abs(bins - target[None, :]) / tau
    logits -= logits.max(axis=0, keepdims=True)
    probs = np.exp(logits)
    probs /= np.maximum(probs.sum(axis=0, keepdims=True), 1.0e-12)
    return probs.astype(np.float32)


def build_soft_segment_sample(
    well_id: str,
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    cfg: SoftSegmentConfig,
    *,
    shuffle_gr: bool = False,
    rng: np.random.Generator | None = None,
) -> SoftSegmentSample | None:
    """Build one schema-safe hidden-interval image sample for a well.

    The target uses true TVT because this function is for train/evaluation.
    Input channels use only shipped-style columns and a known-TVTin bridge;
    hidden true TVT is intentionally not used as a feature.
    """
    feature_columns = ["MD", "X", "Y", "Z", "GR", "TVT_input"]
    assert_schema_safe_columns(feature_columns, context="soft-segment features")

    tvt = pd.to_numeric(horizontal["TVT"], errors="coerce").to_numpy(dtype=np.float32)
    tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(
        dtype=np.float32
    )
    gr_raw = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(dtype=np.float32)
    gr_filled, gr_finite = fill_nan(gr_raw)
    if shuffle_gr:
        rng = rng or np.random.default_rng(0)
        finite_idx = np.flatnonzero(np.asarray(gr_finite) > 0.5)
        if finite_idx.size > 1:
            gr_filled = gr_filled.copy()
            gr_filled[finite_idx] = rng.permutation(gr_filled[finite_idx])
    md = (
        pd.to_numeric(horizontal["MD"], errors="coerce").to_numpy(dtype=np.float32)
        if "MD" in horizontal.columns
        else np.arange(len(horizontal), dtype=np.float32)
    )
    z = (
        pd.to_numeric(horizontal["Z"], errors="coerce").to_numpy(dtype=np.float32)
        if "Z" in horizontal.columns
        else np.zeros(len(horizontal), dtype=np.float32)
    )
    z_filled, _ = fill_nan(z)

    comp_tvt = _compress_nanmean(tvt, cfg.rows_per_step)
    comp_tvt_input = _compress_nanmean(tvt_input, cfg.rows_per_step)
    comp_gr = _compress_nanmean(gr_filled, cfg.rows_per_step)
    comp_gr_finite = _compress_nanmean(gr_finite, cfg.rows_per_step)
    comp_md = _compress_nanmean(md, cfg.rows_per_step)
    comp_z = _compress_nanmean(z_filled, cfg.rows_per_step)

    n_steps = min(
        comp_tvt.size,
        comp_tvt_input.size,
        comp_gr.size,
        comp_gr_finite.size,
        comp_md.size,
        comp_z.size,
    )
    if n_steps == 0:
        return None
    comp_tvt = comp_tvt[:n_steps]
    comp_tvt_input = comp_tvt_input[:n_steps]
    comp_gr = comp_gr[:n_steps]
    comp_gr_finite = comp_gr_finite[:n_steps]
    comp_md = comp_md[:n_steps]
    comp_z = comp_z[:n_steps]

    hidden = ~np.isfinite(comp_tvt_input) & np.isfinite(comp_tvt)
    hidden_steps = np.flatnonzero(hidden)
    if hidden_steps.size < 2:
        return None

    tvt_grid, tw_gr_grid = _fixed_typewell_grid(typewell, cfg.vertical_bins)
    if tvt_grid.size != cfg.vertical_bins:
        return None

    denom = max(float(tvt_grid[-1] - tvt_grid[0]), 1.0e-6)
    true_tvt = comp_tvt[hidden_steps].astype(np.float32)
    target_bins = (true_tvt - float(tvt_grid[0])) / denom * (cfg.vertical_bins - 1)
    target_bins = np.clip(target_bins, 0.0, cfg.vertical_bins - 1).astype(np.float32)
    target = vertical_soft_target(target_bins, cfg.vertical_bins, tau_bins=cfg.tau_bins)

    bridge = _bridge_tvt(comp_md, comp_tvt_input)
    bridge_hidden = bridge[hidden_steps]
    z_hidden = comp_z[hidden_steps]
    plane_prior = bridge_hidden + z_hidden

    valid_bridge = np.isfinite(bridge) & np.isfinite(comp_gr)
    if valid_bridge.sum() >= 2:
        order = np.argsort(bridge[valid_bridge])
        h_plane_gr = np.interp(
            plane_prior,
            bridge[valid_bridge][order],
            comp_gr[valid_bridge][order],
        ).astype(np.float32)
    else:
        h_plane_gr = comp_gr[hidden_steps].astype(np.float32)

    tw_gr_z = _zscore(tw_gr_grid)
    h_gr_z = _zscore(comp_gr)[hidden_steps]
    h_plane_z = _zscore(h_plane_gr)
    bridge_sdf = (tvt_grid[:, None] - bridge_hidden[None, :]) / 100.0
    plane_sdf = (tvt_grid[:, None] - plane_prior[None, :]) / 100.0
    progress = np.linspace(0.0, 1.0, hidden_steps.size, dtype=np.float32)
    finite_gr = np.clip(comp_gr_finite[hidden_steps], 0.0, 1.0).astype(np.float32)
    known_idx = np.flatnonzero(np.isfinite(comp_tvt_input))
    if known_idx.size:
        dist_known = np.min(np.abs(hidden_steps[:, None] - known_idx[None, :]), axis=1)
        dist_known = dist_known.astype(np.float32) / max(float(n_steps), 1.0)
    else:
        dist_known = np.ones(hidden_steps.size, dtype=np.float32)
    z_norm = _zscore(comp_z)[hidden_steps]

    raw_diff = tw_gr_z[:, None] - h_gr_z[None, :]
    plane_diff = tw_gr_z[:, None] - h_plane_z[None, :]
    channel_names_list: list[str] = []
    channels: list[np.ndarray] = []
    if cfg.use_pointwise_gr_diff:
        channel_names_list.extend(
            ["raw_gr_diff", "plane_gr_diff", "raw_gr_absdiff", "plane_gr_absdiff"]
        )
        channels.extend(
            [
                _safe_clip(raw_diff / 4.0),
                _safe_clip(plane_diff / 4.0),
                _safe_clip(np.abs(raw_diff) / 4.0),
                _safe_clip(np.abs(plane_diff) / 4.0),
            ]
        )
    for radius in tuple(int(r) for r in cfg.xcorr_radii):
        channel_names_list.append(f"xcorr_r{radius}")
        channels.append(
            _xcorr_channel(
                comp_gr=comp_gr,
                tw_gr_grid=tw_gr_grid,
                hidden_steps=hidden_steps,
                radius=radius,
                cfg=cfg,
            )
        )
    if cfg.use_sdf_channels:
        channel_names_list.extend(["bridge_sdf", "plane_prior_sdf"])
        channels.extend([_safe_clip(bridge_sdf), _safe_clip(plane_sdf)])
    channel_names_list.extend(["dist_to_known_step", "gr_mask", "md_progress", "z_normalized"])
    channels.extend(
        [
            np.broadcast_to(dist_known[None, :], target.shape).astype(np.float32),
            np.broadcast_to(finite_gr[None, :], target.shape).astype(np.float32),
            np.broadcast_to(progress[None, :], target.shape).astype(np.float32),
            np.broadcast_to(z_norm[None, :], target.shape).astype(np.float32),
        ]
    )
    if cfg.use_typewell_gr_channel:
        channel_names_list.append("typewell_gr")
        channels.append(np.broadcast_to(tw_gr_z[:, None], target.shape).astype(np.float32))
    channel_names = tuple(channel_names_list)
    image = np.stack(channels, axis=0).astype(np.float32)
    image[~np.isfinite(image)] = 0.0

    return SoftSegmentSample(
        well_id=well_id,
        image=image,
        target=target,
        target_bins=target_bins,
        tvt_grid=tvt_grid.astype(np.float32),
        true_tvt=true_tvt,
        step_index=hidden_steps.astype(np.int32),
        channel_names=channel_names,
    )


class SoftSegmentNet(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def vertical_kl_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=1)
    per_step = -(target * logp).sum(dim=1)
    if mask is not None:
        mask = mask.to(dtype=per_step.dtype, device=per_step.device)
        return (per_step * mask).sum() / mask.sum().clamp_min(1.0)
    return per_step.mean()


def viterbi_decode_logprobs(
    logits: np.ndarray,
    *,
    jump_penalty: float = 0.15,
    max_jump_bins: int | None = None,
    normalize_scores: bool = True,
    curvature_penalty: float = 0.0,
) -> np.ndarray:
    del curvature_penalty  # first-order decoder for v0; kept for API stability.
    scores = np.asarray(logits, dtype=np.float32)
    if scores.ndim != 2:
        raise ValueError(f"Expected [H, W] logits, got shape {scores.shape}")
    h, w = scores.shape
    if w == 0:
        return np.array([], dtype=np.int64)
    if normalize_scores:
        shifted = scores - scores.max(axis=0, keepdims=True)
        log_norm = np.log(np.exp(shifted).sum(axis=0, keepdims=True).clip(min=1.0e-12))
        scores = shifted - log_norm
    bins = np.arange(h, dtype=np.float32)
    jump = np.abs(bins[:, None] - bins[None, :])
    transition = float(jump_penalty) * jump
    if max_jump_bins is not None:
        transition = transition.astype(np.float32)
        transition[jump > float(max_jump_bins)] = np.inf
    dp = np.full((w, h), -np.inf, dtype=np.float32)
    prev = np.zeros((w, h), dtype=np.int64)
    dp[0] = scores[:, 0]
    for step in range(1, w):
        values = dp[step - 1][:, None] - transition
        prev_idx = values.argmax(axis=0)
        dp[step] = scores[:, step] + values[prev_idx, np.arange(h)]
        prev[step] = prev_idx
    path = np.zeros(w, dtype=np.int64)
    path[-1] = int(dp[-1].argmax())
    for step in range(w - 1, 0, -1):
        path[step - 1] = prev[step, path[step]]
    return path


def _device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _split_wells(well_ids: list[str], n_folds: int, fold: int) -> tuple[set[str], set[str]]:
    ids = sorted(well_ids)
    n_folds = max(2, int(n_folds))
    fold = int(fold) % n_folds
    valid = {well_id for idx, well_id in enumerate(ids) if idx % n_folds == fold}
    train = set(ids) - valid
    return train, valid


def _sample_batches(
    samples: list[SoftSegmentSample],
    batch_size: int,
    rng: np.random.Generator,
) -> Iterable[list[SoftSegmentSample]]:
    order = np.arange(len(samples))
    rng.shuffle(order)
    for start in range(0, len(order), max(1, int(batch_size))):
        yield [samples[int(idx)] for idx in order[start : start + max(1, int(batch_size))]]


def _collate(
    batch: list[SoftSegmentSample],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    c = batch[0].image.shape[0]
    h = batch[0].image.shape[1]
    max_w = max(sample.image.shape[2] for sample in batch)
    images = np.zeros((len(batch), c, h, max_w), dtype=np.float32)
    targets = np.zeros((len(batch), h, max_w), dtype=np.float32)
    mask = np.zeros((len(batch), max_w), dtype=np.float32)
    for idx, sample in enumerate(batch):
        w = sample.image.shape[2]
        images[idx, :, :, :w] = sample.image
        targets[idx, :, :w] = sample.target
        mask[idx, :w] = 1.0
    return (
        torch.tensor(images, device=device),
        torch.tensor(targets, device=device),
        torch.tensor(mask, device=device),
    )


def _rmse_from_sq(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.sqrt(arr.mean())) if arr.size else float("nan")


def _topk_oracle_sqerr(
    logits: np.ndarray,
    tvt_grid: np.ndarray,
    true_tvt: np.ndarray,
    k: int,
) -> list[float]:
    k = min(int(k), logits.shape[0])
    topk = np.argpartition(-logits, kth=k - 1, axis=0)[:k, :]
    out: list[float] = []
    for step in range(logits.shape[1]):
        preds = tvt_grid[topk[:, step]]
        out.append(float(np.min((preds - true_tvt[step]) ** 2)))
    return out


def _evaluate_samples(
    model: SoftSegmentNet,
    samples: list[SoftSegmentSample],
    cfg: SoftSegmentConfig,
    device: torch.device,
) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()
    rows: list[dict] = []
    top1_sq: list[float] = []
    top3_sq: list[float] = []
    top10_sq: list[float] = []
    dp_sq: list[float] = []
    top3_hits: list[float] = []
    top10_hits: list[float] = []

    with torch.no_grad():
        for sample in samples:
            x = torch.tensor(sample.image[None, ...], device=device)
            logits = model(x)[0].detach().cpu().numpy().astype(np.float32)
            top1_bins = logits.argmax(axis=0).astype(np.int64)
            target_bins_int = np.rint(sample.target_bins).astype(np.int64)
            target_bins_int = np.clip(target_bins_int, 0, sample.tvt_grid.size - 1)
            ranks = np.empty(logits.shape[1], dtype=np.int32)
            order = np.argsort(-logits, axis=0)
            for step in range(logits.shape[1]):
                ranks[step] = int(np.flatnonzero(order[:, step] == target_bins_int[step])[0]) + 1
            dp_bins = viterbi_decode_logprobs(
                logits,
                jump_penalty=cfg.jump_penalty,
                max_jump_bins=cfg.max_jump_bins,
                normalize_scores=cfg.normalize_dp_scores,
                curvature_penalty=cfg.curvature_penalty,
            )
            top1_tvt = sample.tvt_grid[top1_bins]
            dp_tvt = sample.tvt_grid[dp_bins]
            top1_sq.extend(((top1_tvt - sample.true_tvt) ** 2).astype(float).tolist())
            top3_sq.extend(_topk_oracle_sqerr(logits, sample.tvt_grid, sample.true_tvt, 3))
            top10_sq.extend(_topk_oracle_sqerr(logits, sample.tvt_grid, sample.true_tvt, 10))
            dp_sq.extend(((dp_tvt - sample.true_tvt) ** 2).astype(float).tolist())
            top3_hits.extend((ranks <= 3).astype(float).tolist())
            top10_hits.extend((ranks <= 10).astype(float).tolist())
            for idx in range(logits.shape[1]):
                rows.append(
                    {
                        "well_id": sample.well_id,
                        "compressed_step": int(sample.step_index[idx]),
                        "true_tvt": float(sample.true_tvt[idx]),
                        "target_bin": float(sample.target_bins[idx]),
                        "target_rank": int(ranks[idx]),
                        "top1_bin": int(top1_bins[idx]),
                        "top1_tvt": float(top1_tvt[idx]),
                        "dp_bin": int(dp_bins[idx]),
                        "dp_tvt": float(dp_tvt[idx]),
                    }
                )

    metrics = {
        "num_valid_samples": float(len(samples)),
        "num_valid_steps": float(len(rows)),
        "emission_top1_rmse_ft": _rmse_from_sq(top1_sq),
        "emission_top3_oracle_rmse_ft": _rmse_from_sq(top3_sq),
        "emission_top10_oracle_rmse_ft": _rmse_from_sq(top10_sq),
        "emission_top3_rate": float(np.mean(top3_hits)) if top3_hits else float("nan"),
        "emission_top10_rate": float(np.mean(top10_hits)) if top10_hits else float("nan"),
        "dp_path_rmse_ft": _rmse_from_sq(dp_sq),
    }
    return metrics, pd.DataFrame(rows)


def _load_samples(
    cfg: SoftSegmentConfig,
    *,
    shuffle_gr: bool = False,
    seed_offset: int = 0,
) -> list[SoftSegmentSample]:
    wells = discover_wells(DataConfig(data_dir=cfg.data_dir, k_wells=cfg.k_wells))
    samples: list[SoftSegmentSample] = []
    rng = np.random.default_rng(cfg.seed + seed_offset)
    print(
        json.dumps(
            {
                "event": "soft_segment_build_start",
                "wells": len(wells),
                "shuffle_gr": bool(shuffle_gr),
                "xcorr_radii": list(cfg.xcorr_radii),
                "use_sdf_channels": bool(cfg.use_sdf_channels),
            }
        ),
        flush=True,
    )
    skipped = 0
    for idx, well in enumerate(wells, start=1):
        horizontal, typewell = load_well(well)
        sample = build_soft_segment_sample(
            well.well_id,
            horizontal,
            typewell,
            cfg,
            shuffle_gr=shuffle_gr,
            rng=rng,
        )
        if sample is not None:
            samples.append(sample)
        else:
            skipped += 1
        if idx == 1 or idx % 50 == 0 or idx == len(wells):
            print(
                json.dumps(
                    {
                        "event": "soft_segment_build_progress",
                        "processed": idx,
                        "total": len(wells),
                        "built": len(samples),
                        "skipped": skipped,
                        "shuffle_gr": bool(shuffle_gr),
                    }
                ),
                flush=True,
            )
    if not samples:
        raise RuntimeError("No soft-segment samples were built.")
    return samples


def _write_report(cfg: SoftSegmentConfig, metrics: dict[str, float]) -> str:
    lines = [
        "# Soft-Segment / SDF SegFormer v0",
        "",
        "Schema-safe smoke for path-as-posterior segmentation.",
        "",
        "## Config",
        "",
        f"- rows_per_step: `{cfg.rows_per_step}`",
        f"- vertical_bins: `{cfg.vertical_bins}`",
        f"- epochs: `{cfg.epochs}`",
        f"- n_folds: `{cfg.n_folds}`",
        f"- valid_fold: `{cfg.train_fold}`",
        "",
        "## Metrics",
        "",
    ]
    for key, value in metrics.items():
        lines.append(f"- `{key}`: `{value:.6g}`")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Input features exclude hidden `TVT`, `Geology`, and raw formation columns.",
            "- `plane_prior_sdf` is built from known-`TVT_input` bridge plus `Z`; true hidden TVT is target-only.",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_example_figure(
    model: SoftSegmentNet,
    sample: SoftSegmentSample,
    cfg: SoftSegmentConfig,
    device: torch.device,
    output_path: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    model.eval()
    with torch.no_grad():
        logits = (
            model(torch.tensor(sample.image[None, ...], device=device))[0]
            .detach()
            .cpu()
            .numpy()
        )
    dp_bins = viterbi_decode_logprobs(
        logits,
        jump_penalty=cfg.jump_penalty,
        max_jump_bins=cfg.max_jump_bins,
        normalize_scores=cfg.normalize_dp_scores,
        curvature_penalty=cfg.curvature_penalty,
    )
    top1_bins = logits.argmax(axis=0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    preferred = ("plane_gr_diff", "raw_gr_diff", "xcorr_r9", "xcorr_r5", "xcorr_r3", "xcorr_r1")
    channel_name = next((name for name in preferred if name in sample.channel_names), sample.channel_names[0])
    channel_idx = sample.channel_names.index(channel_name)
    im0 = axes[0].imshow(sample.image[channel_idx], aspect="auto", cmap="coolwarm")
    axes[0].plot(sample.target_bins, color="black", linewidth=2, label="true path")
    axes[0].set_title(f"{sample.well_id}: input channel `{channel_name}`")
    axes[0].set_ylabel("TVT bin")
    axes[0].legend(loc="upper right")
    fig.colorbar(im0, ax=axes[0], fraction=0.025, pad=0.01)

    im1 = axes[1].imshow(logits, aspect="auto", cmap="magma")
    axes[1].plot(sample.target_bins, color="cyan", linewidth=2, label="true path")
    axes[1].plot(top1_bins, color="white", linewidth=1, label="emission top1")
    axes[1].plot(dp_bins, color="lime", linewidth=2, label="DP decode")
    axes[1].set_title("learned path logits")
    axes[1].set_xlabel("compressed hidden step")
    axes[1].set_ylabel("TVT bin")
    axes[1].legend(loc="upper right")
    fig.colorbar(im1, ax=axes[1], fraction=0.025, pad=0.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_soft_segment(cfg: SoftSegmentConfig) -> dict[str, float]:
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    samples = _load_samples(cfg)
    train_ids, valid_ids = _split_wells(
        [sample.well_id for sample in samples],
        cfg.n_folds,
        cfg.train_fold,
    )
    train_samples = [sample for sample in samples if sample.well_id in train_ids]
    valid_samples = [sample for sample in samples if sample.well_id in valid_ids]
    print(
        json.dumps(
            {
                "event": "soft_segment_split",
                "train_samples": len(train_samples),
                "valid_samples": len(valid_samples),
                "train_wells": len(train_ids),
                "valid_wells": len(valid_ids),
                "channels": list(train_samples[0].channel_names) if train_samples else [],
            }
        ),
        flush=True,
    )
    shuffled_samples = [
        sample for sample in _load_samples(cfg, shuffle_gr=True, seed_offset=10_000)
        if sample.well_id in valid_ids
    ]
    if not train_samples or not valid_samples:
        raise RuntimeError(
            f"Bad split: train_samples={len(train_samples)} valid_samples={len(valid_samples)}"
        )

    device = _device(cfg.device)
    model = SoftSegmentNet(
        in_channels=train_samples[0].image.shape[0],
        hidden_channels=cfg.hidden_channels,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in _sample_batches(train_samples, cfg.batch_size, rng):
            x, target, mask = _collate(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = vertical_kl_loss(logits, target, mask)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(
            json.dumps(
                {
                    "event": "soft_segment_epoch",
                    "epoch": epoch,
                    "train_loss": float(np.mean(losses)) if losses else float("nan"),
                }
            ),
            flush=True,
        )

    print(json.dumps({"event": "soft_segment_eval_start", "variant": "normal"}), flush=True)
    metrics, predictions = _evaluate_samples(model, valid_samples, cfg, device)
    print(json.dumps({"event": "soft_segment_eval_start", "variant": "shuffled_gr"}), flush=True)
    shuffled_metrics, _ = _evaluate_samples(model, shuffled_samples, cfg, device)
    for key, value in shuffled_metrics.items():
        if key.startswith("num_"):
            continue
        metrics[f"shuffled_gr_{key}"] = value
    metrics["normal_vs_shuffled_top10_rate_gap"] = (
        metrics["emission_top10_rate"] - shuffled_metrics["emission_top10_rate"]
    )
    metrics["normal_vs_shuffled_top10_oracle_rmse_gap_ft"] = (
        shuffled_metrics["emission_top10_oracle_rmse_ft"]
        - metrics["emission_top10_oracle_rmse_ft"]
    )
    metrics.update(
        {
            "num_train_samples": float(len(train_samples)),
            "num_valid_wells": float(len(valid_ids)),
            "num_train_wells": float(len(train_ids)),
        }
    )
    (cfg.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    predictions.to_parquet(cfg.output_dir / "window_predictions.parquet", index=False)
    (cfg.output_dir / "report.md").write_text(_write_report(cfg, metrics), encoding="utf-8")
    _write_example_figure(
        model,
        valid_samples[0],
        cfg,
        device,
        cfg.output_dir / "figures" / "example_alignment.png",
    )
    torch.save(
        {
            "model": model.state_dict(),
            "config": asdict(cfg),
            "channel_names": train_samples[0].channel_names,
        },
        cfg.output_dir / "best.pt",
    )
    return metrics


def load_config(path: Path) -> SoftSegmentConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    allowed = {field.name for field in fields(SoftSegmentConfig)}
    values = {key: value for key, value in raw.items() if key in allowed}
    for key in ("data_dir", "output_dir"):
        if key in values and values[key] is not None:
            values[key] = Path(values[key])
    return SoftSegmentConfig(**values)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train Soft-Segment/SDF v0")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    run_soft_segment(load_config(args.config))


if __name__ == "__main__":
    main()
