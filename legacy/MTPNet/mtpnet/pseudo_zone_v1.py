from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import DataConfig
from .correlation_panel import (
    _apply_search_band,
    _compress_nanmean,
    _json_clean,
    _rank_of_true,
    _regular_typewell_grid,
    _topk_indices,
    known_tail_linear_anchor,
    score_gr_patch_multiscale,
)
from .heatmap import fill_nan
from .io import discover_wells, load_well


COARSE_ZONE_LABELS = (
    "__unknown__",
    "ANCC",
    "ASTNU",
    "ASTNL",
    "EGFDU",
    "EGFDL",
    "BUDA",
    "OLMOS",
    "EGFDL_SUB",
    "OTHER",
)
ZONE_TO_ID = {label: idx for idx, label in enumerate(COARSE_ZONE_LABELS)}
ID_TO_ZONE = {idx: label for label, idx in ZONE_TO_ID.items()}

TEST_SAFE_LATERAL_COLUMNS = (
    "step_frac",
    "md_z",
    "x_z",
    "y_z",
    "z_z",
    "gr_z",
    "gr_valid_frac",
    "tvt_input_delta",
    "known_mask",
    "hidden_progress",
)
TEST_SAFE_TYPEWELL_COLUMNS = ("tvt_frac", "gr_z", "dgr_z")
FORBIDDEN_INFERENCE_COLUMNS = {
    "TVT",
    "Geology",
    "ANCC",
    "ASTNU",
    "ASTNL",
    "EGFDU",
    "EGFDL",
    "BUDA",
}


@dataclass(frozen=True)
class PseudoZoneV1Config:
    rows_per_step: int = 32
    vertical_step_ft: float = 5.0
    n_bins: int = 128
    n_folds: int = 5
    k_wells: int = -1
    seed: int = 42
    neural_epochs: int = 6
    neural_hidden_dim: int = 64
    neural_batch_size: int = 2048
    neural_lr: float = 1.0e-3
    patch_radii: tuple[int, ...] = (2, 4, 8, 16)
    stretch_factors: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5)
    known_tail_radius_ft: float = 120.0
    soft_zone_weight: float = 0.5


@dataclass(frozen=True)
class TemplateZoneModel:
    labels_by_bin: np.ndarray
    n_bins: int


class _ZoneMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class _FoldModels:
    template: TemplateZoneModel
    lateral_model: _ZoneMLP
    typewell_model: _ZoneMLP


def collapse_geology_label(label: object) -> str:
    if label is None:
        return "__unknown__"
    try:
        if isinstance(label, float) and np.isnan(label):
            return "__unknown__"
    except TypeError:
        pass
    value = str(label).strip()
    if value == "" or value.lower() == "nan" or value == "__unknown__":
        return "__unknown__"
    if value in {"ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA", "OLMOS"}:
        return value
    upper = value.upper()
    if any(token in upper for token in ("THL", "TGT", "BHL", "MNSS", "LTGT", "LTHL", "LBHL")):
        return "EGFDL_SUB"
    return "OTHER"


def _zscore(values: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    mask = np.isfinite(arr) if valid is None else (np.asarray(valid, dtype=bool) & np.isfinite(arr))
    if not mask.any():
        return np.zeros_like(arr, dtype=np.float32)
    mean = float(arr[mask].mean())
    std = float(arr[mask].std())
    std = std if std > 1e-6 else 1.0
    return np.where(np.isfinite(arr), (arr - mean) / std, 0.0).astype(np.float32)


def _column_or_default(frame: pd.DataFrame, name: str, fallback: np.ndarray) -> np.ndarray:
    if name in frame.columns:
        return pd.to_numeric(frame[name], errors="coerce").to_numpy(np.float32)
    return np.asarray(fallback, dtype=np.float32)


def _valid_frac_steps(values: np.ndarray, rows_per_step: int) -> np.ndarray:
    usable = (len(values) // rows_per_step) * rows_per_step
    if usable <= 0:
        return np.empty(0, dtype=np.float32)
    arr = np.isfinite(np.asarray(values[:usable], dtype=np.float32)).reshape(-1, rows_per_step)
    return arr.mean(axis=1).astype(np.float32)


def build_lateral_zone_features(
    horizontal: pd.DataFrame, *, rows_per_step: int
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build inference-time lateral features without using train-only columns."""
    n_raw = len(horizontal)
    row_index = np.arange(n_raw, dtype=np.float32)
    md = _column_or_default(horizontal, "MD", row_index)
    x = _column_or_default(horizontal, "X", np.zeros(n_raw, dtype=np.float32))
    y = _column_or_default(horizontal, "Y", np.zeros(n_raw, dtype=np.float32))
    z = _column_or_default(horizontal, "Z", np.zeros(n_raw, dtype=np.float32))
    gr = _column_or_default(horizontal, "GR", np.full(n_raw, np.nan, dtype=np.float32))
    tvt_input = _column_or_default(horizontal, "TVT_input", np.full(n_raw, np.nan, dtype=np.float32))

    comp_md = _compress_nanmean(md, rows_per_step)
    comp_x = _compress_nanmean(x, rows_per_step)
    comp_y = _compress_nanmean(y, rows_per_step)
    comp_z = _compress_nanmean(z, rows_per_step)
    comp_gr = _compress_nanmean(gr, rows_per_step)
    comp_gr_valid = _valid_frac_steps(gr, rows_per_step)
    comp_tvt_input = _compress_nanmean(tvt_input, rows_per_step)
    n_steps = min(
        len(comp_md),
        len(comp_x),
        len(comp_y),
        len(comp_z),
        len(comp_gr),
        len(comp_gr_valid),
        len(comp_tvt_input),
    )
    if n_steps == 0:
        return np.empty((0, len(TEST_SAFE_LATERAL_COLUMNS)), dtype=np.float32), TEST_SAFE_LATERAL_COLUMNS
    comp_md = comp_md[:n_steps]
    comp_x = comp_x[:n_steps]
    comp_y = comp_y[:n_steps]
    comp_z = comp_z[:n_steps]
    comp_gr = comp_gr[:n_steps]
    comp_gr_valid = comp_gr_valid[:n_steps]
    comp_tvt_input = comp_tvt_input[:n_steps]
    gr_filled, _ = fill_nan(comp_gr)
    known = np.isfinite(comp_tvt_input)
    known_idx = np.flatnonzero(known)
    last_known = float(comp_tvt_input[known_idx[-1]]) if known_idx.size else 0.0
    tvt_input_delta = np.where(known, (comp_tvt_input - last_known) / 100.0, 0.0).astype(np.float32)
    hidden_progress = np.zeros(n_steps, dtype=np.float32)
    hidden_idx = np.flatnonzero(~known)
    if hidden_idx.size:
        first_hidden = int(hidden_idx[0])
        denom = max(float(n_steps - first_hidden - 1), 1.0)
        hidden_progress[first_hidden:] = (
            np.arange(first_hidden, n_steps, dtype=np.float32) - first_hidden
        ) / denom
    step_frac = np.arange(n_steps, dtype=np.float32) / max(float(n_steps - 1), 1.0)
    features = np.stack(
        [
            step_frac,
            _zscore(comp_md),
            _zscore(comp_x),
            _zscore(comp_y),
            _zscore(comp_z),
            _zscore(gr_filled, comp_gr_valid > 0),
            comp_gr_valid.astype(np.float32),
            tvt_input_delta,
            known.astype(np.float32),
            hidden_progress,
        ],
        axis=1,
    )
    return features.astype(np.float32), TEST_SAFE_LATERAL_COLUMNS


def build_typewell_zone_features(
    typewell: pd.DataFrame, *, vertical_step_ft: float
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    """Build inference-time typewell features without using Geology."""
    tvt_grid, gr_grid = _regular_typewell_grid(typewell, vertical_step_ft)
    if tvt_grid.size == 0:
        return (
            np.empty((0, len(TEST_SAFE_TYPEWELL_COLUMNS)), dtype=np.float32),
            TEST_SAFE_TYPEWELL_COLUMNS,
            tvt_grid,
        )
    dgr = np.gradient(gr_grid).astype(np.float32)
    tvt_frac = (tvt_grid - float(tvt_grid[0])) / max(float(tvt_grid[-1] - tvt_grid[0]), 1.0)
    features = np.stack([tvt_frac.astype(np.float32), _zscore(gr_grid), _zscore(dgr)], axis=1)
    return features.astype(np.float32), TEST_SAFE_TYPEWELL_COLUMNS, tvt_grid.astype(np.float32)


def _typewell_grid_labels(typewell: pd.DataFrame, tvt_grid: np.ndarray) -> np.ndarray:
    if "Geology" not in typewell.columns or tvt_grid.size == 0:
        return np.full(tvt_grid.size, "__unknown__", dtype=object)
    tvt = pd.to_numeric(typewell["TVT"], errors="coerce").to_numpy(np.float32)
    labels = typewell["Geology"].map(collapse_geology_label).to_numpy(dtype=object)
    finite = np.isfinite(tvt)
    if not finite.any():
        return np.full(tvt_grid.size, "__unknown__", dtype=object)
    order = np.argsort(tvt[finite])
    tvt_sorted = tvt[finite][order]
    labels_sorted = labels[finite][order]
    nearest = np.searchsorted(tvt_sorted, tvt_grid, side="left")
    nearest = np.clip(nearest, 0, len(tvt_sorted) - 1)
    prev = np.clip(nearest - 1, 0, len(tvt_sorted) - 1)
    choose_prev = np.abs(tvt_grid - tvt_sorted[prev]) <= np.abs(tvt_grid - tvt_sorted[nearest])
    idx = np.where(choose_prev, prev, nearest)
    return labels_sorted[idx].astype(object)


def _labels_to_ids(labels: np.ndarray) -> np.ndarray:
    return np.array([ZONE_TO_ID.get(collapse_geology_label(label), ZONE_TO_ID["OTHER"]) for label in labels], dtype=np.int64)


def _relative_tvt_bins(tvt: np.ndarray, n_bins: int) -> np.ndarray:
    finite = np.isfinite(tvt)
    if not finite.any():
        return np.zeros_like(tvt, dtype=np.int32)
    lo = float(np.nanmin(tvt[finite]))
    hi = float(np.nanmax(tvt[finite]))
    denom = max(hi - lo, 1e-6)
    rel = np.clip((tvt.astype(np.float32) - lo) / denom, 0.0, 1.0)
    return np.clip(np.floor(rel * n_bins), 0, n_bins - 1).astype(np.int32)


def _fit_template(typewells: list[pd.DataFrame], *, n_bins: int) -> TemplateZoneModel:
    counts: list[dict[str, int]] = [dict() for _ in range(n_bins)]
    for typewell in typewells:
        if "Geology" not in typewell.columns:
            continue
        tvt = pd.to_numeric(typewell["TVT"], errors="coerce").to_numpy(np.float32)
        labels = typewell["Geology"].map(collapse_geology_label).to_numpy(dtype=object)
        bins = _relative_tvt_bins(tvt, n_bins)
        for bin_idx, label in zip(bins, labels):
            bucket = counts[int(bin_idx)]
            bucket[str(label)] = bucket.get(str(label), 0) + 1
    labels_by_bin = np.full(n_bins, "__unknown__", dtype=object)
    for idx, bucket in enumerate(counts):
        if bucket:
            labels_by_bin[idx] = max(bucket, key=bucket.get)
    known = np.flatnonzero(labels_by_bin != "__unknown__")
    if known.size:
        for idx, label in enumerate(labels_by_bin):
            if label == "__unknown__":
                nearest = known[np.argmin(np.abs(known - idx))]
                labels_by_bin[idx] = labels_by_bin[nearest]
    return TemplateZoneModel(labels_by_bin=labels_by_bin, n_bins=n_bins)


def _predict_template_labels(typewell: pd.DataFrame, model: TemplateZoneModel) -> np.ndarray:
    tvt = pd.to_numeric(typewell["TVT"], errors="coerce").to_numpy(np.float32)
    bins = _relative_tvt_bins(tvt, model.n_bins)
    return model.labels_by_bin[bins].astype(object)


def _predict_template_on_grid(
    typewell: pd.DataFrame, tvt_grid: np.ndarray, model: TemplateZoneModel
) -> np.ndarray:
    if tvt_grid.size == 0:
        return np.empty(0, dtype=object)
    labels_raw = _predict_template_labels(typewell, model)
    tvt = pd.to_numeric(typewell["TVT"], errors="coerce").to_numpy(np.float32)
    finite = np.isfinite(tvt)
    if not finite.any():
        return np.full(tvt_grid.size, "__unknown__", dtype=object)
    order = np.argsort(tvt[finite])
    tvt_sorted = tvt[finite][order]
    labels_sorted = labels_raw[finite][order]
    nearest = np.searchsorted(tvt_sorted, tvt_grid, side="left")
    nearest = np.clip(nearest, 0, len(tvt_sorted) - 1)
    prev = np.clip(nearest - 1, 0, len(tvt_sorted) - 1)
    choose_prev = np.abs(tvt_grid - tvt_sorted[prev]) <= np.abs(tvt_grid - tvt_sorted[nearest])
    return labels_sorted[np.where(choose_prev, prev, nearest)].astype(object)


def _split_folds(n_items: int, n_folds: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(n_items)
    rng.shuffle(indices)
    return [fold.astype(np.int32) for fold in np.array_split(indices, max(1, n_folds))]


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float32)
    logits = logits - np.nanmax(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    denom = np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)
    return (exp / denom).astype(np.float32)


def _train_mlp(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    in_dim: int,
    cfg: PseudoZoneV1Config,
    seed: int,
) -> _ZoneMLP:
    torch.manual_seed(seed)
    model = _ZoneMLP(in_dim, cfg.neural_hidden_dim, len(COARSE_ZONE_LABELS))
    if features.size == 0 or labels.size == 0:
        return model
    x = torch.from_numpy(features.astype(np.float32))
    y = torch.from_numpy(labels.astype(np.int64))
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=cfg.neural_batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.neural_lr, weight_decay=1.0e-4)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _epoch in range(max(1, cfg.neural_epochs)):
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
    return model.eval()


def _collect_training_examples(
    loaded: list[tuple[object, pd.DataFrame, pd.DataFrame]],
    train_indices: Iterable[int],
    cfg: PseudoZoneV1Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lateral_x: list[np.ndarray] = []
    lateral_y: list[np.ndarray] = []
    typewell_x: list[np.ndarray] = []
    typewell_y: list[np.ndarray] = []
    for idx in train_indices:
        _well, horizontal, typewell = loaded[int(idx)]
        lat_features, _ = build_lateral_zone_features(horizontal, rows_per_step=cfg.rows_per_step)
        tw_features, _, tvt_grid = build_typewell_zone_features(
            typewell, vertical_step_ft=cfg.vertical_step_ft
        )
        if tw_features.size and tvt_grid.size:
            tw_labels = _typewell_grid_labels(typewell, tvt_grid)
            typewell_x.append(tw_features)
            typewell_y.append(_labels_to_ids(tw_labels))
        if lat_features.size and tvt_grid.size:
            tvt = pd.to_numeric(horizontal["TVT"], errors="coerce").to_numpy(np.float32)
            tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(
                np.float32
            )
            comp_tvt = _compress_nanmean(tvt, cfg.rows_per_step)
            comp_tvt_input = _compress_nanmean(tvt_input, cfg.rows_per_step)
            n = min(len(comp_tvt), len(comp_tvt_input), len(lat_features))
            hidden = ~np.isfinite(comp_tvt_input[:n]) & np.isfinite(comp_tvt[:n])
            if hidden.any():
                true_bins = np.abs(tvt_grid[:, None] - comp_tvt[:n][None, :]).argmin(axis=0)
                step_labels = _typewell_grid_labels(typewell, tvt_grid)[true_bins]
                lateral_x.append(lat_features[:n][hidden])
                lateral_y.append(_labels_to_ids(step_labels[hidden]))
    lx = np.concatenate(lateral_x, axis=0) if lateral_x else np.empty((0, len(TEST_SAFE_LATERAL_COLUMNS)), dtype=np.float32)
    ly = np.concatenate(lateral_y, axis=0) if lateral_y else np.empty(0, dtype=np.int64)
    tx = np.concatenate(typewell_x, axis=0) if typewell_x else np.empty((0, len(TEST_SAFE_TYPEWELL_COLUMNS)), dtype=np.float32)
    ty = np.concatenate(typewell_y, axis=0) if typewell_y else np.empty(0, dtype=np.int64)
    return lx, ly, tx, ty


def _fit_fold_models(
    loaded: list[tuple[object, pd.DataFrame, pd.DataFrame]],
    train_indices: Iterable[int],
    cfg: PseudoZoneV1Config,
    *,
    seed: int,
) -> _FoldModels:
    train_indices = [int(idx) for idx in train_indices]
    template = _fit_template([loaded[idx][2] for idx in train_indices], n_bins=cfg.n_bins)
    lx, ly, tx, ty = _collect_training_examples(loaded, train_indices, cfg)
    lateral_model = _train_mlp(
        lx,
        ly,
        in_dim=len(TEST_SAFE_LATERAL_COLUMNS),
        cfg=cfg,
        seed=seed,
    )
    typewell_model = _train_mlp(
        tx,
        ty,
        in_dim=len(TEST_SAFE_TYPEWELL_COLUMNS),
        cfg=cfg,
        seed=seed + 17,
    )
    return _FoldModels(template=template, lateral_model=lateral_model, typewell_model=typewell_model)


def _predict_probs(model: _ZoneMLP, features: np.ndarray) -> np.ndarray:
    if features.size == 0:
        return np.empty((0, len(COARSE_ZONE_LABELS)), dtype=np.float32)
    with torch.no_grad():
        logits = model(torch.from_numpy(features.astype(np.float32))).cpu().numpy()
    return _softmax_np(logits)


def _top_zone_columns(prefix: str, probs: np.ndarray, top_n: int = 3) -> dict[str, np.ndarray]:
    if probs.size == 0:
        return {
            f"{prefix}_zone_top{i}": np.empty(0, dtype=object)
            for i in range(1, top_n + 1)
        } | {
            f"{prefix}_prob_top{i}": np.empty(0, dtype=np.float32)
            for i in range(1, top_n + 1)
        }
    order = np.argsort(-probs, axis=1)[:, :top_n]
    out: dict[str, np.ndarray] = {}
    for idx in range(top_n):
        labels = np.array([ID_TO_ZONE[int(item)] for item in order[:, idx]], dtype=object)
        out[f"{prefix}_zone_top{idx + 1}"] = labels
        out[f"{prefix}_prob_top{idx + 1}"] = probs[np.arange(probs.shape[0]), order[:, idx]].astype(np.float32)
    return out


def _template_probs_from_labels(labels: np.ndarray) -> np.ndarray:
    probs = np.zeros((len(labels), len(COARSE_ZONE_LABELS)), dtype=np.float32)
    for idx, label in enumerate(labels):
        probs[idx, ZONE_TO_ID.get(str(label), ZONE_TO_ID["OTHER"])] = 1.0
    return probs


def _known_tail_template_step_labels(
    typewell: pd.DataFrame,
    template_labels: np.ndarray,
    horizontal: pd.DataFrame,
    tvt_grid: np.ndarray,
    *,
    rows_per_step: int,
) -> np.ndarray:
    tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(np.float32)
    comp_tvt_input = _compress_nanmean(tvt_input, rows_per_step)
    anchor = known_tail_linear_anchor(comp_tvt_input)
    if tvt_grid.size == 0 or anchor.size == 0:
        return np.empty(0, dtype=object)
    idx = np.abs(tvt_grid[:, None] - anchor[None, :]).argmin(axis=0)
    idx = np.clip(idx, 0, len(template_labels) - 1)
    return template_labels[idx].astype(object)


def _hidden_true_zone_labels(
    horizontal: pd.DataFrame, typewell: pd.DataFrame, cfg: PseudoZoneV1Config
) -> tuple[np.ndarray, np.ndarray]:
    _, _, tvt_grid = build_typewell_zone_features(typewell, vertical_step_ft=cfg.vertical_step_ft)
    if tvt_grid.size == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=object)
    tvt = pd.to_numeric(horizontal["TVT"], errors="coerce").to_numpy(np.float32)
    tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(np.float32)
    comp_tvt = _compress_nanmean(tvt, cfg.rows_per_step)
    comp_tvt_input = _compress_nanmean(tvt_input, cfg.rows_per_step)
    n = min(len(comp_tvt), len(comp_tvt_input))
    hidden = np.flatnonzero(~np.isfinite(comp_tvt_input[:n]) & np.isfinite(comp_tvt[:n]))
    grid_labels = _typewell_grid_labels(typewell, tvt_grid)
    if hidden.size == 0:
        return hidden.astype(np.int32), np.empty(0, dtype=object)
    true_bins = np.abs(tvt_grid[:, None] - comp_tvt[hidden][None, :]).argmin(axis=0)
    return hidden.astype(np.int32), grid_labels[true_bins].astype(object)


def _zone_rates(pred_top: np.ndarray, true_labels: np.ndarray, *, top_n: int = 1) -> float:
    if true_labels.size == 0 or pred_top.size == 0:
        return float("nan")
    if pred_top.ndim == 1:
        pred = pred_top[:, None]
    else:
        pred = pred_top[:, :top_n]
    return float(np.mean([str(true) in {str(item) for item in row[:top_n]} for true, row in zip(true_labels, pred)]))


def run_pseudo_zone_v1(
    *,
    data_dir: Path,
    output_dir: Path,
    cfg: PseudoZoneV1Config | None = None,
) -> dict[str, object]:
    cfg = cfg or PseudoZoneV1Config()
    output_dir.mkdir(parents=True, exist_ok=True)
    wells = discover_wells(DataConfig(data_dir=data_dir, k_wells=cfg.k_wells))
    loaded = [(well, *load_well(well)) for well in wells]
    folds = _split_folds(len(loaded), cfg.n_folds, cfg.seed)
    prediction_parts: list[pd.DataFrame] = []
    typewell_parts: list[pd.DataFrame] = []
    metrics_rows: list[dict[str, object]] = []
    eval_rows: list[dict[str, object]] = []
    for fold_idx, valid_idx in enumerate(folds):
        valid_set = {int(idx) for idx in valid_idx}
        train_idx = [idx for idx in range(len(loaded)) if idx not in valid_set]
        models = _fit_fold_models(loaded, train_idx, cfg, seed=cfg.seed + fold_idx * 101)
        for idx in valid_idx:
            well, horizontal, typewell = loaded[int(idx)]
            lateral_features, _ = build_lateral_zone_features(
                horizontal, rows_per_step=cfg.rows_per_step
            )
            tw_features, _, tvt_grid = build_typewell_zone_features(
                typewell, vertical_step_ft=cfg.vertical_step_ft
            )
            neural_step_probs = _predict_probs(models.lateral_model, lateral_features)
            neural_tw_probs = _predict_probs(models.typewell_model, tw_features)
            template_tw_labels = _predict_template_on_grid(typewell, tvt_grid, models.template)
            template_tw_probs = _template_probs_from_labels(template_tw_labels)
            template_step_labels = _known_tail_template_step_labels(
                typewell,
                template_tw_labels,
                horizontal,
                tvt_grid,
                rows_per_step=cfg.rows_per_step,
            )
            template_step_probs = _template_probs_from_labels(template_step_labels)
            n_steps = min(lateral_features.shape[0], neural_step_probs.shape[0], template_step_probs.shape[0])
            step_df = pd.DataFrame(
                {
                    "well_id": well.well_id,
                    "fold": int(fold_idx),
                    "step": np.arange(n_steps, dtype=np.int32),
                    "is_hidden": ~np.isfinite(
                        _compress_nanmean(
                            pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(np.float32),
                            cfg.rows_per_step,
                        )[:n_steps]
                    ),
                }
            )
            for label_idx, label in enumerate(COARSE_ZONE_LABELS):
                step_df[f"neural_prob_{label}"] = neural_step_probs[:n_steps, label_idx]
                step_df[f"template_prob_{label}"] = template_step_probs[:n_steps, label_idx]
            for name, values in _top_zone_columns("neural", neural_step_probs[:n_steps]).items():
                step_df[name] = values
            for name, values in _top_zone_columns("template", template_step_probs[:n_steps]).items():
                step_df[name] = values
            prediction_parts.append(step_df)

            bin_df = pd.DataFrame(
                {
                    "well_id": well.well_id,
                    "fold": int(fold_idx),
                    "bin_idx": np.arange(len(tvt_grid), dtype=np.int32),
                    "tvt": tvt_grid.astype(np.float32),
                }
            )
            for label_idx, label in enumerate(COARSE_ZONE_LABELS):
                bin_df[f"neural_prob_{label}"] = neural_tw_probs[:, label_idx] if neural_tw_probs.size else []
                bin_df[f"template_prob_{label}"] = template_tw_probs[:, label_idx] if template_tw_probs.size else []
            for name, values in _top_zone_columns("neural", neural_tw_probs).items():
                bin_df[name] = values
            for name, values in _top_zone_columns("template", template_tw_probs).items():
                bin_df[name] = values
            typewell_parts.append(bin_df)

            hidden_steps, true_labels = _hidden_true_zone_labels(horizontal, typewell, cfg)
            if hidden_steps.size:
                valid_steps = hidden_steps[hidden_steps < n_steps]
                true_labels = true_labels[: len(valid_steps)]
                neural_top = step_df.loc[valid_steps, ["neural_zone_top1", "neural_zone_top2", "neural_zone_top3"]].to_numpy(object)
                template_top = step_df.loc[valid_steps, ["template_zone_top1", "template_zone_top2", "template_zone_top3"]].to_numpy(object)
                for model_name, top in [("neural_zone", neural_top), ("template_baseline", template_top)]:
                    metrics_rows.append(
                        {
                            "model": model_name,
                            "well_id": well.well_id,
                            "hidden_steps": int(len(valid_steps)),
                            "hidden_zone_top1_rate": _zone_rates(top, true_labels, top_n=1),
                            "hidden_zone_top3_rate": _zone_rates(top, true_labels, top_n=3),
                        }
                    )
                    eval_rows.extend(
                        {
                            "model": model_name,
                            "well_id": well.well_id,
                            "true_zone": str(true_label),
                            "pred_zone": str(pred_label),
                        }
                        for true_label, pred_label in zip(true_labels, top[:, 0])
                    )
    predictions = pd.concat(prediction_parts, ignore_index=True) if prediction_parts else pd.DataFrame()
    typewell_bins = pd.concat(typewell_parts, ignore_index=True) if typewell_parts else pd.DataFrame()
    predictions.to_parquet(output_dir / "pseudo_zone_predictions.parquet", index=False)
    typewell_bins.to_parquet(output_dir / "pseudo_zone_typewell_bins.parquet", index=False)
    metrics = _aggregate_zone_metrics(pd.DataFrame(metrics_rows), cfg)
    _write_metrics(output_dir, metrics)
    _write_zone_prediction_figures(output_dir, pd.DataFrame(eval_rows))
    _write_report(output_dir, metrics)
    print(json.dumps(_json_clean(metrics), indent=2), flush=True)
    return metrics


def _aggregate_zone_metrics(rows: pd.DataFrame, cfg: PseudoZoneV1Config) -> dict[str, object]:
    metrics: dict[str, object] = {"config": _json_clean(asdict(cfg))}
    if rows.empty:
        metrics["template_baseline"] = {"hidden_zone_top1_rate": float("nan"), "hidden_zone_top3_rate": float("nan")}
        metrics["neural_zone"] = {"hidden_zone_top1_rate": float("nan"), "hidden_zone_top3_rate": float("nan")}
        return metrics
    for model, group in rows.groupby("model"):
        weights = group["hidden_steps"].to_numpy(np.float64)
        weights = np.where(weights > 0, weights, 1.0)
        metrics[str(model)] = {
            "num_wells": int(group["well_id"].nunique()),
            "hidden_steps": int(group["hidden_steps"].sum()),
            "hidden_zone_top1_rate": float(np.average(group["hidden_zone_top1_rate"], weights=weights)),
            "hidden_zone_top3_rate": float(np.average(group["hidden_zone_top3_rate"], weights=weights)),
        }
    return _json_clean(metrics)


def _write_metrics(output_dir: Path, metrics: dict[str, object]) -> None:
    (output_dir / "pseudo_zone_metrics.json").write_text(json.dumps(_json_clean(metrics), indent=2) + "\n")


def _write_zone_prediction_figures(output_dir: Path, eval_rows: pd.DataFrame) -> None:
    if eval_rows.empty:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    labels = list(COARSE_ZONE_LABELS)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for ax, model in zip(axes, ["template_baseline", "neural_zone"]):
        subset = eval_rows[eval_rows["model"] == model]
        matrix = np.zeros((len(labels), len(labels)), dtype=np.float32)
        for true_zone, pred_zone in zip(subset["true_zone"], subset["pred_zone"]):
            if true_zone in ZONE_TO_ID and pred_zone in ZONE_TO_ID:
                matrix[ZONE_TO_ID[true_zone], ZONE_TO_ID[pred_zone]] += 1.0
        row_sum = matrix.sum(axis=1, keepdims=True)
        norm = np.divide(matrix, row_sum, out=np.zeros_like(matrix), where=row_sum > 0)
        im = ax.imshow(norm, vmin=0.0, vmax=1.0, cmap="YlGnBu")
        ax.set_title(model)
        ax.set_xlabel("predicted zone")
        ax.set_ylabel("true zone")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="row-normalized fraction")
    fig.suptitle("Pseudo-zone hidden-step confusion")
    fig.savefig(figures_dir / "zone_confusion.png", bbox_inches="tight")
    plt.close(fig)


def _write_report(output_dir: Path, metrics: dict[str, object]) -> None:
    lines = [
        "# PSEUDO_ZONE_V1_REPORT",
        "",
        "## Zone Models",
        "| model | hidden top1 | hidden top3 | hidden steps |",
        "|---|---:|---:|---:|",
    ]
    for name in ["template_baseline", "neural_zone"]:
        src = metrics.get(name, {})
        if isinstance(src, dict):
            lines.append(
                f"| {name} | {src.get('hidden_zone_top1_rate')} | {src.get('hidden_zone_top3_rate')} | {src.get('hidden_steps')} |"
            )
    corr = metrics.get("correlation", {})
    if isinstance(corr, dict) and corr:
        lines.extend(["", "## Zone-Restricted Correlation", "| variant | top10 normal | top10 shuffled | gap | bins | train only |", "|---|---:|---:|---:|---:|---:|"])
        for variant, src in corr.items():
            if isinstance(src, dict):
                lines.append(
                    f"| {variant} | {src.get('normal_top10_rate')} | {src.get('shuffled_top10_rate')} | "
                    f"{src.get('normal_vs_shuffled_top10_gap')} | {src.get('normal_candidate_bins_mean')} | {src.get('train_only', False)} |"
                )
    figures_dir = output_dir / "figures"
    figure_files = [
        ("Zone confusion", "zone_confusion.png"),
        ("Normal vs shuffled gap", "normal_vs_shuffled_gap.png"),
        ("Candidate bins retained", "candidate_bins_retained.png"),
        ("Example zone-restricted heatmap", "example_zone_heatmap.png"),
    ]
    existing = [(title, name) for title, name in figure_files if (figures_dir / name).exists()]
    if existing:
        lines.extend(["", "## Figures"])
        for title, name in existing:
            lines.append(f"- [{title}](figures/{name})")
    lines.extend(
        [
            "",
            "## Interpretation",
            "All deployable variants use only test-available inference inputs. `oracle_geology` is a train-only ceiling and must not be used for submission.",
        ]
    )
    (output_dir / "pseudo_zone_report.md").write_text("\n".join(lines) + "\n")


def _load_zone_artifacts(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred_path = output_dir / "pseudo_zone_predictions.parquet"
    tw_path = output_dir / "pseudo_zone_typewell_bins.parquet"
    if not pred_path.exists() or not tw_path.exists():
        raise FileNotFoundError(
            f"Missing pseudo-zone artifacts in {output_dir}; run pseudo-zone-v1 first"
        )
    return pd.read_parquet(pred_path), pd.read_parquet(tw_path)


def zone_mask_from_predictions(
    typewell_zones: np.ndarray, step_zone_probs: np.ndarray, *, top_n: int
) -> np.ndarray:
    zones = np.asarray(typewell_zones, dtype=object).astype(str)
    probs = np.asarray(step_zone_probs, dtype=np.float32)
    if probs.ndim != 2:
        raise ValueError("step_zone_probs must have shape [steps, num_zones]")
    order = np.argsort(-probs, axis=1)[:, :top_n]
    masks = np.zeros((probs.shape[0], zones.shape[0]), dtype=bool)
    for step_idx, labels_idx in enumerate(order):
        allowed = {ID_TO_ZONE[int(idx)] for idx in labels_idx}
        masks[step_idx] = np.isin(zones, list(allowed))
    return masks


def _prob_matrix(frame: pd.DataFrame, prefix: str) -> np.ndarray:
    cols = [f"{prefix}_prob_{label}" for label in COARSE_ZONE_LABELS]
    missing = [col for col in cols if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing pseudo-zone probability columns: {missing[:3]}")
    return frame[cols].to_numpy(np.float32)


def _mask_scores_by_allowed(scores: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    out = np.asarray(scores, dtype=np.float32).copy()
    out[~allowed.astype(bool)] = -np.inf
    return out


def _soft_zone_adjust(
    scores: np.ndarray,
    step_probs: np.ndarray,
    typewell_probs: np.ndarray,
    *,
    weight: float,
) -> np.ndarray:
    affinity = np.maximum(typewell_probs @ step_probs.astype(np.float32), 1.0e-6)
    return (np.asarray(scores, dtype=np.float32) + float(weight) * np.log(affinity)).astype(np.float32)


def _variant_row(
    *,
    well_id: str,
    step: int,
    zone_variant: str,
    gr_variant: str,
    scores: np.ndarray,
    tvt_grid: np.ndarray,
    true_tvt: float,
    true_bin: int,
    train_only: bool,
    topk: int = 10,
) -> dict[str, object]:
    top = _topk_indices(scores, topk)
    top1 = top[:1]
    top3 = top[: min(3, len(top))]
    top10 = top[: min(10, len(top))]
    top1_bin = int(top1[0]) if top1.size else -1
    return {
        "well_id": well_id,
        "step": int(step),
        "zone_variant": zone_variant,
        "gr_variant": gr_variant,
        "train_only": bool(train_only),
        "true_tvt": float(true_tvt),
        "true_bin": int(true_bin),
        "true_rank": _rank_of_true(scores, true_bin),
        "top1_tvt": float(tvt_grid[top1_bin]) if top1_bin >= 0 else np.nan,
        "top3_oracle_sqerr": _nearest_sqerr(tvt_grid[top3], true_tvt),
        "top10_oracle_sqerr": _nearest_sqerr(tvt_grid[top10], true_tvt),
        "candidate_bins": int(np.isfinite(scores).sum()),
    }


def _nearest_sqerr(candidate_tvt: np.ndarray, true_tvt: float) -> float:
    if candidate_tvt.size == 0:
        return float("nan")
    return float(np.min((candidate_tvt.astype(np.float32) - true_tvt) ** 2))


def run_pseudo_zone_corr(
    *,
    data_dir: Path,
    output_dir: Path,
    cfg: PseudoZoneV1Config | None = None,
) -> dict[str, object]:
    cfg = cfg or PseudoZoneV1Config()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions, typewell_bins = _load_zone_artifacts(output_dir)
    wells = discover_wells(DataConfig(data_dir=data_dir, k_wells=cfg.k_wells))
    rows: list[dict[str, object]] = []
    example_heatmap: dict[str, object] | None = None
    for idx, well in enumerate(wells):
        horizontal, typewell = load_well(well)
        step_pred = predictions[predictions["well_id"] == well.well_id].sort_values("step")
        tw_pred = typewell_bins[typewell_bins["well_id"] == well.well_id].sort_values("bin_idx")
        if step_pred.empty or tw_pred.empty:
            continue
        tvt = pd.to_numeric(horizontal["TVT"], errors="coerce").to_numpy(np.float32)
        tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(np.float32)
        gr_raw = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(np.float32)
        gr_filled, gr_finite = fill_nan(gr_raw)
        shuffled_gr = gr_filled.copy()
        finite_idx = np.flatnonzero(gr_finite > 0.5)
        rng = np.random.default_rng(cfg.seed + idx)
        if finite_idx.size:
            shuffled_gr[finite_idx] = rng.permutation(shuffled_gr[finite_idx])
        comp_tvt = _compress_nanmean(tvt, cfg.rows_per_step)
        comp_tvt_input = _compress_nanmean(tvt_input, cfg.rows_per_step)
        comp_gr = _compress_nanmean(gr_filled, cfg.rows_per_step)
        comp_shuffled = _compress_nanmean(shuffled_gr, cfg.rows_per_step)
        anchor = known_tail_linear_anchor(comp_tvt_input)
        tvt_grid, typewell_gr = _regular_typewell_grid(typewell, cfg.vertical_step_ft)
        if tvt_grid.size == 0:
            continue
        grid_labels = _typewell_grid_labels(typewell, tvt_grid)
        neural_tw_probs = _prob_matrix(tw_pred, "neural")[: len(tvt_grid)]
        template_tw_probs = _prob_matrix(tw_pred, "template")[: len(tvt_grid)]
        neural_tw_labels = tw_pred["neural_zone_top1"].to_numpy(object)[: len(tvt_grid)]
        template_tw_labels = tw_pred["template_zone_top1"].to_numpy(object)[: len(tvt_grid)]
        neural_step_probs = _prob_matrix(step_pred, "neural")
        template_step_probs = _prob_matrix(step_pred, "template")
        hidden_steps = np.flatnonzero(~np.isfinite(comp_tvt_input) & np.isfinite(comp_tvt))
        for step in hidden_steps:
            if step >= len(comp_gr) or step >= len(comp_shuffled) or step >= len(step_pred):
                continue
            if not np.isfinite(comp_gr[step]):
                continue
            normal_scores, _ = score_gr_patch_multiscale(
                comp_gr,
                typewell_gr,
                step_index=int(step),
                patch_radii=cfg.patch_radii,
                stretch_factors=cfg.stretch_factors,
            )
            shuffled_scores, _ = score_gr_patch_multiscale(
                comp_shuffled,
                typewell_gr,
                step_index=int(step),
                patch_radii=cfg.patch_radii,
                stretch_factors=cfg.stretch_factors,
            )
            true_tvt = float(comp_tvt[step])
            true_bin = int(np.abs(tvt_grid - true_tvt).argmin())
            anchor_tvt = float(anchor[step]) if step < len(anchor) and np.isfinite(anchor[step]) else None
            variant_scores: dict[str, tuple[np.ndarray, np.ndarray, bool]] = {
                "global": (normal_scores, shuffled_scores, False),
                "known_tail_band": (
                    _apply_search_band(
                        normal_scores,
                        tvt_grid,
                        anchor_tvt=anchor_tvt,
                        search_radius_ft=cfg.known_tail_radius_ft,
                    ),
                    _apply_search_band(
                        shuffled_scores,
                        tvt_grid,
                        anchor_tvt=anchor_tvt,
                        search_radius_ft=cfg.known_tail_radius_ft,
                    ),
                    False,
                ),
            }
            template_masks_top1 = zone_mask_from_predictions(
                template_tw_labels, template_step_probs[[step]], top_n=1
            )[0]
            template_masks_top3 = zone_mask_from_predictions(
                template_tw_labels, template_step_probs[[step]], top_n=3
            )[0]
            neural_masks_top1 = zone_mask_from_predictions(
                neural_tw_labels, neural_step_probs[[step]], top_n=1
            )[0]
            neural_masks_top3 = zone_mask_from_predictions(
                neural_tw_labels, neural_step_probs[[step]], top_n=3
            )[0]
            oracle_allowed = grid_labels == grid_labels[true_bin]
            variant_scores.update(
                {
                    "template_zone_top1": (
                        _mask_scores_by_allowed(normal_scores, template_masks_top1),
                        _mask_scores_by_allowed(shuffled_scores, template_masks_top1),
                        False,
                    ),
                    "template_zone_top3": (
                        _mask_scores_by_allowed(normal_scores, template_masks_top3),
                        _mask_scores_by_allowed(shuffled_scores, template_masks_top3),
                        False,
                    ),
                    "neural_zone_top1": (
                        _mask_scores_by_allowed(normal_scores, neural_masks_top1),
                        _mask_scores_by_allowed(shuffled_scores, neural_masks_top1),
                        False,
                    ),
                    "neural_zone_top3": (
                        _mask_scores_by_allowed(normal_scores, neural_masks_top3),
                        _mask_scores_by_allowed(shuffled_scores, neural_masks_top3),
                        False,
                    ),
                    "neural_zone_soft": (
                        _soft_zone_adjust(
                            normal_scores,
                            neural_step_probs[step],
                            neural_tw_probs,
                            weight=cfg.soft_zone_weight,
                        ),
                        _soft_zone_adjust(
                            shuffled_scores,
                            neural_step_probs[step],
                            neural_tw_probs,
                            weight=cfg.soft_zone_weight,
                        ),
                        False,
                    ),
                    "oracle_geology": (
                        _mask_scores_by_allowed(normal_scores, oracle_allowed),
                        _mask_scores_by_allowed(shuffled_scores, oracle_allowed),
                        True,
                    ),
                }
            )
            if example_heatmap is None:
                example_heatmap = {
                    "well_id": well.well_id,
                    "steps": [],
                    "true_tvt": [],
                    "global_scores": [],
                    "zone_scores": [],
                    "tvt_grid": tvt_grid.copy(),
                }
            if (
                example_heatmap.get("well_id") == well.well_id
                and len(example_heatmap["steps"]) < 96
            ):
                zone_scores = variant_scores["template_zone_top1"][0]
                if not np.isfinite(zone_scores).any():
                    zone_scores = variant_scores["neural_zone_top3"][0]
                example_heatmap["steps"].append(int(step))
                example_heatmap["true_tvt"].append(float(true_tvt))
                example_heatmap["global_scores"].append(normal_scores.copy())
                example_heatmap["zone_scores"].append(zone_scores.copy())
            for zone_variant, (normal_variant, shuffled_variant, train_only) in variant_scores.items():
                rows.append(
                    _variant_row(
                        well_id=well.well_id,
                        step=int(step),
                        zone_variant=zone_variant,
                        gr_variant="normal",
                        scores=normal_variant,
                        tvt_grid=tvt_grid,
                        true_tvt=true_tvt,
                        true_bin=true_bin,
                        train_only=train_only,
                    )
                )
                rows.append(
                    _variant_row(
                        well_id=well.well_id,
                        step=int(step),
                        zone_variant=zone_variant,
                        gr_variant="shuffled_gr",
                        scores=shuffled_variant,
                        tvt_grid=tvt_grid,
                        true_tvt=true_tvt,
                        true_bin=true_bin,
                        train_only=train_only,
                    )
                )
    corr_steps = pd.DataFrame(rows)
    corr_steps.to_parquet(output_dir / "pseudo_zone_corr_steps.parquet", index=False)
    metrics = _read_existing_metrics(output_dir)
    metrics["correlation"] = _aggregate_corr_metrics(corr_steps)
    _write_metrics(output_dir, metrics)
    _write_corr_figures(output_dir, metrics["correlation"], example_heatmap)
    _write_report(output_dir, metrics)
    print(json.dumps(_json_clean(metrics.get("correlation", {})), indent=2), flush=True)
    return metrics


def _write_corr_figures(
    output_dir: Path,
    corr_metrics: dict[str, dict[str, object]],
    example_heatmap: dict[str, object] | None,
) -> None:
    if not corr_metrics:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    variants = [name for name, item in corr_metrics.items() if isinstance(item, dict)]
    gaps = [
        float(corr_metrics[name].get("normal_vs_shuffled_top10_gap", np.nan)) * 100.0
        for name in variants
    ]
    bins = [
        float(corr_metrics[name].get("normal_candidate_bins_mean", np.nan))
        for name in variants
    ]

    order = np.argsort(np.nan_to_num(gaps, nan=-1e9))
    ordered_variants = [variants[int(idx)] for idx in order]
    ordered_gaps = [gaps[int(idx)] for idx in order]
    fig, ax = plt.subplots(figsize=(10, max(4, 0.42 * len(ordered_variants))), dpi=150)
    colors = ["#55A868" if value >= 0 else "#C44E52" for value in ordered_gaps]
    ax.barh(ordered_variants, ordered_gaps, color=colors)
    ax.axvline(0.0, color="#222222", linewidth=1)
    ax.set_xlabel("normal_GR top10 rate - shuffled_GR top10 rate, percentage points")
    ax.set_title("Does GR still matter inside each zone mask?")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(figures_dir / "normal_vs_shuffled_gap.png", bbox_inches="tight")
    plt.close(fig)

    order_bins = np.argsort(np.nan_to_num(bins, nan=np.inf))
    ordered_variants = [variants[int(idx)] for idx in order_bins]
    ordered_bins = [bins[int(idx)] for idx in order_bins]
    fig, ax = plt.subplots(figsize=(10, max(4, 0.42 * len(ordered_variants))), dpi=150)
    ax.barh(ordered_variants, ordered_bins, color="#4C72B0")
    ax.set_xlabel("mean retained typewell bins")
    ax.set_title("How much each zone prior narrows the typewell search")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(figures_dir / "candidate_bins_retained.png", bbox_inches="tight")
    plt.close(fig)

    if example_heatmap is not None and example_heatmap.get("global_scores"):
        steps = np.asarray(example_heatmap["steps"], dtype=np.int32)
        true_tvt = np.asarray(example_heatmap["true_tvt"], dtype=np.float32)
        tvt_grid = np.asarray(example_heatmap["tvt_grid"], dtype=np.float32)
        global_scores = np.vstack(example_heatmap["global_scores"]).T.astype(np.float32)
        zone_scores = np.vstack(example_heatmap["zone_scores"]).T.astype(np.float32)
        finite = np.isfinite(global_scores)
        if finite.any():
            fill_value = float(np.nanpercentile(global_scores[finite], 2))
            zone_plot = np.where(np.isfinite(zone_scores), zone_scores, fill_value)
            vmin = float(np.nanpercentile(global_scores[finite], 5))
            vmax = float(np.nanpercentile(global_scores[finite], 95))
            fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True, dpi=150)
            for ax, matrix, title in [
                (axes[0], global_scores, "global typewell search"),
                (axes[1], zone_plot, "after pseudo-zone mask"),
            ]:
                ax.imshow(
                    matrix,
                    aspect="auto",
                    origin="lower",
                    cmap="viridis",
                    vmin=vmin,
                    vmax=vmax,
                    extent=[steps[0], steps[-1], tvt_grid[0], tvt_grid[-1]],
                )
                ax.plot(steps, true_tvt, color="white", linewidth=1.6, label="true TVT")
                ax.set_title(title)
                ax.set_xlabel("compressed step")
            axes[0].set_ylabel("typewell TVT")
            axes[1].legend(loc="upper right", frameon=True)
            fig.suptitle(f"Correlation panel before/after zone restriction: {example_heatmap.get('well_id')}")
            fig.tight_layout()
            fig.savefig(figures_dir / "example_zone_heatmap.png", bbox_inches="tight")
            plt.close(fig)


def _read_existing_metrics(output_dir: Path) -> dict[str, object]:
    path = output_dir / "pseudo_zone_metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _aggregate_corr_metrics(steps: pd.DataFrame) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    if steps.empty:
        return out
    for variant, group in steps.groupby("zone_variant"):
        normal = group[group["gr_variant"] == "normal"]
        shuffled = group[group["gr_variant"] == "shuffled_gr"]
        normal_metrics = _corr_metrics_for_group(normal)
        shuffled_metrics = _corr_metrics_for_group(shuffled)
        out[str(variant)] = {
            "train_only": bool(group["train_only"].any()),
            "num_steps": int(len(normal)),
            "normal_top1_rmse_ft": normal_metrics["top1_rmse_ft"],
            "shuffled_top1_rmse_ft": shuffled_metrics["top1_rmse_ft"],
            "normal_top10_oracle_rmse_ft": normal_metrics["top10_oracle_rmse_ft"],
            "shuffled_top10_oracle_rmse_ft": shuffled_metrics["top10_oracle_rmse_ft"],
            "normal_top10_rate": normal_metrics["top10_rate"],
            "shuffled_top10_rate": shuffled_metrics["top10_rate"],
            "normal_vs_shuffled_top10_gap": normal_metrics["top10_rate"] - shuffled_metrics["top10_rate"],
            "normal_vs_shuffled_top10_rmse_gap_ft": shuffled_metrics["top10_oracle_rmse_ft"] - normal_metrics["top10_oracle_rmse_ft"],
            "normal_candidate_bins_mean": normal_metrics["candidate_bins_mean"],
            "shuffled_candidate_bins_mean": shuffled_metrics["candidate_bins_mean"],
        }
    return _json_clean(out)


def _corr_metrics_for_group(group: pd.DataFrame) -> dict[str, float]:
    if group.empty:
        return {
            "top1_rmse_ft": float("nan"),
            "top10_oracle_rmse_ft": float("nan"),
            "top10_rate": float("nan"),
            "candidate_bins_mean": float("nan"),
        }
    top1_err = group["top1_tvt"].to_numpy(np.float32) - group["true_tvt"].to_numpy(np.float32)
    top1_finite = np.isfinite(top1_err)
    top10_sqerr = group["top10_oracle_sqerr"].to_numpy(np.float32)
    top10_finite = np.isfinite(top10_sqerr)
    ranks = group["true_rank"].to_numpy(np.float32)
    rank_finite = np.isfinite(ranks)
    candidate_bins = group["candidate_bins"].to_numpy(np.float32)
    candidate_finite = np.isfinite(candidate_bins)
    return {
        "top1_rmse_ft": float(np.sqrt(np.mean(top1_err[top1_finite] ** 2))) if top1_finite.any() else float("nan"),
        "top10_oracle_rmse_ft": float(np.sqrt(np.mean(top10_sqerr[top10_finite]))) if top10_finite.any() else float("nan"),
        "top10_rate": float(np.mean(ranks[rank_finite] <= 10)) if rank_finite.any() else float("nan"),
        "candidate_bins_mean": float(np.mean(candidate_bins[candidate_finite])) if candidate_finite.any() else float("nan"),
    }


def run_pseudo_zone_ga_eval(
    *,
    data_dir: Path,
    output_dir: Path,
    cfg: PseudoZoneV1Config | None = None,
) -> dict[str, object]:
    """Write a lightweight DP-style proxy report from zone-restricted correlation rows.

    This is intentionally a diagnostic bridge: the expensive neural GeoAligner can later
    consume the same zone artifacts, while this command verifies whether zone-restricted
    emissions improve before another model is trained.
    """
    cfg = cfg or PseudoZoneV1Config()
    corr_path = output_dir / "pseudo_zone_corr_steps.parquet"
    if not corr_path.exists():
        run_pseudo_zone_corr(data_dir=data_dir, output_dir=output_dir, cfg=cfg)
    steps = pd.read_parquet(corr_path)
    metrics = _read_existing_metrics(output_dir)
    ga_metrics: dict[str, object] = {}
    for variant, group in steps[steps["gr_variant"] == "normal"].groupby("zone_variant"):
        err = group["top1_tvt"].to_numpy(np.float32) - group["true_tvt"].to_numpy(np.float32)
        ga_metrics[str(variant)] = {
            "emission_top1_rmse_ft": float(np.sqrt(np.nanmean(err**2))) if len(err) else float("nan"),
            "target_top10_rate": float(np.nanmean(group["true_rank"].to_numpy(np.float32) <= 10)) if len(group) else float("nan"),
            "train_only": bool(group["train_only"].any()),
        }
    metrics["ga_zone_eval"] = ga_metrics
    _write_metrics(output_dir, metrics)
    _write_report(output_dir, metrics)
    (output_dir / "pseudo_zone_ga_metrics.json").write_text(
        json.dumps(_json_clean(ga_metrics), indent=2) + "\n"
    )
    print(json.dumps(_json_clean(ga_metrics), indent=2), flush=True)
    return metrics
