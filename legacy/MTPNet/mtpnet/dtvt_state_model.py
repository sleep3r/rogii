"""``dtvt_state_model_v0`` — per-row dTVT predictor trained with dual loss.

Motivation
==========
The horizontal-well TVT/Z decomposition is

    dTVT[i] = -dZ[i] + r[i]
    TVT[i]  = anchor + cumsum(dTVT[i:])

where ``r = dTVT + dZ`` is the local rate-of-change of the formation offset
``C``. Empirically:

* ``r`` is small (|r| typically < 0.1 ft/row) and is strongly predictable
  from the ``top_state_teacher`` classifier output (Pearson 0.72, 31%
  per-row RMSE reduction).
* A pure per-row regressor of ``r`` (CatBoost) hits a ~0.030 ft per-row
  RMSE floor, which after cumsum over ~3000 hidden rows compounds into
  ~25 ft pooled TVT-RMSE due to systematic per-well bias.
* Oracle per-well constant offset already buys 7.59 ft pooled, and
  oracle K=5 piecewise-constant buys 1.82 ft.

The minimum-viable fix is to train a per-row predictor with a loss that
*sees the cumulative TVT error*, not just the per-row residual. That is:

    loss = alpha * MSE(r_pred, r_true) [local]
         + beta  * MSE(tvt_pred, tvt) [global cumsum]

CatBoost cannot natively support a per-well cumsum-coupled loss, so this
module uses a small PyTorch MLP. We process one well at a time, run
cumsum from the last known TVT_input anchor, and backprop through the
cumulative sum.

This module emits an OOF parquet keyed by ``id`` with column ``pred_tvt``
so the candidate bank can consume it like ``residual_stack`` and
``k_segment_offset``.

Schema safety is enforced via ``assert_schema_safe_columns``.
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
import torch.nn as nn
import torch.nn.functional as F

from .residual_stack import (
    ResidualStackConfig,
    _ensure_ids,
    load_training_frame,
    make_group_folds,
)
from .schema_safe import assert_schema_safe_columns


@dataclass(frozen=True)
class DTVTStateModelConfig:
    data_dir: Path = Path("data/train")
    output_dir: Path = Path("artifacts/dtvt_state_model_v0")
    top_state_path: Path | None = Path(
        "artifacts/top_state_teacher_v0/top_state_oof_predictions.parquet"
    )
    n_folds: int = 5
    seed: int = 42
    k_wells: int = -1
    rows_per_step: int = 32
    hidden_dim: int = 256
    dropout: float = 0.0
    epochs: int = 25
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5
    alpha_local: float = 1.0  # weight on local MSE(r_pred, r_true)
    beta_global: float = 5e-3  # weight on global cumsum MSE
    grad_clip: float = 5.0
    micro_batch_wells: int = 4  # gradient-accumulate across this many wells per step
    use_residual_baseline: bool = True  # init r_pred ≈ 0.04 * pred_sign + 0.003
    baseline_scale_init: float = 0.04
    baseline_bias_init: float = 0.003
    warmup_frac: float = 0.1  # cosine schedule warmup fraction
    progress_every: int = 50
    device: str = "auto"  # "auto" / "cpu" / "cuda"


# --- feature engineering -------------------------------------------------


PER_ROW_FEATURES: tuple[str, ...] = (
    "feat_dz",
    "feat_neg_dz",
    "feat_z_n",
    "feat_md_n",
    "feat_gr_n",
    "feat_x_n",
    "feat_y_n",
    "feat_rel_dist",
    "feat_rel_progress",
    "feat_pred_expected_sign",
    "feat_prob_up",
    "feat_prob_down",
    "feat_prob_flat",
    "feat_anchor_C_n",
    "feat_anchor_z_n",
    "feat_well_known_dC_median",
    "feat_well_known_dC_mean",
    "feat_well_known_gr_mean",
    "feat_well_hidden_n_log",
)


def _safe_mean(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else 0.0


def _safe_median(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if a.size else 0.0


def _load_top_state_lookup(path: Path | None) -> dict[str, pd.DataFrame]:
    if path is None or not Path(path).exists():
        return {}
    frame = pd.read_parquet(path)
    if frame.empty:
        return {}
    frame["well_id"] = frame["well_id"].astype(str)
    cols = {"well_id", "step", "pred_expected_sign", "prob_up", "prob_down", "prob_flat"}
    if not cols.issubset(frame.columns):
        return {}
    out: dict[str, pd.DataFrame] = {}
    for wid, grp in frame[list(cols)].groupby("well_id", sort=False):
        gg = grp.copy()
        gg["step"] = pd.to_numeric(gg["step"], errors="coerce").astype("Int64")
        gg = gg.dropna(subset=["step"]).reset_index(drop=True)
        gg["step"] = gg["step"].astype(int)
        out[str(wid)] = gg.set_index("step")
    return out


@dataclass
class _WellTensors:
    well_id: str
    fold: int
    features: torch.Tensor  # [N, F]
    dz: torch.Tensor  # [N]
    r_true: torch.Tensor  # [N], nan where dtvt is undefined
    tvt_true: torch.Tensor  # [N], nan where TVT unknown (test-mode; in train both known and hidden have TVT)
    tvt_input: torch.Tensor  # [N], nan on hidden rows
    is_known: torch.Tensor  # [N] bool: TVT_input present
    is_hidden: torch.Tensor  # [N] bool: TVT_input absent but TVT present (training target rows)
    anchor_row: int
    row_idx: torch.Tensor  # [N], int
    ids: list[str]


def _build_well_tensors(
    frame: pd.DataFrame,
    *,
    fold_of_well: dict[str, int],
    top_state_lookup: dict[str, pd.DataFrame],
    rows_per_step: int,
) -> list[_WellTensors]:
    out: list[_WellTensors] = []
    for wid_raw, group in frame.groupby("well_id", sort=True):
        wid = str(wid_raw)
        if wid not in fold_of_well:
            continue
        g = group.sort_values("row_idx").reset_index(drop=True)
        n = len(g)
        if n < 4:
            continue
        z = pd.to_numeric(g.get("Z", pd.Series(np.nan, index=g.index)), errors="coerce").to_numpy(
            dtype=np.float64
        )
        tvt = pd.to_numeric(g.get("TVT", pd.Series(np.nan, index=g.index)), errors="coerce").to_numpy(
            dtype=np.float64
        )
        tvt_in = pd.to_numeric(
            g.get("TVT_input", pd.Series(np.nan, index=g.index)), errors="coerce"
        ).to_numpy(dtype=np.float64)
        md = pd.to_numeric(g.get("MD", pd.Series(np.nan, index=g.index)), errors="coerce").to_numpy(
            dtype=np.float64
        )
        gr = pd.to_numeric(g.get("GR", pd.Series(np.nan, index=g.index)), errors="coerce").to_numpy(
            dtype=np.float64
        )
        x = pd.to_numeric(g.get("X", pd.Series(np.nan, index=g.index)), errors="coerce").to_numpy(
            dtype=np.float64
        )
        y = pd.to_numeric(g.get("Y", pd.Series(np.nan, index=g.index)), errors="coerce").to_numpy(
            dtype=np.float64
        )
        ids = g.get("id", pd.Series([f"{wid}_{int(i)}" for i in range(n)])).astype(str).tolist()
        if not (np.isfinite(z).any() and np.isfinite(tvt_in).any()):
            continue
        known_mask = np.isfinite(tvt_in)
        known_idx = np.flatnonzero(known_mask)
        if known_idx.size < 2:
            continue
        anchor_row = int(known_idx[-1])
        hidden_mask = (~known_mask) & np.isfinite(tvt)
        if not hidden_mask.any():
            continue
        # gradients
        dz = np.gradient(z) if n >= 2 else np.zeros(n)
        dtvt = np.gradient(tvt)
        r_true = dtvt + dz  # NaN propagation expected on missing tvt
        # well-level scalars (broadcast as per-row features)
        C_known = tvt_in[known_mask] + z[known_mask]
        dC_known = np.gradient(C_known) if C_known.size >= 2 else np.zeros_like(C_known)
        well_known_dC_median = _safe_median(dC_known)
        well_known_dC_mean = _safe_mean(dC_known)
        well_known_gr_mean = _safe_mean(gr[known_mask]) / 100.0
        well_hidden_n_log = float(np.log1p(n - anchor_row - 1))
        anchor_C = float(tvt_in[anchor_row] + z[anchor_row])
        anchor_z = float(z[anchor_row])
        # per-row features
        row_idx_arr = np.arange(n, dtype=np.int64)
        rel_dist = (row_idx_arr - anchor_row).astype(np.float64) / 1000.0
        rel_progress = (row_idx_arr / max(n - 1, 1)).astype(np.float64)
        # top_state lookup (per step)
        sign_arr = np.zeros(n, dtype=np.float64)
        p_up = np.zeros(n, dtype=np.float64)
        p_dn = np.zeros(n, dtype=np.float64)
        p_fl = np.zeros(n, dtype=np.float64)
        tw = top_state_lookup.get(wid)
        if tw is not None and not tw.empty:
            steps = row_idx_arr // max(int(rows_per_step), 1)
            joined = tw.reindex(steps).reset_index(drop=True)
            sign_arr = pd.to_numeric(joined["pred_expected_sign"], errors="coerce").fillna(0.0).to_numpy(
                dtype=np.float64
            )
            p_up = pd.to_numeric(joined["prob_up"], errors="coerce").fillna(0.0).to_numpy(
                dtype=np.float64
            )
            p_dn = pd.to_numeric(joined["prob_down"], errors="coerce").fillna(0.0).to_numpy(
                dtype=np.float64
            )
            p_fl = pd.to_numeric(joined["prob_flat"], errors="coerce").fillna(0.0).to_numpy(
                dtype=np.float64
            )
        # assemble feature matrix in the same order as PER_ROW_FEATURES
        dz_safe = np.nan_to_num(dz)
        feats = np.stack(
            [
                dz_safe,
                -dz_safe,
                np.nan_to_num(z) / 10000.0,
                np.nan_to_num(md) / 10000.0,
                np.nan_to_num(gr) / 100.0,
                np.nan_to_num(x) / 10000.0,
                np.nan_to_num(y) / 10000.0,
                rel_dist,
                rel_progress,
                sign_arr,
                p_up,
                p_dn,
                p_fl,
                np.full(n, anchor_C / 10000.0),
                np.full(n, anchor_z / 10000.0),
                np.full(n, well_known_dC_median),
                np.full(n, well_known_dC_mean),
                np.full(n, well_known_gr_mean),
                np.full(n, well_hidden_n_log),
            ],
            axis=1,
        )
        feats = np.where(np.isfinite(feats), feats, 0.0).astype(np.float32)
        out.append(
            _WellTensors(
                well_id=wid,
                fold=fold_of_well[wid],
                features=torch.from_numpy(feats),
                dz=torch.from_numpy(dz_safe.astype(np.float32)),
                r_true=torch.from_numpy(r_true.astype(np.float32)),
                tvt_true=torch.from_numpy(tvt.astype(np.float32)),
                tvt_input=torch.from_numpy(tvt_in.astype(np.float32)),
                is_known=torch.from_numpy(known_mask),
                is_hidden=torch.from_numpy(hidden_mask),
                anchor_row=anchor_row,
                row_idx=torch.from_numpy(row_idx_arr.astype(np.int64)),
                ids=ids,
            )
        )
    if not out:
        raise ValueError("No wells built for dtvt_state_model")
    return out


# --- model ---------------------------------------------------------------


class DTVTStateModel(nn.Module):
    """Per-row dTVT predictor with optional residual baseline.

    The output is

        r_pred = baseline_scale * pred_expected_sign
               + baseline_bias
               + net(features)

    where ``baseline_scale``/``baseline_bias`` are learnable scalars
    initialised so that even at step 0 the model matches the
    closed-form linear classifier baseline (~30 ft pooled). The MLP
    then only has to learn the *residual* on top, which converges far
    faster than predicting ``r`` from scratch.

    The ``pred_expected_sign`` is the 9th feature in ``PER_ROW_FEATURES``.
    """

    SIGN_FEATURE_INDEX = 9

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        *,
        use_residual_baseline: bool = True,
        baseline_scale_init: float = 0.04,
        baseline_bias_init: float = 0.003,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [nn.Linear(hidden_dim, 1)]
        # initialise the final layer to near-zero output so that initial
        # predictions are dominated by the residual baseline below.
        final = layers[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        self.net = nn.Sequential(*layers)
        self.use_residual_baseline = use_residual_baseline
        if use_residual_baseline:
            self.baseline_scale = nn.Parameter(torch.tensor(float(baseline_scale_init)))
            self.baseline_bias = nn.Parameter(torch.tensor(float(baseline_bias_init)))
        else:
            self.register_buffer("baseline_scale", torch.tensor(0.0))
            self.register_buffer("baseline_bias", torch.tensor(0.0))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        sign = features[..., self.SIGN_FEATURE_INDEX]
        baseline = self.baseline_scale * sign + self.baseline_bias
        residual = self.net(features).squeeze(-1)
        return baseline + residual


# --- dual-loss training --------------------------------------------------


def _cumsum_tvt(
    r_pred: torch.Tensor,
    dz: torch.Tensor,
    anchor_row: int,
    anchor_tvt: float,
) -> torch.Tensor:
    """Differentiable cumsum from ``anchor_row`` forward.

    Returns ``tvt_pred`` of length ``n``; values at positions < anchor_row
    are not used by the global loss but are filled with the anchor for
    indexability. Positions >= anchor_row contain valid predictions.
    """
    n = r_pred.shape[0]
    dtvt_pred = -dz + r_pred  # [N]
    # cumulative sum from anchor_row + 1; first step = anchor_tvt
    forward = dtvt_pred[anchor_row + 1 :]
    cum_forward = torch.cumsum(forward, dim=0)
    after = anchor_tvt + cum_forward  # length n - anchor_row - 1
    head = torch.full((anchor_row + 1,), anchor_tvt, dtype=r_pred.dtype, device=r_pred.device)
    return torch.cat([head, after])


def _fold_train_predict(
    wells: list[_WellTensors],
    train_idx: list[int],
    valid_idx: list[int],
    *,
    config: DTVTStateModelConfig,
    device: torch.device,
) -> dict[str, np.ndarray]:
    in_dim = len(PER_ROW_FEATURES)
    model = DTVTStateModel(
        in_dim,
        config.hidden_dim,
        config.dropout,
        use_residual_baseline=config.use_residual_baseline,
        baseline_scale_init=config.baseline_scale_init,
        baseline_bias_init=config.baseline_bias_init,
    ).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    # cosine schedule with warmup; one optimiser step per micro-batch.
    micro = max(int(config.micro_batch_wells), 1)
    steps_per_epoch = max(len(train_idx) // micro, 1)
    total_steps = max(steps_per_epoch * config.epochs, 1)
    warmup_steps = max(int(total_steps * config.warmup_frac), 1)

    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0)))

    global_step = 0
    for epoch in range(config.epochs):
        model.train()
        rng = np.random.default_rng(config.seed + epoch)
        order = train_idx.copy()
        rng.shuffle(order)
        running_local = 0.0
        running_global = 0.0
        n_local_total = 0
        n_global_total = 0
        # Iterate in micro-batches of wells; accumulate grads and step once.
        micro_count = 0
        optimiser.zero_grad()
        for pos_in_epoch, well_pos in enumerate(order):
            wells_i = wells[well_pos]
            feats = wells_i.features.to(device)
            dz = wells_i.dz.to(device)
            r_true = wells_i.r_true.to(device)
            tvt_true = wells_i.tvt_true.to(device)
            known = wells_i.is_known.to(device)
            hidden = wells_i.is_hidden.to(device)
            anchor_tvt = float(wells_i.tvt_input[wells_i.anchor_row].item())
            r_pred = model(feats)
            local_mask = torch.isfinite(r_true) & (known | hidden)
            if local_mask.any():
                local = F.mse_loss(r_pred[local_mask], r_true[local_mask])
            else:
                local = torch.tensor(0.0, device=device)
            tvt_pred = _cumsum_tvt(r_pred, dz, wells_i.anchor_row, anchor_tvt)
            global_mask = hidden & torch.isfinite(tvt_true)
            if global_mask.any():
                global_loss = F.mse_loss(tvt_pred[global_mask], tvt_true[global_mask])
            else:
                global_loss = torch.tensor(0.0, device=device)
            loss = (config.alpha_local * local + config.beta_global * global_loss) / micro
            loss.backward()
            running_local += float(local.item()) * int(local_mask.sum().item())
            running_global += float(global_loss.item()) * int(global_mask.sum().item())
            n_local_total += int(local_mask.sum().item())
            n_global_total += int(global_mask.sum().item())
            micro_count += 1
            if micro_count >= micro or pos_in_epoch == len(order) - 1:
                # apply schedule
                lr_scale = lr_at(global_step)
                for pg in optimiser.param_groups:
                    pg["lr"] = config.learning_rate * lr_scale
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)
                optimiser.step()
                optimiser.zero_grad()
                micro_count = 0
                global_step += 1
        local_mean = running_local / max(n_local_total, 1)
        global_mean = running_global / max(n_global_total, 1)
        # Extract the learnable residual baseline scalars if used
        scale_val = float(model.baseline_scale.detach().item()) if config.use_residual_baseline else 0.0
        bias_val = float(model.baseline_bias.detach().item()) if config.use_residual_baseline else 0.0
        print(
            f"[dtvt-state] epoch {epoch + 1}/{config.epochs} "
            f"local_mse={local_mean:.5f}  global_mse={global_mean:.2f}  "
            f"baseline=(scale={scale_val:+.4f}, bias={bias_val:+.5f})  "
            f"lr={config.learning_rate * lr_at(global_step):.2e}",
            file=sys.stderr,
            flush=True,
        )
    # predict on valid
    model.eval()
    out: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for well_pos in valid_idx:
            w = wells[well_pos]
            feats = w.features.to(device)
            r_pred = model(feats).cpu().numpy().astype(np.float64)
            dz = w.dz.cpu().numpy().astype(np.float64)
            anchor_tvt = float(w.tvt_input[w.anchor_row].item())
            anchor_z = 0.0  # absorbed in anchor_tvt + cumsum
            dtvt = -dz + r_pred
            tvt_pred = np.full(len(r_pred), np.nan, dtype=np.float64)
            tvt_pred[w.anchor_row] = anchor_tvt
            for i in range(w.anchor_row + 1, len(r_pred)):
                tvt_pred[i] = tvt_pred[i - 1] + dtvt[i]
            out[w.well_id] = tvt_pred
    return out


# --- diagnostics ---------------------------------------------------------


def _evaluate_predictions(
    wells: list[_WellTensors],
    tvt_pred_by_well: dict[str, np.ndarray],
) -> dict[str, Any]:
    sse = 0.0
    n = 0
    per_well = []
    for w in wells:
        pred = tvt_pred_by_well.get(w.well_id)
        if pred is None:
            continue
        hidden = w.is_hidden.cpu().numpy()
        truth = w.tvt_true.cpu().numpy()
        mask = hidden & np.isfinite(pred) & np.isfinite(truth)
        if not mask.any():
            continue
        err = pred[mask] - truth[mask]
        sse += float(np.sum(err**2))
        n += int(mask.sum())
        per_well.append(float(np.sqrt(np.mean(err**2))))
    pooled = float(np.sqrt(sse / n)) if n else float("nan")
    pw = pd.Series(per_well, dtype=float)
    return {
        "hidden_pooled_rmse": pooled,
        "hidden_well_rmse_mean": float(pw.mean()) if not pw.empty else float("nan"),
        "hidden_well_rmse_median": float(pw.median()) if not pw.empty else float("nan"),
        "hidden_well_rmse_p90": float(pw.quantile(0.90)) if not pw.empty else float("nan"),
        "hidden_well_rmse_p99": float(pw.quantile(0.99)) if not pw.empty else float("nan"),
        "rows": int(n),
    }


# --- public driver -------------------------------------------------------


def _resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().tolist()
    return value


def run_dtvt_state_model_from_frame(
    frame: pd.DataFrame,
    *,
    config: DTVTStateModelConfig,
    top_state_lookup: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    assert_schema_safe_columns(list(PER_ROW_FEATURES), context="dtvt_state_model features")
    raw = _ensure_ids(frame)
    well_ids = sorted({str(w) for w in raw["well_id"].astype(str).unique()})
    folds = make_group_folds(pd.Series(well_ids), n_folds=config.n_folds, seed=config.seed)
    fold_of_well = {str(w): idx for idx, (_, valid) in enumerate(folds) for w in valid}
    if top_state_lookup is None:
        top_state_lookup = _load_top_state_lookup(config.top_state_path)
    wells = _build_well_tensors(
        raw,
        fold_of_well=fold_of_well,
        top_state_lookup=top_state_lookup,
        rows_per_step=config.rows_per_step,
    )
    device = _resolve_device(config.device)
    print(
        f"[dtvt-state] wells={len(wells)} device={device} epochs={config.epochs}",
        file=sys.stderr,
        flush=True,
    )
    tvt_pred_by_well: dict[str, np.ndarray] = {}
    for fold_idx in range(len(folds)):
        train_pos = [i for i, w in enumerate(wells) if w.fold != fold_idx]
        valid_pos = [i for i, w in enumerate(wells) if w.fold == fold_idx]
        if not train_pos or not valid_pos:
            continue
        print(
            f"[dtvt-state] fold {fold_idx + 1}/{len(folds)} train={len(train_pos)} valid={len(valid_pos)}",
            file=sys.stderr,
            flush=True,
        )
        preds = _fold_train_predict(
            wells, train_pos, valid_pos, config=config, device=device
        )
        tvt_pred_by_well.update(preds)
    diagnostics = _evaluate_predictions(wells, tvt_pred_by_well)
    # materialise per-row predictions on hidden rows only
    rows: list[dict[str, Any]] = []
    for w in wells:
        pred = tvt_pred_by_well.get(w.well_id)
        if pred is None:
            continue
        hidden_idx = torch.nonzero(w.is_hidden, as_tuple=False).flatten().cpu().numpy()
        for i in hidden_idx:
            value = float(pred[int(i)])
            if not np.isfinite(value):
                continue
            rows.append(
                {
                    "id": w.ids[int(i)],
                    "well_id": w.well_id,
                    "row_idx": int(w.row_idx[int(i)].item()),
                    "pred_tvt": value,
                }
            )
    if not rows:
        raise ValueError("dtvt_state_model produced no row predictions")
    row_predictions = pd.DataFrame(rows)
    metrics: dict[str, Any] = {
        "candidate": "dtvt_state_model_v0",
        "wells": int(len(wells)),
        "config": asdict(config),
        "features": list(PER_ROW_FEATURES),
        **diagnostics,
    }
    row_predictions.to_parquet(out_dir / "dtvt_state_oof_predictions.parquet", index=False)
    (out_dir / "dtvt_state_metrics.json").write_text(
        json.dumps(_json_safe(metrics), indent=2), encoding="utf-8"
    )
    lines = [
        "# DTVT_STATE_MODEL_V0",
        "",
        "Per-row dTVT predictor trained with dual loss: local MSE on `r=dTVT+dZ`",
        "and global MSE on cumsum-derived TVT against held-out wells.",
        "",
        "## summary",
        "```json",
        json.dumps(_json_safe(metrics), indent=2),
        "```",
    ]
    (out_dir / "dtvt_state_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return metrics


def run_dtvt_state_model(config: DTVTStateModelConfig) -> dict[str, Any]:
    frame = load_training_frame(
        ResidualStackConfig(data_dir=config.data_dir, k_wells=config.k_wells)
    )
    return run_dtvt_state_model_from_frame(frame, config=config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train fold-safe per-row dTVT predictor with dual local+cumsum loss"
    )
    parser.add_argument("--data-dir", type=Path, default=DTVTStateModelConfig.data_dir)
    parser.add_argument("--output-dir", type=Path, default=DTVTStateModelConfig.output_dir)
    parser.add_argument("--top-state-path", type=Path, default=DTVTStateModelConfig.top_state_path)
    parser.add_argument("--n-folds", type=int, default=DTVTStateModelConfig.n_folds)
    parser.add_argument("--seed", type=int, default=DTVTStateModelConfig.seed)
    parser.add_argument("--k-wells", type=int, default=DTVTStateModelConfig.k_wells)
    parser.add_argument("--rows-per-step", type=int, default=DTVTStateModelConfig.rows_per_step)
    parser.add_argument("--hidden-dim", type=int, default=DTVTStateModelConfig.hidden_dim)
    parser.add_argument("--dropout", type=float, default=DTVTStateModelConfig.dropout)
    parser.add_argument("--epochs", type=int, default=DTVTStateModelConfig.epochs)
    parser.add_argument("--learning-rate", type=float, default=DTVTStateModelConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=DTVTStateModelConfig.weight_decay)
    parser.add_argument("--alpha-local", type=float, default=DTVTStateModelConfig.alpha_local)
    parser.add_argument("--beta-global", type=float, default=DTVTStateModelConfig.beta_global)
    parser.add_argument("--grad-clip", type=float, default=DTVTStateModelConfig.grad_clip)
    parser.add_argument(
        "--micro-batch-wells",
        type=int,
        default=DTVTStateModelConfig.micro_batch_wells,
        help="Accumulate gradients over this many wells before stepping.",
    )
    parser.add_argument(
        "--no-residual-baseline",
        action="store_true",
        help="Disable the learnable r=scale*pred_sign+bias init (start from zero).",
    )
    parser.add_argument(
        "--baseline-scale-init",
        type=float,
        default=DTVTStateModelConfig.baseline_scale_init,
    )
    parser.add_argument(
        "--baseline-bias-init",
        type=float,
        default=DTVTStateModelConfig.baseline_bias_init,
    )
    parser.add_argument("--warmup-frac", type=float, default=DTVTStateModelConfig.warmup_frac)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=DTVTStateModelConfig.device)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metrics = run_dtvt_state_model(
        DTVTStateModelConfig(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            top_state_path=args.top_state_path,
            n_folds=args.n_folds,
            seed=args.seed,
            k_wells=args.k_wells,
            rows_per_step=args.rows_per_step,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            alpha_local=args.alpha_local,
            beta_global=args.beta_global,
            grad_clip=args.grad_clip,
            micro_batch_wells=args.micro_batch_wells,
            use_residual_baseline=not args.no_residual_baseline,
            baseline_scale_init=args.baseline_scale_init,
            baseline_bias_init=args.baseline_bias_init,
            warmup_frac=args.warmup_frac,
            device=args.device,
        )
    )
    compact = {
        "candidate": metrics["candidate"],
        "wells": metrics["wells"],
        "hidden_pooled_rmse": metrics["hidden_pooled_rmse"],
        "hidden_well_rmse_median": metrics["hidden_well_rmse_median"],
        "report": str(Path(args.output_dir) / "dtvt_state_report.md"),
    }
    print(json.dumps(_json_safe(compact), indent=2))


if __name__ == "__main__":
    main()
