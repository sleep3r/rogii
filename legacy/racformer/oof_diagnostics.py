"""OOF ablation diagnostics for RAC-Former."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import EVENT_THRESHOLD, RACFormerConfig
from .dataset import RACDataset, load_all_wells
from .model import RACFormer, RACFormerOutput, load_state_dict_allowing_top_teacher_heads, materialize_rows

DEFAULT_VARIANTS: tuple[str, ...] = (
    "last_known_tvt",
    "base_no_c0",
    "base_with_c0",
    "model_full",
    "model_no_direct",
    "model_no_s_pred",
)

BUCKET_FIELDS: tuple[str, ...] = (
    "hidden_length",
    "GR_valid_frac",
    "abs_c0",
    "event_count",
    "base_only_rmse",
)


def _rmse(pred: np.ndarray, true: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    valid = np.isfinite(pred) & np.isfinite(true)
    if not np.any(valid):
        return float("nan")
    return float(np.sqrt(np.mean((pred[valid] - true[valid]) ** 2)))


def compute_variant_predictions(
    batch: dict[str, torch.Tensor],
    out: RACFormerOutput,
    k_seg: int,
) -> dict[str, torch.Tensor]:
    """Materialize baseline and inference-time ablation predictions."""
    n_hidden = batch["n_hidden_rows"]
    H_max = batch["base_tvt_hidden"].shape[1]
    anchor_tvt = batch["anchor_tvt"].unsqueeze(1).expand(-1, H_max)
    anchor_z = batch["anchor_z"].unsqueeze(1).expand(-1, H_max)

    base_no_c0 = anchor_tvt - (batch["z_hidden"] - anchor_z)
    base_with_c0 = batch["base_tvt_hidden"]

    def _materialize(s_pred: torch.Tensor, direct: torch.Tensor) -> torch.Tensor:
        return materialize_rows(
            s_pred=s_pred,
            direct_resid_step=direct,
            base_tvt_hidden=base_with_c0,
            n_hidden_rows=n_hidden,
            hidden_row_to_step=batch["hidden_row_to_step"],
            anchor_step=batch["anchor_step"],
            k_seg=k_seg,
            anchor_dr_zero=True,
        )

    return {
        "last_known_tvt": anchor_tvt.clone(),
        "base_no_c0": base_no_c0,
        "base_with_c0": base_with_c0,
        "model_full": _materialize(out.s_pred, out.direct_resid_step),
        "model_no_direct": _materialize(out.s_pred, torch.zeros_like(out.direct_resid_step)),
        "model_no_s_pred": _materialize(torch.zeros_like(out.s_pred), out.direct_resid_step),
    }


def sample_diagnostic_fields(
    sample: Any,
    base_with_c0: np.ndarray,
    tvt_true: np.ndarray,
    event_threshold: float = EVENT_THRESHOLD,
) -> dict[str, float | int]:
    """Compute per-well diagnostic fields used for bucketed OOF summaries."""
    hidden_mask = np.asarray(getattr(sample, "hidden_mask", []), dtype=bool)
    features = np.asarray(getattr(sample, "features", np.empty((0, 27))), dtype=np.float32)
    if features.ndim == 2 and features.shape[1] > 26 and hidden_mask.shape[0] == features.shape[0]:
        gr_vals = features[hidden_mask, 26]
        gr_vals = gr_vals[np.isfinite(gr_vals)]
        gr_valid_frac = float(np.mean(gr_vals)) if len(gr_vals) else 0.0
    else:
        gr_valid_frac = 0.0

    dC_forward = np.asarray(getattr(sample, "dC_forward", []), dtype=np.float32)
    dC_forward = dC_forward[np.isfinite(dC_forward)]
    event_count = int(np.sum(np.abs(dC_forward) > event_threshold))

    return {
        "hidden_length": int(getattr(sample, "n_hidden_rows", len(tvt_true))),
        "GR_valid_frac": gr_valid_frac,
        "abs_c0": float(abs(float(getattr(sample, "c0", 0.0)))),
        "event_count": event_count,
        "base_only_rmse": _rmse(base_with_c0, tvt_true),
    }


def build_row_records(
    fold: int,
    sample: Any,
    variant_predictions: dict[str, torch.Tensor],
    diagnostics: dict[str, float | int],
) -> list[dict[str, float | int | str]]:
    """Build row-level CSV records for one held-out well."""
    H = int(getattr(sample, "n_hidden_rows", len(sample.hidden_row_ids)))
    anchor_row = int(sample.anchor_row)
    rows: list[dict[str, float | int | str]] = []

    variant_np = {
        name: tensor.detach().cpu().numpy()[0, :H]
        for name, tensor in variant_predictions.items()
    }

    for i, row_id in enumerate(sample.hidden_row_ids[:H]):
        row_idx = anchor_row + 1 + i
        record: dict[str, float | int | str] = {
            "fold": int(fold),
            "well_id": str(sample.well_id),
            "id": str(row_id),
            "row_idx": int(row_idx),
            "tvt_true": float(sample.tvt_rows[row_idx]),
        }
        for name in DEFAULT_VARIANTS:
            record[name] = float(variant_np[name][i])
        record.update(diagnostics)
        rows.append(record)

    return rows


def _well_summary_frame(df: pd.DataFrame, variants: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for well_id, g in df.groupby("well_id", sort=False):
        row: dict[str, Any] = {
            "well_id": well_id,
            "n_rows": int(len(g)),
        }
        for field in BUCKET_FIELDS:
            row[field] = float(g[field].iloc[0])
        for variant in variants:
            row[variant] = _rmse(g[variant].to_numpy(), g["tvt_true"].to_numpy())
        rows.append(row)
    return pd.DataFrame(rows)


def _bucket_labels(values: pd.Series, n_bins: int) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    finite = values[np.isfinite(values)]
    if len(finite) < 2 or finite.nunique(dropna=True) < 2:
        return pd.Series(["all"] * len(values), index=values.index)

    q = min(n_bins, int(finite.nunique(dropna=True)))
    try:
        labels = pd.qcut(values, q=q, duplicates="drop")
    except ValueError:
        return pd.Series(["all"] * len(values), index=values.index)
    return labels.astype(str)


def summarize_oof_dataframe(
    df: pd.DataFrame,
    variants: tuple[str, ...] = DEFAULT_VARIANTS,
    n_bins: int = 4,
) -> dict[str, Any]:
    """Summarize row-level OOF predictions into pooled, per-well, and bucket RMSE."""
    valid = df.dropna(subset=["tvt_true"]).copy()
    pooled = {
        variant: _rmse(valid[variant].to_numpy(), valid["tvt_true"].to_numpy())
        for variant in variants
    }

    well_df = _well_summary_frame(valid, variants)
    per_well_rmse = []
    for _, row in well_df.iterrows():
        item: dict[str, Any] = {
            "well_id": str(row["well_id"]),
            "n_rows": int(row["n_rows"]),
        }
        for field in BUCKET_FIELDS:
            item[field] = float(row[field])
        item["rmse"] = {variant: float(row[variant]) for variant in variants}
        per_well_rmse.append(item)

    bucket_rmse: dict[str, list[dict[str, Any]]] = {}
    for field in BUCKET_FIELDS:
        bucketed = well_df.copy()
        bucketed["_bucket"] = _bucket_labels(bucketed[field], n_bins)
        rows = []
        for bucket, g in bucketed.groupby("_bucket", sort=False):
            row_group = valid[valid["well_id"].isin(g["well_id"])]
            rows.append(
                {
                    "bucket": str(bucket),
                    "n_wells": int(len(g)),
                    "n_rows": int(len(row_group)),
                    "field_min": float(g[field].min()),
                    "field_max": float(g[field].max()),
                    "rmse": {
                        variant: _rmse(row_group[variant].to_numpy(), row_group["tvt_true"].to_numpy())
                        for variant in variants
                    },
                }
            )
        bucket_rmse[field] = rows

    return {
        "n_rows": int(len(valid)),
        "n_wells": int(valid["well_id"].nunique()),
        "pooled_rmse": pooled,
        "per_well_rmse": per_well_rmse,
        "bucket_rmse": bucket_rmse,
    }


def _to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
    return out


def _load_checkpoint_model(cfg: RACFormerConfig, ckpt_path: Path, device: torch.device) -> RACFormer:
    ckpt = torch.load(ckpt_path, map_location=device)
    model = RACFormer(cfg.model).to(device)
    use_ema = bool(ckpt.get("use_ema", False)) and bool(ckpt.get("ema_shadow"))
    if use_ema:
        load_state_dict_allowing_top_teacher_heads(model, ckpt["model_state"])
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in ckpt["ema_shadow"]:
                    param.data.copy_(ckpt["ema_shadow"][name].to(param.device))
    else:
        load_state_dict_allowing_top_teacher_heads(model, ckpt["model_state"])
    model.eval()
    return model


@torch.no_grad()
def _predict_fold_diagnostics(
    model: RACFormer,
    samples: list,
    cfg: RACFormerConfig,
    device: torch.device,
    fold: int,
) -> list[dict[str, float | int | str]]:
    ds = RACDataset(
        samples,
        max_seq_len=cfg.data.max_seq_len,
        augment=False,
        k_seg=cfg.model.k_seg,
        rows_per_step=cfg.data.rows_per_step,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    rows: list[dict[str, float | int | str]] = []
    for idx, raw_batch in enumerate(loader):
        batch = _to_device(raw_batch, device)
        out = model(batch)
        variants = compute_variant_predictions(batch, out, cfg.model.k_seg)

        sample = ds.samples[idx]
        H = int(sample.n_hidden_rows)
        true = sample.tvt_rows[sample.anchor_row + 1: sample.anchor_row + 1 + H]
        base_with_c0 = variants["base_with_c0"][0, :H].detach().cpu().numpy()
        diagnostics = sample_diagnostic_fields(sample, base_with_c0, true)
        rows.extend(build_row_records(fold, sample, variants, diagnostics))

    return rows


def run_oof_diagnostics(
    cfg: RACFormerConfig,
    output_dir: Path,
    row_path: Path,
    summary_path: Path,
    n_bins: int = 4,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate fold-safe OOF ablation diagnostics from existing checkpoints."""
    device_str = cfg.train.device
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    output_dir = Path(output_dir)
    all_samples = load_all_wells(cfg, split="train")
    from .train import _group_kfold_splits

    splits = _group_kfold_splits(all_samples, cfg.train.n_folds, cfg.train.seed)
    rows: list[dict[str, float | int | str]] = []

    for fold, (_, val_idx) in enumerate(splits):
        ckpt_path = output_dir / f"fold_{fold}" / "best.pt"
        if not ckpt_path.exists():
            print(f"  [warn] missing {ckpt_path}, skipping", file=sys.stderr)
            continue
        model = _load_checkpoint_model(cfg, ckpt_path, device)
        val_samples = [all_samples[i] for i in val_idx]
        rows.extend(_predict_fold_diagnostics(model, val_samples, cfg, device, fold))

    if not rows:
        raise RuntimeError("No OOF diagnostics generated — check fold checkpoints and data paths")

    row_df = pd.DataFrame(rows)
    summary = summarize_oof_dataframe(row_df, n_bins=n_bins)

    row_path = Path(row_path)
    summary_path = Path(summary_path)
    row_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    row_df.to_csv(row_path, index=False)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"OOF diagnostics pooled RMSE: {summary['pooled_rmse']}")
    print(f"Saved OOF diagnostics rows → {row_path}")
    print(f"Saved OOF diagnostics summary → {summary_path}")
    return row_df, summary
