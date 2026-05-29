from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from .config import GAConfig, load_config
from .dataset import AlignmentSample, collate_alignment_samples, load_alignment_samples, make_variant_samples
from .dp import viterbi_decode
from .model import GeoAligner


def _rmse(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(finite**2)))


def _known_tail_anchor(sample: AlignmentSample) -> np.ndarray:
    anchor = sample.comp_tvt_input.astype(np.float32).copy()
    known = np.flatnonzero(np.isfinite(anchor))
    if known.size == 0:
        return np.full(sample.seq_len, np.nan, dtype=np.float32)
    last = int(known[-1])
    if known.size >= 2:
        prev = int(known[-2])
        slope = float((anchor[last] - anchor[prev]) / max(last - prev, 1))
    else:
        slope = 0.0
    for step in range(last + 1, sample.seq_len):
        anchor[step] = anchor[last] + slope * (step - last)
    return anchor


def _apply_anchor_band(
    log_probs: np.ndarray,
    sample: AlignmentSample,
    *,
    radius_ft: float,
    penalty: float,
) -> np.ndarray:
    if radius_ft <= 0 or penalty <= 0:
        return log_probs
    anchor = _known_tail_anchor(sample)
    adjusted = np.asarray(log_probs, dtype=np.float32).copy()
    for step in range(min(len(anchor), adjusted.shape[0])):
        if not np.isfinite(anchor[step]):
            continue
        distance = np.abs(sample.typewell_tvt.astype(np.float32) - float(anchor[step]))
        outside = np.maximum(distance - float(radius_ft), 0.0)
        adjusted[step] -= float(penalty) * outside
    return adjusted


def _rank_of_target(log_probs: np.ndarray, target_bin: int) -> int:
    order = np.argsort(-log_probs)
    found = np.flatnonzero(order == target_bin)
    if found.size == 0:
        return int(len(log_probs) + 1)
    return int(found[0] + 1)


def _evaluate_variant(
    model: torch.nn.Module,
    samples: list[AlignmentSample],
    device: torch.device,
    *,
    max_jump_bins: int,
    jump_penalty: float,
    anchor_band_radius_ft: float = 0.0,
    anchor_band_penalty: float = 0.0,
    variant: str,
) -> tuple[dict[str, float | int], pd.DataFrame, pd.DataFrame]:
    row_frames: list[pd.DataFrame] = []
    step_rows: list[dict[str, float | int | str]] = []
    emission_top1_errors: list[float] = []
    oracle_errors: dict[int, list[float]] = {3: [], 10: []}
    target_ranks: list[int] = []
    dp_step_errors: list[float] = []
    anchor_errors: list[float] = []
    well_rmse: list[float] = []
    tail_errors: dict[str, list[float]] = {}

    model.eval()
    with torch.no_grad():
        for sample in samples:
            batch = collate_alignment_samples([sample])
            logits = model(
                batch["lateral_features"].to(device),
                batch["typewell_features"].to(device),
                batch["lateral_pad_mask"].to(device),
                batch["typewell_pad_mask"].to(device),
            )[0, : sample.seq_len, : sample.typewell_len]
            log_probs = torch.log_softmax(logits, dim=1).cpu().numpy()
            decode_log_probs = _apply_anchor_band(
                log_probs,
                sample,
                radius_ft=anchor_band_radius_ft,
                penalty=anchor_band_penalty,
            )
            dp_bins = viterbi_decode(
                decode_log_probs, max_jump_bins=max_jump_bins, jump_penalty=jump_penalty
            )
            dp_tvt = sample.typewell_tvt[np.clip(dp_bins, 0, sample.typewell_len - 1)]
            anchor_tvt = _known_tail_anchor(sample)

            hidden_steps = np.flatnonzero(sample.hidden_mask & np.isfinite(sample.target_tvt))
            for step in hidden_steps:
                target_tvt = float(sample.target_tvt[step])
                target_bin = int(np.rint(sample.target_bins[step]))
                top_order = np.argsort(-log_probs[step])
                top1_bin = int(top_order[0])
                top1_tvt = float(sample.typewell_tvt[top1_bin])
                emission_top1_errors.append(top1_tvt - target_tvt)
                target_ranks.append(_rank_of_target(log_probs[step], target_bin))
                for k in oracle_errors:
                    bins = top_order[: min(k, len(top_order))]
                    tvts = sample.typewell_tvt[bins]
                    oracle_errors[k].append(float(tvts[np.argmin(np.abs(tvts - target_tvt))] - target_tvt))
                dp_step_errors.append(float(dp_tvt[step] - target_tvt))
                anchor_errors.append(float(anchor_tvt[step] - target_tvt))
                tail_errors.setdefault(sample.tail_class, []).append(float(dp_tvt[step] - target_tvt))
                step_rows.append(
                    {
                        "well_id": sample.well_id,
                        "variant": variant,
                        "step": int(step),
                        "true_tvt": target_tvt,
                        "top1_tvt": top1_tvt,
                        "dp_tvt": float(dp_tvt[step]),
                        "anchor_tvt": float(anchor_tvt[step]),
                        "true_rank": int(target_ranks[-1]),
                    }
                )

            if sample.hidden_row_ids is not None and sample.hidden_row_steps is not None and sample.hidden_row_tvt is not None:
                valid = sample.hidden_row_steps < len(dp_tvt)
                pred_rows = dp_tvt[sample.hidden_row_steps[valid]]
                true_rows = sample.hidden_row_tvt[valid]
                if len(true_rows):
                    well_rmse.append(_rmse(pred_rows - true_rows))
                    row_frames.append(
                        pd.DataFrame(
                            {
                                "id": sample.hidden_row_ids[valid],
                                "well_id": sample.well_id,
                                "variant": variant,
                                "candidate": f"geoaligner_dp_{variant}",
                                "pred_tvt": pred_rows.astype(np.float32),
                                "TVT": true_rows.astype(np.float32),
                                "tail_class": sample.tail_class,
                            }
                        )
                    )

    rows = pd.concat(row_frames, ignore_index=True) if row_frames else pd.DataFrame()
    steps = pd.DataFrame(step_rows)
    row_errors = rows["pred_tvt"].to_numpy(np.float32) - rows["TVT"].to_numpy(np.float32) if not rows.empty else np.empty(0)
    metrics: dict[str, float | int] = {
        "hidden_steps": int(len(dp_step_errors)),
        "rows": int(len(rows)),
        "n_wells": int(len(samples)),
        "emission_top1_rmse_ft": _rmse(np.asarray(emission_top1_errors)),
        "emission_top3_oracle_rmse_ft": _rmse(np.asarray(oracle_errors[3])),
        "emission_top10_oracle_rmse_ft": _rmse(np.asarray(oracle_errors[10])),
        "target_top3_rate": float(np.mean(np.asarray(target_ranks) <= 3)) if target_ranks else float("nan"),
        "target_top10_rate": float(np.mean(np.asarray(target_ranks) <= 10)) if target_ranks else float("nan"),
        "dp_path_rmse_ft": _rmse(np.asarray(dp_step_errors)),
        "known_tail_anchor_rmse_ft": _rmse(np.asarray(anchor_errors)),
        "row_rmse_ft": _rmse(row_errors),
        "mean_well_rmse": float(np.nanmean(well_rmse)) if well_rmse else float("nan"),
        "p50_well_rmse": float(np.nanpercentile(well_rmse, 50)) if well_rmse else float("nan"),
        "p90_well_rmse": float(np.nanpercentile(well_rmse, 90)) if well_rmse else float("nan"),
        "p95_well_rmse": float(np.nanpercentile(well_rmse, 95)) if well_rmse else float("nan"),
        "worst_well_rmse": float(np.nanmax(well_rmse)) if well_rmse else float("nan"),
    }
    for tail_class, errors in tail_errors.items():
        metrics[f"tail_{tail_class}_dp_rmse_ft"] = _rmse(np.asarray(errors))
    return metrics, rows, steps


def evaluate_model(
    model: torch.nn.Module,
    samples: list[AlignmentSample],
    device: torch.device,
    *,
    variants: Iterable[str] = ("normal",),
    max_jump_bins: int = 6,
    jump_penalty: float = 0.03,
    anchor_band_radius_ft: float = 0.0,
    anchor_band_penalty: float = 0.0,
) -> tuple[dict[str, dict[str, float | int]], pd.DataFrame, pd.DataFrame]:
    all_metrics: dict[str, dict[str, float | int]] = {}
    row_parts: list[pd.DataFrame] = []
    step_parts: list[pd.DataFrame] = []
    for variant in variants:
        variant_samples = samples if variant == "normal" else make_variant_samples(samples, variant)
        metrics, rows, steps = _evaluate_variant(
            model,
            variant_samples,
            device,
            max_jump_bins=max_jump_bins,
            jump_penalty=jump_penalty,
            anchor_band_radius_ft=anchor_band_radius_ft,
            anchor_band_penalty=anchor_band_penalty,
            variant=variant,
        )
        all_metrics[variant] = metrics
        row_parts.append(rows)
        step_parts.append(steps)
    return (
        all_metrics,
        pd.concat(row_parts, ignore_index=True) if row_parts else pd.DataFrame(),
        pd.concat(step_parts, ignore_index=True) if step_parts else pd.DataFrame(),
    )


def _split_samples(samples: list[AlignmentSample], valid_fraction: float, seed: int) -> tuple[list[AlignmentSample], list[AlignmentSample]]:
    rng = np.random.default_rng(seed)
    order = np.arange(len(samples))
    rng.shuffle(order)
    valid_n = max(1, int(round(len(samples) * valid_fraction)))
    valid_idx = set(order[:valid_n].tolist())
    train = [s for i, s in enumerate(samples) if i not in valid_idx]
    valid = [s for i, s in enumerate(samples) if i in valid_idx]
    if not train:
        train, valid = samples, samples
    return train, valid


def evaluate_checkpoint(config_path: Path, checkpoint_path: Path | None = None) -> dict:
    cfg = load_config(config_path)
    device = torch.device("cpu")
    if cfg.train.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
    else:
        device = torch.device(cfg.train.device)
    samples = load_alignment_samples(cfg.data_dir, cfg.data, k_wells=cfg.k_wells)
    _, valid = _split_samples(samples, cfg.train.valid_fraction, cfg.train.seed)
    model = GeoAligner(cfg.model).to(device)
    ckpt = checkpoint_path or cfg.run.output_dir / "checkpoints" / "best.pt"
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"])
    metrics, rows, steps = evaluate_model(
        model,
        valid,
        device,
        variants=("normal", "shuffled_gr", "zero_gr", "typewell_shuffled"),
        max_jump_bins=cfg.decode.max_jump_bins,
        jump_penalty=cfg.decode.jump_penalty,
        anchor_band_radius_ft=cfg.decode.anchor_band_radius_ft,
        anchor_band_penalty=cfg.decode.anchor_band_penalty,
    )
    cfg.run.output_dir.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(cfg.run.output_dir / "geoaligner_row_predictions.parquet", index=False)
    steps.to_parquet(cfg.run.output_dir / "geoaligner_alignment_steps.parquet", index=False)
    (cfg.run.output_dir / "geoaligner_metrics.json").write_text(json.dumps({"valid": metrics}, indent=2), encoding="utf-8")
    return {"valid": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GeoAligner checkpoint")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    print(json.dumps(evaluate_checkpoint(args.config, args.checkpoint), indent=2))


if __name__ == "__main__":
    main()
