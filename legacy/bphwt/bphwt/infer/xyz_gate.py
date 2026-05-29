from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class XYZGateConfig:
    high_alpha: float = 1.0
    fallback_alpha: float = 0.5
    max_abs_p95: float = 40.0
    max_disagreement_rmse: float = 20.0


def choose_gate_alpha(
    full_residual: np.ndarray,
    past_residual: np.ndarray,
    hidden_mask: np.ndarray,
    cfg: XYZGateConfig,
) -> float:
    stats = residual_gate_stats(full_residual, past_residual, hidden_mask)
    if stats["full_abs_p95"] > float(cfg.max_abs_p95):
        return float(cfg.fallback_alpha)
    if stats["full_vs_past_rmse"] > float(cfg.max_disagreement_rmse):
        return float(cfg.fallback_alpha)
    return float(cfg.high_alpha)


def residual_gate_stats(
    full_residual: np.ndarray,
    past_residual: np.ndarray,
    hidden_mask: np.ndarray,
) -> dict[str, float]:
    full = np.asarray(full_residual, dtype=np.float64)
    past = np.asarray(past_residual, dtype=np.float64)
    hidden = np.asarray(hidden_mask, dtype=bool)
    valid = hidden & np.isfinite(full) & np.isfinite(past)
    if not valid.any():
        return {
            "full_abs_mean": 0.0,
            "full_abs_p95": 0.0,
            "full_vs_past_rmse": 0.0,
        }
    full_h = full[valid]
    diff = full_h - past[valid]
    return {
        "full_abs_mean": float(np.mean(np.abs(full_h))),
        "full_abs_p95": float(np.percentile(np.abs(full_h), 95.0)),
        "full_vs_past_rmse": float(np.sqrt(np.mean(diff * diff))),
    }


__all__ = ["XYZGateConfig", "choose_gate_alpha", "residual_gate_stats"]
