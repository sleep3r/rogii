from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, CatBoostRegressor, Pool

from .candidate_bank import build_candidate_bank_from_frames
from .candidate_selector import (
    _boundary_features,
    _candidate_action_features,
    _hidden_only,
    _json_safe,
    _known_boundary_context,
    _mean_abs,
    _mse_pair,
    _p95_abs,
    _rmse_pair,
    _slope,
    _std,
)
from .residual_stack import (
    ResidualStackConfig,
    _add_missing_prior_columns,
    _ensure_ids,
    _rmse,
    load_training_frame,
    make_group_folds,
)
from .schema_safe import assert_schema_safe_columns


@dataclass(frozen=True)
class ChunkPolicyConfig:
    data_dir: Path = Path("data/train")
    output_dir: Path = Path("artifacts/chunk_ranker_dp_v1")
    n_folds: int = 5
    seed: int = 42
    k_wells: int = -1
    chunk_size: int = 128
    iterations: int = 700
    learning_rate: float = 0.04
    depth: int = 5
    l2_leaf_reg: float = 10.0
    ranker_loss: str = "YetiRankPairwise"
    ranker_alpha: float = 5.0
    switch_penalty: float = 500.0
    jump_penalty: float = 1.0
    cost_target: str = "log_mse"  # {"log_mse", "sqrt_mse_weighted"}
    ranker_target: str = "neg_log1p_regret"  # {"neg_log1p_regret", "neg_sqrt_regret"}
    use_selfcal_probes: bool = True
    selfcal_max_probes: int = 4
    use_spatial_priors: bool = True
    use_shared_typewell_priors: bool = False
    shared_typewell_neighbours_path: Path | None = Path(
        "artifacts/shared_typewell_neighbour_v0/typewell_neighbours.parquet"
    )
    progress_every: int = 50
    residual_predictions_path: Path | None = Path(
        "artifacts/residual_stack_v0/oof_predictions.parquet"
    )
    traceback_candidates_path: Path | None = None
    step_predictions_path: Path | None = None
    location_aware_steps_path: Path | None = None
    top_state_predictions_path: Path | None = Path(
        "artifacts/top_state_teacher_v0/top_state_oof_predictions.parquet"
    )
    k_offset_predictions_path: Path | None = Path(
        "artifacts/k_segment_offset_v0/k_offset_oof_predictions.parquet"
    )
    dtvt_state_predictions_path: Path | None = Path(
        "artifacts/dtvt_state_model_v0/dtvt_state_oof_predictions.parquet"
    )


@dataclass
class ChunkPolicyDataset:
    rows: pd.DataFrame
    features: pd.DataFrame
    feature_columns: list[str]


def _prepare_step_predictions(step_predictions: pd.DataFrame | None) -> pd.DataFrame:
    if step_predictions is None or step_predictions.empty:
        return pd.DataFrame()
    frame = step_predictions.copy()
    if "step" not in frame.columns and "compressed_step" in frame.columns:
        frame = frame.rename(columns={"compressed_step": "step"})
    required = {"well_id", "step"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    keep = ["well_id", "step"]
    for column in ("top1_tvt", "dp_tvt"):
        if column in frame.columns:
            keep.append(column)
    out = frame[keep].copy()
    out["well_id"] = out["well_id"].astype(str)
    out["step"] = pd.to_numeric(out["step"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["step"])
    out["step"] = out["step"].astype(int)
    for column in set(keep).difference({"well_id", "step"}):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out.drop_duplicates(["well_id", "step"]).reset_index(drop=True)


def _prepare_location_aware_steps(location_aware_steps: pd.DataFrame | None) -> pd.DataFrame:
    if location_aware_steps is None or location_aware_steps.empty:
        return pd.DataFrame()
    frame = location_aware_steps.copy()
    required = {"well_id", "step", "variant", "top1_tvt"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    out = frame[list(required)].copy()
    out["well_id"] = out["well_id"].astype(str)
    out["variant"] = out["variant"].astype(str)
    out = out[
        ~out["variant"].str.startswith("shuffled")
        & ~out["variant"].str.startswith("zero")
    ].copy()
    out["step"] = pd.to_numeric(out["step"], errors="coerce").astype("Int64")
    out["top1_tvt"] = pd.to_numeric(out["top1_tvt"], errors="coerce")
    out = out.dropna(subset=["step"])
    out["step"] = out["step"].astype(int)
    return out.drop_duplicates(["well_id", "step", "variant"]).reset_index(drop=True)


def _prepare_top_state_predictions(top_state_predictions: pd.DataFrame | None) -> pd.DataFrame:
    if top_state_predictions is None or top_state_predictions.empty:
        return pd.DataFrame()
    frame = top_state_predictions.copy()
    if "step" not in frame.columns and "compressed_step" in frame.columns:
        frame = frame.rename(columns={"compressed_step": "step"})
    required = {"well_id", "step", "prob_down", "prob_flat", "prob_up", "pred_expected_sign"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    out = frame[list(required)].copy()
    out["well_id"] = out["well_id"].astype(str)
    out["step"] = pd.to_numeric(out["step"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["step"])
    out["step"] = out["step"].astype(int)
    for column in required.difference({"well_id", "step"}):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out.drop_duplicates(["well_id", "step"]).reset_index(drop=True)


def _series_for_steps(
    table: pd.DataFrame,
    *,
    well_id: str,
    steps: pd.Series,
    value_column: str,
) -> np.ndarray:
    if table.empty or value_column not in table.columns:
        return np.full(len(steps), np.nan, dtype=np.float64)
    well = table[table["well_id"].astype(str) == str(well_id)]
    if well.empty:
        return np.full(len(steps), np.nan, dtype=np.float64)
    mapping = well.set_index("step")[value_column]
    values = steps.astype("float").map(mapping).to_numpy(dtype=np.float64)
    return values


def _distance_features(prefix: str, pred: np.ndarray, values: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(pred) & np.isfinite(values)
    if not finite.any():
        return {
            f"{prefix}_mean_abs": 0.0,
            f"{prefix}_p95_abs": 0.0,
            f"{prefix}_available_frac": 0.0,
        }
    diff = pred[finite] - values[finite]
    return {
        f"{prefix}_mean_abs": float(np.mean(np.abs(diff))) / 50.0,
        f"{prefix}_p95_abs": float(np.percentile(np.abs(diff), 95.0)) / 50.0,
        f"{prefix}_available_frac": float(finite.mean()),
    }


def _step_prediction_features(
    pred: np.ndarray,
    well_id: str,
    steps: pd.Series,
    step_predictions: pd.DataFrame,
) -> dict[str, float]:
    features: dict[str, float] = {}
    for column, name in (("top1_tvt", "top1"), ("dp_tvt", "dp")):
        values = _series_for_steps(
            step_predictions,
            well_id=well_id,
            steps=steps,
            value_column=column,
        )
        features.update(_distance_features(f"feat_softseg_{name}", pred, values))
    return features


def _location_aware_features(
    pred: np.ndarray,
    well_id: str,
    steps: pd.Series,
    location_aware_steps: pd.DataFrame,
) -> dict[str, float]:
    features: dict[str, float] = {}
    if location_aware_steps.empty:
        return features
    well = location_aware_steps[location_aware_steps["well_id"].astype(str) == str(well_id)]
    for variant in sorted(well["variant"].unique()):
        safe_variant = "".join(ch if ch.isalnum() else "_" for ch in str(variant))
        variant_frame = well[well["variant"].astype(str) == str(variant)]
        values = _series_for_steps(
            variant_frame,
            well_id=well_id,
            steps=steps,
            value_column="top1_tvt",
        )
        features.update(
            _distance_features(f"feat_loccorr_{safe_variant}", pred, values)
        )
    return features


def _top_state_features(
    pred_tvt: np.ndarray,
    well_id: str,
    steps: pd.Series,
    top_state_predictions: pd.DataFrame,
) -> dict[str, float]:
    defaults = {
        "feat_top_state_available_frac": 0.0,
        "feat_top_state_sign_agree": 0.0,
        "feat_top_state_corr": 0.0,
        "feat_top_state_expected_margin_mean": 0.0,
        "feat_top_state_prob_up_mean": 0.0,
        "feat_top_state_prob_down_mean": 0.0,
        "feat_top_state_prob_flat_mean": 0.0,
    }
    if top_state_predictions.empty or pred_tvt.size < 2:
        return defaults
    prob_up = _series_for_steps(
        top_state_predictions,
        well_id=well_id,
        steps=steps,
        value_column="prob_up",
    )
    prob_down = _series_for_steps(
        top_state_predictions,
        well_id=well_id,
        steps=steps,
        value_column="prob_down",
    )
    prob_flat = _series_for_steps(
        top_state_predictions,
        well_id=well_id,
        steps=steps,
        value_column="prob_flat",
    )
    expected = _series_for_steps(
        top_state_predictions,
        well_id=well_id,
        steps=steps,
        value_column="pred_expected_sign",
    )
    finite_rows = np.isfinite(expected)
    row_features = {
        "feat_top_state_available_frac": float(finite_rows.mean()) if finite_rows.size else 0.0,
        "feat_top_state_expected_margin_mean": float(np.nanmean(np.abs(expected)))
        if finite_rows.any()
        else 0.0,
        "feat_top_state_prob_up_mean": float(np.nanmean(prob_up)) if np.isfinite(prob_up).any() else 0.0,
        "feat_top_state_prob_down_mean": float(np.nanmean(prob_down))
        if np.isfinite(prob_down).any()
        else 0.0,
        "feat_top_state_prob_flat_mean": float(np.nanmean(prob_flat))
        if np.isfinite(prob_flat).any()
        else 0.0,
    }
    pred_delta = np.diff(np.asarray(pred_tvt, dtype=np.float64))
    expected_delta = expected[1:]
    finite = np.isfinite(pred_delta) & np.isfinite(expected_delta)
    if not finite.any():
        return {**defaults, **row_features}
    pred_state = np.sign(pred_delta[finite])
    teacher_state = np.sign(expected_delta[finite])
    decisive = (np.abs(pred_delta[finite]) > 1e-9) | (np.abs(expected_delta[finite]) > 1e-9)
    sign_agree = float((pred_state[decisive] == teacher_state[decisive]).mean()) if decisive.any() else 0.0
    if finite.sum() >= 2 and np.std(pred_delta[finite]) > 1e-9 and np.std(expected_delta[finite]) > 1e-9:
        corr = float(np.corrcoef(pred_delta[finite], expected_delta[finite])[0, 1])
    else:
        corr = 0.0
    return {
        **row_features,
        "feat_top_state_sign_agree": sign_agree,
        "feat_top_state_corr": corr if np.isfinite(corr) else 0.0,
    }


def _dz_state_features(pred_tvt: np.ndarray, z_values: np.ndarray) -> dict[str, float]:
    """Describe how candidate local TVT motion agrees with observed trajectory Z motion.

    The annotation-leak audit showed `dTVT` is often state-aligned with `-dZ`,
    but not strongly enough to use as a standalone path formula. These features
    expose that agreement to the chunk selector without using hidden TVT.
    """
    defaults = {
        "feat_dz_state_available_frac": 0.0,
        "feat_dz_state_corr": 0.0,
        "feat_dz_state_sign_agree": 0.0,
        "feat_dz_state_scaled_rmse": 1.0,
        "feat_dz_state_mean_abs_delta_diff": 1.0,
        "feat_dz_state_turn_sign_agree": 0.0,
        "feat_dz_state_mean_delta_ratio": 0.0,
    }
    pred = np.asarray(pred_tvt, dtype=np.float64)
    z = np.asarray(z_values, dtype=np.float64)
    if pred.size < 2 or z.size < 2 or pred.size != z.size:
        return defaults
    valid = np.isfinite(pred[:-1]) & np.isfinite(pred[1:]) & np.isfinite(z[:-1]) & np.isfinite(z[1:])
    if not valid.any():
        return defaults
    pred_delta = np.diff(pred)[valid]
    dz_target = -np.diff(z)[valid]
    available_frac = float(valid.mean())
    pred_std = float(np.std(pred_delta))
    dz_std = float(np.std(dz_target))
    if pred_delta.size >= 2 and pred_std > 1e-9 and dz_std > 1e-9:
        corr = float(np.corrcoef(pred_delta, dz_target)[0, 1])
        pred_scaled = (pred_delta - float(np.mean(pred_delta))) / pred_std
        dz_scaled = (dz_target - float(np.mean(dz_target))) / dz_std
        scaled_rmse = float(np.sqrt(np.mean(np.square(pred_scaled - dz_scaled))))
    else:
        corr = 0.0
        scaled_rmse = 1.0
    pred_sign = np.sign(pred_delta)
    dz_sign = np.sign(dz_target)
    decisive = (np.abs(pred_delta) > 1e-9) | (np.abs(dz_target) > 1e-9)
    sign_agree = float((pred_sign[decisive] == dz_sign[decisive]).mean()) if decisive.any() else 0.0
    if pred_delta.size >= 3:
        pred_turn = np.diff(pred_delta)
        dz_turn = np.diff(dz_target)
        turn_decisive = (np.abs(pred_turn) > 1e-9) | (np.abs(dz_turn) > 1e-9)
        turn_sign_agree = (
            float((np.sign(pred_turn[turn_decisive]) == np.sign(dz_turn[turn_decisive])).mean())
            if turn_decisive.any()
            else 0.0
        )
    else:
        turn_sign_agree = 0.0
    mean_abs_delta_diff = float(np.mean(np.abs(pred_delta - dz_target))) / 10.0
    denom = float(np.mean(np.abs(dz_target)))
    mean_delta_ratio = float(np.mean(pred_delta) / denom) if denom > 1e-9 else 0.0
    return {
        "feat_dz_state_available_frac": available_frac,
        "feat_dz_state_corr": corr if np.isfinite(corr) else 0.0,
        "feat_dz_state_sign_agree": sign_agree,
        "feat_dz_state_scaled_rmse": scaled_rmse if np.isfinite(scaled_rmse) else 1.0,
        "feat_dz_state_mean_abs_delta_diff": mean_abs_delta_diff
        if np.isfinite(mean_abs_delta_diff)
        else 1.0,
        "feat_dz_state_turn_sign_agree": turn_sign_agree,
        "feat_dz_state_mean_delta_ratio": mean_delta_ratio if np.isfinite(mean_delta_ratio) else 0.0,
    }


def _chunk_frame(hidden: pd.DataFrame, chunk_size: int) -> pd.DataFrame:
    chunk = hidden[["id", "row_idx"]].copy()
    chunk = chunk.sort_values("row_idx")
    local_index = np.arange(len(chunk), dtype=np.int64)
    chunk["chunk_id"] = (local_index // max(int(chunk_size), 1)).astype(np.int32)
    denom = max(len(chunk) - 1, 1)
    chunk["hidden_progress"] = local_index / denom
    return chunk


def _bank_stats(bank: pd.DataFrame) -> pd.DataFrame:
    wide = bank.pivot_table(index="id", columns="candidate", values="pred_tvt", aggfunc="first")
    return pd.DataFrame(
        {
            "bank_median": wide.median(axis=1),
            "bank_p25": wide.quantile(0.25, axis=1),
            "bank_p75": wide.quantile(0.75, axis=1),
            "bank_std": wide.std(axis=1, ddof=0),
        },
        index=wide.index,
    )


def build_selfcal_probe_features(
    well: pd.DataFrame,
    *,
    chunk_size: int,
    max_probes: int = 4,
) -> pd.DataFrame:
    """Score candidate families on pseudo-hidden windows inside known TVT_input.

    This is deployable: labels come only from shipped non-null TVT_input values.
    Hidden train TVT is not used.
    """
    raw = _add_missing_prior_columns(_ensure_ids(well)).sort_values("row_idx").reset_index(drop=True)
    if "TVT_input" not in raw.columns:
        return pd.DataFrame()
    tvt_input = pd.to_numeric(raw["TVT_input"], errors="coerce")
    known_idx = np.flatnonzero(tvt_input.notna().to_numpy())
    if known_idx.size < max(4, min(chunk_size, 4)):
        return pd.DataFrame()
    probe_len = max(2, min(int(chunk_size), max(2, known_idx.size // max(max_probes, 1))))
    starts = np.linspace(0, max(known_idx.size - probe_len, 0), num=min(max_probes, known_idx.size), dtype=int)
    records: list[dict[str, Any]] = []
    for probe_id, start in enumerate(sorted(set(starts.tolist()))):
        probe_positions = known_idx[start : start + probe_len]
        if probe_positions.size < 2:
            continue
        probe_ids = set(raw.iloc[probe_positions]["id"].astype(str))
        masked = raw.copy()
        masked.loc[masked["id"].astype(str).isin(probe_ids), "TVT_input"] = np.nan
        try:
            bank = build_candidate_bank_from_frames(masked)
        except Exception:
            continue
        bank = bank[bank["id"].astype(str).isin(probe_ids)].copy()
        if bank.empty:
            continue
        known_truth = raw.set_index("id")["TVT_input"]
        b2_group = bank[bank["candidate"].astype(str) == "b2"]
        b2_mse = (
            _mse_pair(b2_group["pred_tvt"], known_truth.reindex(b2_group["id"].astype(str)))
            if not b2_group.empty
            else np.nan
        )
        probe_scores: list[dict[str, Any]] = []
        for candidate, group in bank.groupby("candidate", sort=True):
            truth = known_truth.reindex(group["id"].astype(str))
            mse = _mse_pair(group["pred_tvt"], truth)
            diff = (
                pd.to_numeric(group["pred_tvt"], errors="coerce").to_numpy(dtype=np.float64)
                - pd.to_numeric(truth, errors="coerce").to_numpy(dtype=np.float64)
            )
            finite_diff = diff[np.isfinite(diff)]
            bias = float(finite_diff.mean()) if finite_diff.size else 0.0
            probe_scores.append(
                {
                    "probe_id": probe_id,
                    "candidate": str(candidate),
                    "probe_mse": mse,
                    "probe_rmse": float(np.sqrt(mse)) if np.isfinite(mse) else np.nan,
                    "probe_bias": bias if np.isfinite(bias) else 0.0,
                    "probe_gain_vs_b2_mse": b2_mse - mse
                    if np.isfinite(b2_mse) and np.isfinite(mse)
                    else np.nan,
                }
            )
        if not probe_scores:
            continue
        score_frame = pd.DataFrame(probe_scores)
        score_frame["probe_rank"] = score_frame.groupby("probe_id")["probe_mse"].rank(
            method="average", ascending=True
        )
        score_frame["probe_win"] = score_frame["probe_rank"] == 1.0
        records.extend(score_frame.to_dict("records"))
    if not records:
        return pd.DataFrame()
    probes = pd.DataFrame(records)
    out = probes.groupby("candidate", sort=True).agg(
        feat_probe_count=("probe_id", "nunique"),
        feat_probe_rmse_mean=("probe_rmse", "mean"),
        feat_probe_rmse_median=("probe_rmse", "median"),
        feat_probe_rank_mean=("probe_rank", "mean"),
        feat_probe_win_frac=("probe_win", "mean"),
        feat_probe_bias_mean=("probe_bias", "mean"),
        feat_probe_gain_vs_b2_mse_mean=("probe_gain_vs_b2_mse", "mean"),
    )
    out = out.reset_index()
    for column in out.columns:
        if column.startswith("feat_probe_"):
            out[column] = pd.to_numeric(out[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    out = out.fillna(0.0)
    out["feat_probe_rmse_mean"] = out["feat_probe_rmse_mean"] / 50.0
    out["feat_probe_rmse_median"] = out["feat_probe_rmse_median"] / 50.0
    out["feat_probe_bias_mean"] = out["feat_probe_bias_mean"] / 50.0
    out["feat_probe_gain_vs_b2_mse_mean"] = out["feat_probe_gain_vs_b2_mse_mean"] / 2500.0
    return out


def _selfcal_defaults() -> dict[str, float]:
    return {
        "feat_probe_count": 0.0,
        "feat_probe_rmse_mean": 1.0,
        "feat_probe_rmse_median": 1.0,
        "feat_probe_rank_mean": 30.0,
        "feat_probe_win_frac": 0.0,
        "feat_probe_bias_mean": 0.0,
        "feat_probe_gain_vs_b2_mse_mean": 0.0,
    }


def build_chunk_policy_dataset(
    hidden_rows: pd.DataFrame,
    *,
    residual_predictions: pd.DataFrame | None = None,
    traceback_candidates: pd.DataFrame | None = None,
    step_predictions: pd.DataFrame | None = None,
    location_aware_steps: pd.DataFrame | None = None,
    top_state_predictions: pd.DataFrame | None = None,
    k_offset_predictions: pd.DataFrame | None = None,
    dtvt_state_predictions: pd.DataFrame | None = None,
    chunk_size: int = 128,
    use_selfcal_probes: bool = True,
    selfcal_max_probes: int = 4,
    progress_every: int = 0,
) -> ChunkPolicyDataset:
    raw = _add_missing_prior_columns(_ensure_ids(hidden_rows))
    residual = residual_predictions.copy() if residual_predictions is not None else None
    if residual is not None and not residual.empty:
        residual["well_id"] = residual["well_id"].astype(str)
    traceback = traceback_candidates.copy() if traceback_candidates is not None else None
    if traceback is not None and not traceback.empty and "well_id" in traceback.columns:
        traceback["well_id"] = traceback["well_id"].astype(str)
    step_preds = _prepare_step_predictions(step_predictions)
    loccorr = _prepare_location_aware_steps(location_aware_steps)
    top_state = _prepare_top_state_predictions(top_state_predictions)
    k_offset = k_offset_predictions.copy() if k_offset_predictions is not None else None
    if k_offset is not None and not k_offset.empty:
        k_offset["well_id"] = k_offset["well_id"].astype(str)
    dtvt_state = dtvt_state_predictions.copy() if dtvt_state_predictions is not None else None
    if dtvt_state is not None and not dtvt_state.empty:
        dtvt_state["well_id"] = dtvt_state["well_id"].astype(str)
    residual_by_well = (
        {str(key): group.copy() for key, group in residual.groupby("well_id", sort=False)}
        if residual is not None and not residual.empty
        else {}
    )
    traceback_by_well = (
        {str(key): group.copy() for key, group in traceback.groupby("well_id", sort=False)}
        if traceback is not None and not traceback.empty and "well_id" in traceback.columns
        else {}
    )
    step_by_well = (
        {str(key): group.copy() for key, group in step_preds.groupby("well_id", sort=False)}
        if not step_preds.empty
        else {}
    )
    loccorr_by_well = (
        {str(key): group.copy() for key, group in loccorr.groupby("well_id", sort=False)}
        if not loccorr.empty
        else {}
    )
    top_state_by_well = (
        {str(key): group.copy() for key, group in top_state.groupby("well_id", sort=False)}
        if not top_state.empty
        else {}
    )
    k_offset_by_well = (
        {str(key): group.copy() for key, group in k_offset.groupby("well_id", sort=False)}
        if k_offset is not None and not k_offset.empty
        else {}
    )
    dtvt_state_by_well = (
        {str(key): group.copy() for key, group in dtvt_state.groupby("well_id", sort=False)}
        if dtvt_state is not None and not dtvt_state.empty
        else {}
    )
    rows: list[dict[str, Any]] = []
    grouped_wells = list(raw.groupby("well_id", sort=True))
    total_wells = len(grouped_wells)
    for index, (well_id, well) in enumerate(grouped_wells, start=1):
        if progress_every > 0 and (index == 1 or index % progress_every == 0 or index == total_wells):
            print(
                f"[chunk-policy] built chunks for {index}/{total_wells} wells",
                file=sys.stderr,
                flush=True,
            )
        hidden = _hidden_only(well).sort_values("row_idx")
        if hidden.empty:
            continue
        well_residual = residual_by_well.get(str(well_id))
        well_traceback = traceback_by_well.get(str(well_id))
        well_step_preds = step_by_well.get(str(well_id), pd.DataFrame())
        well_loccorr = loccorr_by_well.get(str(well_id), pd.DataFrame())
        well_top_state = top_state_by_well.get(str(well_id), pd.DataFrame())
        well_k_offset = k_offset_by_well.get(str(well_id))
        well_dtvt_state = dtvt_state_by_well.get(str(well_id))
        bank = build_candidate_bank_from_frames(
            well,
            residual_predictions=well_residual,
            traceback_candidates=well_traceback,
            step_predictions=well_step_preds,
            k_offset_predictions=well_k_offset,
            dtvt_state_predictions=well_dtvt_state,
        )
        selfcal_defaults = _selfcal_defaults()
        selfcal_map: dict[str, dict[str, float]] = {}
        if use_selfcal_probes:
            selfcal = build_selfcal_probe_features(
                well,
                chunk_size=chunk_size,
                max_probes=selfcal_max_probes,
            )
            if not selfcal.empty:
                selfcal_map = {
                    str(row["candidate"]): {
                        key: float(row[key])
                        for key in selfcal_defaults
                        if key in selfcal.columns
                    }
                    for row in selfcal.to_dict("records")
                }
        chunk_index = _chunk_frame(hidden, chunk_size)
        hidden_chunked = hidden.merge(
            chunk_index[["id", "chunk_id", "hidden_progress"]], on="id", how="inner"
        )
        b2_mse_by_chunk = {
            int(chunk_id): _mse_pair(
                group.get("b2_tvt", pd.Series(np.nan, index=group.index)),
                group["TVT"],
            )
            for chunk_id, group in hidden_chunked.groupby("chunk_id", sort=True)
        }
        bank = bank.merge(chunk_index, on=["id", "row_idx"], how="inner")
        if bank.empty:
            continue
        stats = _bank_stats(bank)
        true_by_id = hidden.set_index("id")["TVT"] if "TVT" in hidden.columns else None
        anchors = hidden.set_index("id")[
            [col for col in ("anchor_tvt", "b2_tvt", "base_tvt", "a_p50_tvt") if col in hidden.columns]
        ]
        boundaries = _known_boundary_context(well, hidden)
        chunk_counts = chunk_index.groupby("chunk_id").size()
        for (chunk_id, candidate), group in bank.groupby(["chunk_id", "candidate"], sort=True):
            ordered = group.sort_values("row_idx")
            pred = pd.to_numeric(ordered["pred_tvt"], errors="coerce").to_numpy(dtype=np.float64)
            by_id = ordered.set_index("id")["pred_tvt"]
            aligned_true = true_by_id.reindex(ordered["id"].astype(str)) if true_by_id is not None else None
            target_mse = _mse_pair(ordered["pred_tvt"], aligned_true) if aligned_true is not None else np.nan
            target_rmse = float(np.sqrt(target_mse)) if np.isfinite(target_mse) else np.nan
            aligned_anchors = anchors.reindex(ordered["id"].astype(str))
            aligned_stats = stats.reindex(ordered["id"].astype(str))
            aligned_steps = pd.to_numeric(ordered["step"], errors="coerce").astype("Int64")
            soft_features = _step_prediction_features(
                pred,
                str(well_id),
                aligned_steps,
                well_step_preds,
            )
            z_values = (
                pd.to_numeric(ordered["Z"], errors="coerce").to_numpy(dtype=np.float64)
                if "Z" in ordered.columns
                else np.full(pred.shape, np.nan, dtype=np.float64)
            )
            loccorr_features = _location_aware_features(
                pred,
                str(well_id),
                aligned_steps,
                well_loccorr,
            )
            top_state_features = _top_state_features(
                pred,
                str(well_id),
                aligned_steps,
                well_top_state,
            )
            median_diff = by_id - aligned_stats["bank_median"]
            p25 = aligned_stats["bank_p25"].to_numpy(dtype=np.float64)
            p75 = aligned_stats["bank_p75"].to_numpy(dtype=np.float64)
            inside_iqr = (
                np.isfinite(pred)
                & np.isfinite(p25)
                & np.isfinite(p75)
                & (pred >= p25)
                & (pred <= p75)
            )
            b2_mse = b2_mse_by_chunk.get(int(chunk_id), float("nan"))
            curvature = np.diff(np.diff(pred[np.isfinite(pred)])) if np.isfinite(pred).sum() >= 3 else np.asarray([])
            rec: dict[str, Any] = {
                "well_id": str(well_id),
                "run_id": 0,
                "chunk_id": int(chunk_id),
                "group_id": f"{well_id}:0:{int(chunk_id)}",
                "candidate": str(candidate),
                "row_count": int(chunk_counts.loc[chunk_id]),
                "target_mse": target_mse,
                "target_rmse": target_rmse,
                "target_b2_mse": b2_mse,
                "target_gain_vs_b2_mse": b2_mse - target_mse
                if np.isfinite(b2_mse) and np.isfinite(target_mse)
                else np.nan,
                "first_pred_tvt": float(pred[0]) if pred.size else np.nan,
                "last_pred_tvt": float(pred[-1]) if pred.size else np.nan,
                **_candidate_action_features(str(candidate)),
                **_boundary_features(ordered, boundaries),
                **{**selfcal_defaults, **selfcal_map.get(str(candidate), {})},
                **soft_features,
                **loccorr_features,
                **top_state_features,
                **_dz_state_features(pred, z_values),
                "feat_chunk_rows_log": float(np.log1p(len(ordered))),
                "feat_chunk_progress_mean": float(
                    pd.to_numeric(ordered["hidden_progress"], errors="coerce").mean()
                ),
                "feat_chunk_progress_start": float(
                    pd.to_numeric(ordered["hidden_progress"], errors="coerce").min()
                ),
                "feat_chunk_progress_end": float(
                    pd.to_numeric(ordered["hidden_progress"], errors="coerce").max()
                ),
                "feat_chunk_bank_std_mean": float(aligned_stats["bank_std"].mean()) / 50.0,
                "feat_pred_mean_delta_anchor": _mean_abs(
                    by_id - aligned_anchors.get("anchor_tvt", pd.Series(np.nan, index=by_id.index))
                )
                / 50.0,
                "feat_pred_std": _std(pred) / 50.0,
                "feat_pred_slope": _slope(pred),
                "feat_pred_curvature_std": _std(curvature) / 10.0,
                "feat_pred_endpoint_span": float(pred[-1] - pred[0]) / 100.0
                if pred.size >= 2 and np.isfinite(pred[[0, -1]]).all()
                else 0.0,
                "feat_candidate_bank_median_mean_abs": _mean_abs(median_diff) / 50.0,
                "feat_candidate_bank_median_p95_abs": _p95_abs(median_diff) / 50.0,
                "feat_candidate_bank_inside_iqr_frac": float(inside_iqr.mean())
                if inside_iqr.size
                else 0.0,
            }
            for name, column in (("b2", "b2_tvt"), ("base", "base_tvt"), ("a50", "a_p50_tvt")):
                if column in aligned_anchors.columns:
                    diff = by_id - aligned_anchors[column]
                    rec[f"feat_mean_abs_to_{name}"] = _mean_abs(diff) / 50.0
                    rec[f"feat_p95_abs_to_{name}"] = _p95_abs(diff) / 50.0
                else:
                    rec[f"feat_mean_abs_to_{name}"] = 0.0
                    rec[f"feat_p95_abs_to_{name}"] = 0.0
            rows.append(rec)
    if not rows:
        raise ValueError("No chunk-policy rows were built")
    frame = pd.DataFrame(rows)
    best_mse = frame.groupby("group_id")["target_mse"].transform("min")
    frame["target_regret_mse"] = frame["target_mse"] - best_mse
    feature_columns = [col for col in frame.columns if col.startswith("feat_")]
    assert_schema_safe_columns(feature_columns, context="ChunkPolicy features")
    features = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    return ChunkPolicyDataset(rows=frame, features=features, feature_columns=feature_columns)


def viterbi_select_candidates(
    chunks: list[dict[str, Any]],
    *,
    switch_penalty: float,
    jump_penalty: float,
) -> list[str]:
    if not chunks:
        return []
    prev_costs: dict[str, float] = {}
    backptrs: list[dict[str, str]] = []
    for idx, chunk in enumerate(chunks):
        current: dict[str, float] = {}
        current_back: dict[str, str] = {}
        for candidate, info in chunk["candidates"].items():
            unary = float(info["cost"])
            if idx == 0:
                current[candidate] = unary
                current_back[candidate] = ""
                continue
            best_prev = None
            best_cost = float("inf")
            for prev_candidate, prev_cost in prev_costs.items():
                prev_info = chunks[idx - 1]["candidates"][prev_candidate]
                jump = float(info["first"] - prev_info["last"])
                transition = jump_penalty * jump * jump
                if prev_candidate != candidate:
                    transition += switch_penalty
                total = prev_cost + unary + transition
                if total < best_cost:
                    best_cost = total
                    best_prev = prev_candidate
            current[candidate] = best_cost
            current_back[candidate] = str(best_prev)
        prev_costs = current
        backptrs.append(current_back)
    last = min(prev_costs, key=prev_costs.get)
    path = [last]
    for idx in range(len(chunks) - 1, 0, -1):
        last = backptrs[idx][last]
        path.append(last)
    return list(reversed(path))


def _train_ranker(dataset: ChunkPolicyDataset, train_wells: list[str], config: ChunkPolicyConfig) -> CatBoostRanker:
    mask = dataset.rows["well_id"].astype(str).isin(set(train_wells)).to_numpy()
    regret = pd.to_numeric(dataset.rows.loc[mask, "target_regret_mse"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    regret_clipped = np.clip(regret, a_min=0.0, a_max=None)
    if config.ranker_target == "neg_sqrt_regret":
        # Preserve disaster signal: regret of 1000 ft² vs 1 ft² maps to -31.6 vs -1.0,
        # so the ranker actually feels the cost of misranking catastrophic chunks.
        target = -np.sqrt(regret_clipped)
    elif config.ranker_target == "neg_log1p_regret":
        target = -np.log1p(regret_clipped)
    else:
        raise ValueError(f"unknown ranker_target: {config.ranker_target}")
    finite = np.isfinite(target)
    pool = Pool(
        dataset.features.loc[mask].iloc[finite],
        label=target[finite],
        group_id=dataset.rows.loc[mask].iloc[finite]["group_id"].astype(str).to_numpy(),
    )
    model = CatBoostRanker(
        loss_function=config.ranker_loss,
        iterations=config.iterations,
        learning_rate=config.learning_rate,
        depth=config.depth,
        l2_leaf_reg=config.l2_leaf_reg,
        random_seed=config.seed,
        allow_writing_files=False,
        verbose=False,
    )
    model.fit(pool)
    return model


def add_fold_safe_spatial_prior_features(
    rows: pd.DataFrame,
    features: pd.DataFrame,
    feature_columns: list[str],
    *,
    train_wells: list[str],
    valid_wells: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    out_rows = rows.copy()
    out_features = features.copy()
    train_mask = out_rows["well_id"].astype(str).isin(set(train_wells))
    train = out_rows.loc[train_mask]
    global_mse = float(pd.to_numeric(train["target_mse"], errors="coerce").mean())
    global_gain = float(pd.to_numeric(train["target_gain_vs_b2_mse"], errors="coerce").mean())
    by_candidate = train.groupby("candidate", sort=True).agg(
        prior_mse=("target_mse", "mean"),
        prior_regret=("target_regret_mse", "mean"),
        prior_gain=("target_gain_vs_b2_mse", "mean"),
        prior_win_frac=("target_regret_mse", lambda s: float((pd.to_numeric(s, errors="coerce") <= 1e-9).mean())),
    )
    candidate = out_rows["candidate"].astype(str)
    out_features["feat_spatial_candidate_prior_mse"] = np.log1p(
        candidate.map(by_candidate["prior_mse"]).fillna(global_mse).astype(float)
    )
    out_features["feat_spatial_candidate_prior_regret"] = np.log1p(
        candidate.map(by_candidate["prior_regret"]).fillna(global_mse).clip(lower=0.0).astype(float)
    )
    out_features["feat_spatial_candidate_prior_gain"] = (
        candidate.map(by_candidate["prior_gain"]).fillna(global_gain).astype(float) / 2500.0
    )
    out_features["feat_spatial_candidate_prior_win_frac"] = (
        candidate.map(by_candidate["prior_win_frac"]).fillna(0.0).astype(float)
    )
    new_columns = [
        "feat_spatial_candidate_prior_mse",
        "feat_spatial_candidate_prior_regret",
        "feat_spatial_candidate_prior_gain",
        "feat_spatial_candidate_prior_win_frac",
    ]
    out_features.loc[
        ~out_rows["well_id"].astype(str).isin(set(train_wells) | set(valid_wells)),
        new_columns,
    ] = 0.0
    merged_columns = list(dict.fromkeys([*feature_columns, *new_columns]))
    assert_schema_safe_columns(merged_columns, context="ChunkPolicy spatial features")
    out_features = out_features.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    return out_rows, out_features, merged_columns


def add_fold_safe_shared_typewell_features(
    rows: pd.DataFrame,
    features: pd.DataFrame,
    feature_columns: list[str],
    *,
    neighbours: pd.DataFrame | None,
    train_wells: list[str],
    valid_wells: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Add test-safe typewell-neighbour context to candidate/chunk rows.

    The neighbour table is built from typewell GR signatures and geometry only. It does
    not carry hidden TVT labels, so these are priors/features rather than oracle costs.
    """
    out_rows = rows.copy()
    out_features = features.copy()
    new_columns = [
        "feat_stw_top1_corr",
        "feat_stw_mean_corr",
        "feat_stw_corr_std",
        "feat_stw_top1_shift_bins",
        "feat_stw_mean_abs_shift_bins",
        "feat_stw_min_xyz_distance_log",
        "feat_stw_corr_x_action_shift",
        "feat_stw_corr_x_action_drift",
        "feat_stw_corr_x_action_slope",
        "feat_stw_abs_shift_x_action_shift",
        "feat_stw_signed_shift_x_shift_amount",
    ]
    for column in new_columns:
        out_features[column] = 0.0
    if neighbours is not None and not neighbours.empty:
        required = {"query_well_id", "typewell_corr", "typewell_shift_bins", "xyz_distance", "rank"}
        missing = required.difference(neighbours.columns)
        if missing:
            raise ValueError(f"shared typewell neighbour table missing columns: {sorted(missing)}")
        table = neighbours.copy()
        table["query_well_id"] = table["query_well_id"].astype(str)
        for column in ("typewell_corr", "typewell_shift_bins", "xyz_distance", "rank"):
            table[column] = pd.to_numeric(table[column], errors="coerce")
        table = table.replace([np.inf, -np.inf], np.nan)
        top1 = (
            table.sort_values(["query_well_id", "rank"])
            .groupby("query_well_id", sort=True)
            .first(numeric_only=True)
        )
        stats = table.groupby("query_well_id", sort=True).agg(
            stw_mean_corr=("typewell_corr", "mean"),
            stw_corr_std=("typewell_corr", "std"),
            stw_mean_abs_shift_bins=("typewell_shift_bins", lambda s: float(np.nanmean(np.abs(s)))),
            stw_min_xyz_distance=("xyz_distance", "min"),
        )
        stats["stw_top1_corr"] = top1["typewell_corr"]
        stats["stw_top1_shift_bins"] = top1["typewell_shift_bins"]
        stats = stats.fillna(0.0)
        well = out_rows["well_id"].astype(str)
        out_features["feat_stw_top1_corr"] = well.map(stats["stw_top1_corr"]).fillna(0.0).astype(float)
        out_features["feat_stw_mean_corr"] = well.map(stats["stw_mean_corr"]).fillna(0.0).astype(float)
        out_features["feat_stw_corr_std"] = well.map(stats["stw_corr_std"]).fillna(0.0).astype(float)
        out_features["feat_stw_top1_shift_bins"] = (
            well.map(stats["stw_top1_shift_bins"]).fillna(0.0).astype(float) / 32.0
        )
        out_features["feat_stw_mean_abs_shift_bins"] = (
            well.map(stats["stw_mean_abs_shift_bins"]).fillna(0.0).astype(float) / 32.0
        )
        out_features["feat_stw_min_xyz_distance_log"] = (
            np.log1p(well.map(stats["stw_min_xyz_distance"]).fillna(0.0).astype(float)) / 12.0
        )
    active = out_rows["well_id"].astype(str).isin(set(train_wells) | set(valid_wells))
    for column in new_columns:
        out_features.loc[~active, column] = 0.0

    def _feature(name: str) -> pd.Series:
        if name in out_features.columns:
            return pd.to_numeric(out_features[name], errors="coerce").fillna(0.0).astype(float)
        return pd.Series(0.0, index=out_features.index, dtype=float)

    corr = _feature("feat_stw_top1_corr")
    abs_shift = _feature("feat_stw_mean_abs_shift_bins")
    signed_shift = _feature("feat_stw_top1_shift_bins")
    out_features["feat_stw_corr_x_action_shift"] = corr * _feature("feat_action_is_shift")
    out_features["feat_stw_corr_x_action_drift"] = corr * _feature("feat_action_is_drift")
    out_features["feat_stw_corr_x_action_slope"] = corr * _feature("feat_action_is_slope")
    out_features["feat_stw_abs_shift_x_action_shift"] = abs_shift * _feature("feat_action_is_shift")
    out_features["feat_stw_signed_shift_x_shift_amount"] = signed_shift * (
        _feature("feat_shift_amount") / 80.0
    )

    merged_columns = list(dict.fromkeys([*feature_columns, *new_columns]))
    assert_schema_safe_columns(merged_columns, context="ChunkPolicy shared typewell features")
    out_features = out_features.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    return out_rows, out_features, merged_columns


def _train_cost_model(
    dataset: ChunkPolicyDataset, train_wells: list[str], config: ChunkPolicyConfig
) -> tuple[CatBoostRegressor, str]:
    """Train per-chunk per-candidate MSE estimator.

    Two modes:
      - "log_mse": legacy. predict log1p(target_mse), unweighted, RMSE loss.
        Severely under-predicts disaster chunks (target RMSE >15 → pred RMSE ~7).
      - "sqrt_mse_weighted": predict sqrt(target_mse) (i.e. per-chunk RMSE)
        with MAE loss and sample_weight=row_count. Preserves tail signal and
        matches the row-weighted RMSE metric we ultimately optimize.

    Returns (model, mode) so the caller knows how to decode predictions.
    """
    mask = dataset.rows["well_id"].astype(str).isin(set(train_wells)).to_numpy()
    raw_mse = pd.to_numeric(dataset.rows.loc[mask, "target_mse"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    row_count = pd.to_numeric(dataset.rows.loc[mask, "row_count"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if config.cost_target == "log_mse":
        target = np.log1p(np.clip(raw_mse, a_min=0.0, a_max=None))
        loss_function = "RMSE"
        weights = None
    elif config.cost_target == "sqrt_mse_weighted":
        target = np.sqrt(np.clip(raw_mse, a_min=0.0, a_max=None))
        loss_function = "MAE"
        weights = row_count
    else:
        raise ValueError(f"unknown cost_target: {config.cost_target}")
    finite = np.isfinite(target)
    model = CatBoostRegressor(
        loss_function=loss_function,
        iterations=config.iterations,
        learning_rate=config.learning_rate,
        depth=config.depth,
        l2_leaf_reg=config.l2_leaf_reg,
        random_seed=config.seed,
        allow_writing_files=False,
        verbose=False,
    )
    fit_kwargs: dict[str, Any] = {}
    if weights is not None:
        fit_kwargs["sample_weight"] = weights[finite]
    model.fit(dataset.features.loc[mask].iloc[finite], target[finite], **fit_kwargs)
    return model, config.cost_target


def _decode_pred_mse(raw_pred: np.ndarray, cost_target: str) -> np.ndarray:
    """Convert cost_model output back to MSE units used by DP unary cost."""
    if cost_target == "log_mse":
        return np.expm1(raw_pred).clip(min=0.0)
    if cost_target == "sqrt_mse_weighted":
        return np.square(raw_pred.clip(min=0.0))
    raise ValueError(f"unknown cost_target: {cost_target}")


def _add_ranker_cost(local: pd.DataFrame, ranker_alpha: float) -> pd.DataFrame:
    out = local.copy()
    out["ranker_z"] = out.groupby("group_id")["ranker_score"].transform(
        lambda s: (s - s.mean()) / max(float(s.std(ddof=0)), 1e-6)
    )
    out["dp_unary_cost"] = (
        pd.to_numeric(out["row_count"], errors="coerce")
        * pd.to_numeric(out["pred_mse"], errors="coerce")
        - ranker_alpha * pd.to_numeric(out["row_count"], errors="coerce") * out["ranker_z"]
    )
    return out


def _select_dp_chunks(local: pd.DataFrame, config: ChunkPolicyConfig) -> pd.DataFrame:
    selections: list[dict[str, Any]] = []
    for well_id, well in local.groupby("well_id", sort=True):
        chunks: list[dict[str, Any]] = []
        chunk_ids: list[int] = []
        for chunk_id, chunk in well.groupby("chunk_id", sort=True):
            candidates = {
                str(row.candidate): {
                    "cost": float(row.dp_unary_cost),
                    "first": float(row.first_pred_tvt),
                    "last": float(row.last_pred_tvt),
                }
                for row in chunk.itertuples(index=False)
            }
            chunks.append({"chunk_id": int(chunk_id), "candidates": candidates})
            chunk_ids.append(int(chunk_id))
        path = viterbi_select_candidates(
            chunks,
            switch_penalty=config.switch_penalty,
            jump_penalty=config.jump_penalty,
        )
        for chunk_id, candidate in zip(chunk_ids, path, strict=True):
            row = well[(well["chunk_id"].astype(int) == chunk_id) & (well["candidate"].astype(str) == candidate)].iloc[0]
            selections.append(
                {
                    "well_id": str(well_id),
                    "run_id": int(row.run_id),
                    "chunk_id": int(chunk_id),
                    "selected_candidate": str(candidate),
                    "selected_target_mse": float(row.target_mse),
                    "selected_true_rmse": float(row.target_rmse),
                    "fold": int(row.fold),
                }
            )
    return pd.DataFrame(selections)


def _row_predictions_for_chunk_selection(
    frame: pd.DataFrame,
    selected_chunks: pd.DataFrame,
    *,
    residual_predictions: pd.DataFrame | None,
    traceback_candidates: pd.DataFrame | None,
    step_predictions: pd.DataFrame | None,
    k_offset_predictions: pd.DataFrame | None,
    dtvt_state_predictions: pd.DataFrame | None,
    chunk_size: int,
) -> pd.DataFrame:
    raw = _add_missing_prior_columns(_ensure_ids(frame))
    residual = residual_predictions.copy() if residual_predictions is not None else None
    if residual is not None and not residual.empty:
        residual["well_id"] = residual["well_id"].astype(str)
    traceback = traceback_candidates.copy() if traceback_candidates is not None else None
    if traceback is not None and not traceback.empty and "well_id" in traceback.columns:
        traceback["well_id"] = traceback["well_id"].astype(str)
    step_preds = _prepare_step_predictions(step_predictions)
    k_offset = k_offset_predictions.copy() if k_offset_predictions is not None else None
    if k_offset is not None and not k_offset.empty:
        k_offset["well_id"] = k_offset["well_id"].astype(str)
    dtvt_state = dtvt_state_predictions.copy() if dtvt_state_predictions is not None else None
    if dtvt_state is not None and not dtvt_state.empty:
        dtvt_state["well_id"] = dtvt_state["well_id"].astype(str)
    rows: list[pd.DataFrame] = []
    for well_id, well in raw.groupby("well_id", sort=True):
        selected = selected_chunks[selected_chunks["well_id"].astype(str) == str(well_id)]
        if selected.empty:
            continue
        hidden = _hidden_only(well).sort_values("row_idx")
        chunk_index = _chunk_frame(hidden, chunk_size)
        well_residual = None
        if residual is not None and not residual.empty:
            well_residual = residual[residual["well_id"].astype(str) == str(well_id)]
        well_traceback = None
        if traceback is not None and not traceback.empty and "well_id" in traceback.columns:
            well_traceback = traceback[traceback["well_id"].astype(str) == str(well_id)]
        well_step_preds = None
        if not step_preds.empty:
            well_step_preds = step_preds[step_preds["well_id"].astype(str) == str(well_id)]
        well_k_offset = None
        if k_offset is not None and not k_offset.empty:
            well_k_offset = k_offset[k_offset["well_id"].astype(str) == str(well_id)]
        well_dtvt_state = None
        if dtvt_state is not None and not dtvt_state.empty:
            well_dtvt_state = dtvt_state[dtvt_state["well_id"].astype(str) == str(well_id)]
        bank = build_candidate_bank_from_frames(
            well,
            residual_predictions=well_residual,
            traceback_candidates=well_traceback,
            step_predictions=well_step_preds,
            k_offset_predictions=well_k_offset,
            dtvt_state_predictions=well_dtvt_state,
        )
        bank = bank.merge(chunk_index[["id", "chunk_id"]], on="id", how="inner")
        for row in selected.itertuples(index=False):
            part = bank[
                (bank["chunk_id"].astype(int) == int(row.chunk_id))
                & (bank["candidate"].astype(str) == str(row.selected_candidate))
            ].copy()
            if not part.empty:
                part["selected_candidate"] = str(row.selected_candidate)
                rows.append(part)
    if not rows:
        raise ValueError("Chunk policy produced no row predictions")
    out = pd.concat(rows, ignore_index=True)
    out["candidate"] = "chunk_ranker_dp_v1"
    return out


def _row_metrics(predictions: pd.DataFrame, hidden_rows: pd.DataFrame) -> dict[str, Any]:
    hidden = _hidden_only(_add_missing_prior_columns(_ensure_ids(hidden_rows)))
    b2 = pd.to_numeric(hidden["b2_tvt"], errors="coerce")
    true_hidden = pd.to_numeric(hidden["TVT"], errors="coerce")
    metrics: dict[str, Any] = {
        "row_rmse": _rmse(
            pd.to_numeric(predictions["pred_tvt"], errors="coerce").to_numpy(dtype=np.float64)
            - pd.to_numeric(predictions["TVT"], errors="coerce").to_numpy(dtype=np.float64)
        ),
        "mean_well_rmse": float(
            predictions.groupby("well_id").apply(lambda g: _rmse_pair(g["pred_tvt"], g["TVT"])).mean()
        ),
        "b2_row_rmse": _rmse(b2.to_numpy(dtype=np.float64) - true_hidden.to_numpy(dtype=np.float64)),
    }
    if {"tail_class", "b2_tvt"}.issubset(predictions.columns):
        tail_metrics: dict[str, dict[str, float]] = {}
        for tail_class, group in predictions.groupby("tail_class", dropna=False, sort=True):
            label = "__missing__" if pd.isna(tail_class) else str(tail_class)
            policy_rmse = _rmse_pair(group["pred_tvt"], group["TVT"])
            b2_rmse = _rmse_pair(group["b2_tvt"], group["TVT"])
            tail_metrics[label] = {
                "rows": int(len(group)),
                "policy_row_rmse": float(policy_rmse),
                "b2_row_rmse": float(b2_rmse),
                "delta_b2_minus_policy": float(b2_rmse - policy_rmse),
            }
        metrics["tail_class"] = tail_metrics
    return metrics


def _write_report(output_dir: Path, metrics: dict[str, Any], selected_chunks: pd.DataFrame) -> None:
    lines = [
        "CHUNK_RANKER_DP_V1_REPORT",
        "",
        "summary:",
        json.dumps(_json_safe(metrics), indent=2),
        "",
        "selected candidate counts:",
        selected_chunks["selected_candidate"].value_counts().to_string(),
    ]
    (output_dir / "chunk_policy_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_chunk_policy_from_frames(
    hidden_rows: pd.DataFrame,
    *,
    config: ChunkPolicyConfig,
    residual_predictions: pd.DataFrame | None = None,
    traceback_candidates: pd.DataFrame | None = None,
    step_predictions: pd.DataFrame | None = None,
    location_aware_steps: pd.DataFrame | None = None,
    top_state_predictions: pd.DataFrame | None = None,
    k_offset_predictions: pd.DataFrame | None = None,
    dtvt_state_predictions: pd.DataFrame | None = None,
) -> dict[str, Any]:
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    shared_neighbours = None
    if config.use_shared_typewell_priors:
        if config.shared_typewell_neighbours_path is None:
            raise ValueError("shared typewell priors enabled but no neighbour path was provided")
        neighbour_path = Path(config.shared_typewell_neighbours_path)
        if not neighbour_path.exists():
            raise FileNotFoundError(f"shared typewell neighbour file not found: {neighbour_path}")
        shared_neighbours = pd.read_parquet(neighbour_path)
        print(
            f"[chunk-policy] loaded shared typewell neighbours rows={len(shared_neighbours):,} "
            f"from {neighbour_path}",
            file=sys.stderr,
            flush=True,
        )
    print("[chunk-policy] building chunk-level dataset", file=sys.stderr, flush=True)
    dataset = build_chunk_policy_dataset(
        hidden_rows,
        residual_predictions=residual_predictions,
        traceback_candidates=traceback_candidates,
        step_predictions=step_predictions,
        location_aware_steps=location_aware_steps,
        top_state_predictions=top_state_predictions,
        k_offset_predictions=k_offset_predictions,
        dtvt_state_predictions=dtvt_state_predictions,
        chunk_size=config.chunk_size,
        use_selfcal_probes=config.use_selfcal_probes,
        selfcal_max_probes=config.selfcal_max_probes,
        progress_every=config.progress_every,
    )
    print(
        f"[chunk-policy] dataset rows={len(dataset.rows):,} groups={dataset.rows['group_id'].nunique()} "
        f"wells={dataset.rows['well_id'].nunique()} features={len(dataset.feature_columns)}",
        file=sys.stderr,
        flush=True,
    )
    folds = make_group_folds(dataset.rows["well_id"], n_folds=config.n_folds, seed=config.seed)
    prediction_parts: list[pd.DataFrame] = []
    selected_parts: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, Any]] = []
    feature_columns_used = list(dataset.feature_columns)
    for fold_idx, (train_wells, valid_wells) in enumerate(folds):
        print(
            f"[chunk-policy] fold {fold_idx + 1}/{len(folds)} train_wells={len(train_wells)} "
            f"valid_wells={len(valid_wells)}",
            file=sys.stderr,
            flush=True,
        )
        fold_rows = dataset.rows
        fold_features = dataset.features
        fold_feature_columns = dataset.feature_columns
        if config.use_spatial_priors:
            fold_rows, fold_features, fold_feature_columns = add_fold_safe_spatial_prior_features(
                fold_rows,
                fold_features,
                fold_feature_columns,
                train_wells=train_wells,
                valid_wells=valid_wells,
            )
        if config.use_shared_typewell_priors:
            fold_rows, fold_features, fold_feature_columns = add_fold_safe_shared_typewell_features(
                fold_rows,
                fold_features,
                fold_feature_columns,
                neighbours=shared_neighbours,
                train_wells=train_wells,
                valid_wells=valid_wells,
            )
        feature_columns_used = list(dict.fromkeys([*feature_columns_used, *fold_feature_columns]))
        fold_dataset = ChunkPolicyDataset(
            rows=fold_rows,
            features=fold_features[fold_feature_columns],
            feature_columns=fold_feature_columns,
        )
        ranker = _train_ranker(fold_dataset, train_wells, config)
        cost_model, cost_target = _train_cost_model(fold_dataset, train_wells, config)
        valid_mask = fold_dataset.rows["well_id"].astype(str).isin(set(valid_wells)).to_numpy()
        local_cols = [
            "well_id",
            "run_id",
            "chunk_id",
            "group_id",
            "candidate",
            "row_count",
            "target_mse",
            "target_rmse",
            "first_pred_tvt",
            "last_pred_tvt",
        ]
        local = fold_dataset.rows.loc[valid_mask, local_cols].copy()
        local["fold"] = fold_idx
        local["ranker_score"] = ranker.predict(fold_dataset.features.loc[valid_mask])
        local["pred_mse"] = _decode_pred_mse(
            cost_model.predict(fold_dataset.features.loc[valid_mask]),
            cost_target,
        )
        local = _add_ranker_cost(local, config.ranker_alpha)
        prediction_parts.append(local)
        selected = _select_dp_chunks(local, config)
        selected_parts.append(selected)
        selected_row_count = pd.to_numeric(
            selected.merge(
                local[["well_id", "chunk_id", "candidate", "row_count"]].rename(
                    columns={"candidate": "selected_candidate"}
                ),
                on=["well_id", "chunk_id", "selected_candidate"],
                how="left",
            )["row_count"],
            errors="coerce",
        ).fillna(0.0)
        selected_mse = pd.to_numeric(selected["selected_target_mse"], errors="coerce").fillna(0.0)
        weighted_den = float(selected_row_count.sum())
        selected_row_weighted_rmse = (
            float(np.sqrt(float((selected_row_count * selected_mse).sum()) / weighted_den))
            if weighted_den > 0
            else float("nan")
        )
        selected_chunk_rmse = pd.to_numeric(selected["selected_true_rmse"], errors="coerce")
        fold_metrics.append(
            {
                "fold": fold_idx,
                "train_wells": train_wells,
                "valid_wells": valid_wells,
                "selected_mean_true_rmse": float(selected["selected_true_rmse"].mean()),
                "selected_row_weighted_rmse": selected_row_weighted_rmse,
                "selected_p90_chunk_rmse": float(selected_chunk_rmse.quantile(0.90)),
                "selected_worst_chunk_rmse": float(selected_chunk_rmse.max()),
            }
        )
        print(
            f"[chunk-policy] fold {fold_idx + 1}/{len(folds)} selected_mean_true_rmse="
            f"{float(selected['selected_true_rmse'].mean()):.4f} "
            f"selected_row_weighted_rmse={selected_row_weighted_rmse:.4f} "
            f"p90_chunk_rmse={float(selected_chunk_rmse.quantile(0.90)):.4f} "
            f"worst_chunk_rmse={float(selected_chunk_rmse.max()):.4f}",
            file=sys.stderr,
            flush=True,
        )
    chunk_predictions = pd.concat(prediction_parts, ignore_index=True)
    selected_chunks = pd.concat(selected_parts, ignore_index=True)
    row_predictions = _row_predictions_for_chunk_selection(
        hidden_rows,
        selected_chunks,
        residual_predictions=residual_predictions,
        traceback_candidates=traceback_candidates,
        step_predictions=step_predictions,
        k_offset_predictions=k_offset_predictions,
        dtvt_state_predictions=dtvt_state_predictions,
        chunk_size=config.chunk_size,
    )
    dp_metrics = _row_metrics(row_predictions, hidden_rows)
    oracle_chunks = dataset.rows.loc[
        dataset.rows.groupby("group_id")["target_mse"].idxmin(),
        ["well_id", "run_id", "chunk_id", "candidate", "target_mse", "target_rmse"],
    ].rename(
        columns={
            "candidate": "selected_candidate",
            "target_mse": "selected_target_mse",
            "target_rmse": "selected_true_rmse",
        }
    )
    oracle_chunks["fold"] = -1
    oracle_rows = _row_predictions_for_chunk_selection(
        hidden_rows,
        oracle_chunks,
        residual_predictions=residual_predictions,
        traceback_candidates=traceback_candidates,
        step_predictions=step_predictions,
        k_offset_predictions=k_offset_predictions,
        dtvt_state_predictions=dtvt_state_predictions,
        chunk_size=config.chunk_size,
    )
    oracle_metrics = _row_metrics(oracle_rows, hidden_rows)
    b2_row_rmse = dp_metrics["b2_row_rmse"]
    denominator = b2_row_rmse**2 - oracle_metrics["row_rmse"] ** 2
    metrics: dict[str, Any] = {
        "candidate": "chunk_ranker_dp_v1",
        "rows": int(len(row_predictions)),
        "wells": int(row_predictions["well_id"].nunique()),
        "chunks": int(selected_chunks[["well_id", "chunk_id"]].drop_duplicates().shape[0]),
        "folds": int(len(fold_metrics)),
        "chunk_size": int(config.chunk_size),
        "feature_columns": feature_columns_used,
        "use_selfcal_probes": bool(config.use_selfcal_probes),
        "use_spatial_priors": bool(config.use_spatial_priors),
        "use_shared_typewell_priors": bool(config.use_shared_typewell_priors),
        "use_step_predictions": step_predictions is not None and not step_predictions.empty,
        "use_location_aware_steps": location_aware_steps is not None and not location_aware_steps.empty,
        "use_top_state_predictions": top_state_predictions is not None and not top_state_predictions.empty,
        "use_traceback_candidates": traceback_candidates is not None and not traceback_candidates.empty,
        "use_k_offset_predictions": k_offset_predictions is not None and not k_offset_predictions.empty,
        "use_dtvt_state_predictions": dtvt_state_predictions is not None and not dtvt_state_predictions.empty,
        "cost_target": config.cost_target,
        "ranker_target": config.ranker_target,
        "shared_typewell_neighbours_path": str(config.shared_typewell_neighbours_path)
        if config.shared_typewell_neighbours_path is not None
        else None,
        "dp": dp_metrics,
        "b2": {"row_rmse": b2_row_rmse},
        "oracle": {"chunk_oracle_row_rmse": oracle_metrics["row_rmse"]},
        "mse_gain_capture": float((b2_row_rmse**2 - dp_metrics["row_rmse"] ** 2) / denominator)
        if denominator > 0
        else float("nan"),
        "fold_metrics": fold_metrics,
    }
    row_predictions.to_parquet(out / "chunk_policy_row_predictions.parquet", index=False)
    selected_chunks.to_csv(out / "chunk_policy_selected_chunks.csv", index=False)
    chunk_predictions.to_parquet(out / "chunk_policy_chunk_predictions.parquet", index=False)
    dataset.rows.to_parquet(out / "chunk_policy_dataset.parquet", index=False)
    (out / "chunk_policy_metrics.json").write_text(
        json.dumps(_json_safe(metrics), indent=2), encoding="utf-8"
    )
    _write_report(out, metrics, selected_chunks)
    return metrics


def run_chunk_policy(config: ChunkPolicyConfig) -> dict[str, Any]:
    print(f"[chunk-policy] loading training frame from {config.data_dir}", file=sys.stderr, flush=True)
    frame = load_training_frame(ResidualStackConfig(data_dir=config.data_dir, k_wells=config.k_wells))
    print(
        f"[chunk-policy] loaded frame rows={len(frame):,} wells={frame['well_id'].nunique()}",
        file=sys.stderr,
        flush=True,
    )
    residual = (
        pd.read_parquet(config.residual_predictions_path)
        if config.residual_predictions_path is not None and Path(config.residual_predictions_path).exists()
        else None
    )
    if residual is not None:
        print(f"[chunk-policy] loaded residual rows={len(residual):,}", file=sys.stderr, flush=True)
    traceback = (
        pd.read_parquet(config.traceback_candidates_path)
        if config.traceback_candidates_path is not None
        and Path(config.traceback_candidates_path).exists()
        else None
    )
    if traceback is not None:
        print(
            f"[chunk-policy] loaded traceback candidates rows={len(traceback):,}",
            file=sys.stderr,
            flush=True,
        )
    step_predictions = (
        pd.read_parquet(config.step_predictions_path)
        if config.step_predictions_path is not None and Path(config.step_predictions_path).exists()
        else None
    )
    if step_predictions is not None:
        print(
            f"[chunk-policy] loaded step predictions rows={len(step_predictions):,}",
            file=sys.stderr,
            flush=True,
        )
    location_aware_steps = (
        pd.read_parquet(config.location_aware_steps_path)
        if config.location_aware_steps_path is not None and Path(config.location_aware_steps_path).exists()
        else None
    )
    if location_aware_steps is not None:
        print(
            f"[chunk-policy] loaded location-aware steps rows={len(location_aware_steps):,}",
            file=sys.stderr,
            flush=True,
        )
    top_state_predictions = (
        pd.read_parquet(config.top_state_predictions_path)
        if config.top_state_predictions_path is not None and Path(config.top_state_predictions_path).exists()
        else None
    )
    if top_state_predictions is not None:
        print(
            f"[chunk-policy] loaded top-state predictions rows={len(top_state_predictions):,}",
            file=sys.stderr,
            flush=True,
        )
    k_offset_predictions = (
        pd.read_parquet(config.k_offset_predictions_path)
        if config.k_offset_predictions_path is not None
        and Path(config.k_offset_predictions_path).exists()
        else None
    )
    if k_offset_predictions is not None:
        print(
            f"[chunk-policy] loaded k-offset predictions rows={len(k_offset_predictions):,}",
            file=sys.stderr,
            flush=True,
        )
    dtvt_state_predictions = (
        pd.read_parquet(config.dtvt_state_predictions_path)
        if config.dtvt_state_predictions_path is not None
        and Path(config.dtvt_state_predictions_path).exists()
        else None
    )
    if dtvt_state_predictions is not None:
        print(
            f"[chunk-policy] loaded dtvt-state predictions rows={len(dtvt_state_predictions):,}",
            file=sys.stderr,
            flush=True,
        )
    metrics = run_chunk_policy_from_frames(
        frame,
        config=config,
        residual_predictions=residual,
        traceback_candidates=traceback,
        step_predictions=step_predictions,
        location_aware_steps=location_aware_steps,
        top_state_predictions=top_state_predictions,
        k_offset_predictions=k_offset_predictions,
        dtvt_state_predictions=dtvt_state_predictions,
    )
    compact = {
        "candidate": metrics["candidate"],
        "rows": metrics["rows"],
        "wells": metrics["wells"],
        "chunks": metrics["chunks"],
        "folds": metrics["folds"],
        "dp": metrics["dp"],
        "b2": metrics["b2"],
        "oracle": metrics["oracle"],
        "mse_gain_capture": metrics["mse_gain_capture"],
        "fold_selected_mean_true_rmse": [
            {
                "fold": item["fold"],
                "train_wells": len(item["train_wells"]),
                "valid_wells": len(item["valid_wells"]),
                "selected_mean_true_rmse": item["selected_mean_true_rmse"],
                "selected_row_weighted_rmse": item["selected_row_weighted_rmse"],
                "selected_p90_chunk_rmse": item["selected_p90_chunk_rmse"],
                "selected_worst_chunk_rmse": item["selected_worst_chunk_rmse"],
            }
            for item in metrics["fold_metrics"]
        ],
    }
    print(json.dumps(_json_safe(compact), indent=2))
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train fold-safe chunk ranker + DP policy")
    parser.add_argument("--data-dir", type=Path, default=ChunkPolicyConfig.data_dir)
    parser.add_argument("--output-dir", type=Path, default=ChunkPolicyConfig.output_dir)
    parser.add_argument("--residual-predictions", type=Path, default=ChunkPolicyConfig.residual_predictions_path)
    parser.add_argument("--traceback-candidates", type=Path, default=ChunkPolicyConfig.traceback_candidates_path)
    parser.add_argument("--step-predictions", type=Path, default=ChunkPolicyConfig.step_predictions_path)
    parser.add_argument("--location-aware-steps", type=Path, default=ChunkPolicyConfig.location_aware_steps_path)
    parser.add_argument("--top-state-predictions", type=Path, default=ChunkPolicyConfig.top_state_predictions_path)
    parser.add_argument(
        "--k-offset-predictions",
        type=Path,
        default=ChunkPolicyConfig.k_offset_predictions_path,
        help="Per-row k-segment offset OOF predictions (id, well_id, row_idx, pred_tvt).",
    )
    parser.add_argument(
        "--dtvt-state-predictions",
        type=Path,
        default=ChunkPolicyConfig.dtvt_state_predictions_path,
        help="Per-row dTVT state-model OOF predictions (id, well_id, row_idx, pred_tvt).",
    )
    parser.add_argument("--n-folds", type=int, default=ChunkPolicyConfig.n_folds)
    parser.add_argument("--seed", type=int, default=ChunkPolicyConfig.seed)
    parser.add_argument("--k-wells", type=int, default=ChunkPolicyConfig.k_wells)
    parser.add_argument("--chunk-size", type=int, default=ChunkPolicyConfig.chunk_size)
    parser.add_argument("--iterations", type=int, default=ChunkPolicyConfig.iterations)
    parser.add_argument("--learning-rate", type=float, default=ChunkPolicyConfig.learning_rate)
    parser.add_argument("--depth", type=int, default=ChunkPolicyConfig.depth)
    parser.add_argument("--l2-leaf-reg", type=float, default=ChunkPolicyConfig.l2_leaf_reg)
    parser.add_argument("--ranker-loss", default=ChunkPolicyConfig.ranker_loss)
    parser.add_argument("--ranker-alpha", type=float, default=ChunkPolicyConfig.ranker_alpha)
    parser.add_argument("--switch-penalty", type=float, default=ChunkPolicyConfig.switch_penalty)
    parser.add_argument("--jump-penalty", type=float, default=ChunkPolicyConfig.jump_penalty)
    parser.add_argument(
        "--cost-target",
        choices=["log_mse", "sqrt_mse_weighted"],
        default=ChunkPolicyConfig.cost_target,
        help="Loss for the per-chunk per-candidate MSE estimator.",
    )
    parser.add_argument(
        "--ranker-target",
        choices=["neg_log1p_regret", "neg_sqrt_regret"],
        default=ChunkPolicyConfig.ranker_target,
        help="Target transform used inside the ranking model.",
    )
    parser.add_argument("--no-selfcal-probes", action="store_true")
    parser.add_argument("--selfcal-max-probes", type=int, default=ChunkPolicyConfig.selfcal_max_probes)
    parser.add_argument("--no-spatial-priors", action="store_true")
    parser.add_argument("--use-shared-typewell-priors", action="store_true")
    parser.add_argument(
        "--shared-typewell-neighbours",
        type=Path,
        default=ChunkPolicyConfig.shared_typewell_neighbours_path,
    )
    parser.add_argument("--progress-every", type=int, default=ChunkPolicyConfig.progress_every)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_chunk_policy(
        ChunkPolicyConfig(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            residual_predictions_path=args.residual_predictions,
            traceback_candidates_path=args.traceback_candidates,
            step_predictions_path=args.step_predictions,
            location_aware_steps_path=args.location_aware_steps,
            top_state_predictions_path=args.top_state_predictions,
            k_offset_predictions_path=args.k_offset_predictions,
            dtvt_state_predictions_path=args.dtvt_state_predictions,
            n_folds=args.n_folds,
            seed=args.seed,
            k_wells=args.k_wells,
            chunk_size=args.chunk_size,
            iterations=args.iterations,
            learning_rate=args.learning_rate,
            depth=args.depth,
            l2_leaf_reg=args.l2_leaf_reg,
            ranker_loss=args.ranker_loss,
            ranker_alpha=args.ranker_alpha,
            switch_penalty=args.switch_penalty,
            jump_penalty=args.jump_penalty,
            cost_target=args.cost_target,
            ranker_target=args.ranker_target,
            use_selfcal_probes=not args.no_selfcal_probes,
            selfcal_max_probes=args.selfcal_max_probes,
            use_spatial_priors=not args.no_spatial_priors,
            use_shared_typewell_priors=args.use_shared_typewell_priors,
            shared_typewell_neighbours_path=args.shared_typewell_neighbours,
            progress_every=args.progress_every,
        )
    )


if __name__ == "__main__":
    main()
