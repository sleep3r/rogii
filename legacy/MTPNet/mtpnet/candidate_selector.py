from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, CatBoostRegressor, Pool

from .candidate_bank import build_candidate_bank_from_frames
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
class CandidateSelectorConfig:
    data_dir: Path = Path("data/train")
    output_dir: Path = Path("artifacts/candidate_selector_v0")
    n_folds: int = 5
    seed: int = 42
    k_wells: int = -1
    iterations: int = 900
    learning_rate: float = 0.035
    depth: int = 5
    l2_leaf_reg: float = 10.0
    progress_every: int = 50
    mode: str = "regressor"
    ranker_loss: str = "YetiRankPairwise"
    guard_gain_threshold: float = 0.0
    dangerous_guard_gain_threshold: float = 5.0
    residual_predictions_path: Path | None = Path(
        "artifacts/residual_stack_v0/oof_predictions.parquet"
    )
    traceback_candidates_path: Path | None = None


@dataclass
class SelectorDataset:
    rows: pd.DataFrame
    features: pd.DataFrame
    feature_columns: list[str]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _hidden_only(frame: pd.DataFrame) -> pd.DataFrame:
    if "TVT_input" not in frame.columns:
        return frame.copy()
    return frame[pd.to_numeric(frame["TVT_input"], errors="coerce").isna()].copy()


def _rmse_pair(pred: pd.Series | np.ndarray, true: pd.Series | np.ndarray) -> float:
    pred_arr = np.asarray(pred, dtype=np.float64)
    true_arr = np.asarray(true, dtype=np.float64)
    mask = np.isfinite(pred_arr) & np.isfinite(true_arr)
    if not mask.any():
        return float("nan")
    return float(np.sqrt(np.mean(np.square(pred_arr[mask] - true_arr[mask]))))


def _mse_pair(pred: pd.Series | np.ndarray, true: pd.Series | np.ndarray) -> float:
    pred_arr = np.asarray(pred, dtype=np.float64)
    true_arr = np.asarray(true, dtype=np.float64)
    mask = np.isfinite(pred_arr) & np.isfinite(true_arr)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.square(pred_arr[mask] - true_arr[mask])))


def _mean_abs(values: pd.Series | np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    return float(np.mean(np.abs(arr)))


def _p95_abs(values: pd.Series | np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    return float(np.percentile(np.abs(arr), 95.0))


def _slope(values: pd.Series | np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    if finite.sum() < 2:
        return 0.0
    return float(np.polyfit(np.arange(arr.size, dtype=np.float64)[finite], arr[finite], 1)[0])


def _std(values: pd.Series | np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return 0.0
    return float(arr.std())


def _candidate_action_features(candidate: str) -> dict[str, float]:
    features = {
        "feat_action_is_anchor": 0.0,
        "feat_action_is_shift": 0.0,
        "feat_action_is_drift": 0.0,
        "feat_action_is_slope": 0.0,
        "feat_action_is_residual": 0.0,
        "feat_action_is_softseg": 0.0,
        "feat_action_is_traceback": 0.0,
        "feat_action_is_template_envelope": 0.0,
        "feat_anchor_b2": 0.0,
        "feat_anchor_base": 0.0,
        "feat_anchor_a50": 0.0,
        "feat_shift_amount": 0.0,
        "feat_slope_delta": 0.0,
        "feat_action_is_dangerous": 0.0,
        "feat_action_global_risk": 0.0,
    }
    if candidate == "residual_stack_v0":
        features["feat_action_is_residual"] = 1.0
        return features
    if candidate.startswith("softseg_"):
        features["feat_action_is_softseg"] = 1.0
        features["feat_action_global_risk"] = 0.5
        return features
    if candidate.startswith("traceback_"):
        features["feat_action_is_traceback"] = 1.0
        features["feat_action_is_dangerous"] = 1.0
        features["feat_action_global_risk"] = 0.8
        return features
    if candidate.startswith("template_envelope_"):
        features["feat_action_is_template_envelope"] = 1.0
        features["feat_action_is_dangerous"] = 1.0
        features["feat_action_global_risk"] = 0.7
        return features
    if candidate.startswith("slope"):
        features["feat_action_is_slope"] = 1.0
        match = re.search(r"slope([+-]\d+(?:\.\d+)?)", candidate)
        features["feat_slope_delta"] = float(match.group(1)) if match else 0.0
        abs_delta = abs(features["feat_slope_delta"])
        features["feat_action_is_dangerous"] = float(abs_delta >= 2.0)
        features["feat_action_global_risk"] = min(abs_delta / 4.0, 1.0)
        return features
    anchor = candidate
    shift = re.search(r"^(b2|base|a_p50)_shift([+-]\d+(?:\.\d+)?)$", candidate)
    drift = re.search(r"^(b2|base|a_p50)_drift_end([+-]\d+(?:\.\d+)?)$", candidate)
    if shift:
        features["feat_action_is_shift"] = 1.0
        anchor = shift.group(1)
        features["feat_shift_amount"] = float(shift.group(2))
    elif drift:
        features["feat_action_is_drift"] = 1.0
        anchor = drift.group(1)
        features["feat_shift_amount"] = float(drift.group(2))
    else:
        features["feat_action_is_anchor"] = 1.0
    if anchor == "b2":
        features["feat_anchor_b2"] = 1.0
    elif anchor == "base":
        features["feat_anchor_base"] = 1.0
    elif anchor == "a_p50":
        features["feat_anchor_a50"] = 1.0
    abs_shift = abs(features["feat_shift_amount"])
    features["feat_action_is_dangerous"] = float(
        features["feat_action_is_dangerous"] or abs_shift >= 80.0
    )
    features["feat_action_global_risk"] = max(
        features["feat_action_global_risk"],
        min(abs_shift / 80.0, 1.0),
    )
    return features


def _well_context(well: pd.DataFrame, hidden: pd.DataFrame, bank: pd.DataFrame) -> dict[str, float]:
    known = well[pd.to_numeric(well.get("TVT_input", pd.Series(np.nan, index=well.index)), errors="coerce").notna()]
    gr_hidden = pd.to_numeric(hidden.get("GR", pd.Series(np.nan, index=hidden.index)), errors="coerce")
    gr_arr = gr_hidden.to_numpy(dtype=np.float64)
    gr_finite = gr_arr[np.isfinite(gr_arr)]
    disagreement = bank.groupby("id")["pred_tvt"].std(ddof=0)
    return {
        "feat_well_hidden_rows_log": float(np.log1p(len(hidden))),
        "feat_well_known_rows_log": float(np.log1p(len(known))),
        "feat_well_hidden_frac": float(len(hidden) / max(len(well), 1)),
        "feat_well_gr_nan_frac": float(gr_hidden.isna().mean()) if len(hidden) else 0.0,
        "feat_well_gr_volatility": float(np.mean(np.abs(np.diff(gr_finite)))) / 50.0
        if gr_finite.size >= 2
        else 0.0,
        "feat_well_known_slope": _slope(pd.to_numeric(known.get("TVT_input", pd.Series(dtype=float)), errors="coerce")),
        "feat_bank_disagreement_mean": float(disagreement.mean()) / 50.0
        if not disagreement.empty
        else 0.0,
        "feat_bank_disagreement_p95": float(disagreement.quantile(0.95)) / 50.0
        if not disagreement.empty
        else 0.0,
    }


def _known_boundary_context(well: pd.DataFrame, hidden: pd.DataFrame) -> dict[str, float]:
    if hidden.empty:
        return {
            "left_row": np.nan,
            "left_tvt": np.nan,
            "left_slope": 0.0,
            "right_row": np.nan,
            "right_tvt": np.nan,
            "right_slope": 0.0,
        }
    tvt_input = pd.to_numeric(
        well.get("TVT_input", pd.Series(np.nan, index=well.index)), errors="coerce"
    )
    row_idx = pd.to_numeric(well["row_idx"], errors="coerce")
    first_hidden = float(pd.to_numeric(hidden["row_idx"], errors="coerce").min())
    last_hidden = float(pd.to_numeric(hidden["row_idx"], errors="coerce").max())
    known = well.loc[tvt_input.notna()].copy()
    known["_row_idx_num"] = row_idx.loc[known.index].to_numpy(dtype=np.float64)
    known["_tvt_input_num"] = tvt_input.loc[known.index].to_numpy(dtype=np.float64)
    left = known[known["_row_idx_num"] < first_hidden].tail(1)
    right = known[known["_row_idx_num"] > last_hidden].head(1)
    left_hist = known[known["_row_idx_num"] < first_hidden].tail(5)
    right_hist = known[known["_row_idx_num"] > last_hidden].head(5)
    return {
        "left_row": float(left["_row_idx_num"].iloc[0]) if not left.empty else np.nan,
        "left_tvt": float(left["_tvt_input_num"].iloc[0]) if not left.empty else np.nan,
        "left_slope": _slope(left_hist["_tvt_input_num"]) if len(left_hist) >= 2 else 0.0,
        "right_row": float(right["_row_idx_num"].iloc[0]) if not right.empty else np.nan,
        "right_tvt": float(right["_tvt_input_num"].iloc[0]) if not right.empty else np.nan,
        "right_slope": _slope(right_hist["_tvt_input_num"]) if len(right_hist) >= 2 else 0.0,
    }


def _boundary_features(group: pd.DataFrame, boundaries: dict[str, float]) -> dict[str, float]:
    ordered = group.sort_values("row_idx")
    rows = pd.to_numeric(ordered["row_idx"], errors="coerce").to_numpy(dtype=np.float64)
    pred = pd.to_numeric(ordered["pred_tvt"], errors="coerce").to_numpy(dtype=np.float64)
    has_left = np.isfinite(boundaries["left_row"]) and np.isfinite(boundaries["left_tvt"])
    has_right = np.isfinite(boundaries["right_row"]) and np.isfinite(boundaries["right_tvt"])
    features = {
        "feat_boundary_has_left": float(has_left),
        "feat_boundary_has_right": float(has_right),
        "feat_boundary_left_jump": 0.0,
        "feat_boundary_right_jump": 0.0,
        "feat_bridge_mean_abs": 0.0,
        "feat_bridge_p95_abs": 0.0,
        "feat_endpoint_span_vs_boundary_span": 0.0,
    }
    if rows.size == 0 or pred.size == 0:
        return features
    bridge = np.full_like(pred, np.nan, dtype=np.float64)
    if has_left:
        left_extrap = boundaries["left_tvt"] + boundaries["left_slope"] * (rows - boundaries["left_row"])
        features["feat_boundary_left_jump"] = float(pred[0] - left_extrap[0]) / 50.0
        bridge = left_extrap
    if has_right:
        right_extrap = boundaries["right_tvt"] + boundaries["right_slope"] * (
            rows - boundaries["right_row"]
        )
        features["feat_boundary_right_jump"] = float(pred[-1] - right_extrap[-1]) / 50.0
        if not has_left:
            bridge = right_extrap
    if has_left and has_right and boundaries["right_row"] != boundaries["left_row"]:
        frac = (rows - boundaries["left_row"]) / (boundaries["right_row"] - boundaries["left_row"])
        bridge = boundaries["left_tvt"] + frac * (boundaries["right_tvt"] - boundaries["left_tvt"])
        boundary_span = boundaries["right_tvt"] - boundaries["left_tvt"]
        if pred.size >= 2:
            features["feat_endpoint_span_vs_boundary_span"] = float(
                (pred[-1] - pred[0]) - boundary_span
            ) / 100.0
    diff = pred - bridge
    features["feat_bridge_mean_abs"] = _mean_abs(diff) / 50.0
    features["feat_bridge_p95_abs"] = _p95_abs(diff) / 50.0
    return features


def build_selector_dataset(
    hidden_rows: pd.DataFrame,
    *,
    residual_predictions: pd.DataFrame | None = None,
    traceback_candidates: pd.DataFrame | None = None,
    progress_every: int = 0,
) -> SelectorDataset:
    raw = _add_missing_prior_columns(_ensure_ids(hidden_rows))
    if "step" not in raw.columns:
        raw["step"] = pd.to_numeric(raw["row_idx"], errors="coerce").astype(int)
    residual = residual_predictions.copy() if residual_predictions is not None else None
    if residual is not None and not residual.empty:
        residual["well_id"] = residual["well_id"].astype(str)
    traceback = traceback_candidates.copy() if traceback_candidates is not None else None
    if traceback is not None and not traceback.empty and "well_id" in traceback.columns:
        traceback["well_id"] = traceback["well_id"].astype(str)
    rows: list[dict[str, Any]] = []
    grouped_wells = list(raw.groupby("well_id", sort=True))
    total_wells = len(grouped_wells)
    for index, (well_id, well) in enumerate(grouped_wells, start=1):
        if progress_every > 0 and (index == 1 or index % progress_every == 0 or index == total_wells):
            print(
                f"[selector] built candidates for {index}/{total_wells} wells",
                file=sys.stderr,
                flush=True,
            )
        well_residual = None
        if residual is not None and not residual.empty and "well_id" in residual.columns:
            well_residual = residual[residual["well_id"].astype(str) == str(well_id)]
        well_traceback = None
        if traceback is not None and not traceback.empty and "well_id" in traceback.columns:
            well_traceback = traceback[traceback["well_id"].astype(str) == str(well_id)]
        bank = build_candidate_bank_from_frames(
            well,
            residual_predictions=well_residual,
            traceback_candidates=well_traceback,
        )
        hidden = _hidden_only(well)
        context = _well_context(well, hidden, bank)
        boundaries = _known_boundary_context(well, hidden)
        true_by_id = hidden.set_index("id")["TVT"] if "TVT" in hidden.columns else None
        b2_mse = (
            _mse_pair(hidden.get("b2_tvt", pd.Series(np.nan, index=hidden.index)), hidden["TVT"])
            if "TVT" in hidden.columns
            else np.nan
        )
        anchors = hidden.set_index("id")[
            [col for col in ("anchor_tvt", "b2_tvt", "base_tvt", "a_p50_tvt") if col in hidden.columns]
        ]
        bank_wide = bank.pivot_table(
            index="id",
            columns="candidate",
            values="pred_tvt",
            aggfunc="first",
        )
        bank_stats = pd.DataFrame(
            {
                "bank_median": bank_wide.median(axis=1),
                "bank_p25": bank_wide.quantile(0.25, axis=1),
                "bank_p75": bank_wide.quantile(0.75, axis=1),
            },
            index=bank_wide.index,
        )
        for candidate, group in bank.groupby("candidate", sort=True):
            pred = pd.to_numeric(group["pred_tvt"], errors="coerce").to_numpy(dtype=np.float64)
            by_id = group.set_index("id")["pred_tvt"]
            aligned_anchors = anchors.reindex(group["id"].astype(str))
            aligned_bank_stats = bank_stats.reindex(group["id"].astype(str))
            aligned_true = (
                true_by_id.reindex(group["id"].astype(str)) if true_by_id is not None else None
            )
            target_mse = (
                _mse_pair(group["pred_tvt"], aligned_true)
                if aligned_true is not None
                else np.nan
            )
            target_rmse = (
                float(np.sqrt(target_mse))
                if np.isfinite(target_mse)
                else np.nan
            )
            slope = _slope(pred)
            curvature = np.diff(np.diff(pred[np.isfinite(pred)])) if np.isfinite(pred).sum() >= 3 else np.asarray([])
            median_diff = by_id - aligned_bank_stats["bank_median"]
            p25 = aligned_bank_stats["bank_p25"].to_numpy(dtype=np.float64)
            p75 = aligned_bank_stats["bank_p75"].to_numpy(dtype=np.float64)
            inside_iqr = np.isfinite(pred) & np.isfinite(p25) & np.isfinite(p75) & (pred >= p25) & (pred <= p75)
            rec: dict[str, Any] = {
                "well_id": str(well_id),
                "candidate": str(candidate),
                "target_rmse": target_rmse,
                "target_mse": target_mse,
                "target_b2_mse": b2_mse,
                "target_gain_vs_b2_mse": b2_mse - target_mse
                if np.isfinite(b2_mse) and np.isfinite(target_mse)
                else np.nan,
                **context,
                **_candidate_action_features(str(candidate)),
                **_boundary_features(group, boundaries),
                "feat_pred_mean_delta_anchor": _mean_abs(
                    by_id - aligned_anchors.get("anchor_tvt", pd.Series(np.nan, index=by_id.index))
                )
                / 50.0,
                "feat_pred_std": _std(pred) / 50.0,
                "feat_pred_slope": slope,
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
            for name, column in (
                ("b2", "b2_tvt"),
                ("base", "base_tvt"),
                ("a50", "a_p50_tvt"),
            ):
                if column in aligned_anchors.columns:
                    diff = by_id - aligned_anchors[column]
                    rec[f"feat_mean_abs_to_{name}"] = _mean_abs(diff) / 50.0
                    rec[f"feat_p95_abs_to_{name}"] = _p95_abs(diff) / 50.0
                else:
                    rec[f"feat_mean_abs_to_{name}"] = 0.0
                    rec[f"feat_p95_abs_to_{name}"] = 0.0
            rows.append(rec)
    if not rows:
        raise ValueError("No candidate selector rows were built")
    frame = pd.DataFrame(rows)
    best_mse = frame.groupby("well_id")["target_mse"].transform("min")
    frame["target_regret_mse"] = frame["target_mse"] - best_mse
    feature_columns = [col for col in frame.columns if col.startswith("feat_")]
    assert_schema_safe_columns(feature_columns, context="CandidateSelector features")
    features = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    return SelectorDataset(rows=frame, features=features, feature_columns=feature_columns)


def _train_model(
    dataset: SelectorDataset,
    *,
    train_wells: list[str],
    config: CandidateSelectorConfig,
) -> CatBoostRegressor:
    assert_schema_safe_columns(dataset.feature_columns, context="CandidateSelector features")
    mask = dataset.rows["well_id"].astype(str).isin(set(train_wells)).to_numpy()
    target = np.log1p(
        pd.to_numeric(dataset.rows.loc[mask, "target_rmse"], errors="coerce").to_numpy(dtype=np.float64)
    )
    finite = np.isfinite(target)
    if not finite.any():
        raise ValueError("No finite selector targets in train fold")
    model = CatBoostRegressor(
        loss_function="RMSE",
        iterations=config.iterations,
        learning_rate=config.learning_rate,
        depth=config.depth,
        l2_leaf_reg=config.l2_leaf_reg,
        random_seed=config.seed,
        allow_writing_files=False,
        verbose=False,
    )
    model.fit(dataset.features.loc[mask].iloc[finite], target[finite])
    return model


def _train_cost_model(
    dataset: SelectorDataset,
    *,
    train_wells: list[str],
    config: CandidateSelectorConfig,
) -> CatBoostRegressor:
    assert_schema_safe_columns(dataset.feature_columns, context="CandidateSelector features")
    mask = dataset.rows["well_id"].astype(str).isin(set(train_wells)).to_numpy()
    target = np.log1p(
        pd.to_numeric(dataset.rows.loc[mask, "target_mse"], errors="coerce").to_numpy(
            dtype=np.float64
        )
    )
    finite = np.isfinite(target)
    if not finite.any():
        raise ValueError("No finite selector MSE targets in train fold")
    model = CatBoostRegressor(
        loss_function="RMSE",
        iterations=config.iterations,
        learning_rate=config.learning_rate,
        depth=config.depth,
        l2_leaf_reg=config.l2_leaf_reg,
        random_seed=config.seed,
        allow_writing_files=False,
        verbose=False,
    )
    model.fit(dataset.features.loc[mask].iloc[finite], target[finite])
    return model


def _train_ranker_model(
    dataset: SelectorDataset,
    *,
    train_wells: list[str],
    config: CandidateSelectorConfig,
) -> CatBoostRanker:
    assert_schema_safe_columns(dataset.feature_columns, context="CandidateSelector features")
    mask = dataset.rows["well_id"].astype(str).isin(set(train_wells)).to_numpy()
    target = -np.log1p(
        pd.to_numeric(dataset.rows.loc[mask, "target_regret_mse"], errors="coerce").to_numpy(
            dtype=np.float64
        )
    )
    finite = np.isfinite(target)
    if not finite.any():
        raise ValueError("No finite selector ranker targets in train fold")
    train_rows = dataset.rows.loc[mask].iloc[finite].copy()
    train_pool = Pool(
        dataset.features.loc[mask].iloc[finite],
        label=target[finite],
        group_id=train_rows["well_id"].astype(str).to_numpy(),
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
    model.fit(train_pool)
    return model


def _select_ranker_guard_wells(
    local: pd.DataFrame,
    *,
    guard_gain_threshold: float,
    dangerous_guard_gain_threshold: float,
) -> pd.DataFrame:
    selections: list[dict[str, Any]] = []
    for well_id, group in local.groupby("well_id", sort=True):
        ordered = group.sort_values(["ranker_score", "pred_mse"], ascending=[False, True])
        best = ordered.iloc[0]
        b2_rows = group[group["candidate"].astype(str) == "b2"]
        b2 = b2_rows.iloc[0] if not b2_rows.empty else ordered.sort_values("pred_mse").iloc[0]
        pred_gain = float(b2["pred_mse"] - best["pred_mse"])
        selected = best
        reason = "ranker"
        if str(best["candidate"]) != "b2":
            dangerous = float(best.get("feat_action_is_dangerous", 0.0)) > 0.0
            if dangerous and pred_gain < dangerous_guard_gain_threshold:
                selected = b2
                reason = "dangerous_guard"
            elif pred_gain < guard_gain_threshold:
                selected = b2
                reason = "weak_predicted_gain"
        selections.append(
            {
                "well_id": str(well_id),
                "selected_candidate": str(selected["candidate"]),
                "pred_mse": float(selected["pred_mse"]),
                "pred_gain_vs_b2_mse": pred_gain,
                "selected_true_rmse": float(selected["target_rmse"]),
                "selected_target_mse": float(selected.get("target_mse", np.nan)),
                "ranker_score": float(selected.get("ranker_score", np.nan)),
                "fold": int(selected.get("fold", -1)),
                "guard_reason": reason,
            }
        )
    return pd.DataFrame(selections)


def _selected_row_predictions(
    frame: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    residual_predictions: pd.DataFrame | None,
    traceback_candidates: pd.DataFrame | None = None,
    output_candidate_name: str = "candidate_selector_v0",
) -> pd.DataFrame:
    raw = _add_missing_prior_columns(_ensure_ids(frame))
    residual = residual_predictions.copy() if residual_predictions is not None else None
    if residual is not None and not residual.empty:
        residual["well_id"] = residual["well_id"].astype(str)
    traceback = traceback_candidates.copy() if traceback_candidates is not None else None
    if traceback is not None and not traceback.empty and "well_id" in traceback.columns:
        traceback["well_id"] = traceback["well_id"].astype(str)
    rows: list[pd.DataFrame] = []
    selected_map = dict(zip(selected["well_id"].astype(str), selected["selected_candidate"].astype(str)))
    for well_id, well in raw.groupby("well_id", sort=True):
        candidate = selected_map.get(str(well_id))
        if candidate is None:
            continue
        well_residual = None
        if residual is not None and not residual.empty and "well_id" in residual.columns:
            well_residual = residual[residual["well_id"].astype(str) == str(well_id)]
        well_traceback = None
        if traceback is not None and not traceback.empty and "well_id" in traceback.columns:
            well_traceback = traceback[traceback["well_id"].astype(str) == str(well_id)]
        bank = build_candidate_bank_from_frames(
            well,
            residual_predictions=well_residual,
            traceback_candidates=well_traceback,
        )
        part = bank[bank["candidate"].astype(str) == candidate].copy()
        if not part.empty:
            rows.append(part)
    if not rows:
        raise ValueError("Selector did not select any row predictions")
    out = pd.concat(rows, ignore_index=True)
    out["candidate"] = output_candidate_name
    out["selected_candidate"] = out.groupby("well_id")["selected_candidate"].transform("first") if "selected_candidate" in out.columns else np.nan
    return out


def _row_metrics(predictions: pd.DataFrame, hidden_rows: pd.DataFrame) -> dict[str, float]:
    pred = pd.to_numeric(predictions["pred_tvt"], errors="coerce")
    true = pd.to_numeric(predictions["TVT"], errors="coerce")
    hidden = _hidden_only(_add_missing_prior_columns(_ensure_ids(hidden_rows)))
    b2 = pd.to_numeric(hidden["b2_tvt"], errors="coerce")
    true_hidden = pd.to_numeric(hidden["TVT"], errors="coerce")
    return {
        "row_rmse": _rmse(pred.to_numpy(dtype=np.float64) - true.to_numpy(dtype=np.float64)),
        "mean_well_rmse": float(
            predictions.groupby("well_id")
            .apply(lambda g: _rmse_pair(g["pred_tvt"], g["TVT"]))
            .mean()
        ),
        "b2_row_rmse": _rmse(b2.to_numpy(dtype=np.float64) - true_hidden.to_numpy(dtype=np.float64)),
    }


def _write_report(output_dir: Path, metrics: dict[str, Any], selected: pd.DataFrame) -> None:
    title = (
        "CANDIDATE_SELECTOR_V1_RANKER_GUARD_REPORT"
        if metrics.get("mode") == "ranker_guard"
        else "CANDIDATE_SELECTOR_V0_REPORT"
    )
    lines = [
        title,
        "",
        "summary:",
        json.dumps(_json_safe(metrics), indent=2),
        "",
        "selected candidate counts:",
        selected["selected_candidate"].value_counts().to_string(),
    ]
    (output_dir / "selector_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_candidate_selector_from_frames(
    hidden_rows: pd.DataFrame,
    *,
    config: CandidateSelectorConfig,
    residual_predictions: pd.DataFrame | None = None,
    traceback_candidates: pd.DataFrame | None = None,
) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_name = (
        "candidate_selector_v1_ranker_guard"
        if config.mode == "ranker_guard"
        else "candidate_selector_v0"
    )
    print("[selector] building candidate-level dataset", file=sys.stderr, flush=True)
    dataset = build_selector_dataset(
        hidden_rows,
        residual_predictions=residual_predictions,
        traceback_candidates=traceback_candidates,
        progress_every=config.progress_every,
    )
    print(
        f"[selector] dataset rows={len(dataset.rows):,} wells={dataset.rows['well_id'].nunique()} "
        f"features={len(dataset.feature_columns)}",
        file=sys.stderr,
        flush=True,
    )
    folds = make_group_folds(dataset.rows["well_id"], n_folds=config.n_folds, seed=config.seed)
    prediction_parts: list[pd.DataFrame] = []
    selected_parts: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, Any]] = []
    for fold_idx, (train_wells, valid_wells) in enumerate(folds):
        print(
            f"[selector] fold {fold_idx + 1}/{len(folds)} train_wells={len(train_wells)} "
            f"valid_wells={len(valid_wells)}",
            file=sys.stderr,
            flush=True,
        )
        valid_mask = dataset.rows["well_id"].astype(str).isin(set(valid_wells)).to_numpy()
        base_cols = [
            "well_id",
            "candidate",
            "target_rmse",
            "target_mse",
            "target_regret_mse",
            "target_gain_vs_b2_mse",
            "feat_action_is_dangerous",
        ]
        local = dataset.rows.loc[valid_mask, [col for col in base_cols if col in dataset.rows.columns]].copy()
        local["fold"] = fold_idx
        prediction_parts.append(local)
        if config.mode == "ranker_guard":
            ranker = _train_ranker_model(dataset, train_wells=train_wells, config=config)
            cost_model = _train_cost_model(dataset, train_wells=train_wells, config=config)
            local["ranker_score"] = ranker.predict(dataset.features.loc[valid_mask])
            local["pred_log_mse"] = cost_model.predict(dataset.features.loc[valid_mask])
            local["pred_mse"] = np.expm1(local["pred_log_mse"].to_numpy(dtype=np.float64)).clip(
                min=0.0
            )
            selected = _select_ranker_guard_wells(
                local,
                guard_gain_threshold=config.guard_gain_threshold,
                dangerous_guard_gain_threshold=config.dangerous_guard_gain_threshold,
            )
        else:
            model = _train_model(dataset, train_wells=train_wells, config=config)
            local["pred_log_rmse"] = model.predict(dataset.features.loc[valid_mask])
            local["pred_rmse"] = np.expm1(local["pred_log_rmse"].to_numpy(dtype=np.float64))
            idx = local.groupby("well_id")["pred_rmse"].idxmin()
            selected = local.loc[
                idx, ["well_id", "candidate", "pred_rmse", "target_rmse", "target_mse", "fold"]
            ].copy()
            selected = selected.rename(
                columns={
                    "candidate": "selected_candidate",
                    "target_rmse": "selected_true_rmse",
                    "target_mse": "selected_target_mse",
                }
            )
            selected["guard_reason"] = "regressor"
        selected_parts.append(selected)
        fold_metrics.append(
            {
                "fold": fold_idx,
                "train_wells": train_wells,
                "valid_wells": valid_wells,
                "selected_mean_true_rmse": float(selected["selected_true_rmse"].mean()),
            }
        )
        print(
            f"[selector] fold {fold_idx + 1}/{len(folds)} selected_mean_true_rmse="
            f"{float(selected['selected_true_rmse'].mean()):.4f}",
            file=sys.stderr,
            flush=True,
        )
    candidate_predictions = pd.concat(prediction_parts, ignore_index=True)
    selected_wells = pd.concat(selected_parts, ignore_index=True)
    row_predictions = _selected_row_predictions(
        hidden_rows,
        selected_wells,
        residual_predictions=residual_predictions,
        traceback_candidates=traceback_candidates,
        output_candidate_name=candidate_name,
    )
    row_predictions = row_predictions.merge(
        selected_wells[["well_id", "selected_candidate", "fold"]],
        on="well_id",
        how="left",
        suffixes=("", "_selected"),
    )
    selected_metrics = _row_metrics(row_predictions, hidden_rows)
    b2_row_rmse = selected_metrics["b2_row_rmse"]
    oracle_row_rmse = float(
        np.sqrt(
            np.average(
                np.square(
                    dataset.rows.groupby("well_id")["target_rmse"].min().to_numpy(dtype=np.float64)
                ),
                weights=_hidden_only(_add_missing_prior_columns(_ensure_ids(hidden_rows))).groupby("well_id").size().reindex(
                    dataset.rows.groupby("well_id")["target_rmse"].min().index
                ).to_numpy(dtype=np.float64),
            )
        )
    )
    denominator = b2_row_rmse**2 - oracle_row_rmse**2
    metrics = {
        "candidate": candidate_name,
        "mode": config.mode,
        "rows": int(len(row_predictions)),
        "wells": int(row_predictions["well_id"].nunique()),
        "folds": int(len(fold_metrics)),
        "feature_columns": dataset.feature_columns,
        "selected": selected_metrics,
        "b2": {"row_rmse": b2_row_rmse},
        "oracle": {
            "row_rmse": oracle_row_rmse,
        },
        "mse_gain_capture": float(
            (b2_row_rmse**2 - selected_metrics["row_rmse"] ** 2) / denominator
        )
        if denominator > 0
        else float("nan"),
        "guard_counts": selected_wells.get("guard_reason", pd.Series(dtype=str)).value_counts().to_dict(),
        "fold_metrics": fold_metrics,
    }
    row_predictions.to_parquet(output_dir / "selector_row_predictions.parquet", index=False)
    selected_wells.to_csv(output_dir / "selector_well_predictions.csv", index=False)
    candidate_predictions.to_parquet(output_dir / "selector_candidate_predictions.parquet", index=False)
    dataset.rows.to_parquet(output_dir / "selector_mode_dataset.parquet", index=False)
    (output_dir / "selector_metrics.json").write_text(
        json.dumps(_json_safe(metrics), indent=2), encoding="utf-8"
    )
    _write_report(output_dir, metrics, selected_wells)
    return metrics


def run_candidate_selector(config: CandidateSelectorConfig) -> dict[str, Any]:
    print(f"[selector] loading training frame from {config.data_dir}", file=sys.stderr, flush=True)
    frame = load_training_frame(ResidualStackConfig(data_dir=config.data_dir, k_wells=config.k_wells))
    print(
        f"[selector] loaded frame rows={len(frame):,} wells={frame['well_id'].nunique()}",
        file=sys.stderr,
        flush=True,
    )
    print(
        f"[selector] loading residual predictions from {config.residual_predictions_path}",
        file=sys.stderr,
        flush=True,
    )
    residual = (
        pd.read_parquet(config.residual_predictions_path)
        if config.residual_predictions_path is not None and Path(config.residual_predictions_path).exists()
        else None
    )
    if residual is None:
        print("[selector] residual predictions missing; continuing without residual candidate", file=sys.stderr, flush=True)
    else:
        print(f"[selector] loaded residual rows={len(residual):,}", file=sys.stderr, flush=True)
    traceback = (
        pd.read_parquet(config.traceback_candidates_path)
        if config.traceback_candidates_path is not None
        and Path(config.traceback_candidates_path).exists()
        else None
    )
    if traceback is not None:
        print(f"[selector] loaded traceback candidates rows={len(traceback):,}", file=sys.stderr, flush=True)
    metrics = run_candidate_selector_from_frames(
        frame,
        config=config,
        residual_predictions=residual,
        traceback_candidates=traceback,
    )
    compact = {
        "candidate": metrics["candidate"],
        "rows": metrics["rows"],
        "wells": metrics["wells"],
        "folds": metrics["folds"],
        "selected": metrics["selected"],
        "b2": metrics["b2"],
        "oracle": metrics["oracle"],
        "fold_selected_mean_true_rmse": [
            {
                "fold": item["fold"],
                "train_wells": len(item["train_wells"]),
                "valid_wells": len(item["valid_wells"]),
                "selected_mean_true_rmse": item["selected_mean_true_rmse"],
            }
            for item in metrics["fold_metrics"]
        ],
    }
    print(json.dumps(_json_safe(compact), indent=2))
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train fold-safe candidate-bank selector")
    parser.add_argument("--data-dir", type=Path, default=CandidateSelectorConfig.data_dir)
    parser.add_argument("--output-dir", type=Path, default=CandidateSelectorConfig.output_dir)
    parser.add_argument("--n-folds", type=int, default=CandidateSelectorConfig.n_folds)
    parser.add_argument("--seed", type=int, default=CandidateSelectorConfig.seed)
    parser.add_argument("--k-wells", type=int, default=CandidateSelectorConfig.k_wells)
    parser.add_argument("--iterations", type=int, default=CandidateSelectorConfig.iterations)
    parser.add_argument("--learning-rate", type=float, default=CandidateSelectorConfig.learning_rate)
    parser.add_argument("--depth", type=int, default=CandidateSelectorConfig.depth)
    parser.add_argument("--l2-leaf-reg", type=float, default=CandidateSelectorConfig.l2_leaf_reg)
    parser.add_argument("--progress-every", type=int, default=CandidateSelectorConfig.progress_every)
    parser.add_argument("--residual-predictions", type=Path, default=CandidateSelectorConfig.residual_predictions_path)
    parser.add_argument("--traceback-candidates", type=Path, default=CandidateSelectorConfig.traceback_candidates_path)
    parser.add_argument("--mode", choices=["regressor", "ranker_guard"], default=CandidateSelectorConfig.mode)
    parser.add_argument("--ranker-loss", default=CandidateSelectorConfig.ranker_loss)
    parser.add_argument("--guard-gain-threshold", type=float, default=CandidateSelectorConfig.guard_gain_threshold)
    parser.add_argument(
        "--dangerous-guard-gain-threshold",
        type=float,
        default=CandidateSelectorConfig.dangerous_guard_gain_threshold,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_candidate_selector(
        CandidateSelectorConfig(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            n_folds=args.n_folds,
            seed=args.seed,
            k_wells=args.k_wells,
            iterations=args.iterations,
            learning_rate=args.learning_rate,
            depth=args.depth,
            l2_leaf_reg=args.l2_leaf_reg,
            progress_every=args.progress_every,
            residual_predictions_path=args.residual_predictions,
            traceback_candidates_path=args.traceback_candidates,
            mode=args.mode,
            ranker_loss=args.ranker_loss,
            guard_gain_threshold=args.guard_gain_threshold,
            dangerous_guard_gain_threshold=args.dangerous_guard_gain_threshold,
        )
    )


if __name__ == "__main__":
    main()
