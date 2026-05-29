"""Discrete global-offset TVT candidate experiment.

This is the small, literal version of the Kaggle discussion idea:

    TVT[i] = TVT_anchor - (Z[i] - Z_anchor) + offset * (i - anchor)

Instead of searching an unconstrained TVT path, we search a small fixed grid
of formation-offset slopes.  The train-only target/oracle chooses the best
offset on hidden rows; deployable selectors can only use test-safe geometry,
known ``TVT_input``, GR and typewell GR match features.
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
from catboost import CatBoostRegressor

from .residual_stack import make_group_folds
from .schema_safe import assert_schema_safe_columns


@dataclass(frozen=True)
class DiscreteOffsetConfig:
    data_dir: Path = Path("data/train")
    output_dir: Path = Path("artifacts/discrete_offset_v0")
    offset_grid: str = "-0.16:0.16:0.001"
    n_folds: int = 5
    seed: int = 42
    k_wells: int = -1
    iterations: int = 300
    learning_rate: float = 0.04
    depth: int = 5
    l2_leaf_reg: float = 8.0
    progress_every: int = 100
    include_shuffled: bool = True
    n_panel_wells: int = 6


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def parse_offset_grid(text: str) -> np.ndarray:
    """Parse comma list or inclusive ``start:stop:step`` grid."""
    value = str(text).strip()
    if ":" in value and "," not in value:
        parts = [float(part) for part in value.split(":")]
        if len(parts) != 3:
            raise ValueError("offset range must be start:stop:step")
        start, stop, step = parts
        if step == 0:
            raise ValueError("offset step must be non-zero")
        # Inclusive with small epsilon so 0.12:... contains the right edge.
        n = int(np.floor((stop - start) / step + 0.5)) + 1
        grid = start + step * np.arange(max(n, 1), dtype=np.float64)
        grid = grid[(grid >= min(start, stop) - 1e-12) & (grid <= max(start, stop) + 1e-12)]
    else:
        grid = np.asarray([float(part) for part in value.split(",") if part.strip()], dtype=np.float64)
    if grid.size == 0:
        raise ValueError("offset grid is empty")
    return np.asarray(sorted(set(np.round(grid, 10).tolist())), dtype=np.float64)


def _rmse_from_err(err: np.ndarray) -> float:
    err = np.asarray(err, dtype=np.float64)
    err = err[np.isfinite(err)]
    if err.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(err * err)))


def _metrics_from_errors(errors_by_well: list[np.ndarray]) -> dict[str, Any]:
    sq_parts: list[np.ndarray] = []
    well_rmse: list[float] = []
    rows = 0
    for err in errors_by_well:
        e = np.asarray(err, dtype=np.float64)
        e = e[np.isfinite(e)]
        if e.size == 0:
            continue
        sq_parts.append(e * e)
        well_rmse.append(float(np.sqrt(np.mean(e * e))))
        rows += int(e.size)
    if not sq_parts:
        return {"rows": 0, "wells": 0, "row_rmse": float("nan")}
    sq = np.concatenate(sq_parts)
    wr = np.asarray(well_rmse, dtype=np.float64)
    return {
        "rows": int(rows),
        "wells": int(wr.size),
        "row_rmse": float(np.sqrt(np.mean(sq))),
        "mean_well_rmse": float(np.mean(wr)),
        "p50_well_rmse": float(np.quantile(wr, 0.50)),
        "p90_well_rmse": float(np.quantile(wr, 0.90)),
        "p95_well_rmse": float(np.quantile(wr, 0.95)),
        "worst_well_rmse": float(np.max(wr)),
    }


def _safe_stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(aa) & np.isfinite(bb)
    if mask.sum() < 3:
        return 0.0
    aa = aa[mask]
    bb = bb[mask]
    if float(np.std(aa)) < 1e-9 or float(np.std(bb)) < 1e-9:
        return 0.0
    return float(np.corrcoef(aa, bb)[0, 1])


def _safe_mad(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(aa) & np.isfinite(bb)
    if not mask.any():
        return 0.0
    return float(np.median(np.abs(aa[mask] - bb[mask])))


@dataclass
class _WellArrays:
    well_id: str
    row_idx: np.ndarray
    ids: np.ndarray
    md: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    gr: np.ndarray
    tvt: np.ndarray
    tvt_input: np.ndarray
    anchor_row: int
    hidden_idx: np.ndarray
    fold: int
    typewell_tvt: np.ndarray
    typewell_gr: np.ndarray


def _load_typewell(data_dir: Path, well_id: str) -> tuple[np.ndarray, np.ndarray]:
    path = data_dir / f"{well_id}__typewell.csv"
    if not path.exists():
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    frame = pd.read_csv(path)
    if "TVT" not in frame.columns or "GR" not in frame.columns:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    tvt = pd.to_numeric(frame["TVT"], errors="coerce").to_numpy(dtype=np.float64)
    gr = pd.to_numeric(frame["GR"], errors="coerce").to_numpy(dtype=np.float64)
    mask = np.isfinite(tvt) & np.isfinite(gr)
    if mask.sum() < 3:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    order = np.argsort(tvt[mask])
    return tvt[mask][order], gr[mask][order]


def _load_wells(data_dir: Path, *, k_wells: int, fold_of_well: dict[str, int]) -> list[_WellArrays]:
    paths = sorted(Path(data_dir).glob("*__horizontal_well.csv"))
    if k_wells > 0:
        paths = paths[:k_wells]
    wells: list[_WellArrays] = []
    for path in paths:
        well_id = path.name.replace("__horizontal_well.csv", "")
        if well_id not in fold_of_well:
            continue
        frame = pd.read_csv(path)
        n = len(frame)
        row_idx = np.arange(n, dtype=np.int64)
        ids = np.asarray([f"{well_id}_{int(i)}" for i in row_idx], dtype=object)
        values: dict[str, np.ndarray] = {}
        for col in ("MD", "X", "Y", "Z", "GR", "TVT", "TVT_input"):
            if col in frame.columns:
                values[col] = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float64)
            else:
                values[col] = np.full(n, np.nan, dtype=np.float64)
        tvt_input = values["TVT_input"]
        known = np.flatnonzero(np.isfinite(tvt_input))
        hidden = np.flatnonzero((~np.isfinite(tvt_input)) & np.isfinite(values["TVT"]))
        if known.size < 2 or hidden.size < 2:
            continue
        anchor = int(known[-1])
        # Real train/test shape is a single hidden tail. Keep the invariant
        # explicit so a future schema change does not silently misalign paths.
        hidden = hidden[hidden > anchor]
        if hidden.size < 2:
            continue
        tw_tvt, tw_gr = _load_typewell(Path(data_dir), well_id)
        wells.append(
            _WellArrays(
                well_id=well_id,
                row_idx=row_idx,
                ids=ids,
                md=values["MD"],
                x=values["X"],
                y=values["Y"],
                z=values["Z"],
                gr=values["GR"],
                tvt=values["TVT"],
                tvt_input=tvt_input,
                anchor_row=anchor,
                hidden_idx=hidden,
                fold=fold_of_well[well_id],
                typewell_tvt=tw_tvt,
                typewell_gr=tw_gr,
            )
        )
    return wells


def _known_dc_stats(well: _WellArrays, *, tail_window: int = 500) -> dict[str, float]:
    known = np.flatnonzero(np.isfinite(well.tvt_input) & (np.arange(well.tvt_input.size) <= well.anchor_row))
    known_tail = known[known >= max(0, well.anchor_row - tail_window)]
    c_tail = well.tvt_input[known_tail] + well.z[known_tail]
    dc_tail = np.diff(c_tail)
    dc_tail = dc_tail[np.isfinite(dc_tail)]
    c_all = well.tvt_input[known] + well.z[known]
    dc_all = np.diff(c_all)
    dc_all = dc_all[np.isfinite(dc_all)]

    def _median(arr: np.ndarray) -> float:
        return float(np.median(arr)) if arr.size else 0.0

    def _mean(arr: np.ndarray) -> float:
        return float(np.mean(arr)) if arr.size else 0.0

    def _std(arr: np.ndarray) -> float:
        return float(np.std(arr)) if arr.size > 1 else 0.0

    return {
        "known_c0_median500": _median(dc_tail),
        "known_c0_mean500": _mean(dc_tail),
        "known_c0_std500": _std(dc_tail),
        "known_c0_median_all": _median(dc_all),
        "known_c0_mean_all": _mean(dc_all),
        "known_c0_std_all": _std(dc_all),
    }


def _candidate_path(well: _WellArrays, offset: float) -> np.ndarray:
    idx = well.hidden_idx
    anchor = well.anchor_row
    return (
        float(well.tvt_input[anchor])
        - (well.z[idx] - float(well.z[anchor]))
        + float(offset) * (idx - anchor)
    )


def _sample_typewell_gr(well: _WellArrays, path_tvt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if well.typewell_tvt.size < 3:
        return np.full_like(path_tvt, np.nan, dtype=np.float64), np.zeros_like(path_tvt, dtype=bool)
    lo = float(np.min(well.typewell_tvt))
    hi = float(np.max(well.typewell_tvt))
    inside = np.isfinite(path_tvt) & (path_tvt >= lo) & (path_tvt <= hi)
    sampled = np.full_like(path_tvt, np.nan, dtype=np.float64)
    if inside.any():
        sampled[inside] = np.interp(path_tvt[inside], well.typewell_tvt, well.typewell_gr)
    return sampled, inside


def _candidate_gr_features(
    well: _WellArrays, path_tvt: np.ndarray, *, rng: np.random.Generator | None = None
) -> dict[str, float]:
    hidden_gr = well.gr[well.hidden_idx].astype(np.float64)
    if rng is not None:
        finite = np.flatnonzero(np.isfinite(hidden_gr))
        if finite.size > 1:
            shuffled = hidden_gr.copy()
            shuffled[finite] = shuffled[rng.permutation(finite)]
            hidden_gr = shuffled
    sampled, inside = _sample_typewell_gr(well, path_tvt)
    mask = np.isfinite(hidden_gr) & np.isfinite(sampled)
    if mask.sum() < 3:
        return {
            "feat_gr_corr": 0.0,
            "feat_gr_dcorr": 0.0,
            "feat_gr_mad": 0.0,
            "feat_gr_valid_frac": 0.0,
            "feat_tvt_oob_frac": float(1.0 - inside.mean()) if inside.size else 1.0,
        }
    hg = hidden_gr[mask]
    tg = sampled[mask]
    return {
        "feat_gr_corr": _safe_corr(hg, tg),
        "feat_gr_dcorr": _safe_corr(np.diff(hg), np.diff(tg)),
        "feat_gr_mad": _safe_mad(hg, tg) / 100.0,
        "feat_gr_valid_frac": float(mask.mean()),
        "feat_tvt_oob_frac": float(1.0 - inside.mean()) if inside.size else 1.0,
    }


def _well_features(well: _WellArrays, dc: dict[str, float]) -> dict[str, float]:
    idx = well.hidden_idx
    z_hidden = well.z[idx]
    dz_hidden = np.gradient(well.z)[idx] if well.z.size >= 2 else np.zeros_like(idx, dtype=np.float64)
    gr_hidden = well.gr[idx]
    gr_known = well.gr[np.isfinite(well.tvt_input)]
    z_stats = _safe_stats(z_hidden)
    dz_stats = _safe_stats(dz_hidden)
    gr_h = _safe_stats(gr_hidden)
    gr_k = _safe_stats(gr_known)
    anchor = well.anchor_row
    return {
        "feat_hidden_n_log": float(np.log1p(idx.size)),
        "feat_hidden_n": float(idx.size),
        "feat_anchor_row_frac": float(anchor / max(well.row_idx.size - 1, 1)),
        "feat_hidden_z_span": float(z_hidden[-1] - z_hidden[0]) / 100.0,
        "feat_hidden_md_span": float(well.md[idx[-1]] - well.md[idx[0]]) / 1000.0,
        "feat_hidden_xy_span": float(
            np.hypot(well.x[idx[-1]] - well.x[idx[0]], well.y[idx[-1]] - well.y[idx[0]])
        )
        / 1000.0,
        "feat_hidden_z_mean": z_stats["mean"] / 10000.0,
        "feat_hidden_z_std": z_stats["std"] / 100.0,
        "feat_hidden_dz_mean": dz_stats["mean"],
        "feat_hidden_dz_std": dz_stats["std"],
        "feat_hidden_dz_min": dz_stats["min"],
        "feat_hidden_dz_max": dz_stats["max"],
        "feat_hidden_gr_mean": gr_h["mean"] / 100.0,
        "feat_hidden_gr_std": gr_h["std"] / 50.0,
        "feat_known_gr_mean": gr_k["mean"] / 100.0,
        "feat_known_gr_std": gr_k["std"] / 50.0,
        "feat_anchor_tvt": float(well.tvt_input[anchor]) / 10000.0,
        "feat_anchor_z": float(well.z[anchor]) / 10000.0,
        "feat_anchor_C": float(well.tvt_input[anchor] + well.z[anchor]) / 10000.0,
        "feat_known_c0_median500": dc["known_c0_median500"],
        "feat_known_c0_mean500": dc["known_c0_mean500"],
        "feat_known_c0_std500": dc["known_c0_std500"],
        "feat_known_c0_median_all": dc["known_c0_median_all"],
        "feat_known_c0_mean_all": dc["known_c0_mean_all"],
        "feat_known_c0_std_all": dc["known_c0_std_all"],
    }


def _selfcal_offset_features(well: _WellArrays, offset: float) -> dict[str, float]:
    """Score an offset on pseudo-hidden windows inside known ``TVT_input``.

    This is deployable: test wells also expose non-null ``TVT_input`` before the
    hidden tail.  The feature asks: if this offset had been used to predict a
    recent known window from the row just before it, how badly would it drift?
    """
    known = np.flatnonzero(np.isfinite(well.tvt_input) & (np.arange(well.tvt_input.size) <= well.anchor_row))
    if known.size < 12:
        return {
            "feat_selfcal_rmse_mean": 0.0,
            "feat_selfcal_rmse_min": 0.0,
            "feat_selfcal_rmse_tail": 0.0,
            "feat_selfcal_bias_mean": 0.0,
            "feat_selfcal_last_err_mean": 0.0,
            "feat_selfcal_n_windows": 0.0,
        }
    # Real data uses a contiguous known prefix.  Still derive windows from the
    # known array so synthetic/test fixtures with short prefixes work.
    prefix_end = int(known[-1]) + 1
    n_known = int(known.size)
    lengths = sorted(
        {
            max(8, min(64, n_known // 4)),
            max(8, min(256, n_known // 2)),
            max(8, min(512, (3 * n_known) // 4)),
        }
    )
    windows: list[tuple[int, int]] = []
    for length in lengths:
        end = prefix_end
        start = end - int(length)
        if start >= 1 and end - start >= 4:
            windows.append((start, end))
        # Add one earlier window of the same length to avoid using only the
        # last few known rows when enough prefix exists.
        earlier_end = max(2, start)
        earlier_start = earlier_end - int(length)
        if earlier_start >= 1 and earlier_end - earlier_start >= 4:
            windows.append((earlier_start, earlier_end))
    rmses: list[float] = []
    biases: list[float] = []
    last_errs: list[float] = []
    for start, end in windows:
        probe_anchor = start - 1
        rows = np.arange(start, end, dtype=np.int64)
        truth = well.tvt_input[rows]
        if not np.isfinite(well.tvt_input[probe_anchor]) or not np.isfinite(truth).all():
            continue
        pred = (
            float(well.tvt_input[probe_anchor])
            - (well.z[rows] - float(well.z[probe_anchor]))
            + float(offset) * (rows - probe_anchor)
        )
        err = pred - truth
        rmses.append(_rmse_from_err(err))
        biases.append(float(np.mean(err)))
        last_errs.append(float(err[-1]))
    if not rmses:
        return {
            "feat_selfcal_rmse_mean": 0.0,
            "feat_selfcal_rmse_min": 0.0,
            "feat_selfcal_rmse_tail": 0.0,
            "feat_selfcal_bias_mean": 0.0,
            "feat_selfcal_last_err_mean": 0.0,
            "feat_selfcal_n_windows": 0.0,
        }
    return {
        "feat_selfcal_rmse_mean": float(np.mean(rmses)),
        "feat_selfcal_rmse_min": float(np.min(rmses)),
        "feat_selfcal_rmse_tail": float(rmses[0]),
        "feat_selfcal_bias_mean": float(np.mean(biases)),
        "feat_selfcal_last_err_mean": float(np.mean(last_errs)),
        "feat_selfcal_n_windows": float(len(rmses)),
    }


def build_discrete_offset_dataset(
    wells: list[_WellArrays],
    *,
    offset_grid: np.ndarray,
    seed: int,
    shuffled: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Build one row per well × offset with train-only candidate losses."""
    records: list[dict[str, Any]] = []
    for well in wells:
        dc = _known_dc_stats(well)
        base_features = _well_features(well, dc)
        rng = np.random.default_rng(seed + abs(hash(well.well_id)) % 1_000_000) if shuffled else None
        candidate_rows: list[dict[str, Any]] = []
        truth = well.tvt[well.hidden_idx]
        for offset in offset_grid:
            pred = _candidate_path(well, float(offset))
            err = pred - truth
            mse = float(np.nanmean(err * err))
            rmse = float(np.sqrt(mse))
            gr_feats = _candidate_gr_features(well, pred, rng=rng)
            selfcal_feats = _selfcal_offset_features(well, float(offset))
            rec = {
                "well_id": well.well_id,
                "fold": well.fold,
                "offset": float(offset),
                "candidate": f"discrete_offset_{float(offset):+.3f}",
                "candidate_mse": mse,
                "candidate_rmse": rmse,
                "rows": int(well.hidden_idx.size),
                "feat_offset": float(offset),
                "feat_abs_offset": abs(float(offset)),
                "feat_offset_minus_known_median500": float(offset - dc["known_c0_median500"]),
                "feat_offset_minus_known_mean500": float(offset - dc["known_c0_mean500"]),
                "feat_path_tvt_span": float(pred[-1] - pred[0]) / 100.0 if pred.size else 0.0,
                "feat_path_tvt_min": float(np.nanmin(pred)) / 10000.0,
                "feat_path_tvt_max": float(np.nanmax(pred)) / 10000.0,
                **base_features,
                **gr_feats,
                **selfcal_feats,
            }
            candidate_rows.append(rec)
        if not candidate_rows:
            continue
        best_mse = min(row["candidate_mse"] for row in candidate_rows)
        for row in candidate_rows:
            row["target_regret"] = float(row["candidate_mse"] - best_mse)
            row["target_log_mse"] = float(np.log1p(row["candidate_mse"]))
            row["is_best_offset"] = int(row["candidate_mse"] <= best_mse + 1e-12)
            records.append(row)
    if not records:
        raise ValueError("No discrete offset candidate rows could be built")
    frame = pd.DataFrame(records).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    feature_columns = [col for col in frame.columns if col.startswith("feat_")]
    assert_schema_safe_columns(feature_columns, context="DiscreteOffset features")
    return frame, feature_columns


def _select_offsets_from_candidate_rows(
    rows: pd.DataFrame,
    *,
    score_column: str,
    minimize: bool,
) -> dict[str, float]:
    chosen: dict[str, float] = {}
    for wid, group in rows.groupby("well_id", sort=False):
        ordered = group.sort_values(score_column, ascending=minimize)
        chosen[str(wid)] = float(ordered.iloc[0]["offset"])
    return chosen


def _select_known_grid(wells: list[_WellArrays], offset_grid: np.ndarray, key: str) -> dict[str, float]:
    chosen: dict[str, float] = {}
    for well in wells:
        dc = _known_dc_stats(well)
        value = dc[key]
        chosen[well.well_id] = float(offset_grid[np.argmin(np.abs(offset_grid - value))])
    return chosen


def _evaluate_selected_offsets(
    wells: list[_WellArrays],
    selected: dict[str, float],
    *,
    candidate: str,
) -> dict[str, Any]:
    errors: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    offsets: list[float] = []
    for well in wells:
        if well.well_id not in selected:
            continue
        offset = float(selected[well.well_id])
        pred = _candidate_path(well, offset)
        truth = well.tvt[well.hidden_idx]
        err = pred - truth
        errors.append(err)
        offsets.append(offset)
        rows.append({"well_id": well.well_id, "offset": offset, "rmse": _rmse_from_err(err)})
    metrics = _metrics_from_errors(errors)
    metrics.update(
        {
            "candidate": candidate,
            "selected_offset_mean": float(np.mean(offsets)) if offsets else float("nan"),
            "selected_offset_std": float(np.std(offsets)) if offsets else float("nan"),
        }
    )
    return metrics


def _materialise_predictions(
    wells: list[_WellArrays],
    selected: dict[str, float],
    *,
    candidate: str,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for well in wells:
        if well.well_id not in selected:
            continue
        pred = _candidate_path(well, float(selected[well.well_id]))
        parts.append(
            pd.DataFrame(
                {
                    "id": well.ids[well.hidden_idx],
                    "well_id": well.well_id,
                    "row_idx": well.hidden_idx.astype(np.int64),
                    "pred_tvt": pred.astype(np.float64),
                    "candidate": candidate,
                    "selected_offset": float(selected[well.well_id]),
                }
            )
        )
    if not parts:
        return pd.DataFrame(columns=["id", "well_id", "row_idx", "pred_tvt", "candidate", "selected_offset"])
    return pd.concat(parts, ignore_index=True)


def _train_oof_cost_selector(
    candidates: pd.DataFrame,
    feature_columns: list[str],
    *,
    config: DiscreteOffsetConfig,
    shuffled_candidates: pd.DataFrame | None = None,
) -> tuple[dict[str, float], dict[str, float], pd.DataFrame]:
    """Train fold-safe cost model and select lowest predicted log-MSE offset."""
    pred_frames: list[pd.DataFrame] = []
    selected: dict[str, float] = {}
    selected_shuffled: dict[str, float] = {}
    all_folds = sorted(candidates["fold"].astype(int).unique().tolist())
    for fold in all_folds:
        train_mask = candidates["fold"].to_numpy(dtype=int) != int(fold)
        valid_mask = candidates["fold"].to_numpy(dtype=int) == int(fold)
        if not train_mask.any() or not valid_mask.any():
            continue
        print(
            f"[discrete-offset] fold {fold + 1}/{len(all_folds)} "
            f"train_rows={int(train_mask.sum())} valid_rows={int(valid_mask.sum())}",
            file=sys.stderr,
            flush=True,
        )
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
        model.fit(candidates.loc[train_mask, feature_columns], candidates.loc[train_mask, "target_log_mse"])
        valid = candidates.loc[valid_mask].copy()
        valid["pred_log_mse"] = model.predict(valid[feature_columns])
        pred_frames.append(valid)
        selected.update(_select_offsets_from_candidate_rows(valid, score_column="pred_log_mse", minimize=True))
        if shuffled_candidates is not None:
            shuf_valid = shuffled_candidates.loc[valid_mask].copy()
            shuf_valid["pred_log_mse"] = model.predict(shuf_valid[feature_columns])
            selected_shuffled.update(
                _select_offsets_from_candidate_rows(shuf_valid, score_column="pred_log_mse", minimize=True)
            )
    if not pred_frames:
        raise ValueError("No OOF cost predictions were produced")
    return selected, selected_shuffled, pd.concat(pred_frames, ignore_index=True)


def _offset_oracle_from_rows(candidates: pd.DataFrame) -> dict[str, float]:
    return _select_offsets_from_candidate_rows(candidates, score_column="candidate_mse", minimize=True)


def _best_gr_score_from_rows(candidates: pd.DataFrame) -> dict[str, float]:
    # Simple deployable no-train selector: favor candidate whose sampled typewell GR
    # has the best correlation and low MAD.
    rows = candidates.copy()
    rows["gr_score"] = (
        pd.to_numeric(rows["feat_gr_corr"], errors="coerce").fillna(0.0)
        + 0.5 * pd.to_numeric(rows["feat_gr_dcorr"], errors="coerce").fillna(0.0)
        - 0.2 * pd.to_numeric(rows["feat_gr_mad"], errors="coerce").fillna(0.0)
        - 0.5 * pd.to_numeric(rows["feat_tvt_oob_frac"], errors="coerce").fillna(1.0)
    )
    return _select_offsets_from_candidate_rows(rows, score_column="gr_score", minimize=False)


def run_discrete_offset(config: DiscreteOffsetConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    fig_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(config.data_dir)
    paths = sorted(data_dir.glob("*__horizontal_well.csv"))
    if config.k_wells > 0:
        paths = paths[: config.k_wells]
    well_ids = [path.name.replace("__horizontal_well.csv", "") for path in paths]
    folds = make_group_folds(well_ids, n_folds=config.n_folds, seed=config.seed)
    fold_of_well = {str(w): idx for idx, (_, valid) in enumerate(folds) for w in valid}
    offset_grid = parse_offset_grid(config.offset_grid)
    wells = _load_wells(data_dir, k_wells=config.k_wells, fold_of_well=fold_of_well)
    print(
        f"[discrete-offset] wells={len(wells)} offsets={len(offset_grid)} "
        f"grid=[{offset_grid[0]:.4f},{offset_grid[-1]:.4f}]",
        file=sys.stderr,
        flush=True,
    )
    candidates, feature_columns = build_discrete_offset_dataset(
        wells, offset_grid=offset_grid, seed=config.seed, shuffled=False
    )
    shuffled_candidates = None
    if config.include_shuffled:
        shuffled_candidates, _ = build_discrete_offset_dataset(
            wells, offset_grid=offset_grid, seed=config.seed, shuffled=True
        )
    candidates.to_parquet(output_dir / "discrete_offset_candidates.parquet", index=False)

    known_median = _select_known_grid(wells, offset_grid, "known_c0_median500")
    known_mean = _select_known_grid(wells, offset_grid, "known_c0_mean500")
    oracle = _offset_oracle_from_rows(candidates)
    gr_selected = _best_gr_score_from_rows(candidates)
    gr_shuffled_selected = (
        _best_gr_score_from_rows(shuffled_candidates) if shuffled_candidates is not None else {}
    )
    model_selected, model_shuffled_selected, oof_candidate_scores = _train_oof_cost_selector(
        candidates,
        feature_columns,
        config=config,
        shuffled_candidates=shuffled_candidates,
    )
    oof_candidate_scores.to_parquet(output_dir / "discrete_offset_oof_candidate_scores.parquet", index=False)

    selected_predictions = _materialise_predictions(
        wells, model_selected, candidate="discrete_offset_cost_selector"
    )
    selected_predictions.to_parquet(output_dir / "discrete_offset_oof_predictions.parquet", index=False)
    oracle_predictions = _materialise_predictions(wells, oracle, candidate="discrete_offset_grid_oracle")
    oracle_predictions.to_parquet(output_dir / "discrete_offset_grid_oracle_predictions.parquet", index=False)

    candidate_metrics = [
        _evaluate_selected_offsets(wells, known_median, candidate="known_median500_snapped"),
        _evaluate_selected_offsets(wells, known_mean, candidate="known_mean500_snapped"),
        _evaluate_selected_offsets(wells, gr_selected, candidate="gr_match_selector"),
        _evaluate_selected_offsets(wells, model_selected, candidate="cost_selector_oof"),
        _evaluate_selected_offsets(wells, oracle, candidate="grid_oracle"),
    ]
    if shuffled_candidates is not None:
        candidate_metrics.insert(
            4,
            _evaluate_selected_offsets(wells, gr_shuffled_selected, candidate="shuffled_gr_match_selector"),
        )
        candidate_metrics.insert(
            5,
            _evaluate_selected_offsets(
                wells, model_shuffled_selected, candidate="shuffled_cost_selector_oof"
            ),
        )
    metrics: dict[str, Any] = {
        "experiment": "discrete_offset_v0",
        "config": asdict(config),
        "offset_grid": offset_grid.tolist(),
        "wells": int(len(wells)),
        "candidate_rows": int(len(candidates)),
        "feature_columns": feature_columns,
        "candidates": candidate_metrics,
        "best_deployable": min(
            [m for m in candidate_metrics if "oracle" not in str(m["candidate"])],
            key=lambda item: item.get("row_rmse", float("inf")),
        ),
        "grid_oracle": next(item for item in candidate_metrics if item["candidate"] == "grid_oracle"),
    }
    (output_dir / "discrete_offset_metrics.json").write_text(
        json.dumps(_json_safe(metrics), indent=2), encoding="utf-8"
    )
    _write_report(output_dir, metrics)
    _write_figures(output_dir, metrics, wells, oracle, model_selected, known_median)
    return metrics


def _write_report(output_dir: Path, metrics: dict[str, Any]) -> None:
    lines = [
        "# DISCRETE_OFFSET_V0",
        "",
        "Fixed-grid implementation of the `cumsum(-dZ + offset)` Kaggle idea.",
        "Candidate generation is test-safe: only `MD/X/Y/Z/GR/TVT_input` and",
        "the well's typewell `TVT/GR` are used. Hidden `TVT` is used only for",
        "train labels, oracle diagnostics, and OOF scoring.",
        "",
        "## summary",
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
            "## interpretation",
            "",
            "- `grid_oracle` is the direct ceiling for the fixed offset grid.",
            "- `known_*_snapped` shows whether the known prefix offset transfers.",
            "- `gr_match_selector` is a no-train constrained typewell-GR selector.",
            "- `cost_selector_oof` is fold-safe learned selection over offset candidates.",
            "",
            "## artifacts",
            "",
            "- `discrete_offset_candidates.parquet`",
            "- `discrete_offset_oof_candidate_scores.parquet`",
            "- `discrete_offset_oof_predictions.parquet`",
            "- `discrete_offset_grid_oracle_predictions.parquet`",
            "- `figures/`",
            "",
            "## raw metrics",
            "",
            "```json",
            json.dumps(_json_safe(metrics), indent=2),
            "```",
        ]
    )
    (output_dir / "discrete_offset_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_figures(
    output_dir: Path,
    metrics: dict[str, Any],
    wells: list[_WellArrays],
    oracle: dict[str, float],
    selected: dict[str, float],
    known: dict[str, float],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting is optional in lean envs
        print(f"[discrete-offset] plotting skipped: {exc}", file=sys.stderr, flush=True)
        return
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)
    items = metrics["candidates"]
    labels = [str(item["candidate"]) for item in items]
    vals = [float(item["row_rmse"]) for item in items]
    ax.barh(labels, vals, color="#4C72B0")
    for y, val in enumerate(vals):
        ax.text(val + 0.3, y, f"{val:.2f}", va="center", fontsize=9)
    ax.set_xlabel("row RMSE ft")
    ax.set_title("Discrete offset selectors")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_dir / "selector_rmse_bar.png")
    plt.close(fig)

    # Pick panels by oracle RMSE spread: best/median/worst under selected model.
    panel_rows: list[tuple[float, _WellArrays]] = []
    for well in wells:
        if well.well_id not in selected:
            continue
        err = _candidate_path(well, selected[well.well_id]) - well.tvt[well.hidden_idx]
        panel_rows.append((_rmse_from_err(err), well))
    if not panel_rows:
        return
    panel_rows = sorted(panel_rows, key=lambda x: x[0])
    picks = [panel_rows[0], panel_rows[len(panel_rows) // 2], panel_rows[-1]]
    for tag, (_, well) in zip(("best", "median", "worst"), picks, strict=False):
        idx = well.hidden_idx
        md = well.md[idx]
        truth = well.tvt[idx]
        pred_sel = _candidate_path(well, selected[well.well_id])
        pred_oracle = _candidate_path(well, oracle[well.well_id])
        pred_known = _candidate_path(well, known[well.well_id])
        dtvt = np.gradient(well.tvt)
        neg_dz = -np.gradient(well.z)
        lo = max(0, well.anchor_row - 300)
        hi = min(well.tvt.size, idx[-1] + 1)
        fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), dpi=150, sharex=False)
        axes[0].plot(well.md[lo:hi], dtvt[lo:hi], label="dTVT", color="#DD8452", lw=1.0)
        axes[0].plot(well.md[lo:hi], neg_dz[lo:hi], label="-dZ", color="#4C72B0", lw=1.0)
        axes[0].axvline(well.md[well.anchor_row], color="red", lw=1, alpha=0.8, label="last known")
        axes[0].legend(ncol=3, fontsize=8)
        axes[0].set_title(f"{well.well_id}: derivative relation and offset candidates")
        axes[0].set_ylabel("ft/row")
        axes[1].plot(md, truth, color="black", lw=1.3, label="true TVT")
        axes[1].plot(
            md,
            pred_known,
            color="#DD8452",
            lw=1.0,
            label=f"known offset {known[well.well_id]:+.3f} ({_rmse_from_err(pred_known - truth):.1f})",
        )
        axes[1].plot(
            md,
            pred_sel,
            color="#4C72B0",
            lw=1.0,
            label=f"OOF selected {selected[well.well_id]:+.3f} ({_rmse_from_err(pred_sel - truth):.1f})",
        )
        axes[1].plot(
            md,
            pred_oracle,
            color="#55A868",
            lw=1.0,
            label=f"grid oracle {oracle[well.well_id]:+.3f} ({_rmse_from_err(pred_oracle - truth):.1f})",
        )
        axes[1].legend(ncol=3, fontsize=8)
        axes[1].set_xlabel("MD")
        axes[1].set_ylabel("TVT ft")
        for ax in axes:
            ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(fig_dir / f"panel_{tag}_{well.well_id}.png")
        plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run fixed-grid discrete offset experiment")
    parser.add_argument("--data-dir", type=Path, default=DiscreteOffsetConfig.data_dir)
    parser.add_argument("--output-dir", type=Path, default=DiscreteOffsetConfig.output_dir)
    parser.add_argument("--offset-grid", type=str, default=DiscreteOffsetConfig.offset_grid)
    parser.add_argument("--n-folds", type=int, default=DiscreteOffsetConfig.n_folds)
    parser.add_argument("--seed", type=int, default=DiscreteOffsetConfig.seed)
    parser.add_argument("--k-wells", type=int, default=DiscreteOffsetConfig.k_wells)
    parser.add_argument("--iterations", type=int, default=DiscreteOffsetConfig.iterations)
    parser.add_argument("--learning-rate", type=float, default=DiscreteOffsetConfig.learning_rate)
    parser.add_argument("--depth", type=int, default=DiscreteOffsetConfig.depth)
    parser.add_argument("--l2-leaf-reg", type=float, default=DiscreteOffsetConfig.l2_leaf_reg)
    parser.add_argument("--progress-every", type=int, default=DiscreteOffsetConfig.progress_every)
    parser.add_argument("--no-shuffled", action="store_true")
    parser.add_argument("--n-panel-wells", type=int, default=DiscreteOffsetConfig.n_panel_wells)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metrics = run_discrete_offset(
        DiscreteOffsetConfig(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            offset_grid=args.offset_grid,
            n_folds=args.n_folds,
            seed=args.seed,
            k_wells=args.k_wells,
            iterations=args.iterations,
            learning_rate=args.learning_rate,
            depth=args.depth,
            l2_leaf_reg=args.l2_leaf_reg,
            progress_every=args.progress_every,
            include_shuffled=not args.no_shuffled,
            n_panel_wells=args.n_panel_wells,
        )
    )
    print(
        json.dumps(
            _json_safe(
                {
                    "experiment": metrics["experiment"],
                    "best_deployable": metrics["best_deployable"],
                    "grid_oracle": metrics["grid_oracle"],
                    "report": str(Path(args.output_dir) / "discrete_offset_report.md"),
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
