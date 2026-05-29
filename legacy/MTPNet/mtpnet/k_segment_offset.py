"""K-segment per-well constant-offset candidate.

Background
==========
The horizontal-well TVT signal decomposes as

    TVT[i] = -Z[i] + C[i]

where ``C`` is the cumulative formation offset. Empirically ``dC/drow`` is
sparse: it is close to zero except at ~15 formation-top "control points" per
well. So predicting a small number of *per-segment* offsets and cumsumming

    dtvt_pred[i] = -dz[i] + c_segment(i)
    tvt_pred[i]  = anchor + cumsum(dtvt_pred[i])

is a fold-safe alternative path that the existing candidate bank (b2 / a_p50 /
residual_stack) does not directly model.

Oracle ceilings (per-well, knowing hidden TVT, using equal-spaced segments):
  K=1:  7.59 ft pooled
  K=3:  3.02 ft
  K=5:  1.82 ft   <- beats the bank-oracle of the chunk policy (2.38 ft)
  K=15: 0.65 ft

This module trains a CatBoost regressor that predicts each segment's constant
offset ``c_k`` from test-safe features only (MD/X/Y/Z/GR/TVT_input plus the
``top_state_teacher`` OOF probabilities). It then writes per-row OOF
predictions in the same ``(id, well_id, row_idx, pred_tvt)`` schema as
``residual_stack`` so the candidate bank can consume them without further
plumbing.

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
from catboost import CatBoostRegressor

from .residual_stack import (
    ResidualStackConfig,
    _ensure_ids,
    load_training_frame,
    make_group_folds,
)
from .schema_safe import assert_schema_safe_columns


@dataclass(frozen=True)
class KSegmentOffsetConfig:
    data_dir: Path = Path("data/train")
    output_dir: Path = Path("artifacts/k_segment_offset_v0")
    top_state_path: Path | None = Path(
        "artifacts/top_state_teacher_v0/top_state_oof_predictions.parquet"
    )
    n_folds: int = 5
    seed: int = 42
    k_wells: int = -1
    K: int = 5
    rows_per_step: int = 32
    iterations: int = 800
    learning_rate: float = 0.04
    depth: int = 6
    l2_leaf_reg: float = 5.0
    progress_every: int = 100


@dataclass
class _WellGeometry:
    """Cached per-well arrays used both at feature build and prediction time."""

    well_id: str
    z: np.ndarray
    tvt: np.ndarray  # may contain NaN
    tvt_input: np.ndarray
    md: np.ndarray
    gr: np.ndarray
    x: np.ndarray
    y: np.ndarray
    row_idx: np.ndarray
    dz: np.ndarray
    last_known_row: int
    hidden_row_idx: np.ndarray  # rows we need to predict
    bounds: np.ndarray  # K+1 segment boundaries in [last_known+1, len(z)]
    fold: int


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


def _segment_boundaries(last_known: int, n: int, K: int) -> np.ndarray:
    """Equal-row boundaries over ``[last_known+1, n)``.

    The empirical study showed equal-spaced segments dominate top_state-driven
    boundaries at the same K, because top_state knows direction but not the
    *position* of the underlying control points.
    """
    if last_known + 1 >= n or K < 1:
        return np.asarray([last_known + 1, n], dtype=np.int64)
    return np.linspace(last_known + 1, n, K + 1, dtype=np.int64)


def _segment_design_matrix(
    bounds: np.ndarray, hidden_row_idx: np.ndarray
) -> np.ndarray:
    """Build ``M`` such that ``cumsum(c)[hidden] == M @ c``.

    Specifically, ``M[i, k]`` is the count of segment-k rows up to and
    including hidden row ``hidden_row_idx[i]``. The TVT prediction at hidden
    row ``i`` is then ``anchor - (z[i] - z[anchor]) + (M @ c)[i]``.
    """
    K = bounds.size - 1
    M = np.zeros((hidden_row_idx.size, K), dtype=np.float64)
    for k in range(K):
        lo = int(bounds[k])
        hi = int(bounds[k + 1])
        if hi <= lo:
            continue
        # rows in segment k that are <= row i
        # M[i, k] = min(i, hi-1) - lo + 1 for i >= lo, else 0
        contains = hidden_row_idx >= lo
        in_seg_count = np.minimum(hidden_row_idx, hi - 1) - lo + 1
        M[contains, k] = np.maximum(0.0, in_seg_count[contains].astype(np.float64))
    return M


def _oracle_segment_offsets(
    z: np.ndarray,
    tvt: np.ndarray,
    tvt_input: np.ndarray,
    bounds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Fit per-segment offsets by least-squares against true hidden TVT.

    Returns ``(c, hidden_row_idx)``. Only used during training because it
    requires hidden TVT.
    """
    known = np.flatnonzero(np.isfinite(tvt_input))
    if known.size < 2:
        return None
    last = int(known[-1])
    hidden_idx = np.flatnonzero((~np.isfinite(tvt_input)) & np.isfinite(tvt))
    if hidden_idx.size < bounds.size - 1:
        return None
    anchor_tvt = float(tvt_input[last])
    anchor_z = float(z[last])
    M = _segment_design_matrix(bounds, hidden_idx)
    y = tvt[hidden_idx] - (anchor_tvt - z[hidden_idx] + anchor_z)
    c, *_ = np.linalg.lstsq(M, y, rcond=None)
    return c.astype(np.float64), hidden_idx


def _load_top_state_lookup(path: Path | None) -> dict[str, pd.DataFrame]:
    if path is None or not Path(path).exists():
        return {}
    frame = pd.read_parquet(path)
    if frame.empty:
        return {}
    frame["well_id"] = frame["well_id"].astype(str)
    needed = {"well_id", "step", "pred_expected_sign", "prob_up", "prob_down", "prob_flat"}
    if not needed.issubset(frame.columns):
        return {}
    keep = list(needed | {"pred_state"} & set(frame.columns))
    out: dict[str, pd.DataFrame] = {}
    for wid, group in frame[keep].groupby("well_id", sort=False):
        gg = group.copy()
        gg["step"] = pd.to_numeric(gg["step"], errors="coerce").astype("Int64")
        gg = gg.dropna(subset=["step"]).reset_index(drop=True)
        gg["step"] = gg["step"].astype(int)
        out[str(wid)] = gg.set_index("step")
    return out


def _well_geometries(
    frame: pd.DataFrame,
    *,
    K: int,
    fold_of_well: dict[str, int],
) -> list[_WellGeometry]:
    out: list[_WellGeometry] = []
    for wid, group in frame.groupby("well_id", sort=True):
        wid_s = str(wid)
        if wid_s not in fold_of_well:
            continue
        g = group.sort_values("row_idx")
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
        row_idx = pd.to_numeric(g["row_idx"], errors="coerce").to_numpy(dtype=np.int64)
        n = len(z)
        if n == 0:
            continue
        if not (np.isfinite(z).any() and np.isfinite(tvt_in).any()):
            continue
        known = np.flatnonzero(np.isfinite(tvt_in))
        if known.size < 2:
            continue
        last_known = int(known[-1])
        hidden_row_idx = np.flatnonzero((~np.isfinite(tvt_in)) & np.isfinite(tvt))
        if hidden_row_idx.size < K:
            continue
        bounds = _segment_boundaries(last_known, n, K)
        dz = np.gradient(z) if n >= 2 else np.zeros(n)
        out.append(
            _WellGeometry(
                well_id=wid_s,
                z=z,
                tvt=tvt,
                tvt_input=tvt_in,
                md=md,
                gr=gr,
                x=x,
                y=y,
                row_idx=row_idx,
                dz=dz,
                last_known_row=last_known,
                hidden_row_idx=hidden_row_idx,
                bounds=bounds,
                fold=fold_of_well[wid_s],
            )
        )
    return out


def _segment_features(
    geom: _WellGeometry,
    k: int,
    *,
    top_state_lookup: dict[str, pd.DataFrame],
    rows_per_step: int,
) -> dict[str, float]:
    lo = int(geom.bounds[k])
    hi = int(geom.bounds[k + 1])
    rng = np.arange(lo, hi)
    z = geom.z
    md = geom.md
    gr = geom.gr
    x = geom.x
    y = geom.y
    dz_seg = geom.dz[rng] if rng.size else np.asarray([0.0])
    anchor = geom.last_known_row
    anchor_z = float(z[anchor])
    anchor_md = float(md[anchor]) if np.isfinite(md[anchor]) else 0.0
    anchor_tvt = float(geom.tvt_input[anchor])
    known = np.flatnonzero(np.isfinite(geom.tvt_input))
    C_known = geom.tvt_input[known] + z[known] if known.size else np.asarray([0.0])
    dC_known = np.gradient(C_known) if C_known.size >= 2 else np.zeros_like(C_known)

    seg_sign = 0.0
    seg_abs_sign = 0.0
    seg_p_up = 0.0
    seg_p_down = 0.0
    seg_p_flat = 0.0
    seg_ts_steps = 0.0
    tw = top_state_lookup.get(geom.well_id)
    if tw is not None and rng.size:
        steps = (rng // max(int(rows_per_step), 1)).astype(int)
        unique_steps = np.unique(steps)
        tw_seg = tw.reindex(unique_steps)
        if len(tw_seg) > 0:
            signs = pd.to_numeric(tw_seg["pred_expected_sign"], errors="coerce").to_numpy()
            signs = signs[np.isfinite(signs)]
            if signs.size:
                seg_sign = float(np.mean(signs))
                seg_abs_sign = float(np.mean(np.abs(signs)))
            for col, target in (
                ("prob_up", "seg_p_up"),
                ("prob_down", "seg_p_down"),
                ("prob_flat", "seg_p_flat"),
            ):
                values = pd.to_numeric(tw_seg[col], errors="coerce").to_numpy()
                values = values[np.isfinite(values)]
                if values.size:
                    if target == "seg_p_up":
                        seg_p_up = float(values.mean())
                    elif target == "seg_p_down":
                        seg_p_down = float(values.mean())
                    elif target == "seg_p_flat":
                        seg_p_flat = float(values.mean())
            seg_ts_steps = float(len(tw_seg))

    def _safe_mean(arr: np.ndarray) -> float:
        a = arr[np.isfinite(arr)]
        return float(a.mean()) if a.size else 0.0

    def _safe_std(arr: np.ndarray) -> float:
        a = arr[np.isfinite(arr)]
        return float(a.std()) if a.size > 1 else 0.0

    return {
        "feat_seg_k": float(k),
        "feat_seg_k_norm": float(k) / float(max(geom.bounds.size - 1, 1)),
        "feat_seg_progress": float((lo + hi) / 2 - anchor) / 5000.0,
        "feat_seg_rel_start": float(lo - anchor) / 5000.0,
        "feat_seg_length": float(hi - lo) / 1000.0,
        "feat_seg_dz_mean": _safe_mean(dz_seg),
        "feat_seg_dz_std": _safe_std(dz_seg),
        "feat_seg_z_start": float(z[lo]) / 10000.0 if rng.size else 0.0,
        "feat_seg_z_end": float(z[hi - 1]) / 10000.0 if rng.size else 0.0,
        "feat_seg_z_span": float(z[hi - 1] - z[lo]) / 100.0 if rng.size else 0.0,
        "feat_seg_md_dist_anchor": float(md[lo] - md[anchor]) / 1000.0
        if rng.size and np.isfinite(md[lo])
        else 0.0,
        "feat_seg_xy_dist_anchor": float(
            np.sqrt((x[lo] - x[anchor]) ** 2 + (y[lo] - y[anchor]) ** 2)
        )
        / 1000.0
        if rng.size and np.isfinite(x[lo]) and np.isfinite(y[lo])
        else 0.0,
        "feat_seg_gr_mean": _safe_mean(gr[rng]) / 100.0 if rng.size else 0.0,
        "feat_seg_gr_std": _safe_std(gr[rng]) / 50.0 if rng.size else 0.0,
        "feat_seg_ts_sign": seg_sign,
        "feat_seg_ts_abs_sign": seg_abs_sign,
        "feat_seg_ts_p_up": seg_p_up,
        "feat_seg_ts_p_down": seg_p_down,
        "feat_seg_ts_p_flat": seg_p_flat,
        "feat_seg_ts_steps": seg_ts_steps,
        "feat_well_known_dC_median": float(np.median(dC_known)),
        "feat_well_known_dC_mean": float(np.mean(dC_known)),
        "feat_well_known_dC_std": float(np.std(dC_known)) if dC_known.size > 1 else 0.0,
        "feat_well_anchor_C": float(anchor_tvt + anchor_z) / 10000.0,
        "feat_well_anchor_z": anchor_z / 10000.0,
        "feat_well_anchor_md": anchor_md / 10000.0,
        "feat_well_hidden_z_span": float(z[int(geom.bounds[-1]) - 1] - z[int(geom.bounds[0])])
        / 100.0
        if int(geom.bounds[-1]) > int(geom.bounds[0])
        else 0.0,
        "feat_well_hidden_n": float(int(geom.bounds[-1]) - int(geom.bounds[0])),
        "feat_well_known_n": float(known.size),
        "feat_well_gr_mean_known": _safe_mean(gr[known]) / 100.0 if known.size else 0.0,
        "feat_well_gr_std_known": _safe_std(gr[known]) / 50.0 if known.size else 0.0,
    }


def build_k_segment_dataset(
    frame: pd.DataFrame,
    *,
    K: int,
    rows_per_step: int,
    fold_of_well: dict[str, int],
    top_state_lookup: dict[str, pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, list[_WellGeometry], list[str]]:
    """Build per-(well, segment) training table plus per-well geometry cache."""
    if top_state_lookup is None:
        top_state_lookup = {}
    raw = _ensure_ids(frame)
    geometries = _well_geometries(raw, K=K, fold_of_well=fold_of_well)
    records: list[dict[str, Any]] = []
    for geom in geometries:
        oracle = _oracle_segment_offsets(geom.z, geom.tvt, geom.tvt_input, geom.bounds)
        if oracle is None:
            continue
        c_target, _ = oracle
        for k in range(geom.bounds.size - 1):
            feats = _segment_features(
                geom, k, top_state_lookup=top_state_lookup, rows_per_step=rows_per_step
            )
            rec = {
                "well_id": geom.well_id,
                "fold": geom.fold,
                "segment_k": k,
                "target_c": float(c_target[k]),
                **feats,
            }
            records.append(rec)
    if not records:
        raise ValueError("No K-segment offset rows could be built")
    df = pd.DataFrame(records).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    assert_schema_safe_columns(feat_cols, context="KSegmentOffset features")
    return df, geometries, feat_cols


def _train_and_predict_c(
    df: pd.DataFrame,
    feat_cols: list[str],
    *,
    n_folds: int,
    config: KSegmentOffsetConfig,
) -> np.ndarray:
    """5-fold OOF predicted c values aligned with ``df`` row order."""
    out = np.full(len(df), np.nan, dtype=np.float64)
    for fold_idx in range(n_folds):
        train_mask = df["fold"].to_numpy() != fold_idx
        valid_mask = df["fold"].to_numpy() == fold_idx
        if not train_mask.any() or not valid_mask.any():
            continue
        print(
            f"[k-seg] fold {fold_idx + 1}/{n_folds} "
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
        model.fit(df.loc[train_mask, feat_cols], df.loc[train_mask, "target_c"])
        out[valid_mask] = model.predict(df.loc[valid_mask, feat_cols])
    return out


def _materialise_row_predictions(
    geometries: list[_WellGeometry],
    c_by_well: dict[str, np.ndarray],
    frame_index: pd.DataFrame,
) -> pd.DataFrame:
    """Convert per-segment c into per-row ``pred_tvt`` on hidden rows.

    Builds the long table per-well in numpy, then does a single vectorised
    merge with the (well_id, row_idx) → id lookup. The previous per-row
    ``DataFrame.loc`` lookup made this O(N_hidden) MultiIndex queries which
    was the dominant cost on full 773-well runs.
    """
    parts: list[pd.DataFrame] = []
    for geom in geometries:
        c_pred = c_by_well.get(geom.well_id)
        if c_pred is None:
            continue
        anchor_tvt = float(geom.tvt_input[geom.last_known_row])
        anchor_z = float(geom.z[geom.last_known_row])
        M = _segment_design_matrix(geom.bounds, geom.hidden_row_idx)
        delta = M @ c_pred
        pred_tvt = anchor_tvt - (geom.z[geom.hidden_row_idx] - anchor_z) + delta
        parts.append(
            pd.DataFrame(
                {
                    "well_id": geom.well_id,
                    "row_idx": geom.hidden_row_idx.astype(np.int64),
                    "pred_tvt": pred_tvt.astype(np.float64),
                }
            )
        )
    if not parts:
        raise ValueError("Materialised row predictions are empty")
    out = pd.concat(parts, ignore_index=True)
    if frame_index is not None and not frame_index.empty:
        # Vectorised id lookup — single merge instead of per-row .loc.
        id_lookup = frame_index[["well_id", "row_idx", "id"]].copy()
        id_lookup["well_id"] = id_lookup["well_id"].astype(str)
        id_lookup["row_idx"] = pd.to_numeric(id_lookup["row_idx"], errors="coerce").astype("Int64")
        id_lookup = id_lookup.dropna(subset=["row_idx"]).copy()
        id_lookup["row_idx"] = id_lookup["row_idx"].astype(np.int64)
        id_lookup = id_lookup.drop_duplicates(["well_id", "row_idx"])
        out = out.merge(id_lookup, on=["well_id", "row_idx"], how="left")
    if "id" not in out.columns or out["id"].isna().any():
        # Fallback for any rows the lookup missed (synthetic frames in tests).
        synthetic = out["well_id"].astype(str) + "_" + out["row_idx"].astype(int).astype(str)
        if "id" in out.columns:
            out["id"] = out["id"].fillna(synthetic)
        else:
            out["id"] = synthetic
    return out[["id", "well_id", "row_idx", "pred_tvt"]]


def run_k_segment_offset_from_frame(
    frame: pd.DataFrame,
    *,
    config: KSegmentOffsetConfig,
    top_state_lookup: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """Train K-segment offset model on ``frame`` and emit OOF candidate parquet."""
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = _ensure_ids(frame)
    well_ids = sorted({str(w) for w in raw["well_id"].astype(str).unique()})
    folds = make_group_folds(pd.Series(well_ids), n_folds=config.n_folds, seed=config.seed)
    fold_of_well = {str(w): idx for idx, (_, valid) in enumerate(folds) for w in valid}
    if top_state_lookup is None:
        top_state_lookup = _load_top_state_lookup(config.top_state_path)
    df, geometries, feat_cols = build_k_segment_dataset(
        raw,
        K=config.K,
        rows_per_step=config.rows_per_step,
        fold_of_well=fold_of_well,
        top_state_lookup=top_state_lookup,
    )
    print(
        f"[k-seg] wells_in_train={len(geometries)} (K={config.K}) "
        f"segment_rows={len(df):,} feats={len(feat_cols)}",
        file=sys.stderr,
        flush=True,
    )
    c_pred_flat = _train_and_predict_c(
        df, feat_cols, n_folds=len(folds), config=config
    )
    df = df.copy()
    df["c_pred"] = c_pred_flat
    c_by_well: dict[str, np.ndarray] = {}
    for wid, grp in df.groupby("well_id"):
        ordered = grp.sort_values("segment_k")
        c_by_well[str(wid)] = ordered["c_pred"].to_numpy(dtype=np.float64)
    id_frame = raw[["well_id", "row_idx", "id"]].copy() if "id" in raw.columns else None
    if id_frame is not None:
        id_frame["well_id"] = id_frame["well_id"].astype(str)
        id_frame["row_idx"] = pd.to_numeric(id_frame["row_idx"], errors="coerce").astype("Int64")
        id_frame = id_frame.dropna(subset=["row_idx"]).copy()
        id_frame["row_idx"] = id_frame["row_idx"].astype(int)
    row_predictions = _materialise_row_predictions(
        geometries, c_by_well, id_frame if id_frame is not None else pd.DataFrame()
    )
    # Diagnostics on hidden rows
    metrics = _evaluate_against_hidden(geometries, c_by_well, df=df)
    metrics.update(
        {
            "candidate": "k_segment_offset_v0",
            "K": int(config.K),
            "wells_trained": int(len(geometries)),
            "segment_rows": int(len(df)),
            "feature_columns": feat_cols,
            "config": asdict(config),
        }
    )
    row_predictions.to_parquet(out_dir / "k_offset_oof_predictions.parquet", index=False)
    (out_dir / "k_offset_metrics.json").write_text(
        json.dumps(_json_safe(metrics), indent=2), encoding="utf-8"
    )
    _write_report(out_dir, metrics)
    return metrics


def _evaluate_against_hidden(
    geometries: list[_WellGeometry],
    c_by_well: dict[str, np.ndarray],
    *,
    df: pd.DataFrame,
) -> dict[str, Any]:
    sse_total = 0.0
    n_total = 0
    per_well: list[float] = []
    for geom in geometries:
        c_pred = c_by_well.get(geom.well_id)
        if c_pred is None:
            continue
        M = _segment_design_matrix(geom.bounds, geom.hidden_row_idx)
        anchor_tvt = float(geom.tvt_input[geom.last_known_row])
        anchor_z = float(geom.z[geom.last_known_row])
        pred = anchor_tvt - (geom.z[geom.hidden_row_idx] - anchor_z) + (M @ c_pred)
        truth = geom.tvt[geom.hidden_row_idx]
        valid = np.isfinite(pred) & np.isfinite(truth)
        if not valid.any():
            continue
        err = pred[valid] - truth[valid]
        sse_total += float(np.sum(err**2))
        n_total += int(valid.sum())
        per_well.append(float(np.sqrt(np.mean(err**2))))
    pooled_rmse = float(np.sqrt(sse_total / n_total)) if n_total else float("nan")
    well_series = pd.Series(per_well, dtype=float)
    c_mae = float(np.mean(np.abs(df["target_c"] - df["c_pred"]))) if "c_pred" in df else float("nan")
    c_corr = (
        float(np.corrcoef(df["target_c"], df["c_pred"])[0, 1])
        if "c_pred" in df and df["target_c"].std() > 0 and df["c_pred"].std() > 0
        else float("nan")
    )
    return {
        "hidden_pooled_rmse": pooled_rmse,
        "hidden_well_rmse_mean": float(well_series.mean()) if not well_series.empty else float("nan"),
        "hidden_well_rmse_median": float(well_series.median())
        if not well_series.empty
        else float("nan"),
        "hidden_well_rmse_p90": float(well_series.quantile(0.90))
        if not well_series.empty
        else float("nan"),
        "hidden_well_rmse_p99": float(well_series.quantile(0.99))
        if not well_series.empty
        else float("nan"),
        "c_mae": c_mae,
        "c_pearson": c_corr,
    }


def _write_report(output_dir: Path, metrics: dict[str, Any]) -> None:
    lines = [
        "# K_SEGMENT_OFFSET_V0",
        "",
        "Fold-safe per-well per-segment offset predictor. Each well's hidden",
        "lateral section is split into K equal-row segments; one constant",
        "offset is learned per segment. Predictions are materialised as",
        f"a `{metrics.get('candidate')}` candidate via the candidate bank.",
        "",
        "## summary",
        "```json",
        json.dumps(_json_safe(metrics), indent=2),
        "```",
    ]
    (output_dir / "k_offset_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_k_segment_offset(config: KSegmentOffsetConfig) -> dict[str, Any]:
    frame = load_training_frame(
        ResidualStackConfig(data_dir=config.data_dir, k_wells=config.k_wells)
    )
    print(
        f"[k-seg] loaded frame rows={len(frame):,} wells={frame['well_id'].nunique()}",
        file=sys.stderr,
        flush=True,
    )
    return run_k_segment_offset_from_frame(frame, config=config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train fold-safe K-segment per-well constant-offset candidate"
    )
    parser.add_argument("--data-dir", type=Path, default=KSegmentOffsetConfig.data_dir)
    parser.add_argument("--output-dir", type=Path, default=KSegmentOffsetConfig.output_dir)
    parser.add_argument(
        "--top-state-path",
        type=Path,
        default=KSegmentOffsetConfig.top_state_path,
        help="Path to top_state_teacher OOF predictions parquet (optional).",
    )
    parser.add_argument("--n-folds", type=int, default=KSegmentOffsetConfig.n_folds)
    parser.add_argument("--seed", type=int, default=KSegmentOffsetConfig.seed)
    parser.add_argument("--k-wells", type=int, default=KSegmentOffsetConfig.k_wells)
    parser.add_argument("--K", type=int, default=KSegmentOffsetConfig.K)
    parser.add_argument("--rows-per-step", type=int, default=KSegmentOffsetConfig.rows_per_step)
    parser.add_argument("--iterations", type=int, default=KSegmentOffsetConfig.iterations)
    parser.add_argument("--learning-rate", type=float, default=KSegmentOffsetConfig.learning_rate)
    parser.add_argument("--depth", type=int, default=KSegmentOffsetConfig.depth)
    parser.add_argument("--l2-leaf-reg", type=float, default=KSegmentOffsetConfig.l2_leaf_reg)
    parser.add_argument("--progress-every", type=int, default=KSegmentOffsetConfig.progress_every)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metrics = run_k_segment_offset(
        KSegmentOffsetConfig(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            top_state_path=args.top_state_path,
            n_folds=args.n_folds,
            seed=args.seed,
            k_wells=args.k_wells,
            K=args.K,
            rows_per_step=args.rows_per_step,
            iterations=args.iterations,
            learning_rate=args.learning_rate,
            depth=args.depth,
            l2_leaf_reg=args.l2_leaf_reg,
            progress_every=args.progress_every,
        )
    )
    print(
        json.dumps(
            _json_safe(
                {
                    "candidate": metrics["candidate"],
                    "K": metrics["K"],
                    "wells_trained": metrics["wells_trained"],
                    "hidden_pooled_rmse": metrics["hidden_pooled_rmse"],
                    "hidden_well_rmse_median": metrics["hidden_well_rmse_median"],
                    "c_pearson": metrics["c_pearson"],
                    "report": str(Path(args.output_dir) / "k_offset_report.md"),
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
