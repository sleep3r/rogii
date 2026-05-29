"""``k_offset_gated_v0`` — confidence-gated single candidate from k_segment_offset.

Background
==========
``k_segment_offset_v0`` produces per-row predictions with Pearson 0.96 against the
oracle ``c`` and ``c_MAE ≈ 0.008``, but its per-row TVT cumulates this MAE into
~15.6 ft pooled. On per-chunk evaluation:

* beats ``b2`` on **34%** of chunks overall, **52%** of disaster chunks
* is the new-best (beats all 30 existing candidates) on **9%** of chunks
* but on the ~66% of chunks where ``b2`` is already good, ``k_offset`` is
  substantially worse and the DP ranker on small folds confidently picks it
  in the wrong places.

Smoke runs showed adding raw ``k_offset`` as a bank candidate **hurts** DP
(9.327 → 9.684 ft on 50-well sqrt_mse smoke) despite a strictly smaller
bank-oracle (-0.110 ft). This is *selector* risk — we have a useful but rare
specialist, and DP can't tell when to trust it on a small training set.

Strategy
========
Train a per-chunk gate classifier that predicts whether ``k_offset`` will beat
``b2`` on this chunk. Materialise a single new candidate

    pred_gated[row] = b2[row] + gate * (k_offset[row] - b2[row])

where ``gate = P(k_offset_better | chunk_features)`` is in ``[0, 1]``. The
candidate degenerates to ``b2`` on low-confidence chunks and to ``k_offset``
on high-confidence chunks. This converts the bank-oracle gain into something
the DP can actually use without overfitting.

The gate classifier uses only test-safe per-chunk features: top_state
aggregates, k_offset-vs-b2 disagreement (both test-safe), c_pred stats,
geometry, GR validity. Schema safety is enforced via
``assert_schema_safe_columns``.

Output schema mirrors ``k_segment_offset`` and ``residual_stack``:
``(id, well_id, row_idx, pred_tvt)``. The bank registers it as the new
candidate label ``k_offset_gated_v0`` so it lives next to (not replacing)
the raw ``k_segment_offset_v0`` slot.
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
from catboost import CatBoostClassifier

from .residual_stack import (
    ResidualStackConfig,
    _ensure_ids,
    load_training_frame,
    make_group_folds,
)
from .schema_safe import assert_schema_safe_columns


@dataclass(frozen=True)
class KOffsetGatedConfig:
    data_dir: Path = Path("data/train")
    output_dir: Path = Path("artifacts/k_offset_gated_v0")
    k_offset_path: Path = Path(
        "artifacts/k_segment_offset_v0/k_offset_oof_predictions.parquet"
    )
    b2_path: Path = Path(
        "../old/artifacts/formation_b2_danger_guard_a2_full_schema10/guarded_predictions.parquet"
    )
    b2_column: str = "b2_guarded_submit"
    top_state_path: Path | None = Path(
        "artifacts/top_state_teacher_v0/top_state_oof_predictions.parquet"
    )
    rows_per_step: int = 32
    chunk_size: int = 512
    n_folds: int = 5
    seed: int = 42
    k_wells: int = -1
    iterations: int = 600
    learning_rate: float = 0.05
    depth: int = 5
    l2_leaf_reg: float = 5.0
    # Gate target sharpness: a chunk counts as "k_offset better" only if its MSE
    # is strictly better than b2 by at least this many ft^2 (~0.5 ft RMSE band).
    gate_target_eps_ft2: float = 0.5
    # Optional gate squashing — clip P(class=1) below ``gate_floor`` and above
    # ``gate_ceiling`` so the gated candidate cannot fully collapse to either
    # b2 or k_offset.
    gate_floor: float = 0.0
    gate_ceiling: float = 1.0
    progress_every: int = 100


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
    return value


def _load_b2(b2_path: Path, b2_column: str) -> pd.DataFrame:
    if not Path(b2_path).exists():
        raise FileNotFoundError(f"b2 predictions parquet not found: {b2_path}")
    b2 = pd.read_parquet(b2_path)
    needed = {"id", b2_column}
    if not needed.issubset(b2.columns):
        raise ValueError(f"b2 parquet missing required columns {sorted(needed)}; has {list(b2.columns)}")
    out = b2[["id", b2_column]].rename(columns={b2_column: "b2_tvt"}).copy()
    out["id"] = out["id"].astype(str)
    out["b2_tvt"] = pd.to_numeric(out["b2_tvt"], errors="coerce")
    return out.drop_duplicates("id")


def _load_top_state(path: Path | None) -> dict[str, pd.DataFrame]:
    if path is None or not Path(path).exists():
        return {}
    frame = pd.read_parquet(path)
    if frame.empty:
        return {}
    needed = {"well_id", "step", "pred_expected_sign", "prob_up", "prob_down", "prob_flat"}
    if not needed.issubset(frame.columns):
        return {}
    frame = frame.copy()
    frame["well_id"] = frame["well_id"].astype(str)
    out: dict[str, pd.DataFrame] = {}
    for wid, grp in frame[list(needed)].groupby("well_id", sort=False):
        gg = grp.copy()
        gg["step"] = pd.to_numeric(gg["step"], errors="coerce").astype("Int64")
        gg = gg.dropna(subset=["step"]).reset_index(drop=True)
        gg["step"] = gg["step"].astype(int)
        out[str(wid)] = gg.set_index("step").sort_index()
    return out


@dataclass
class _ChunkRecord:
    well_id: str
    chunk_id: int
    fold: int
    rows: np.ndarray  # row_idx values of hidden rows in this chunk
    ids: list[str]
    b2: np.ndarray
    k_offset: np.ndarray
    tvt_true: np.ndarray  # NaN at inference
    b2_mse: float  # NaN at inference
    k_offset_mse: float  # NaN at inference
    features: dict[str, float]


def _chunk_features(
    *,
    rows: np.ndarray,
    z: np.ndarray,
    md: np.ndarray,
    gr: np.ndarray,
    b2: np.ndarray,
    k_offset: np.ndarray,
    well_tvt_input: np.ndarray,
    last_known_row: int,
    well_hidden_count: int,
    top_state_step_df: pd.DataFrame | None,
    rows_per_step: int,
) -> dict[str, float]:
    def _safe_mean(arr: np.ndarray) -> float:
        a = arr[np.isfinite(arr)]
        return float(a.mean()) if a.size else 0.0

    def _safe_std(arr: np.ndarray) -> float:
        a = arr[np.isfinite(arr)]
        return float(a.std()) if a.size > 1 else 0.0

    chunk_z = z[rows]
    chunk_md = md[rows]
    chunk_gr = gr[rows]
    chunk_dz = np.gradient(z)[rows] if z.size >= 2 else np.zeros(rows.size)
    diff = k_offset - b2
    finite_diff = diff[np.isfinite(diff)]
    diff_abs = np.abs(finite_diff)
    rel_start = float(rows.min() - last_known_row) / 1000.0
    rel_end = float(rows.max() - last_known_row) / 1000.0
    progress_start = (
        float(rows.min() - last_known_row) / max(well_hidden_count, 1)
    )
    progress_end = float(rows.max() - last_known_row) / max(well_hidden_count, 1)

    # top_state aggregates over the steps the chunk overlaps
    sign_arr = np.zeros(0)
    prob_up = prob_down = prob_flat = np.zeros(0)
    if top_state_step_df is not None and not top_state_step_df.empty:
        chunk_steps = np.unique((rows // max(int(rows_per_step), 1)).astype(int))
        sub = top_state_step_df.reindex(chunk_steps).dropna(subset=["pred_expected_sign"])
        if not sub.empty:
            sign_arr = pd.to_numeric(sub["pred_expected_sign"], errors="coerce").to_numpy(dtype=np.float64)
            sign_arr = sign_arr[np.isfinite(sign_arr)]
            prob_up = pd.to_numeric(sub["prob_up"], errors="coerce").to_numpy(dtype=np.float64)
            prob_down = pd.to_numeric(sub["prob_down"], errors="coerce").to_numpy(dtype=np.float64)
            prob_flat = pd.to_numeric(sub["prob_flat"], errors="coerce").to_numpy(dtype=np.float64)

    def _sign_consensus(sign_arr: np.ndarray) -> float:
        if sign_arr.size == 0:
            return 0.0
        pos = float((sign_arr > 0).sum())
        neg = float((sign_arr < 0).sum())
        return abs(pos - neg) / max(sign_arr.size, 1)

    return {
        # k_offset vs b2 disagreement profile — primary signal of when to
        # trust k_offset over b2 at all.
        "feat_diff_mean_abs": float(np.mean(diff_abs)) / 50.0 if diff_abs.size else 0.0,
        "feat_diff_std": _safe_std(finite_diff) / 50.0,
        "feat_diff_p95_abs": float(np.percentile(diff_abs, 95)) / 50.0
        if diff_abs.size else 0.0,
        "feat_diff_signed_mean": _safe_mean(finite_diff) / 50.0,
        "feat_diff_endpoint": float(diff[-1] - diff[0]) / 50.0
        if diff.size >= 2 and np.isfinite(diff[0]) and np.isfinite(diff[-1])
        else 0.0,
        "feat_diff_n_finite_frac": float(np.isfinite(diff).mean()),
        # Top_state confidence within the chunk
        "feat_ts_sign_abs_mean": float(np.mean(np.abs(sign_arr))) if sign_arr.size else 0.0,
        "feat_ts_sign_signed_mean": _safe_mean(sign_arr),
        "feat_ts_sign_std": _safe_std(sign_arr),
        "feat_ts_sign_consensus": _sign_consensus(sign_arr),
        "feat_ts_prob_flat_mean": _safe_mean(prob_flat),
        "feat_ts_prob_up_mean": _safe_mean(prob_up),
        "feat_ts_prob_down_mean": _safe_mean(prob_down),
        "feat_ts_n_steps_in_chunk": float(sign_arr.size),
        # Geometry of the chunk
        "feat_z_span": float(chunk_z[-1] - chunk_z[0]) / 100.0 if chunk_z.size >= 2 else 0.0,
        "feat_dz_mean": _safe_mean(chunk_dz),
        "feat_dz_std": _safe_std(chunk_dz),
        "feat_md_span": float(chunk_md[-1] - chunk_md[0]) / 1000.0
        if chunk_md.size >= 2 else 0.0,
        "feat_chunk_rows_log": float(np.log1p(rows.size)),
        "feat_progress_start": float(progress_start),
        "feat_progress_end": float(progress_end),
        "feat_rel_start": rel_start,
        "feat_rel_end": rel_end,
        # GR availability — proxy for whether other GR-driven priors are reliable.
        "feat_gr_valid_frac": float(np.isfinite(chunk_gr).mean()),
        "feat_gr_mean_n": _safe_mean(chunk_gr) / 100.0,
        "feat_gr_std_n": _safe_std(chunk_gr) / 50.0,
        # Well-level: how much of TVT_input we had (anchor strength)
        "feat_well_known_frac": float(np.isfinite(well_tvt_input).mean()),
    }


FEATURE_COLUMNS: tuple[str, ...] = (
    "feat_diff_mean_abs",
    "feat_diff_std",
    "feat_diff_p95_abs",
    "feat_diff_signed_mean",
    "feat_diff_endpoint",
    "feat_diff_n_finite_frac",
    "feat_ts_sign_abs_mean",
    "feat_ts_sign_signed_mean",
    "feat_ts_sign_std",
    "feat_ts_sign_consensus",
    "feat_ts_prob_flat_mean",
    "feat_ts_prob_up_mean",
    "feat_ts_prob_down_mean",
    "feat_ts_n_steps_in_chunk",
    "feat_z_span",
    "feat_dz_mean",
    "feat_dz_std",
    "feat_md_span",
    "feat_chunk_rows_log",
    "feat_progress_start",
    "feat_progress_end",
    "feat_rel_start",
    "feat_rel_end",
    "feat_gr_valid_frac",
    "feat_gr_mean_n",
    "feat_gr_std_n",
    "feat_well_known_frac",
)


def build_chunk_dataset(
    frame: pd.DataFrame,
    *,
    k_offset_preds: pd.DataFrame,
    b2_preds: pd.DataFrame,
    top_state_lookup: dict[str, pd.DataFrame],
    chunk_size: int,
    rows_per_step: int,
    fold_of_well: dict[str, int],
    gate_target_eps_ft2: float,
) -> tuple[pd.DataFrame, list[_ChunkRecord]]:
    """Per-(well, chunk) feature table + materialisation cache.

    Each row represents one chunk. Features are test-safe. Target columns
    (``b2_mse``, ``k_offset_mse``, ``gate_target``) are derived from training
    TVT — present in train mode, NaN at inference.
    """
    raw = _ensure_ids(frame)
    raw = raw.sort_values(["well_id", "row_idx"]).reset_index(drop=True)
    raw["well_id"] = raw["well_id"].astype(str)

    # Per-id maps for fast lookup
    k_off_map = dict(
        zip(k_offset_preds["id"].astype(str), pd.to_numeric(k_offset_preds["pred_tvt"], errors="coerce"))
    )
    b2_map = dict(zip(b2_preds["id"].astype(str), pd.to_numeric(b2_preds["b2_tvt"], errors="coerce")))

    records: list[_ChunkRecord] = []
    for wid, well in raw.groupby("well_id", sort=False):
        if str(wid) not in fold_of_well:
            continue
        well_sorted = well.sort_values("row_idx").reset_index(drop=True)
        z = pd.to_numeric(well_sorted.get("Z", pd.Series(dtype=float)), errors="coerce").to_numpy(
            dtype=np.float64
        )
        md = pd.to_numeric(well_sorted.get("MD", pd.Series(dtype=float)), errors="coerce").to_numpy(
            dtype=np.float64
        )
        gr = pd.to_numeric(well_sorted.get("GR", pd.Series(dtype=float)), errors="coerce").to_numpy(
            dtype=np.float64
        )
        tvt = pd.to_numeric(well_sorted.get("TVT", pd.Series(dtype=float)), errors="coerce").to_numpy(
            dtype=np.float64
        )
        tvt_in = pd.to_numeric(
            well_sorted.get("TVT_input", pd.Series(dtype=float)), errors="coerce"
        ).to_numpy(dtype=np.float64)
        ids = well_sorted["id"].astype(str).tolist()
        row_idx = pd.to_numeric(well_sorted["row_idx"], errors="coerce").astype(int).to_numpy()
        known = np.flatnonzero(np.isfinite(tvt_in))
        if known.size < 2:
            continue
        last_known = int(known[-1])
        hidden_mask = (~np.isfinite(tvt_in)) & np.isfinite(tvt)
        if not hidden_mask.any():
            hidden_mask = ~np.isfinite(tvt_in)
            if not hidden_mask.any():
                continue
        hidden_local_idx = np.flatnonzero(hidden_mask)
        local_within_hidden = np.arange(hidden_local_idx.size)
        chunk_ids_arr = (local_within_hidden // max(int(chunk_size), 1)).astype(int)
        well_hidden_count = int(hidden_local_idx.size)
        ts_df = top_state_lookup.get(str(wid))
        for cid in np.unique(chunk_ids_arr):
            local_slice = hidden_local_idx[chunk_ids_arr == cid]
            chunk_ids_str = [ids[i] for i in local_slice]
            chunk_b2 = np.array([b2_map.get(i, np.nan) for i in chunk_ids_str], dtype=np.float64)
            chunk_ko = np.array([k_off_map.get(i, np.nan) for i in chunk_ids_str], dtype=np.float64)
            chunk_tvt = tvt[local_slice]
            valid_truth = np.isfinite(chunk_tvt)
            b2_err = chunk_b2[valid_truth] - chunk_tvt[valid_truth]
            ko_err = chunk_ko[valid_truth] - chunk_tvt[valid_truth]
            b2_finite = b2_err[np.isfinite(b2_err)]
            ko_finite = ko_err[np.isfinite(ko_err)]
            b2_mse = float(np.mean(b2_finite**2)) if b2_finite.size else float("nan")
            ko_mse = float(np.mean(ko_finite**2)) if ko_finite.size else float("nan")
            feats = _chunk_features(
                rows=row_idx[local_slice],
                z=z, md=md, gr=gr,
                b2=chunk_b2, k_offset=chunk_ko,
                well_tvt_input=tvt_in,
                last_known_row=last_known,
                well_hidden_count=well_hidden_count,
                top_state_step_df=ts_df,
                rows_per_step=rows_per_step,
            )
            records.append(
                _ChunkRecord(
                    well_id=str(wid),
                    chunk_id=int(cid),
                    fold=fold_of_well[str(wid)],
                    rows=row_idx[local_slice],
                    ids=chunk_ids_str,
                    b2=chunk_b2,
                    k_offset=chunk_ko,
                    tvt_true=chunk_tvt,
                    b2_mse=b2_mse,
                    k_offset_mse=ko_mse,
                    features=feats,
                )
            )
    if not records:
        raise ValueError("No chunks were built")
    df = pd.DataFrame(
        {
            "well_id": [r.well_id for r in records],
            "chunk_id": [r.chunk_id for r in records],
            "fold": [r.fold for r in records],
            "b2_mse": [r.b2_mse for r in records],
            "k_offset_mse": [r.k_offset_mse for r in records],
            **{
                col: [r.features.get(col, 0.0) for r in records]
                for col in FEATURE_COLUMNS
            },
        }
    )
    df = df.replace([np.inf, -np.inf], np.nan)
    df[list(FEATURE_COLUMNS)] = df[list(FEATURE_COLUMNS)].fillna(0.0)
    # Gate target: 1 if k_offset_mse strictly better than b2_mse - eps,
    # else 0. NaN target for chunks lacking valid truth (test-mode).
    target = np.where(
        np.isfinite(df["b2_mse"]) & np.isfinite(df["k_offset_mse"])
        & (df["k_offset_mse"] < df["b2_mse"] - float(gate_target_eps_ft2)),
        1, 0,
    )
    target_valid = np.isfinite(df["b2_mse"]) & np.isfinite(df["k_offset_mse"])
    df["gate_target"] = pd.Series(target, index=df.index)
    df.loc[~target_valid, "gate_target"] = pd.NA
    assert_schema_safe_columns(list(FEATURE_COLUMNS), context="KOffsetGated features")
    return df, records


def _train_and_predict_gate(
    df: pd.DataFrame,
    *,
    config: KOffsetGatedConfig,
) -> np.ndarray:
    """5-fold OOF posterior ``P(k_offset_better_than_b2 | features)`` per chunk."""
    n_folds = int(df["fold"].max()) + 1 if len(df) else 0
    if n_folds <= 0:
        raise ValueError("no folds available")
    out = np.full(len(df), np.nan, dtype=np.float64)
    feat_cols = list(FEATURE_COLUMNS)
    folds_arr = df["fold"].to_numpy()
    target_notna = df["gate_target"].notna().to_numpy()
    for fold_idx in range(n_folds):
        train_mask = (folds_arr != fold_idx) & target_notna
        valid_mask = folds_arr == fold_idx
        if not train_mask.any() or not valid_mask.any():
            continue
        train_targets = df.loc[train_mask, "gate_target"].astype(int).to_numpy()
        n_unique = int(np.unique(train_targets).size)
        if n_unique < 2:
            # Degenerate fold: predict the constant prior probability.
            prior = float(np.mean(train_targets)) if train_targets.size else 0.5
            print(
                f"[k-offset-gated] fold {fold_idx + 1}/{n_folds} degenerate (single class) — "
                f"using prior gate={prior:.3f}",
                file=sys.stderr,
                flush=True,
            )
            out[valid_mask] = prior
            continue
        pos = int(train_targets.sum())
        n_train = int(train_targets.size)
        print(
            f"[k-offset-gated] fold {fold_idx + 1}/{n_folds} "
            f"train_chunks={n_train} positives={pos} ({100*pos/max(n_train,1):.1f}%)",
            file=sys.stderr,
            flush=True,
        )
        model = CatBoostClassifier(
            loss_function="Logloss",
            iterations=config.iterations,
            learning_rate=config.learning_rate,
            depth=config.depth,
            l2_leaf_reg=config.l2_leaf_reg,
            random_seed=config.seed,
            allow_writing_files=False,
            verbose=False,
            auto_class_weights="Balanced",  # gate target is imbalanced (~30% positives)
        )
        model.fit(
            df.loc[train_mask, feat_cols],
            df.loc[train_mask, "gate_target"].astype(int),
        )
        proba = model.predict_proba(df.loc[valid_mask, feat_cols])
        # CatBoost guarantees column order matches model.classes_ which we sort
        pos_index = int(np.where(model.classes_ == 1)[0][0]) if 1 in model.classes_ else 1
        out[valid_mask] = proba[:, pos_index]
    if np.isnan(out).any():
        raise ValueError("gate posterior contains NaN — fold coverage incomplete")
    return out


def _materialise_predictions(
    records: list[_ChunkRecord],
    *,
    gate_by_chunk: dict[tuple[str, int], float],
    floor: float,
    ceiling: float,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for r in records:
        gate = float(gate_by_chunk.get((r.well_id, r.chunk_id), 0.0))
        gate = float(np.clip(gate, floor, ceiling))
        # Per-row: pred = b2 + gate * (k_offset - b2). Where either is NaN,
        # fall back to the other; if both NaN, NaN propagates and the row
        # is excluded from the output.
        b2 = r.b2.astype(np.float64)
        ko = r.k_offset.astype(np.float64)
        pred = b2 + gate * (ko - b2)
        # If b2 missing but k_offset present: take k_offset
        mask_b2_nan = np.isnan(b2) & ~np.isnan(ko)
        pred = np.where(mask_b2_nan, ko, pred)
        # If k_offset missing but b2 present: take b2
        mask_ko_nan = np.isnan(ko) & ~np.isnan(b2)
        pred = np.where(mask_ko_nan, b2, pred)
        parts.append(
            pd.DataFrame(
                {
                    "id": r.ids,
                    "well_id": r.well_id,
                    "row_idx": r.rows.astype(np.int64),
                    "pred_tvt": pred,
                    "gate": gate,
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def run_k_offset_gated_from_frame(
    frame: pd.DataFrame,
    *,
    config: KOffsetGatedConfig,
    k_offset_preds: pd.DataFrame,
    b2_preds: pd.DataFrame,
    top_state_lookup: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if top_state_lookup is None:
        top_state_lookup = _load_top_state(config.top_state_path)
    raw = _ensure_ids(frame)
    well_ids = sorted({str(w) for w in raw["well_id"].astype(str).unique()})
    folds = make_group_folds(pd.Series(well_ids), n_folds=config.n_folds, seed=config.seed)
    fold_of_well = {str(w): idx for idx, (_, valid) in enumerate(folds) for w in valid}
    df, records = build_chunk_dataset(
        raw,
        k_offset_preds=k_offset_preds,
        b2_preds=b2_preds,
        top_state_lookup=top_state_lookup,
        chunk_size=config.chunk_size,
        rows_per_step=config.rows_per_step,
        fold_of_well=fold_of_well,
        gate_target_eps_ft2=config.gate_target_eps_ft2,
    )
    print(
        f"[k-offset-gated] chunks={len(df):,} wells={df['well_id'].nunique()} "
        f"gate_positives={int((df['gate_target']==1).sum())} "
        f"({100*float((df['gate_target']==1).mean()):.1f}%)",
        file=sys.stderr,
        flush=True,
    )
    gate_proba = _train_and_predict_gate(df, config=config)
    df = df.copy()
    df["gate_pred"] = gate_proba
    gate_by_chunk = {
        (str(row.well_id), int(row.chunk_id)): float(row.gate_pred)
        for row in df.itertuples(index=False)
    }
    row_preds = _materialise_predictions(
        records,
        gate_by_chunk=gate_by_chunk,
        floor=config.gate_floor,
        ceiling=config.gate_ceiling,
    )
    metrics = _evaluate(df, records, row_preds)
    metrics.update(
        {
            "candidate": "k_offset_gated_v0",
            "wells": int(row_preds["well_id"].nunique()),
            "chunks": int(len(df)),
            "feature_columns": list(FEATURE_COLUMNS),
            "config": asdict(config),
        }
    )
    row_preds[["id", "well_id", "row_idx", "pred_tvt"]].to_parquet(
        out_dir / "k_offset_gated_oof_predictions.parquet", index=False
    )
    # also persist the gate per chunk for diagnostics
    df[["well_id", "chunk_id", "gate_target", "gate_pred", "b2_mse", "k_offset_mse"]].to_parquet(
        out_dir / "k_offset_gated_chunks.parquet", index=False
    )
    (out_dir / "k_offset_gated_metrics.json").write_text(
        json.dumps(_json_safe(metrics), indent=2), encoding="utf-8"
    )
    lines = [
        "# K_OFFSET_GATED_V0",
        "",
        "Confidence-gated single candidate. For each chunk, predicts the",
        "probability that `k_offset` will beat `b2`, then materialises",
        "`pred = b2 + gate * (k_offset - b2)` per row. Schema-safe (uses only",
        "test-safe features: top_state aggregates, b2/k_offset disagreement,",
        "GR validity, geometry).",
        "",
        "## summary",
        "```json",
        json.dumps(_json_safe(metrics), indent=2),
        "```",
    ]
    (out_dir / "k_offset_gated_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return metrics


def _binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Fast O(n log n) AUC using the rank-based formulation."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=np.float64)
    valid = np.isfinite(y_score) & ((y_true == 0) | (y_true == 1))
    y_true = y_true[valid]
    y_score = y_score[valid]
    if y_true.size == 0:
        return float("nan")
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # rankdata-style ranks with ties → average rank
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_scores = y_score[order]
    n = y_score.size
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank
        ranks[order[i : j + 1]] = avg
        i = j + 1
    sum_pos_ranks = float(ranks[y_true == 1].sum())
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _evaluate(
    df: pd.DataFrame,
    records: list[_ChunkRecord],
    row_preds: pd.DataFrame,
) -> dict[str, Any]:
    # Gate classifier accuracy / AUC (only on chunks with valid target)
    valid = df["gate_target"].notna()
    if valid.any():
        auc = _binary_auc(
            df.loc[valid, "gate_target"].astype(int).to_numpy(),
            df.loc[valid, "gate_pred"].to_numpy(),
        )
    else:
        auc = float("nan")
    # Per-chunk RMSE of gated candidate vs b2 vs k_offset
    chunk_rmse = []
    for r in records:
        gate = float(df.loc[(df.well_id == r.well_id) & (df.chunk_id == r.chunk_id), "gate_pred"].iloc[0])
        valid_truth = np.isfinite(r.tvt_true)
        if not valid_truth.any():
            continue
        b2 = r.b2[valid_truth]
        ko = r.k_offset[valid_truth]
        truth = r.tvt_true[valid_truth]
        pred = b2 + gate * (ko - b2)
        finite = np.isfinite(pred) & np.isfinite(truth)
        if not finite.any():
            continue
        gated_mse = float(np.mean((pred[finite] - truth[finite]) ** 2))
        b2_finite = b2[finite & np.isfinite(b2)]
        ko_finite = ko[finite & np.isfinite(ko)]
        b2_mse = (
            float(np.mean((b2_finite - truth[finite & np.isfinite(b2)]) ** 2)) if b2_finite.size else float("nan")
        )
        ko_mse = (
            float(np.mean((ko_finite - truth[finite & np.isfinite(ko)]) ** 2)) if ko_finite.size else float("nan")
        )
        chunk_rmse.append(
            {
                "well_id": r.well_id,
                "chunk_id": r.chunk_id,
                "rows": int(finite.sum()),
                "b2_mse": b2_mse,
                "ko_mse": ko_mse,
                "gated_mse": gated_mse,
                "gate": gate,
            }
        )
    cdf = pd.DataFrame(chunk_rmse)
    if cdf.empty:
        return {"gate_auc": auc, "rows_evaluated": 0}
    # Pooled (row-weighted)
    def _pooled(col: str) -> float:
        return float(
            np.sqrt(
                (cdf[col].fillna(0) * cdf["rows"]).sum() / cdf["rows"].sum()
            )
        )
    return {
        "gate_auc": auc,
        "rows_evaluated": int(cdf["rows"].sum()),
        "chunks_evaluated": int(len(cdf)),
        "b2_pooled_rmse": _pooled("b2_mse"),
        "k_offset_pooled_rmse": _pooled("ko_mse"),
        "gated_pooled_rmse": _pooled("gated_mse"),
        "beats_b2_pct": float((cdf["gated_mse"] < cdf["b2_mse"]).mean()),
        "improves_vs_b2_ft": float(
            np.sqrt((cdf["b2_mse"].fillna(0) * cdf["rows"]).sum() / cdf["rows"].sum())
            - np.sqrt((cdf["gated_mse"].fillna(0) * cdf["rows"]).sum() / cdf["rows"].sum())
        ),
        "gate_mean": float(cdf["gate"].mean()),
        "gate_median": float(cdf["gate"].median()),
        "gate_above_0.5_pct": float((cdf["gate"] > 0.5).mean()),
    }


def run_k_offset_gated(config: KOffsetGatedConfig) -> dict[str, Any]:
    frame = load_training_frame(
        ResidualStackConfig(data_dir=config.data_dir, k_wells=config.k_wells)
    )
    print(
        f"[k-offset-gated] loaded frame rows={len(frame):,} wells={frame['well_id'].nunique()}",
        file=sys.stderr,
        flush=True,
    )
    k_offset_preds = pd.read_parquet(config.k_offset_path)
    print(f"[k-offset-gated] k_offset rows={len(k_offset_preds):,}", file=sys.stderr, flush=True)
    b2_preds = _load_b2(config.b2_path, config.b2_column)
    print(f"[k-offset-gated] b2 rows={len(b2_preds):,}", file=sys.stderr, flush=True)
    return run_k_offset_gated_from_frame(
        frame,
        config=config,
        k_offset_preds=k_offset_preds,
        b2_preds=b2_preds,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train fold-safe confidence gate for k_segment_offset_v0."
    )
    parser.add_argument("--data-dir", type=Path, default=KOffsetGatedConfig.data_dir)
    parser.add_argument("--output-dir", type=Path, default=KOffsetGatedConfig.output_dir)
    parser.add_argument("--k-offset-path", type=Path, default=KOffsetGatedConfig.k_offset_path)
    parser.add_argument("--b2-path", type=Path, default=KOffsetGatedConfig.b2_path)
    parser.add_argument("--b2-column", default=KOffsetGatedConfig.b2_column)
    parser.add_argument("--top-state-path", type=Path, default=KOffsetGatedConfig.top_state_path)
    parser.add_argument("--rows-per-step", type=int, default=KOffsetGatedConfig.rows_per_step)
    parser.add_argument("--chunk-size", type=int, default=KOffsetGatedConfig.chunk_size)
    parser.add_argument("--n-folds", type=int, default=KOffsetGatedConfig.n_folds)
    parser.add_argument("--seed", type=int, default=KOffsetGatedConfig.seed)
    parser.add_argument("--k-wells", type=int, default=KOffsetGatedConfig.k_wells)
    parser.add_argument("--iterations", type=int, default=KOffsetGatedConfig.iterations)
    parser.add_argument("--learning-rate", type=float, default=KOffsetGatedConfig.learning_rate)
    parser.add_argument("--depth", type=int, default=KOffsetGatedConfig.depth)
    parser.add_argument("--l2-leaf-reg", type=float, default=KOffsetGatedConfig.l2_leaf_reg)
    parser.add_argument("--gate-target-eps-ft2", type=float, default=KOffsetGatedConfig.gate_target_eps_ft2)
    parser.add_argument("--gate-floor", type=float, default=KOffsetGatedConfig.gate_floor)
    parser.add_argument("--gate-ceiling", type=float, default=KOffsetGatedConfig.gate_ceiling)
    parser.add_argument("--progress-every", type=int, default=KOffsetGatedConfig.progress_every)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metrics = run_k_offset_gated(
        KOffsetGatedConfig(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            k_offset_path=args.k_offset_path,
            b2_path=args.b2_path,
            b2_column=args.b2_column,
            top_state_path=args.top_state_path,
            rows_per_step=args.rows_per_step,
            chunk_size=args.chunk_size,
            n_folds=args.n_folds,
            seed=args.seed,
            k_wells=args.k_wells,
            iterations=args.iterations,
            learning_rate=args.learning_rate,
            depth=args.depth,
            l2_leaf_reg=args.l2_leaf_reg,
            gate_target_eps_ft2=args.gate_target_eps_ft2,
            gate_floor=args.gate_floor,
            gate_ceiling=args.gate_ceiling,
            progress_every=args.progress_every,
        )
    )
    compact = {
        "candidate": metrics["candidate"],
        "wells": metrics["wells"],
        "chunks": metrics["chunks"],
        "gate_auc": metrics.get("gate_auc"),
        "b2_pooled_rmse": metrics.get("b2_pooled_rmse"),
        "k_offset_pooled_rmse": metrics.get("k_offset_pooled_rmse"),
        "gated_pooled_rmse": metrics.get("gated_pooled_rmse"),
        "improves_vs_b2_ft": metrics.get("improves_vs_b2_ft"),
        "gate_mean": metrics.get("gate_mean"),
        "gate_above_0.5_pct": metrics.get("gate_above_0.5_pct"),
        "report": str(Path(args.output_dir) / "k_offset_gated_report.md"),
    }
    print(json.dumps(_json_safe(compact), indent=2))


if __name__ == "__main__":
    main()
