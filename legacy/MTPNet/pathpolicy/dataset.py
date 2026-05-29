"""
pathpolicy/dataset.py — WellChunkDataset for PolicyFormer training.

Each sample is one (well, chunk) pair from oracle_chunks.parquet.

Context window: last PAST_LEN known rows + CHUNK_LEN hidden rows → [T, D_FEAT].
Action segments: [N_ACTION_TYPES, CHUNK_LEN] normalized delta from B2.
Labels: oracle_label (int in 0..N_ACTION_TYPES-1), value_target (oracle_rmse).

Features per row (D_FEAT = 7):
  [0]  GR normalized (0 if missing)
  [1]  gr_valid (1 if GR present)
  [2]  b2_delta   = (b2_tvt  - last_known_tvt) / 100  (0 if no B2)
  [3]  base_delta = (base_tvt - last_known_tvt) / 100  (0 if no base)
  [4]  a_delta    = (a_p50_tvt - last_known_tvt) / 100 (0 if no A)
  [5]  tvt_delta  = (TVT_input - last_known_tvt) / 100 (0 for hidden rows)
  [6]  is_hidden  (0/1)

NOTE: Formation columns (ANCC/ASTNU/ASTNL/EGFDU/EGFDL/BUDA) are intentionally
excluded — they are present in train CSVs but absent from test CSVs and therefore
cannot be used at inference time.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from pathpolicy.actions import (
    CHUNK_LEN,
    N_ACTION_TYPES,
    ACTION_TO_IDX,
    MTP_NAMES,
    generate_action_segments,
    normalize_action_segs,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PAST_LEN: int = 64  # context rows from known section
D_FEAT: int = 7     # features per row (no formation cols — not available at test time)

_GR_MEAN: float = 111.0
_GR_STD: float = 24.0

_PRIOR_NAMES_CTX: list[str] = ["b2", "base", "a_p50"]  # used in context features

# Default paths (relative to repo root)
_DATA_DIR = Path("data/train")
_BASE_PATH = Path("../old/artifacts/oof_baseline/schema10_oof.parquet")
_B2_PATH = Path("../old/artifacts/formation_b2_danger_guard_a2_full_schema10/guarded_predictions.parquet")
_A_PATH = Path("../old/artifacts/formation_plane_knn/oof_candidates.parquet")
_MTP_PATH = Path("artifacts/mtp_v3_gr_forced/track_row_predictions.parquet")
_ORACLE_CHUNKS = Path("artifacts/oracle_v0/oracle_chunks.parquet")


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_well_csvs(data_dir: Path, well_ids: list[str]) -> dict[str, pd.DataFrame]:
    """Load all well CSVs into memory, keyed by well_id."""
    wells: dict[str, pd.DataFrame] = {}
    for wid in well_ids:
        fp = data_dir / f"{wid}__horizontal_well.csv"
        if fp.exists():
            df = pd.read_csv(fp, usecols=["MD", "GR", "TVT", "TVT_input"])
            wells[wid] = df
    return wells


def load_priors(
    base_path: Path = _BASE_PATH,
    b2_path: Path = _B2_PATH,
    a_path: Path = _A_PATH,
) -> dict[str, pd.Series]:
    base_df = pd.read_parquet(base_path, columns=["id", "schema10_oof_pp"]).set_index("id")["schema10_oof_pp"]
    b2_df   = pd.read_parquet(b2_path,   columns=["id", "b2_guarded_submit"]).set_index("id")["b2_guarded_submit"]
    a_df    = pd.read_parquet(a_path,    columns=["id", "formation_sample_median",
                                                        "formation_sample_p10",
                                                        "formation_sample_p90"]).set_index("id")
    return {
        "base":  base_df,
        "b2":    b2_df,
        "a_p50": a_df["formation_sample_median"],
        "a_p10": a_df["formation_sample_p10"],
        "a_p90": a_df["formation_sample_p90"],
    }


def load_mtp_pivot(track_path: Path = _MTP_PATH) -> pd.DataFrame:
    tr = pd.read_parquet(track_path, columns=["well_id", "row_idx", "pred_tvt", "candidate"])
    piv = tr.pivot_table(index=["well_id", "row_idx"], columns="candidate", values="pred_tvt")
    piv.columns = [f"mtp_{c}" for c in piv.columns]
    return piv


def _extract_prior_arrays(
    priors: dict[str, pd.Series],
    wells: dict[str, pd.DataFrame],
) -> dict[str, dict[str, np.ndarray]]:
    """Pre-extract prior TVT values per well into numpy arrays indexed by row number.

    Returns: {well_id: {pname: np.ndarray[n_rows]}} with NaN where missing.
    This converts the slow string-indexed pandas reindex into fast numpy indexing.
    """
    result: dict[str, dict[str, np.ndarray]] = {}
    for wid, df in wells.items():
        n_rows = len(df)
        well_arrs: dict[str, np.ndarray] = {}
        ids = [f"{wid}_{r}" for r in range(n_rows)]
        for pname, series in priors.items():
            arr = series.reindex(ids).values.astype(np.float32)
            well_arrs[pname] = arr
        result[wid] = well_arrs
    return result


def _extract_mtp_arrays(
    mtp_piv: pd.DataFrame,
    wells: dict[str, pd.DataFrame],
) -> dict[str, dict[str, np.ndarray]]:
    """Pre-extract MTP TVT values per well into numpy arrays indexed by row number."""
    result: dict[str, dict[str, np.ndarray]] = {}
    mtp_well_ids = set(mtp_piv.index.get_level_values("well_id").unique())
    for wid, df in wells.items():
        if wid not in mtp_well_ids:
            result[wid] = {}
            continue
        n_rows = len(df)
        well_mtp = mtp_piv.xs(wid, level="well_id")  # index=row_idx
        well_arrs: dict[str, np.ndarray] = {}
        for col in well_mtp.columns:
            arr = np.full(n_rows, np.nan, dtype=np.float32)
            row_idxs = well_mtp.index.values
            valid = (row_idxs >= 0) & (row_idxs < n_rows)
            arr[row_idxs[valid]] = well_mtp[col].values[valid].astype(np.float32)
            well_arrs[col] = arr
        result[wid] = well_arrs
    return result


# ---------------------------------------------------------------------------
# Feature extraction (numpy-only, no pandas after dataset init)
# ---------------------------------------------------------------------------

def extract_context_features_fast(
    well_df: pd.DataFrame,
    prior_arrs: dict[str, np.ndarray],   # pname → [n_rows]
    chunk_rows: np.ndarray,              # hidden row indices for this chunk
    known_indices: np.ndarray,
    last_known_tvt: float,
) -> np.ndarray:
    """Return context feature matrix [PAST_LEN + len(chunk_rows), D_FEAT].

    D_FEAT = 7: GR_norm, gr_valid, b2_delta, base_delta, a_p50_delta, tvt_delta, is_hidden.
    Formation columns are excluded — they are absent from test data.
    """
    chunk_start_row = int(chunk_rows[0])

    # Past context: last PAST_LEN known rows before hidden section
    past_rows = known_indices[known_indices < chunk_start_row]
    if len(past_rows) >= PAST_LEN:
        past_rows = past_rows[-PAST_LEN:]
    else:
        pad_idx = past_rows[0] if len(past_rows) > 0 else 0
        pad = np.full(PAST_LEN - len(past_rows), pad_idx, dtype=int)
        past_rows = np.concatenate([pad, past_rows]).astype(int)

    all_rows = np.concatenate([past_rows, chunk_rows]).astype(int)
    T = len(all_rows)
    is_hidden = np.zeros(T, dtype=np.float32)
    is_hidden[PAST_LEN:] = 1.0

    feat = np.zeros((T, D_FEAT), dtype=np.float32)

    # [0] GR normalized, [1] gr_valid
    gr_arr = well_df["GR"].values.astype(np.float32)
    gr_vals = gr_arr[all_rows]
    gr_valid = np.isfinite(gr_vals).astype(np.float32)
    feat[:, 0] = np.where(gr_valid, (gr_vals - _GR_MEAN) / _GR_STD, 0.0)
    feat[:, 1] = gr_valid

    # [2-4] Prior deltas: b2, base, a_p50 (numpy array lookup, no pandas)
    for pi, pname in enumerate(_PRIOR_NAMES_CTX):
        parr = prior_arrs.get(pname)
        if parr is not None:
            pvals = parr[all_rows]
            feat[:, 2 + pi] = np.where(np.isfinite(pvals), (pvals - last_known_tvt) / 100.0, 0.0)

    # [5] TVT delta (0 for hidden rows since TVT_input is NaN there)
    tvt_arr = well_df["TVT_input"].values.astype(np.float32)
    tvt_vals = tvt_arr[all_rows]
    feat[:, 5] = np.where(np.isfinite(tvt_vals), (tvt_vals - last_known_tvt) / 100.0, 0.0)

    # [6] is_hidden
    feat[:, 6] = is_hidden

    return feat  # [T, D_FEAT]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PolicyDataset(Dataset):
    """One sample per (well, chunk) from oracle_chunks.parquet."""

    def __init__(
        self,
        oracle_chunks: pd.DataFrame,
        wells: dict[str, pd.DataFrame],
        priors: dict[str, pd.Series],
        mtp_piv: pd.DataFrame,
        prior_arrs: dict[str, dict[str, np.ndarray]],   # pre-extracted
        mtp_arrs: dict[str, dict[str, np.ndarray]],     # pre-extracted
        well_ids: list[str] | None = None,
    ) -> None:
        self.wells = wells
        self.prior_arrs = prior_arrs   # {well_id: {pname: np.ndarray[n_rows]}}
        self.mtp_arrs   = mtp_arrs     # {well_id: {col:   np.ndarray[n_rows]}}

        if well_ids is not None:
            oracle_chunks = oracle_chunks[oracle_chunks["well_id"].isin(set(well_ids))].copy()
        self.chunks = oracle_chunks.reset_index(drop=True)

        self.chunks["oracle_label"] = self.chunks["oracle_action"].map(ACTION_TO_IDX)
        missing = self.chunks["oracle_label"].isna().sum()
        if missing > 0:
            print(f"[dataset] WARNING: {missing} chunks have unrecognized oracle_action; dropping.")
            self.chunks = self.chunks[self.chunks["oracle_label"].notna()].reset_index(drop=True)
        self.chunks["oracle_label"] = self.chunks["oracle_label"].astype(int)

        # Per-action RMSE matrix [n_chunks, N_ACTION_TYPES] — from oracle_v1 columns rmse_0..rmse_N-1.
        # Used for soft-label KL training. NaN = action unavailable for that chunk.
        rmse_cols = [f"rmse_{i}" for i in range(N_ACTION_TYPES)]
        available_rmse_cols = [c for c in rmse_cols if c in self.chunks.columns]
        if available_rmse_cols:
            self._action_rmses = np.full(
                (len(self.chunks), N_ACTION_TYPES), np.nan, dtype=np.float32
            )
            for col in available_rmse_cols:
                idx = int(col.split("_")[1])
                self._action_rmses[:, idx] = self.chunks[col].values.astype(np.float32)
            print(
                f"[dataset] loaded per-action RMSE for {len(available_rmse_cols)}/{N_ACTION_TYPES} actions",
                flush=True,
            )
        else:
            self._action_rmses = None
            print("[dataset] WARNING: no per-action RMSE cols found; soft-label KL will be skipped", flush=True)

        # Pre-compute per-well hidden/known indices
        self._well_hidden: dict[str, np.ndarray] = {}
        self._well_known: dict[str, np.ndarray] = {}
        for wid, df in wells.items():
            hidden_mask = df["TVT_input"].isna().values
            row_indices = np.arange(len(df))
            self._well_known[wid] = row_indices[~hidden_mask]
            self._well_hidden[wid] = row_indices[hidden_mask]

        # Cache numpy arrays for well CSV columns (avoid repeated .values calls)
        self._well_tvt_input: dict[str, np.ndarray] = {
            wid: df["TVT_input"].values.astype(np.float32) for wid, df in wells.items()
        }

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> dict:
        row = self.chunks.iloc[idx]
        well_id: str = row["well_id"]
        c_start: int = int(row["chunk_start_hidden"])
        oracle_label: int = int(row["oracle_label"])
        oracle_rmse: float = float(row["oracle_rmse"])

        well_df = self.wells[well_id]
        hidden_indices = self._well_hidden[well_id]
        known_indices  = self._well_known[well_id]
        tvt_input_arr  = self._well_tvt_input[well_id]
        prior_arrs     = self.prior_arrs[well_id]
        mtp_arrs       = self.mtp_arrs.get(well_id, {})

        # Chunk rows
        c_end = min(c_start + CHUNK_LEN, len(hidden_indices))
        chunk_rows = hidden_indices[c_start:c_end]
        L = len(chunk_rows)

        # Last known TVT and slope (from known section)
        last_known_idx = int(known_indices[-1])
        last_known_tvt = float(tvt_input_arr[last_known_idx])
        if len(known_indices) >= 3:
            k3 = known_indices[-3:]
            slope_per_row = float(np.polyfit(np.arange(3), tvt_input_arr[k3], 1)[0])
        else:
            slope_per_row = 0.0

        # B2-anchored slope for subsequent chunks
        if c_start > 0:
            b2_arr = prior_arrs.get("b2")
            if b2_arr is not None:
                prev_idxs = hidden_indices[max(0, c_start - CHUNK_LEN): c_start]
                prev_b2 = b2_arr[prev_idxs]
                valid_b2 = prev_b2[np.isfinite(prev_b2)]
                if len(valid_b2) >= 2:
                    last_known_tvt = float(valid_b2[-1])
                    slope_per_row = float(np.polyfit(np.arange(len(valid_b2)), valid_b2, 1)[0])
                elif len(valid_b2) == 1:
                    last_known_tvt = float(valid_b2[-1])

        # Build prior_segs dict for action generation
        prior_segs: dict[str, np.ndarray | None] = {}
        for pname, parr in prior_arrs.items():
            seg = parr[chunk_rows]
            prior_segs[pname] = seg if np.isfinite(seg).any() else None
        for col, marr in mtp_arrs.items():
            seg = marr[chunk_rows]
            if np.isfinite(seg).any():
                prior_segs[col] = seg

        # Action segments [N_ACTION_TYPES, L]
        segs, mask = generate_action_segments(prior_segs, last_known_tvt, slope_per_row, L=L)

        # Pad to CHUNK_LEN if last chunk is shorter — use edge-value fill, not zeros.
        # Zero-padding creates a spurious drop to 0 at chunk end that the model can
        # exploit as a positional signal; edge-fill (repeat last row) is neutral.
        if L < CHUNK_LEN:
            edge = segs[:, L - 1 : L]  # [N_ACTION_TYPES, 1]
            pad  = np.repeat(edge, CHUNK_LEN - L, axis=1)  # [N_ACTION_TYPES, CHUNK_LEN-L]
            segs = np.concatenate([segs, pad], axis=1)
            b2_seg_raw = prior_segs.get("b2")
            if b2_seg_raw is not None:
                b2_edge = b2_seg_raw[-1:] if np.isfinite(b2_seg_raw[-1]) else np.array([np.nan], dtype=np.float32)
                b2_seg = np.concatenate([b2_seg_raw, np.full(CHUNK_LEN - L, float(b2_edge[0]), dtype=np.float32)])
            else:
                b2_seg = None
        else:
            b2_seg = prior_segs.get("b2")

        # Normalize: delta from B2 anchor
        delta_segs = normalize_action_segs(segs, mask, b2_seg, last_known_tvt)

        # Context features [PAST_LEN + L, D_FEAT]
        context = extract_context_features_fast(
            well_df=well_df,
            prior_arrs=prior_arrs,
            chunk_rows=chunk_rows,
            known_indices=known_indices,
            last_known_tvt=float(tvt_input_arr[known_indices[-1]]),
        )

        return {
            "context":      torch.from_numpy(context),       # [PAST+L, D_FEAT]
            "action_segs":  torch.from_numpy(delta_segs),    # [N_ACTION_TYPES, CHUNK_LEN]
            "action_mask":  torch.from_numpy(mask),          # [N_ACTION_TYPES] bool
            "oracle_label": torch.tensor(oracle_label, dtype=torch.long),
            "value_target": torch.tensor(oracle_rmse, dtype=torch.float32),
            "action_rmses": torch.from_numpy(
                self._action_rmses[idx] if self._action_rmses is not None
                else np.full(N_ACTION_TYPES, np.nan, dtype=np.float32)
            ),  # [N_ACTION_TYPES] float, NaN = unavailable
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_datasets(
    oracle_chunks_path: Path = _ORACLE_CHUNKS,
    data_dir: Path = _DATA_DIR,
    base_path: Path = _BASE_PATH,
    b2_path: Path = _B2_PATH,
    a_path: Path = _A_PATH,
    mtp_path: Path = _MTP_PATH,
    val_frac: float = 0.2,
    seed: int = 42,
    tail_oversample: int = 1,
) -> tuple["PolicyDataset", "PolicyDataset", list[str], list[str]]:
    """Return (train_dataset, val_dataset, train_well_ids, val_well_ids).

    tail_oversample: repeat tail-well chunks this many extra times in training
    (e.g. tail_oversample=3 means tail chunks appear 3× as often as regular chunks).
    """
    print("[dataset] loading oracle chunks...", flush=True)
    chunks = pd.read_parquet(oracle_chunks_path)

    print("[dataset] loading priors...", flush=True)
    priors = load_priors(base_path, b2_path, a_path)

    print("[dataset] loading MTP pivot...", flush=True)
    mtp_piv = load_mtp_pivot(mtp_path)

    well_ids = chunks["well_id"].unique().tolist()
    print(f"[dataset] loading {len(well_ids)} well CSVs...", flush=True)
    wells = load_well_csvs(data_dir, well_ids)

    print("[dataset] pre-extracting prior arrays...", flush=True)
    prior_arrs = _extract_prior_arrays(priors, wells)

    print("[dataset] pre-extracting MTP arrays...", flush=True)
    mtp_arrs = _extract_mtp_arrays(mtp_piv, wells)

    # Well-level split — stratified by tail class so tail wells appear proportionally in train
    rng = np.random.default_rng(seed)
    tail_audit_path = Path("artifacts/tail_audit_v1/well_tail_audit.csv")
    wids = sorted(wells.keys())

    if tail_audit_path.exists():
        tail_df = pd.read_csv(tail_audit_path, usecols=["well_id"])
        tail_set = set(tail_df["well_id"])
        tail_wids    = [w for w in wids if w in tail_set]
        nontail_wids = [w for w in wids if w not in tail_set]
        rng.shuffle(tail_wids)
        rng.shuffle(nontail_wids)
        n_val_tail    = max(1, int(len(tail_wids)    * val_frac))
        n_val_nontail = max(1, int(len(nontail_wids) * val_frac))
        val_ids   = set(tail_wids[:n_val_tail] + nontail_wids[:n_val_nontail])
        train_ids = [w for w in wids if w not in val_ids]
        print(
            f"[dataset] stratified split: "
            f"tail train={len(tail_wids)-n_val_tail}/{len(tail_wids)}, "
            f"nontail train={len(nontail_wids)-n_val_nontail}/{len(nontail_wids)}",
            flush=True,
        )
    else:
        rng.shuffle(wids)
        n_val = max(1, int(len(wids) * val_frac))
        val_ids   = set(wids[:n_val])
        train_ids = [w for w in wids if w not in val_ids]

    print(f"[dataset] {len(train_ids)} train wells / {len(val_ids)} val wells", flush=True)

    # Tail-well oversampling: duplicate tail-well chunks in train split
    train_chunks = chunks[chunks["well_id"].isin(set(train_ids))].copy()
    if tail_oversample > 1:
        tail_audit_path = Path("artifacts/tail_audit_v1/well_tail_audit.csv")
        if tail_audit_path.exists():
            tail_df = pd.read_csv(tail_audit_path, usecols=["well_id"])
            tail_train_ids = set(tail_df["well_id"]) & set(train_ids)
            tail_chunks = train_chunks[train_chunks["well_id"].isin(tail_train_ids)]
            extra = pd.concat([tail_chunks] * (tail_oversample - 1), ignore_index=True)
            train_chunks = pd.concat([train_chunks, extra], ignore_index=True)
            print(
                f"[dataset] tail oversample ×{tail_oversample}: "
                f"{len(tail_train_ids)} tail wells, "
                f"{len(extra)} extra chunks added",
                flush=True,
            )

    train_ds = PolicyDataset(train_chunks, wells, priors, mtp_piv, prior_arrs, mtp_arrs)
    val_ds   = PolicyDataset(chunks, wells, priors, mtp_piv, prior_arrs, mtp_arrs, well_ids=list(val_ids))

    print(f"[dataset] {len(train_ds)} train chunks / {len(val_ds)} val chunks", flush=True)
    return train_ds, val_ds, sorted(train_ids), sorted(val_ids)
