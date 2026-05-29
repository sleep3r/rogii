"""PathFormer dataset: full-well feature engineering.

One training sample = one well.
Features are built at compressed step resolution (rows_per_step raw rows → 1 step).

Feature layout (N_FEATURES = 28, must match config.py):
  [0]  step_frac              — position in sequence, 0→1
  [1]  is_known               — 1.0 if TVT_input is available, 0.0 if hidden
  [2]  known_progress         — step / first_hidden_step  (0→1 for known, 1 for hidden)
  [3]  hidden_progress        — (step - first_hidden) / hidden_len  (0 for known, 0→1 for hidden)
  [4]  gr_mean                — nanmean(GR) / 100.0   (0 if all NaN)
  [5]  gr_valid_frac          — fraction of valid GR samples in this step
  [6]  gr_std                 — nanstd(GR) / 50.0     (0 if < 2 valid samples)
  [7]  tvt_delta_step         — (TVT_input[t] - TVT_input[t-1]) / 10.0  for known, 0 for hidden
  [8]  tvt_from_last_known    — (TVT_input[t] - last_known_tvt) / 100.0  for known, 0 for hidden
  [9]  ancc_rel               — (ANCC - last_known_tvt) / 100.0
  [10] astnu_rel              — (ASTNU - last_known_tvt) / 100.0
  [11] astnl_rel              — (ASTNL - last_known_tvt) / 100.0
  [12] egfdu_rel              — (EGFDU - last_known_tvt) / 100.0
  [13] egfdl_rel              — (EGFDL - last_known_tvt) / 100.0
  [14] buda_rel               — (BUDA - last_known_tvt) / 100.0
  [15] boundary_valid         — 1.0 if all 6 boundaries are finite, else 0.0
  [16] b2_delta               — (b2_tvt - last_known_tvt) / 100.0  (0 if prior absent/dropped)
  [17] base_delta             — (base_tvt - last_known_tvt) / 100.0
  [18] a_p50_delta            — (a_p50_tvt - last_known_tvt) / 100.0
  [19] a_p10_delta
  [20] a_p90_delta
  [21] b2_valid               — 1.0 if b2 prior available and not dropped
  [22] base_valid             — 1.0 if base prior available and not dropped
  [23] a_valid                — 1.0 if a prior available and not dropped
  [24] a_spread               — (a_p90 - a_p10) / 100.0  (0 if a absent/dropped)
  [25] hidden_frac            — hidden_steps / total_steps  (global, broadcast)
  [26] gr_nan_frac_hidden     — frac of hidden steps with NaN GR  (global, broadcast)
  [27] log_total_steps        — log(total_steps) / 8.0  (global, broadcast)

Target: tvt_delta = TVT - last_known_tvt  (in ft)
  — shape: (seq_len,)
  — only hidden steps are used for loss

Padding: sequences are padded to max_seq_len with zeros; pad_mask marks real steps.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import PathFormerConfig, PFPriorConfig, N_FEATURES

# Formation boundary columns in the horizontal CSV (absolute TVT depths).
# These columns are not available in the competition test schema, so this
# PathFormer branch is explicitly diagnostic until retrained without them.
FORMATION_COLS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
DEPLOYABLE = False
DIAGNOSTIC_ONLY_REASON = (
    "PathFormer v0 reads horizontal formation boundary columns; those are "
    "train-only and must not be used by deployable inference models."
)

# Scale factors for normalization (ft)
GR_SCALE = 100.0
GR_STD_SCALE = 50.0
TVT_STEP_SCALE = 10.0   # per-step delta
TVT_SCALE = 100.0       # larger deltas (from last known, priors)


# ---------------------------------------------------------------------------
# Raw prior loading (lightweight, re-uses same parquet paths as MTPNet)
# ---------------------------------------------------------------------------

def _load_prior_frame(cfg: PFPriorConfig) -> pd.DataFrame | None:
    """Load all prior columns into a single DataFrame indexed by row id."""
    if not cfg.enabled:
        return None

    parts: list[pd.DataFrame] = []

    def _read(path: Path, src_col: str, dst_col: str) -> pd.DataFrame:
        p = Path(path)
        if p.suffix.lower() == ".csv":
            df = pd.read_csv(p, usecols=[cfg.id_column, src_col])
        else:
            df = pd.read_parquet(p, columns=[cfg.id_column, src_col])
        df = df.rename(columns={src_col: dst_col})
        df[cfg.id_column] = df[cfg.id_column].astype(str)
        return df.drop_duplicates(subset=[cfg.id_column]).set_index(cfg.id_column)

    if cfg.base_path is not None:
        parts.append(_read(cfg.base_path, cfg.base_column, "base_tvt"))
    if cfg.b2_path is not None:
        parts.append(_read(cfg.b2_path, cfg.b2_column, "b2_tvt"))
    if cfg.a_path is not None:
        for src, dst in [
            (cfg.a_p50_column, "a_p50_tvt"),
            (cfg.a_p10_column, "a_p10_tvt"),
            (cfg.a_p90_column, "a_p90_tvt"),
        ]:
            parts.append(_read(cfg.a_path, src, dst))

    if not parts:
        return None

    result = parts[0]
    for part in parts[1:]:
        result = result.join(part, how="outer")
    return result


# ---------------------------------------------------------------------------
# Per-step compression helpers
# ---------------------------------------------------------------------------

def _nanmean_steps(values: np.ndarray, rows_per_step: int) -> np.ndarray:
    usable = (len(values) // rows_per_step) * rows_per_step
    if usable == 0:
        return np.empty(0, dtype=np.float32)
    arr = np.asarray(values[:usable], dtype=np.float32).reshape(-1, rows_per_step)
    finite = np.isfinite(arr)
    sums = np.where(finite, arr, 0.0).sum(axis=1)
    counts = finite.sum(axis=1).astype(np.float32)
    out = np.full(arr.shape[0], np.nan, dtype=np.float32)
    np.divide(sums, counts, out=out, where=counts > 0)
    return out


def _mean_steps(values: np.ndarray, rows_per_step: int) -> np.ndarray:
    usable = (len(values) // rows_per_step) * rows_per_step
    if usable == 0:
        return np.empty(0, dtype=np.float32)
    return np.asarray(values[:usable], dtype=np.float32).reshape(-1, rows_per_step).mean(axis=1)


def _valid_frac_steps(values: np.ndarray, rows_per_step: int) -> np.ndarray:
    """Fraction of finite values per step."""
    usable = (len(values) // rows_per_step) * rows_per_step
    if usable == 0:
        return np.empty(0, dtype=np.float32)
    arr = np.isfinite(np.asarray(values[:usable], dtype=np.float32)).reshape(-1, rows_per_step)
    return arr.mean(axis=1).astype(np.float32)


def _nanstd_steps(values: np.ndarray, rows_per_step: int) -> np.ndarray:
    usable = (len(values) // rows_per_step) * rows_per_step
    if usable == 0:
        return np.empty(0, dtype=np.float32)
    arr = np.asarray(values[:usable], dtype=np.float32).reshape(-1, rows_per_step)
    out = np.zeros(arr.shape[0], dtype=np.float32)
    for i in range(arr.shape[0]):
        row = arr[i]
        finite = row[np.isfinite(row)]
        if len(finite) >= 2:
            out[i] = float(np.std(finite))
    return out


# ---------------------------------------------------------------------------
# WellSample dataclass
# ---------------------------------------------------------------------------

@dataclass
class WellSample:
    well_id: str
    features: np.ndarray      # (seq_len, N_FEATURES) float32
    target_delta: np.ndarray  # (seq_len,) float32 = TVT - last_known_tvt
    hidden_mask: np.ndarray   # (seq_len,) bool — True = hidden step
    seq_len: int
    last_known_tvt: float
    first_hidden_step: int
    tail_class: str            # from tail audit; "unknown" if not available
    hidden_row_ids: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=object)
    )
    hidden_row_idx: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int32)
    )
    hidden_row_steps: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int32)
    )
    hidden_row_tvt: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32)
    )
    hidden_row_gr: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32)
    )


# ---------------------------------------------------------------------------
# Feature engineering for one well
# ---------------------------------------------------------------------------

def _build_well_sample(
    well_id: str,
    horizontal: pd.DataFrame,
    prior_frame: pd.DataFrame | None,
    rows_per_step: int,
    max_seq_len: int,
    tail_class: str,
) -> WellSample | None:
    """Build a WellSample from raw CSVs + priors.  Returns None if well is unusable."""

    n_raw = len(horizontal)
    n_steps = n_raw // rows_per_step
    if n_steps < 4:
        return None

    # ---- compress core columns ----
    tvt_raw = pd.to_numeric(horizontal["TVT"], errors="coerce").to_numpy(np.float32)
    tvt_input_raw = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(np.float32)
    gr_raw = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(np.float32)

    comp_tvt = _nanmean_steps(tvt_raw, rows_per_step)
    comp_tvt_input = _nanmean_steps(tvt_input_raw, rows_per_step)
    comp_gr_mean = _nanmean_steps(gr_raw, rows_per_step)
    comp_gr_valid = _valid_frac_steps(gr_raw, rows_per_step)
    comp_gr_std = _nanstd_steps(gr_raw, rows_per_step)

    # ---- formation boundaries ----
    form_arrays: dict[str, np.ndarray] = {}
    for col in FORMATION_COLS:
        if col in horizontal.columns:
            raw = pd.to_numeric(horizontal[col], errors="coerce").to_numpy(np.float32)
            form_arrays[col] = _nanmean_steps(raw, rows_per_step)
        else:
            form_arrays[col] = np.full(n_steps, np.nan, dtype=np.float32)

    # ---- priors ----
    ids = [f"{well_id}_{i}" for i in range(n_raw)]
    prior_cols: dict[str, np.ndarray] = {}
    for pname in ["b2_tvt", "base_tvt", "a_p50_tvt", "a_p10_tvt", "a_p90_tvt"]:
        if prior_frame is not None and pname in prior_frame.columns:
            vals = prior_frame.reindex(ids)[pname].to_numpy(np.float32)
            prior_cols[pname] = _nanmean_steps(vals, rows_per_step)
        else:
            prior_cols[pname] = np.full(n_steps, np.nan, dtype=np.float32)

    # ---- identify known / hidden boundary ----
    hidden = ~np.isfinite(comp_tvt_input)
    hidden_indices = np.flatnonzero(hidden)
    if len(hidden_indices) == 0:
        return None   # no hidden rows — skip
    first_hidden = int(hidden_indices[0])
    hidden_len = int(hidden.sum())
    total_steps = n_steps

    # last known TVT — the anchor for all delta features
    known_indices = np.flatnonzero(~hidden)
    if len(known_indices) == 0:
        return None
    last_known_tvt = float(comp_tvt_input[known_indices[-1]])
    if not np.isfinite(last_known_tvt):
        return None

    # ground-truth TVT must be finite for all hidden steps (training only)
    hidden_tvt = comp_tvt[hidden]
    if not np.all(np.isfinite(hidden_tvt)):
        # some hidden steps lack ground truth; only skip if ALL hidden are missing
        if not np.any(np.isfinite(hidden_tvt)):
            return None

    step_offset = 0

    # ---- truncate to max_seq_len (from the back — keep hidden section) ----
    if total_steps > max_seq_len:
        trim = total_steps - max_seq_len
        step_offset = trim
        # keep last max_seq_len steps (includes all hidden + some known context)
        sl = slice(trim, None)
        comp_tvt = comp_tvt[sl]
        comp_tvt_input = comp_tvt_input[sl]
        comp_gr_mean = comp_gr_mean[sl]
        comp_gr_valid = comp_gr_valid[sl]
        comp_gr_std = comp_gr_std[sl]
        for col in FORMATION_COLS:
            form_arrays[col] = form_arrays[col][sl]
        for pname in prior_cols:
            prior_cols[pname] = prior_cols[pname][sl]
        # recompute hidden/known
        hidden = ~np.isfinite(comp_tvt_input)
        hidden_indices = np.flatnonzero(hidden)
        known_indices = np.flatnonzero(~hidden)
        first_hidden = int(hidden_indices[0]) if len(hidden_indices) > 0 else 0
        hidden_len = int(hidden.sum())
        total_steps = len(comp_tvt)
        if len(known_indices) > 0:
            last_known_tvt = float(comp_tvt_input[known_indices[-1]])

    # ---- build features ----
    feats = np.zeros((total_steps, N_FEATURES), dtype=np.float32)

    # global scalars (same for every step)
    hidden_frac = hidden_len / max(1, total_steps)
    hidden_gr = comp_gr_valid[hidden]
    gr_nan_frac_hidden = float(1.0 - hidden_gr.mean()) if len(hidden_gr) > 0 else 0.0
    log_total = float(np.log(max(1, total_steps)) / 8.0)

    # per-step TVT_input delta (for known steps only)
    tvt_input_padded = comp_tvt_input.copy()
    # forward-fill: propagate last known value into hidden (for delta calculation)
    for t in range(1, total_steps):
        if not np.isfinite(tvt_input_padded[t]):
            tvt_input_padded[t] = tvt_input_padded[t - 1]

    for t in range(total_steps):
        is_known_t = float(not hidden[t])

        # position features
        feats[t, 0] = t / max(1, total_steps - 1)
        feats[t, 1] = is_known_t
        feats[t, 2] = min(1.0, t / max(1, first_hidden))         # known_progress
        if hidden[t]:
            feats[t, 3] = (t - first_hidden) / max(1, hidden_len - 1)  # hidden_progress
        else:
            feats[t, 3] = 0.0

        # GR
        gr_m = comp_gr_mean[t] if np.isfinite(comp_gr_mean[t]) else 0.0
        feats[t, 4] = gr_m / GR_SCALE
        feats[t, 5] = comp_gr_valid[t]
        feats[t, 6] = comp_gr_std[t] / GR_STD_SCALE

        # TVT input context (known steps only)
        if not hidden[t]:
            if t > 0 and np.isfinite(comp_tvt_input[t - 1]):
                feats[t, 7] = (comp_tvt_input[t] - comp_tvt_input[t - 1]) / TVT_STEP_SCALE
            feats[t, 8] = (comp_tvt_input[t] - last_known_tvt) / TVT_SCALE
        # hidden: stays 0

        # Formation boundaries relative to last_known_tvt
        all_finite = True
        for fi, col in enumerate(FORMATION_COLS):
            val = form_arrays[col][t]
            if np.isfinite(val):
                feats[t, 9 + fi] = (val - last_known_tvt) / TVT_SCALE
            else:
                feats[t, 9 + fi] = 0.0
                all_finite = False
        feats[t, 15] = 1.0 if all_finite else 0.0

        # global features (broadcast)
        feats[t, 25] = hidden_frac
        feats[t, 26] = gr_nan_frac_hidden
        feats[t, 27] = log_total

    # Priors (indices 16-24) — filled separately so dropout can zero groups
    b2 = prior_cols["b2_tvt"]
    base = prior_cols["base_tvt"]
    a_p50 = prior_cols["a_p50_tvt"]
    a_p10 = prior_cols["a_p10_tvt"]
    a_p90 = prior_cols["a_p90_tvt"]

    b2_avail = np.isfinite(b2)
    base_avail = np.isfinite(base)
    a_avail = np.isfinite(a_p50)

    for t in range(total_steps):
        if b2_avail[t]:
            feats[t, 16] = (b2[t] - last_known_tvt) / TVT_SCALE
            feats[t, 21] = 1.0
        if base_avail[t]:
            feats[t, 17] = (base[t] - last_known_tvt) / TVT_SCALE
            feats[t, 22] = 1.0
        if a_avail[t]:
            feats[t, 18] = (a_p50[t] - last_known_tvt) / TVT_SCALE
            feats[t, 23] = 1.0
            if np.isfinite(a_p10[t]):
                feats[t, 19] = (a_p10[t] - last_known_tvt) / TVT_SCALE
            if np.isfinite(a_p90[t]):
                feats[t, 20] = (a_p90[t] - last_known_tvt) / TVT_SCALE
            if np.isfinite(a_p10[t]) and np.isfinite(a_p90[t]):
                feats[t, 24] = (a_p90[t] - a_p10[t]) / TVT_SCALE

    # ---- target: TVT - last_known_tvt ----
    target_delta = (comp_tvt - last_known_tvt).astype(np.float32)

    if "id" in horizontal.columns:
        raw_ids = horizontal["id"].astype(str).to_numpy(dtype=object)
    else:
        raw_ids = np.asarray([f"{well_id}_{idx}" for idx in range(n_raw)], dtype=object)
    raw_row_idx = np.arange(n_raw, dtype=np.int32)
    raw_steps_old = (raw_row_idx // rows_per_step).astype(np.int32)
    raw_hidden = np.isnan(tvt_input_raw) & np.isfinite(tvt_raw)
    raw_in_kept_steps = (raw_steps_old >= step_offset) & (
        raw_steps_old < step_offset + total_steps
    )
    raw_mask = raw_hidden & raw_in_kept_steps
    hidden_row_steps = (raw_steps_old[raw_mask] - step_offset).astype(np.int32)

    return WellSample(
        well_id=well_id,
        features=feats,
        target_delta=target_delta,
        hidden_mask=hidden,
        seq_len=total_steps,
        last_known_tvt=last_known_tvt,
        first_hidden_step=first_hidden,
        tail_class=tail_class,
        hidden_row_ids=raw_ids[raw_mask],
        hidden_row_idx=raw_row_idx[raw_mask],
        hidden_row_steps=hidden_row_steps,
        hidden_row_tvt=tvt_raw[raw_mask].astype(np.float32),
        hidden_row_gr=gr_raw[raw_mask].astype(np.float32),
    )


# ---------------------------------------------------------------------------
# Apply prior dropout augmentation (in-place on a copy)
# ---------------------------------------------------------------------------

PRIOR_FEAT_INDICES = {
    "b2":   ([16], [21]),       # (value_idx_list, valid_idx_list)
    "base": ([17], [22]),
    "a":    ([18, 19, 20, 24], [23]),
}


def apply_prior_dropout(
    features: np.ndarray,
    rng: np.random.Generator,
    drop_b2_prob: float,
    drop_base_prob: float,
    drop_a_prob: float,
    drop_all_priors_prob: float,
) -> np.ndarray:
    """Zero out prior feature groups randomly.  Returns a new array."""
    feats = features.copy()
    if rng.random() < drop_all_priors_prob:
        # zero everything
        all_val = [16, 17, 18, 19, 20, 24]
        all_valid = [21, 22, 23]
        feats[:, all_val] = 0.0
        feats[:, all_valid] = 0.0
        return feats
    for name, (val_idxs, valid_idxs) in PRIOR_FEAT_INDICES.items():
        p = {"b2": drop_b2_prob, "base": drop_base_prob, "a": drop_a_prob}[name]
        if rng.random() < p:
            feats[:, val_idxs] = 0.0
            feats[:, valid_idxs] = 0.0
    return feats


# ---------------------------------------------------------------------------
# Main loading function
# ---------------------------------------------------------------------------

def load_all_wells(
    cfg: PathFormerConfig,
) -> tuple[list[WellSample], pd.DataFrame | None]:
    """Load all wells and build WellSamples.

    Returns (samples, tail_audit_df).
    """
    from mtpnet.io import discover_wells, load_well

    # Monkey-patch DataConfig for io reuse.
    # Pass data_dir as train_dir so resolve_train_dir() doesn't append /train again.
    from mtpnet.config import DataConfig
    data_cfg = DataConfig(
        data_dir=Path(cfg.data_dir).parent,
        train_dir=Path(cfg.data_dir),
        k_wells=cfg.k_wells,
    )
    wells = discover_wells(data_cfg)

    # Load tail audit for tail class labels
    tail_map: dict[str, str] = {}
    tail_df: pd.DataFrame | None = None
    if cfg.tail_audit_path is not None and Path(cfg.tail_audit_path).exists():
        tail_df = pd.read_csv(cfg.tail_audit_path)
        tail_map = dict(zip(tail_df["well_id"].astype(str), tail_df["tail_class"].astype(str)))

    # Load prior tables
    prior_frame = _load_prior_frame(cfg.priors)

    samples: list[WellSample] = []
    skipped = 0
    for well_paths in wells:
        try:
            horizontal, _typewell = load_well(well_paths)
        except Exception as e:
            print(f"[dataset] skipping {well_paths.well_id}: {e}", file=sys.stderr)
            skipped += 1
            continue
        tail_class = tail_map.get(well_paths.well_id, "unknown")
        sample = _build_well_sample(
            well_id=well_paths.well_id,
            horizontal=horizontal,
            prior_frame=prior_frame,
            rows_per_step=cfg.data.rows_per_step,
            max_seq_len=cfg.data.max_seq_len,
            tail_class=tail_class,
        )
        if sample is None:
            skipped += 1
            continue
        samples.append(sample)

    if skipped:
        print(f"[dataset] skipped {skipped} wells (no hidden rows or too short)", file=sys.stderr)
    print(f"[dataset] loaded {len(samples)} wells", file=sys.stderr)
    return samples, tail_df


# ---------------------------------------------------------------------------
# PyTorch Dataset with padding and augmentation
# ---------------------------------------------------------------------------

def _pad_sample(
    sample: WellSample,
    max_seq_len: int,
) -> dict[str, torch.Tensor]:
    """Pad sample to max_seq_len.  Returns dict of tensors."""
    T = sample.seq_len
    pad = max_seq_len - T
    assert pad >= 0, f"seq_len {T} > max_seq_len {max_seq_len}"

    feats = sample.features             # (T, F)
    target = sample.target_delta        # (T,)
    mask = sample.hidden_mask           # (T,) bool

    if pad > 0:
        feats = np.concatenate([feats, np.zeros((pad, N_FEATURES), dtype=np.float32)], axis=0)
        target = np.concatenate([target, np.zeros(pad, dtype=np.float32)])
        mask = np.concatenate([mask, np.zeros(pad, dtype=bool)])

    pad_mask = np.zeros(max_seq_len, dtype=bool)
    pad_mask[T:] = True   # True = padding position (ignored by attention)

    return {
        "features": torch.from_numpy(feats).float(),        # (L, F)
        "target_delta": torch.from_numpy(target).float(),   # (L,)
        "hidden_mask": torch.from_numpy(mask),               # (L,) bool
        "pad_mask": torch.from_numpy(pad_mask),              # (L,) bool
        "seq_len": torch.tensor(T, dtype=torch.long),
    }


class WellDataset(Dataset):
    def __init__(
        self,
        samples: list[WellSample],
        max_seq_len: int,
        augment: bool = False,
        aug_cfg=None,
        seed: int = 42,
    ):
        self.samples = samples
        self.max_seq_len = max_seq_len
        self.augment = augment
        self.aug_cfg = aug_cfg
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        feats = sample.features
        if self.augment and self.aug_cfg is not None:
            feats = apply_prior_dropout(
                feats,
                self._rng,
                drop_b2_prob=self.aug_cfg.drop_b2_prob,
                drop_base_prob=self.aug_cfg.drop_base_prob,
                drop_a_prob=self.aug_cfg.drop_a_prob,
                drop_all_priors_prob=self.aug_cfg.drop_all_priors_prob,
            )
        item = _pad_sample(
            WellSample(
                well_id=sample.well_id,
                features=feats,
                target_delta=sample.target_delta,
                hidden_mask=sample.hidden_mask,
                seq_len=sample.seq_len,
                last_known_tvt=sample.last_known_tvt,
                first_hidden_step=sample.first_hidden_step,
                tail_class=sample.tail_class,
                hidden_row_ids=sample.hidden_row_ids,
                hidden_row_idx=sample.hidden_row_idx,
                hidden_row_steps=sample.hidden_row_steps,
                hidden_row_tvt=sample.hidden_row_tvt,
                hidden_row_gr=sample.hidden_row_gr,
            ),
            self.max_seq_len,
        )
        item["well_id"] = sample.well_id
        item["last_known_tvt"] = torch.tensor(sample.last_known_tvt, dtype=torch.float32)
        return item


# ---------------------------------------------------------------------------
# Tail-balanced sampler
# ---------------------------------------------------------------------------

def make_tail_balanced_indices(
    samples: list[WellSample],
    oversample_factor: float,
    rng: np.random.Generator,
) -> list[int]:
    """Return index list with tail wells oversampled.

    tail classes G, A, D, B, C are oversampled by oversample_factor.
    OK_or_mixed and unknown are sampled once.
    """
    normal_idx = [i for i, s in enumerate(samples) if s.tail_class in {"OK_or_mixed", "unknown"}]
    tail_idx = [i for i, s in enumerate(samples) if s.tail_class not in {"OK_or_mixed", "unknown"}]

    extra_tail = rng.choice(tail_idx, size=int(len(tail_idx) * (oversample_factor - 1)), replace=True).tolist() if tail_idx else []
    all_idx = normal_idx + tail_idx + extra_tail
    rng.shuffle(all_idx)
    return all_idx
