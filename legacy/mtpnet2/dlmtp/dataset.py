"""
dlmtp/dataset.py  —  PyTorch Dataset for heatmap-based TVT prediction.

Each item is:
    heatmap    : (N_CHANNELS, L, J) float32 tensor
    true_bins  : (L,) int64 tensor   — target bin indices (training only)
    valid_mask : (L,) float32 tensor — 1 for rows with valid target
    well_idx   : int                 — index into the samples list

Training: random L-row crop from each well.
Inference: full well (padded to multiple of 2^depth).
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from mtpnet.offsets import tvt_from_ksegment_offsets
from mtpnet.local_gr_search import smooth_gr
from dlmtp.heatmap import build_heatmap, N_CHANNELS, _smooth_gr


# ---------------------------------------------------------------------------
# Typewell loader (cached in caller)
# ---------------------------------------------------------------------------

def load_typewell(well_id: str, data_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (tw_tvt, tw_gr) sorted by TVT ascending."""
    import pandas as pd
    from pathlib import Path
    path = Path(data_dir) / f"{well_id}__typewell.csv"
    df = pd.read_csv(path)
    df = df.dropna(subset=["TVT", "GR"]).sort_values("TVT")
    return df["TVT"].to_numpy(np.float64), df["GR"].to_numpy(np.float64)


def compute_prior_tvt(sample, k_offsets: np.ndarray) -> np.ndarray:
    """Integrate K-segment offsets to get per-hidden-row prior TVT."""
    return tvt_from_ksegment_offsets(
        sample.z, sample.anchor_row, sample.anchor_tvt,
        sample.hidden_rows, k_offsets,
    ).astype(np.float64)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class HeatmapDataset(Dataset):
    """One sample = one random crop from one well.

    Parameters
    ----------
    samples      : list of OffsetSample (train split only — must have has_true)
    oof_k3       : (n_wells, K) float array of K3 OOF offset priors
    tw_cache     : dict well_id → (tw_tvt, tw_gr)
    cfg          : dict with keys: crop_len, tvt_bins, bin_ft, gr_smooth_window
    n_crops      : how many crops to sample per well per epoch (default 4)
    train        : if True, apply random crop; else return full well
    gr_tvt_cache : optional dict well_id → (nh,) greedy-GR TVT (Run 2 channel 8)
    """

    def __init__(
        self,
        samples: list,
        oof_k3: np.ndarray,
        tw_cache: dict,
        cfg: dict,
        n_crops: int = 4,
        train: bool = True,
        gr_tvt_cache: dict | None = None,
    ):
        self.train    = train
        self.n_crops  = n_crops
        self.cfg      = cfg

        # Keep only samples that have typewell cache
        paired = [(s, oof_k3[i]) for i, s in enumerate(samples)
                  if s.well_id in tw_cache and s.has_true]
        self.samples   = [p[0] for p in paired]
        self.k3_priors = [p[1] for p in paired]
        self.tw_cache  = tw_cache
        self.gr_tvt_cache = gr_tvt_cache  # may be None (Run 1) or dict (Run 2)

        # Precompute prior TVT for each well (expensive to redo each epoch)
        self.prior_tvts = [
            compute_prior_tvt(s, k)
            for s, k in zip(self.samples, self.k3_priors)
        ]

        # Precompute smoothed GR per well (avoids redundant savgol per crop)
        win = cfg.get("gr_smooth_window", 101)
        self.smooth_grs = [
            _smooth_gr(s.gr.astype(np.float64), win)
            for s in self.samples
        ]

        # Build index: (well_idx, crop_idx) for training
        self._build_index()

    def _build_index(self):
        self._index = []
        for wi, s in enumerate(self.samples):
            nh = len(s.hidden_rows)
            if self.train:
                for _ in range(self.n_crops):
                    self._index.append(wi)
            else:
                self._index.append(wi)

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        wi = self._index[idx]
        s  = self.samples[wi]
        nh = len(s.hidden_rows)

        crop_len = self.cfg["crop_len"]
        J        = self.cfg["tvt_bins"]
        bin_ft   = self.cfg["bin_ft"]

        tw_tvt, tw_gr = self.tw_cache[s.well_id]
        prior_tvt     = self.prior_tvts[wi]
        gr_smooth     = self.smooth_grs[wi]

        # ---- Crop start -----------------------------------------------------
        if self.train and nh > crop_len:
            crop_start = int(np.random.randint(0, nh - crop_len + 1))
        else:
            crop_start = 0
        actual_end = min(crop_start + crop_len, nh)

        # ---- Build heatmap --------------------------------------------------
        gr_tvt_cache = getattr(self, 'gr_tvt_cache', None)
        gr_prior_tvt = (gr_tvt_cache.get(s.well_id)
                        if gr_tvt_cache is not None else None)
        heatmap = build_heatmap(
            gr_well      = s.gr,
            z_well       = s.z,
            hidden_rows  = s.hidden_rows,
            prior_tvt    = prior_tvt,
            tw_tvt       = tw_tvt,
            tw_gr        = tw_gr,
            crop_start   = crop_start,
            crop_len     = crop_len,
            tvt_bins     = J,
            bin_ft       = bin_ft,
            gr_smooth_window = self.cfg.get("gr_smooth_window", 101),
            gr_smooth_precomputed = gr_smooth,
            gr_prior_tvt = gr_prior_tvt,
        )  # (8 or 9, crop_len, J)

        # ---- Target ---------------------------------------------------------
        true_tvt_crop = s.tvt_true[s.hidden_rows[crop_start:actual_end]]
        prior_crop    = prior_tvt[crop_start:actual_end]
        dev           = (true_tvt_crop - prior_crop) / bin_ft
        raw_bins      = np.round(dev).astype(np.int64) + J // 2

        valid_mask = np.zeros(crop_len, dtype=np.float32)
        true_bins  = np.full(crop_len, J // 2, dtype=np.int64)
        actual_len = actual_end - crop_start
        valid_mask[:actual_len] = ((raw_bins >= 0) & (raw_bins < J)).astype(np.float32)
        true_bins[:actual_len]  = np.clip(raw_bins, 0, J - 1)

        return (
            torch.from_numpy(heatmap),                       # (C, L, J)
            torch.from_numpy(true_bins),                     # (L,)
            torch.from_numpy(valid_mask),                    # (L,)
            wi,                                              # int
        )


# ---------------------------------------------------------------------------
# Inference helper: build full-well heatmap padded to multiple of pad_to
# ---------------------------------------------------------------------------

def build_inference_heatmap(
    sample,
    prior_tvt: np.ndarray,
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
    cfg: dict,
    pad_to: int = 8,
    gr_smooth_precomputed: np.ndarray | None = None,
    gr_prior_tvt: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """Return (heatmap, padded_len) where heatmap is (C, padded_len, J) float32."""
    nh = len(sample.hidden_rows)
    J  = cfg["tvt_bins"]

    # Pad nh to multiple of pad_to
    padded_len = int(np.ceil(nh / pad_to) * pad_to) if nh % pad_to != 0 else nh

    heatmap = build_heatmap(
        gr_well     = sample.gr,
        z_well      = sample.z,
        hidden_rows = sample.hidden_rows,
        prior_tvt   = prior_tvt,
        tw_tvt      = tw_tvt,
        tw_gr       = tw_gr,
        crop_start  = 0,
        crop_len    = padded_len,
        tvt_bins    = J,
        bin_ft      = cfg["bin_ft"],
        gr_smooth_window = cfg.get("gr_smooth_window", 101),
        gr_smooth_precomputed = gr_smooth_precomputed,
        gr_prior_tvt = gr_prior_tvt,
    )  # (C, padded_len, J)

    return heatmap, padded_len
