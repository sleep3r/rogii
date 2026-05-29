"""OffsetSample dataclass and data loader for the MTPNet offset-state decoder.

Each OffsetSample captures:
  - Full per-row arrays (MD, X, Y, Z, GR, TVT_input, TVT_true)
  - Anchor information (last known row, anchor TVT/Z)
  - Hidden row indices
  - c0 drift (from known prefix, test-safe)
  - Train-only oracle targets (global_offset_star, kseg_offset_star)

Loading:
    samples = load_offset_samples("data/train")
    folds   = make_group_kfold(samples, n_folds=5)
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

# Last-N-rows window for anchor statistics (test-safe)
LAST_KNOWN_WINDOW: int = 512


@dataclass
class OffsetSample:
    """Single-well data for the offset-state decoder."""

    well_id: str

    # ---- per-row arrays (full well, length N) ----
    md: np.ndarray           # (N,) float32 — measured depth [ft]
    x: np.ndarray            # (N,) float32 — easting
    y: np.ndarray            # (N,) float32 — northing
    z: np.ndarray            # (N,) float32 — TVD (negative = deeper in sign convention)
    gr: np.ndarray           # (N,) float32 — gamma ray, may contain NaN
    tvt_input: np.ndarray    # (N,) float32 — TVT_input, NaN for hidden rows
    tvt_true: np.ndarray | None  # (N,) float32 — true TVT, None for test split

    # ---- anchor (last row with finite TVT_input) ----
    anchor_row: int
    anchor_tvt: float
    anchor_z: float

    # ---- hidden rows ----
    hidden_rows: np.ndarray  # (H,) int64 — indices of rows to predict

    # ---- test-safe C-field drift ----
    c0: float   # median dC = median d(TVT_input + Z) over last LAST_KNOWN_WINDOW known rows

    # ---- known C-field statistics (from known prefix) ----
    dC_last64_median: float = 0.0
    dC_last128_median: float = 0.0
    dC_last256_median: float = 0.0
    dC_last512_median: float = 0.0
    dC_last512_mean: float = 0.0
    dC_last512_std: float = 0.0
    dC_last512_slope: float = 0.0
    anchor_C: float = 0.0   # TVT_anchor + Z_anchor

    # ---- train-only oracle targets (populated by oracle.py) ----
    global_offset_star: float | None = None
    kseg_offset_star: dict[int, np.ndarray] = field(default_factory=dict)

    @property
    def n_hidden(self) -> int:
        return len(self.hidden_rows)

    @property
    def n_rows(self) -> int:
        return len(self.md)

    @property
    def has_true(self) -> bool:
        return self.tvt_true is not None

    @property
    def tvt_hidden_true(self) -> np.ndarray | None:
        """True TVT for hidden rows, or None if not available."""
        if self.tvt_true is None:
            return None
        return self.tvt_true[self.hidden_rows]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_c0_and_stats(
    tvt_input: np.ndarray,
    z: np.ndarray,
    anchor_row: int,
    window: int = LAST_KNOWN_WINDOW,
) -> dict[str, float]:
    """Compute C-field drift statistics from known prefix (test-safe)."""
    known = np.flatnonzero(np.isfinite(tvt_input))
    if len(known) < 2:
        return {
            "c0": 0.0,
            "dC_last64_median": 0.0,
            "dC_last128_median": 0.0,
            "dC_last256_median": 0.0,
            "dC_last512_median": 0.0,
            "dC_last512_mean": 0.0,
            "dC_last512_std": 0.0,
            "dC_last512_slope": 0.0,
        }

    # Restrict to window before anchor
    win_start = max(0, anchor_row - window + 1)
    known_in_win = known[known >= win_start]

    def _dC_stats(n: int) -> tuple[float, float, float, float]:
        k_recent = known_in_win[-n:] if len(known_in_win) >= 2 else known_in_win
        if len(k_recent) < 2:
            return 0.0, 0.0, 0.0, 0.0
        C = (tvt_input[k_recent] + z[k_recent]).astype(np.float64)
        dC = np.diff(C)
        dC = dC[np.isfinite(dC)]
        if len(dC) == 0:
            return 0.0, 0.0, 0.0, 0.0
        med = float(np.median(dC))
        mn = float(np.mean(dC))
        std = float(np.std(dC)) if len(dC) > 1 else 0.0
        if len(dC) >= 3:
            xs = np.arange(len(dC), dtype=np.float64)
            slope = float(np.polyfit(xs, dC, 1)[0])
        else:
            slope = 0.0
        return med, mn, std, slope

    med512, mean512, std512, slope512 = _dC_stats(512)
    med256, _, _, _ = _dC_stats(256)
    med128, _, _, _ = _dC_stats(128)
    med64, _, _, _ = _dC_stats(64)

    return {
        "c0": med512,
        "dC_last64_median": med64,
        "dC_last128_median": med128,
        "dC_last256_median": med256,
        "dC_last512_median": med512,
        "dC_last512_mean": mean512,
        "dC_last512_std": std512,
        "dC_last512_slope": slope512,
    }


# ---------------------------------------------------------------------------
# Single-well builder
# ---------------------------------------------------------------------------

def load_offset_sample(
    well_id: str,
    horizontal: pd.DataFrame,
) -> OffsetSample | None:
    """Build an OffsetSample from a raw horizontal well DataFrame.

    Returns None if the well is degenerate (no known rows, no hidden rows,
    or missing anchor).
    """
    n = len(horizontal)
    if n < 8:
        return None

    def _col(name: str) -> np.ndarray:
        if name in horizontal.columns:
            return pd.to_numeric(horizontal[name], errors="coerce").to_numpy(np.float32)
        return np.full(n, np.nan, dtype=np.float32)

    md = _col("MD")
    x = _col("X")
    y = _col("Y")
    z = _col("Z")
    gr = _col("GR")
    tvt_input = _col("TVT_input")
    tvt_true: np.ndarray | None = _col("TVT") if "TVT" in horizontal.columns else None

    # If all TVT values are NaN, it's a test file
    if tvt_true is not None and not np.any(np.isfinite(tvt_true)):
        tvt_true = None

    known_mask = np.isfinite(tvt_input)
    hidden_mask = ~known_mask
    known_idx = np.flatnonzero(known_mask)
    hidden_idx = np.flatnonzero(hidden_mask)

    if len(known_idx) == 0 or len(hidden_idx) == 0:
        return None

    anchor_row = int(known_idx[-1])
    anchor_tvt = float(tvt_input[anchor_row])
    anchor_z = float(z[anchor_row])

    if not (np.isfinite(anchor_tvt) and np.isfinite(anchor_z)):
        return None

    # Only hidden rows AFTER anchor matter for prediction
    post_anchor_hidden = hidden_idx[hidden_idx > anchor_row]
    if len(post_anchor_hidden) == 0:
        return None

    stats = _compute_c0_and_stats(tvt_input, z, anchor_row)

    return OffsetSample(
        well_id=well_id,
        md=md, x=x, y=y, z=z, gr=gr,
        tvt_input=tvt_input,
        tvt_true=tvt_true,
        anchor_row=anchor_row,
        anchor_tvt=anchor_tvt,
        anchor_z=anchor_z,
        hidden_rows=post_anchor_hidden.astype(np.int64),
        c0=stats["c0"],
        dC_last64_median=stats["dC_last64_median"],
        dC_last128_median=stats["dC_last128_median"],
        dC_last256_median=stats["dC_last256_median"],
        dC_last512_median=stats["dC_last512_median"],
        dC_last512_mean=stats["dC_last512_mean"],
        dC_last512_std=stats["dC_last512_std"],
        dC_last512_slope=stats["dC_last512_slope"],
        anchor_C=anchor_tvt + anchor_z,
    )


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_offset_samples(
    data_dir: Path | str,
    k_wells: int = -1,
    cache_path: Path | str | None = None,
    verbose: bool = True,
) -> list[OffsetSample]:
    """Load all training wells as OffsetSample objects.

    Args:
        data_dir   : directory containing *__horizontal_well.csv files
        k_wells    : if > 0, load only the first k_wells (for smoke tests)
        cache_path : if given, load from / save to a pickle cache
        verbose    : print progress

    Returns:
        List of OffsetSample objects (one per well), sorted by well_id.
    """
    data_dir = Path(data_dir)

    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            if verbose:
                print(f"Loading cached samples from {cache_path}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)

    horizontal_files = sorted(data_dir.glob("*__horizontal_well.csv"))
    if not horizontal_files:
        raise FileNotFoundError(f"No horizontal well files in {data_dir}")

    if k_wells > 0:
        horizontal_files = horizontal_files[:k_wells]

    samples: list[OffsetSample] = []
    skipped = 0
    for path in horizontal_files:
        well_id = path.name.split("__horizontal_well")[0]
        try:
            horizontal = pd.read_csv(path)
        except Exception as e:
            if verbose:
                print(f"  [warn] Could not load {path}: {e}")
            skipped += 1
            continue

        sample = load_offset_sample(well_id, horizontal)
        if sample is None:
            skipped += 1
            continue
        samples.append(sample)

    if verbose:
        print(
            f"Loaded {len(samples)} wells from {data_dir}"
            + (f" (skipped {skipped})" if skipped > 0 else "")
        )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(samples, f)
        if verbose:
            print(f"Saved cache → {cache_path}")

    return samples


# ---------------------------------------------------------------------------
# Cross-validation split
# ---------------------------------------------------------------------------

def make_group_kfold(
    samples: list[OffsetSample],
    n_folds: int = 5,
    seed: int = 42,
) -> list[tuple[list[int], list[int]]]:
    """Return fold splits with GroupKFold by well_id.

    Each well appears in exactly one validation fold.

    Args:
        samples : list of OffsetSample
        n_folds : number of folds
        seed    : random seed (for reproducibility with shuffle)

    Returns:
        List of (train_indices, val_indices) tuples.
    """
    n = len(samples)
    X = np.zeros((n, 1))
    # Each well is its own group → GroupKFold ≡ KFold here,
    # but GroupKFold makes the intent explicit.
    groups = np.arange(n)

    gkf = GroupKFold(n_splits=n_folds)
    splits: list[tuple[list[int], list[int]]] = []
    for tr_idx, va_idx in gkf.split(X, groups=groups):
        splits.append((tr_idx.tolist(), va_idx.tolist()))
    return splits


# ---------------------------------------------------------------------------
# Convenience: load folds from JSON cache
# ---------------------------------------------------------------------------

def save_folds(folds: list[tuple[list[int], list[int]]], path: Path | str) -> None:
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [{"train": tr, "val": va} for tr, va in folds]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_folds(path: Path | str) -> list[tuple[list[int], list[int]]]:
    import json
    with open(path) as f:
        data = json.load(f)
    return [(d["train"], d["val"]) for d in data]
