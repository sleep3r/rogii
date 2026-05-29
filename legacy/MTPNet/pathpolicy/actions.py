"""
pathpolicy/actions.py — Canonical action vocabulary for PolicyFormer.

Fixed 44-type vocabulary layout:
  [0-4]   prior anchors:  b2, base, a_p50, a_p10, a_p90
  [5-12]  MTP candidates: 8 sorted names
  [13-20] b2 level shifts: -80, -60, -40, -20, +20, +40, +60, +80
  [21-28] base level shifts
  [29-36] a_p50 level shifts
  [37-43] slope deltas:   -4, -2, -1, 0, +1, +2, +4  ft/step
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Vocabulary constants
# ---------------------------------------------------------------------------

PRIOR_NAMES: list[str] = ["b2", "base", "a_p50", "a_p10", "a_p90"]

# MTP candidate names as they appear in oracle (prefixed with "mtp_")
MTP_CANDIDATES_RAW: list[str] = sorted([
    "mtp_track_top1",
    "mtp_track_weighted",
    "mtp_track_anchored_weighted_a0.1_clip20",
    "mtp_track_anchored_weighted_a0.1_clip30",
    "mtp_track_anchored_weighted_a0.2_clip20",
    "mtp_track_anchored_weighted_a0.2_clip30",
    "mtp_track_anchored_weighted_a0.3_clip20",
    "mtp_track_anchored_weighted_a0.3_clip30",
])
MTP_NAMES: list[str] = [f"mtp_{c}" for c in MTP_CANDIDATES_RAW]

LEVEL_SHIFTS: list[float] = [-80.0, -60.0, -40.0, -20.0, 20.0, 40.0, 60.0, 80.0]
SHIFT_ANCHORS: list[str] = ["b2", "base", "a_p50"]
SLOPE_DELTAS: list[float] = [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0]

ROWS_PER_STEP: int = 32
CHUNK_LEN: int = 256


def _build_vocab() -> list[str]:
    vocab: list[str] = []
    vocab += PRIOR_NAMES
    vocab += MTP_NAMES
    for anchor in SHIFT_ANCHORS:
        for sh in LEVEL_SHIFTS:
            vocab.append(f"{anchor}{sh:+.0f}")
    for sd in SLOPE_DELTAS:
        vocab.append(f"slope{sd:+.0f}")
    return vocab


ACTION_VOCAB: list[str] = _build_vocab()
ACTION_TO_IDX: dict[str, int] = {name: i for i, name in enumerate(ACTION_VOCAB)}
N_ACTION_TYPES: int = len(ACTION_VOCAB)  # 44


# ---------------------------------------------------------------------------
# Fill helpers
# ---------------------------------------------------------------------------

def _ffill_bfill(arr: np.ndarray) -> np.ndarray:
    """Forward-fill then backward-fill NaN values in a 1D float array.

    Any segment with at least one finite value will be fully finite after this
    call.  All-NaN segments are returned unchanged (caller must not set mask=True
    for them).

    Implementation is O(n) using vectorised index propagation — no Python loops.
    """
    arr = arr.astype(np.float32).copy()
    finite = np.isfinite(arr)
    if finite.all():
        return arr
    if not finite.any():
        return arr  # nothing to propagate from

    # Forward fill: each NaN slot inherits the last finite index
    idx = np.where(finite, np.arange(len(arr)), 0)
    np.maximum.accumulate(idx, out=idx)
    arr = arr[idx]

    # Backward fill: any still-NaN slots are leading NaNs before first finite value;
    # fill them from the first finite value.
    finite2 = np.isfinite(arr)
    if not finite2.all():
        first_finite = int(np.argmax(finite2))
        arr[:first_finite] = arr[first_finite]

    return arr


# ---------------------------------------------------------------------------
# Segment generation (shared between oracle and PolicyFormer)
# ---------------------------------------------------------------------------

def generate_action_segments(
    prior_segs: dict[str, np.ndarray | None],
    last_tvt: float,
    last_slope_per_row: float,
    L: int = CHUNK_LEN,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (segments [N_ACTION_TYPES, L], mask [N_ACTION_TYPES]).

    segments[i] is the TVT path for action i (absolute values).
    mask[i] is True if action i is available for this chunk.
    Unavailable action segments are zero-filled.
    """
    segs = np.zeros((N_ACTION_TYPES, L), dtype=np.float32)
    mask = np.zeros(N_ACTION_TYPES, dtype=bool)

    # --- anchor priors + MTP ---
    for name in PRIOR_NAMES + MTP_NAMES:
        if name not in ACTION_TO_IDX:
            continue
        idx = ACTION_TO_IDX[name]
        seg = prior_segs.get(name)
        if seg is not None and len(seg) == L and np.isfinite(seg).any():
            # Full ffill+bfill: guarantees mask=True ↔ all-finite segment.
            # Forward-only fill (old behaviour) left leading NaN intact, which
            # propagated as NaN into Conv1d / Transformer in ActionEncoder.
            seg = _ffill_bfill(seg)
            assert np.isfinite(seg).all(), (
                f"_ffill_bfill produced NaN in segment '{name}' — this is a bug"
            )
            segs[idx] = seg
            mask[idx] = True

    # --- level shifts ---
    for anchor in SHIFT_ANCHORS:
        base_seg = prior_segs.get(anchor)
        if base_seg is None or len(base_seg) != L or not np.isfinite(base_seg).any():
            continue
        # Fill NaN in the base segment before shifting so shift results are all-finite.
        base_seg_filled = _ffill_bfill(base_seg)
        for sh in LEVEL_SHIFTS:
            name = f"{anchor}{sh:+.0f}"
            idx = ACTION_TO_IDX[name]
            segs[idx] = (base_seg_filled + sh).astype(np.float32)
            mask[idx] = True

    # --- slope continuation ---
    steps = np.arange(1, L + 1, dtype=np.float32)
    for sd in SLOPE_DELTAS:
        name = f"slope{sd:+.0f}"
        idx = ACTION_TO_IDX[name]
        slope_row = last_slope_per_row + sd / ROWS_PER_STEP
        segs[idx] = (last_tvt + steps * slope_row).astype(np.float32)
        mask[idx] = True

    return segs, mask


def normalize_action_segs(
    segs: np.ndarray,         # [N_ACTION_TYPES, L] absolute TVT
    mask: np.ndarray,         # [N_ACTION_TYPES] bool
    b2_seg: np.ndarray | None,  # [L] B2 prior for this chunk (anchor)
    last_tvt: float,
    scale: float = 100.0,
) -> np.ndarray:
    """Return normalized delta segments [N_ACTION_TYPES, L].

    delta[i] = (segs[i] - anchor) / scale
    where anchor = b2_seg if available, else last_tvt (broadcast).
    Unavailable actions are left as zeros.
    """
    K, L = segs.shape
    if b2_seg is not None and len(b2_seg) == L and np.isfinite(b2_seg).any():
        b2 = b2_seg.copy()
        # fill NaN in b2 with last_tvt
        b2[~np.isfinite(b2)] = last_tvt
        anchor = b2[np.newaxis, :]  # [1, L]
    else:
        anchor = np.full((1, L), last_tvt, dtype=np.float32)

    delta = np.zeros((K, L), dtype=np.float32)
    delta[mask] = (segs[mask] - anchor) / scale
    return delta
