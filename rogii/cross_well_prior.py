"""Cross-well TVT prior.

For each test well, build a synthetic typewell path from the ``k`` nearest
train wells in a standardized signature space (XY centroid, formation
surface depths near the hidden entry, GR statistics, last-known Z and
TVT). For each row in the hidden interval we interpolate ``delta_tvt`` at
the row's MD-offset and Z-offset relative to the last-known anchor of each
neighbor, then aggregate by median across neighbors.

This is the Tier-1 lever from ``RESEARCH_PLAN.md`` item 4. The expected
public LB impact is ``+0.4 - +1.0`` ft. The whole point is to introduce a
prior that is not pinned to the submission anchor and not already
encoded in the GBM features.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

FORMATIONS: tuple[str, ...] = ("ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA")

SIGNATURE_KEYS: tuple[str, ...] = (
    "xy_x",
    "xy_y",
    "last_known_z",
    "last_known_tvt",
    "gr_mean_hidden",
    "gr_std_hidden",
    *(f"surf_{surf}" for surf in FORMATIONS),
)


@dataclass(frozen=True)
class TrainWellPath:
    """Full known TVT/MD/Z curves and a hidden-entry signature."""

    well: str
    md: np.ndarray
    z: np.ndarray
    tvt: np.ndarray
    last_known_md: float
    last_known_tvt: float
    last_known_z: float
    signature: dict[str, float]


def _safe_mean(series: pd.Series) -> float:
    arr = pd.to_numeric(series, errors="coerce").to_numpy(float)
    if not np.isfinite(arr).any():
        return float("nan")
    return float(np.nanmean(arr))


def compute_signature(
    frame: pd.DataFrame,
    *,
    last_idx: int,
    hidden_indices: np.ndarray,
) -> dict[str, float]:
    """Signature used for nearest-neighbor lookup.

    All features are computed at, or just after, the last known TVT row so
    train and test signatures live in the same coordinate frame.
    """
    sig: dict[str, float] = {}
    x = pd.to_numeric(frame.get("X", pd.Series(np.nan, index=frame.index)), errors="coerce").to_numpy(float)
    y = pd.to_numeric(frame.get("Y", pd.Series(np.nan, index=frame.index)), errors="coerce").to_numpy(float)
    z = pd.to_numeric(frame.get("Z", pd.Series(np.nan, index=frame.index)), errors="coerce").to_numpy(float)
    tvt_input = pd.to_numeric(frame.get("TVT_input", pd.Series(np.nan, index=frame.index)), errors="coerce").to_numpy(float)
    gr = pd.to_numeric(frame.get("GR", pd.Series(np.nan, index=frame.index)), errors="coerce").to_numpy(float)

    sig["xy_x"] = float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")
    sig["xy_y"] = float(np.nanmean(y)) if np.isfinite(y).any() else float("nan")
    sig["last_known_z"] = float(z[int(last_idx)]) if 0 <= int(last_idx) < len(z) and np.isfinite(z[int(last_idx)]) else float("nan")
    sig["last_known_tvt"] = (
        float(tvt_input[int(last_idx)])
        if 0 <= int(last_idx) < len(tvt_input) and np.isfinite(tvt_input[int(last_idx)])
        else float("nan")
    )
    hidden_idx = np.asarray(hidden_indices, dtype=int)
    hidden_idx = hidden_idx[(hidden_idx >= 0) & (hidden_idx < len(frame))]
    if len(hidden_idx):
        gr_hidden = gr[hidden_idx]
        sig["gr_mean_hidden"] = float(np.nanmean(gr_hidden)) if np.isfinite(gr_hidden).any() else float("nan")
        sig["gr_std_hidden"] = float(np.nanstd(gr_hidden)) if np.isfinite(gr_hidden).any() else float("nan")
    else:
        sig["gr_mean_hidden"] = float("nan")
        sig["gr_std_hidden"] = float("nan")
    for surf in FORMATIONS:
        sig[f"surf_{surf}"] = _safe_mean(frame.get(surf, pd.Series(np.nan, index=frame.index)))
    return sig


def signature_to_vector(sig: dict[str, float]) -> np.ndarray:
    return np.array([sig.get(key, np.nan) for key in SIGNATURE_KEYS], dtype=float)


def _hidden_entry_index(tvt_input: np.ndarray) -> tuple[int | None, np.ndarray]:
    """Return ``(last_idx, hidden_indices)`` from a TVT_input column."""
    finite = np.flatnonzero(np.isfinite(tvt_input))
    if len(finite) == 0:
        return None, np.zeros(0, dtype=int)
    last_idx = int(finite[-1])
    hidden = np.flatnonzero(~np.isfinite(tvt_input) & (np.arange(len(tvt_input)) > last_idx))
    return last_idx, hidden


def collect_train_paths(data_dir: Path) -> list[TrainWellPath]:
    """Read all train wells and return their TVT/MD/Z curves plus a signature.

    A train well only contributes to the cross-well prior if it has both
    a non-empty hidden interval (so its signature is meaningful) and a
    finite ground-truth TVT curve for interpolation.
    """
    base = Path(data_dir) / "train"
    if not base.is_dir():
        return []
    paths: list[TrainWellPath] = []
    for csv in sorted(base.glob("*__horizontal_well.csv")):
        well = csv.name.replace("__horizontal_well.csv", "")
        try:
            df = pd.read_csv(csv)
        except Exception:
            continue
        md = pd.to_numeric(df.get("MD"), errors="coerce").to_numpy(float)
        z = pd.to_numeric(df.get("Z"), errors="coerce").to_numpy(float)
        tvt = pd.to_numeric(df.get("TVT"), errors="coerce").to_numpy(float)
        tvt_input = pd.to_numeric(df.get("TVT_input"), errors="coerce").to_numpy(float)
        if not (np.isfinite(md).any() and np.isfinite(z).any() and np.isfinite(tvt).any()):
            continue
        last_idx, hidden = _hidden_entry_index(tvt_input)
        if last_idx is None or len(hidden) == 0:
            continue
        signature = compute_signature(df, last_idx=last_idx, hidden_indices=hidden)
        valid = np.isfinite(md) & np.isfinite(z) & np.isfinite(tvt)
        if int(valid.sum()) < 16:
            continue
        md_v = md[valid]
        z_v = z[valid]
        tvt_v = tvt[valid]
        order = np.argsort(md_v)
        paths.append(
            TrainWellPath(
                well=well,
                md=md_v[order],
                z=z_v[order],
                tvt=tvt_v[order],
                last_known_md=float(md[last_idx]),
                last_known_tvt=float(tvt_input[last_idx]),
                last_known_z=float(z[last_idx]),
                signature=signature,
            )
        )
    return paths


def _standardize(train_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(train_matrix, axis=0)
    std = np.nanstd(train_matrix, axis=0)
    std = np.where(std > 1e-6, std, 1.0)
    return mean, std


def _impute_nan_with_zero(values: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(values), values, 0.0)


def nearest_train_wells(
    test_signature: dict[str, float],
    train_paths: list[TrainWellPath],
    *,
    k: int,
    self_well: str | None,
) -> tuple[list[TrainWellPath], np.ndarray]:
    """Return the ``k`` nearest train wells by standardized signature distance."""
    if not train_paths:
        return [], np.zeros(0, dtype=float)
    filtered = [path for path in train_paths if path.well != self_well]
    if not filtered:
        return [], np.zeros(0, dtype=float)
    train_matrix = np.stack([signature_to_vector(p.signature) for p in filtered])
    mean, std = _standardize(train_matrix)
    test_vec = signature_to_vector(test_signature)
    train_std = _impute_nan_with_zero((train_matrix - mean) / std)
    test_std = _impute_nan_with_zero((test_vec - mean) / std)
    distances = np.linalg.norm(train_std - test_std, axis=1)
    if not np.isfinite(distances).any():
        return [], np.zeros(0, dtype=float)
    order = np.argsort(distances)
    k_eff = max(1, min(int(k), len(filtered)))
    chosen = [filtered[int(i)] for i in order[:k_eff]]
    return chosen, distances[order[:k_eff]]


def _interp_sorted(query: np.ndarray, ref_x: np.ndarray, ref_y: np.ndarray) -> np.ndarray:
    """Numpy interp on a reference that we sort defensively."""
    order = np.argsort(ref_x)
    x = ref_x[order]
    y = ref_y[order]
    return np.interp(query, x, y, left=y[0], right=y[-1]).astype(float)


def cross_well_typewell_path(
    *,
    test_md: np.ndarray,
    test_z: np.ndarray,
    hidden_indices: np.ndarray,
    last_idx: int,
    last_tvt: float,
    last_md: float,
    last_z: float,
    test_signature: dict[str, float],
    train_paths: list[TrainWellPath],
    k: int = 8,
    self_well: str | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Build two synthetic TVT paths from the ``k`` nearest train wells.

    Returns ``(md_aligned_path, z_aligned_path, diagnostics)``.

    Both paths predict ``tvt[i] = last_tvt + median_over_neighbors(delta_tvt[i])``
    where ``delta_tvt`` is interpolated either against neighbor MD offsets
    or against neighbor Z offsets relative to the neighbor's last-known
    anchor.
    """
    n = len(test_md)
    fallback = np.full(n, last_tvt, dtype=float)
    if n == 0 or len(hidden_indices) == 0 or not train_paths:
        return fallback, fallback.copy(), {
            "crosswell_k": 0.0,
            "crosswell_nearest_distance": float("nan"),
            "crosswell_farthest_neighbor_distance": float("nan"),
            "crosswell_neighbor_distance_std": float("nan"),
            "crosswell_neighbors": "",
        }
    neighbors, distances = nearest_train_wells(
        test_signature, train_paths, k=k, self_well=self_well
    )
    if not neighbors:
        return fallback, fallback.copy(), {
            "crosswell_k": 0.0,
            "crosswell_nearest_distance": float("nan"),
            "crosswell_farthest_neighbor_distance": float("nan"),
            "crosswell_neighbor_distance_std": float("nan"),
            "crosswell_neighbors": "",
        }

    md_deltas: list[np.ndarray] = []
    z_deltas: list[np.ndarray] = []
    delta_md_query = test_md - float(last_md)
    delta_z_query = test_z - float(last_z)
    for nb in neighbors:
        delta_md_nb = nb.md - nb.last_known_md
        delta_tvt_nb = nb.tvt - nb.last_known_tvt
        md_interp = _interp_sorted(delta_md_query, delta_md_nb, delta_tvt_nb)
        md_deltas.append(md_interp)
        delta_z_nb = nb.z - nb.last_known_z
        z_interp = _interp_sorted(delta_z_query, delta_z_nb, delta_tvt_nb)
        z_deltas.append(z_interp)

    md_matrix = np.stack(md_deltas)
    z_matrix = np.stack(z_deltas)
    md_path = fallback.copy()
    z_path = fallback.copy()
    md_path[:] = float(last_tvt) + np.nanmedian(md_matrix, axis=0)
    z_path[:] = float(last_tvt) + np.nanmedian(z_matrix, axis=0)
    md_path[: int(last_idx) + 1] = float(last_tvt)
    z_path[: int(last_idx) + 1] = float(last_tvt)

    diag = {
        "crosswell_k": float(len(neighbors)),
        "crosswell_nearest_distance": float(distances[0]) if len(distances) else float("nan"),
        "crosswell_farthest_neighbor_distance": (
            float(distances[-1]) if len(distances) else float("nan")
        ),
        "crosswell_neighbor_distance_std": (
            float(np.std(distances)) if len(distances) > 1 else float("nan")
        ),
        "crosswell_neighbors": ",".join(p.well for p in neighbors),
    }
    return md_path, z_path, diag
