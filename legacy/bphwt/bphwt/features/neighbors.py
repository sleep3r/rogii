"""Neighbour-well template prior features.

For each query well, find K spatially- and typewell-similar train wells,
then aggregate their TVT residuals as a soft prior.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


class NeighborPrior:
    """Stores train well fingerprints and computes neighbour-based TVT prior."""

    def __init__(self) -> None:
        self.train_fingerprints: list[dict] = []
        self.fitted = False

    @classmethod
    def build_from_train(cls, train_wells: list[dict], verbose: bool = False) -> NeighborPrior:
        """
        train_wells — each dict must contain:
            'well_id', 'typewell_id', 'X0', 'Y0', 'Z0'
            'gr_head_mean', 'gr_head_std'   — GR stats in known-head region
            'tvt_range'                      — (tvt_min, tvt_max)
            'hmm_tvt_mu'                     — [n_rows] HMM prior TVT
            'linear_tvt'                     — [n_rows] linear prior TVT
            'tvt_true'                        — [n_rows] true TVT (train only)
            'hidden_mask'                     — [n_rows] bool
        """
        obj = cls()
        for w in train_wells:
            fp: dict = {
                "well_id": w["well_id"],
                "typewell_id": w.get("typewell_id", ""),
                "X0": float(w["X0"]),
                "Y0": float(w["Y0"]),
                "Z0": float(w.get("Z0", 0.0)),
                "gr_head_mean": float(w.get("gr_head_mean", 0.0)),
                "gr_head_std": float(w.get("gr_head_std", 1.0)),
                "tvt_min": float(w["tvt_range"][0]),
                "tvt_max": float(w["tvt_range"][1]),
                # residual from linear prior (for the hidden zone)
                "hidden_tvt_residual": _safe_residual(w),
                "n_hidden": int(w.get("hidden_mask", np.array([])).sum()),
            }
            obj.train_fingerprints.append(fp)
        obj.fitted = True
        if verbose:
            print(f"NeighborPrior: {len(obj.train_fingerprints)} fingerprints")
        return obj

    def query(
        self,
        X0: float,
        Y0: float,
        typewell_id: str,
        gr_head_mean: float,
        gr_head_std: float,
        exclude_well_id: str | None = None,
        k: int = 8,
        spatial_weight: float = 1.0,
        typewell_weight: float = 2.0,
    ) -> dict[str, float]:
        """Return aggregated neighbour statistics."""
        if not self.train_fingerprints:
            return _empty_neighbor_feats()

        scored = []
        for fp in self.train_fingerprints:
            if exclude_well_id is not None and fp["well_id"] == exclude_well_id:
                continue
            dist_spatial = np.sqrt((fp["X0"] - X0) ** 2 + (fp["Y0"] - Y0) ** 2) / 5000.0
            same_tw = 1.0 if fp["typewell_id"] == typewell_id else 0.0
            gr_diff = abs(fp["gr_head_mean"] - gr_head_mean) / (gr_head_std + 1.0)
            score = spatial_weight * dist_spatial - typewell_weight * same_tw + 0.3 * gr_diff
            scored.append((score, fp))

        if not scored:
            return _empty_neighbor_feats()
        neighbors = [fp for _, fp in sorted(scored, key=lambda item: item[0])[:k]]

        residuals = [fp["hidden_tvt_residual"] for fp in neighbors if fp["hidden_tvt_residual"] is not None]
        n_same_tw = sum(1 for fp in neighbors if fp["typewell_id"] == typewell_id)
        spatial_dists = [np.sqrt((fp["X0"] - X0) ** 2 + (fp["Y0"] - Y0) ** 2) for fp in neighbors]

        return {
            "neighbor_n_same_typewell": float(n_same_tw),
            "neighbor_mean_spatial_dist": float(np.mean(spatial_dists)),
            "neighbor_residual_mean": float(np.mean(residuals)) if residuals else 0.0,
            "neighbor_residual_std": float(np.std(residuals)) if len(residuals) > 1 else 5.0,
            "neighbor_count": float(len(neighbors)),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> NeighborPrior:
        with open(path, "rb") as f:
            return pickle.load(f)


def _safe_residual(w: dict) -> float | None:
    try:
        tvt_true = np.asarray(w["tvt_true"], dtype=np.float64)
        linear = np.asarray(w["linear_tvt"], dtype=np.float64)
        mask = np.asarray(w["hidden_mask"], dtype=bool)
        if mask.sum() == 0:
            return None
        return float(np.mean(tvt_true[mask] - linear[mask]))
    except Exception:
        return None


def _empty_neighbor_feats() -> dict[str, float]:
    return {
        "neighbor_n_same_typewell": 0.0,
        "neighbor_mean_spatial_dist": 9999.0,
        "neighbor_residual_mean": 0.0,
        "neighbor_residual_std": 10.0,
        "neighbor_count": 0.0,
    }
