from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .constants import FORMATIONS
from .io import well_name

class KaggleTopContext:
    """Spatial priors inspired by the current public Kaggle top notebooks."""

    def __init__(self, train_paths: list[Path], config: dict[str, Any]) -> None:
        top_cfg = config["features"].get("kaggle_top", {})
        self.spatial_k = int(top_cfg.get("spatial_k", 10))
        self.dense_k = int(top_cfg.get("dense_k", 20))
        self.dense_samples_per_well = int(top_cfg.get("dense_samples_per_well", 60))
        self.formation_tree: cKDTree | None = None
        self.dense_tree: cKDTree | None = None
        self._build_formation_tree(train_paths)
        self._build_dense_ancc_tree(train_paths)

    def _build_formation_tree(self, train_paths: list[Path]) -> None:
        rows: list[dict[str, float | str]] = []
        for path in train_paths:
            try:
                df = pd.read_csv(path, usecols=["X", "Y", *FORMATIONS]).dropna()
            except Exception:
                continue
            if df.empty:
                continue
            row: dict[str, float | str] = {
                "well": well_name(path),
                "x": float(df["X"].median()),
                "y": float(df["Y"].median()),
            }
            for formation in FORMATIONS:
                row[formation] = float(df[formation].median())
            rows.append(row)

        self.formation_df = pd.DataFrame(rows)
        if self.formation_df.empty:
            self.formation_xy = np.empty((0, 2), dtype=float)
            self.formation_values = np.empty((0, len(FORMATIONS)), dtype=float)
            self.formation_wells = np.array([], dtype=str)
            self.formation_scale = np.ones(2, dtype=float)
            return

        self.formation_xy = self.formation_df[["x", "y"]].to_numpy(dtype=float)
        self.formation_values = self.formation_df[FORMATIONS].to_numpy(dtype=float)
        self.formation_wells = self.formation_df["well"].astype(str).to_numpy()
        scale = np.nanstd(self.formation_xy, axis=0)
        self.formation_scale = np.where(scale < 1e-6, 1.0, scale)
        self.formation_tree = cKDTree(self.formation_xy / self.formation_scale)

    def _build_dense_ancc_tree(self, train_paths: list[Path]) -> None:
        xy_parts: list[np.ndarray] = []
        ancc_parts: list[np.ndarray] = []
        well_parts: list[np.ndarray] = []
        for path in train_paths:
            try:
                df = pd.read_csv(path, usecols=["X", "Y", "ANCC"]).dropna()
            except Exception:
                continue
            if df.empty:
                continue
            take = np.linspace(0, len(df) - 1, min(self.dense_samples_per_well, len(df)), dtype=int)
            sample = df.iloc[take]
            xy_parts.append(sample[["X", "Y"]].to_numpy(dtype=float))
            ancc_parts.append(sample["ANCC"].to_numpy(dtype=float))
            well_parts.append(np.full(len(sample), well_name(path), dtype=object))

        if not xy_parts:
            self.dense_xy = np.empty((0, 2), dtype=float)
            self.dense_ancc = np.empty(0, dtype=float)
            self.dense_wells = np.array([], dtype=object)
            self.dense_scale = np.ones(2, dtype=float)
            return

        self.dense_xy = np.vstack(xy_parts)
        self.dense_ancc = np.concatenate(ancc_parts)
        self.dense_wells = np.concatenate(well_parts)
        scale = np.nanstd(self.dense_xy, axis=0)
        self.dense_scale = np.where(scale < 1e-6, 1.0, scale)
        self.dense_tree = cKDTree(self.dense_xy / self.dense_scale)

    def impute_formations(self, xy: np.ndarray, self_well: str | None) -> tuple[np.ndarray, np.ndarray]:
        if self.formation_tree is None or len(self.formation_values) == 0:
            return (
                np.full((len(xy), len(FORMATIONS)), np.nan, dtype=float),
                np.full(len(xy), np.nan, dtype=float),
            )
        k_fetch = min(len(self.formation_values), self.spatial_k + 8)
        dist, idx = self.formation_tree.query(xy / self.formation_scale, k=k_fetch)
        dist = np.atleast_2d(dist)
        idx = np.atleast_2d(idx)
        if len(xy) == 1:
            dist = dist.reshape(1, -1)
            idx = idx.reshape(1, -1)
        if self_well is not None:
            dist = np.where(self.formation_wells[idx] == self_well, np.inf, dist)

        pred = np.empty((len(xy), len(FORMATIONS)), dtype=float)
        nearest_dist = np.empty(len(xy), dtype=float)
        global_mean = np.nanmean(self.formation_values, axis=0)
        for row in range(len(xy)):
            order = np.argsort(dist[row])[: self.spatial_k]
            valid = np.isfinite(dist[row, order])
            if not valid.any():
                pred[row] = global_mean
                nearest_dist[row] = np.nan
                continue
            chosen = order[valid]
            weights = 1.0 / (dist[row, chosen] + 1e-3)
            weights /= weights.sum()
            pred[row] = weights @ self.formation_values[idx[row, chosen]]
            nearest_dist[row] = float(np.nanmin(dist[row, chosen]))
        return pred, nearest_dist

    def impute_dense_ancc(self, xy: np.ndarray, self_well: str | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.dense_tree is None or len(self.dense_ancc) == 0:
            return (
                np.full(len(xy), np.nan, dtype=float),
                np.full(len(xy), np.nan, dtype=float),
                np.full(len(xy), np.nan, dtype=float),
            )
        k_fetch = min(len(self.dense_ancc), max(self.dense_k + 80, self.dense_k))
        dist, idx = self.dense_tree.query(xy / self.dense_scale, k=k_fetch)
        dist = np.atleast_2d(dist)
        idx = np.atleast_2d(idx)
        if len(xy) == 1:
            dist = dist.reshape(1, -1)
            idx = idx.reshape(1, -1)
        if self_well is not None:
            dist = np.where(self.dense_wells[idx] == self_well, np.inf, dist)

        pred = np.empty(len(xy), dtype=float)
        std = np.empty(len(xy), dtype=float)
        nearest_dist = np.empty(len(xy), dtype=float)
        global_mean = float(np.nanmean(self.dense_ancc))
        for row in range(len(xy)):
            order = np.argsort(dist[row])[: self.dense_k]
            valid = np.isfinite(dist[row, order])
            if not valid.any():
                pred[row] = global_mean
                std[row] = np.nan
                nearest_dist[row] = np.nan
                continue
            chosen = order[valid]
            values = self.dense_ancc[idx[row, chosen]]
            weights = 1.0 / (dist[row, chosen] + 1e-3)
            weights /= weights.sum()
            mean = float(weights @ values)
            pred[row] = mean
            std[row] = float(np.sqrt(weights @ ((values - mean) ** 2)))
            nearest_dist[row] = float(np.nanmin(dist[row, chosen]))
        return pred, std, nearest_dist
