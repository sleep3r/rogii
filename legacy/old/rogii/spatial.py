from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .constants import FORMATIONS
from .io import well_name


def context_key_for_paths(train_paths: list[Path], config: dict[str, Any]) -> str:
    path_stats = []
    for path in sorted(train_paths, key=well_name):
        path_stats.append(
            {
                "well": well_name(path),
                "name": path.name,
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
        )
    payload = {
        "wells": path_stats,
        "kaggle_top": config.get("features", {}).get("kaggle_top", {}),
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:16]


def context_well_overlap(context: "KaggleTopContext", paths: list[Path]) -> set[str]:
    return set(context.context_wells).intersection(well_name(path) for path in paths)


class KaggleTopContext:
    """Spatial priors inspired by the current public Kaggle top notebooks."""

    def __init__(self, train_paths: list[Path], config: dict[str, Any]) -> None:
        top_cfg = config["features"].get("kaggle_top", {})
        self.context_wells = frozenset(well_name(path) for path in train_paths)
        self.context_key = context_key_for_paths(train_paths, config)
        self.spatial_k = int(top_cfg.get("spatial_k", 10))
        self.dense_k = int(top_cfg.get("dense_k", 20))
        self.dense_fetch = int(top_cfg.get("dense_fetch", 5000))
        self.dense_query_chunk = int(top_cfg.get("dense_query_chunk", 512))
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
            take = np.linspace(
                0, len(df) - 1, min(self.dense_samples_per_well, len(df)), dtype=int
            )
            sample = df.iloc[take]
            xy_parts.append(sample[["X", "Y"]].to_numpy(dtype=float))
            ancc_parts.append(sample["ANCC"].to_numpy(dtype=float))
            well_parts.append(np.full(len(sample), well_name(path), dtype=object))

        if not xy_parts:
            self.dense_xy = np.empty((0, 2), dtype=float)
            self.dense_ancc = np.empty(0, dtype=float)
            self.dense_wells = np.array([], dtype=object)
            self.dense_well_counts: dict[str, int] = {}
            self.dense_scale = np.ones(2, dtype=float)
            return

        self.dense_xy = np.vstack(xy_parts)
        self.dense_ancc = np.concatenate(ancc_parts)
        self.dense_wells = np.concatenate(well_parts)
        unique_wells, well_counts = np.unique(self.dense_wells, return_counts=True)
        self.dense_well_counts = {
            str(well): int(count) for well, count in zip(unique_wells, well_counts)
        }
        scale = np.nanstd(self.dense_xy, axis=0)
        self.dense_scale = np.where(scale < 1e-6, 1.0, scale)
        self.dense_tree = cKDTree(self.dense_xy / self.dense_scale)

    def _dense_fetch_count(self, self_well: str | None) -> int:
        if len(self.dense_ancc) == 0:
            return 0
        self_count = (
            self.dense_well_counts.get(self_well, 0) if self_well is not None else 0
        )
        needed_for_exact_self_exclusion = self.dense_k + self_count + 8
        return min(
            len(self.dense_ancc), max(self.dense_k, needed_for_exact_self_exclusion)
        )

    def impute_formations(
        self, xy: np.ndarray, self_well: str | None
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.formation_tree is None or len(self.formation_values) == 0:
            return (
                np.full((len(xy), len(FORMATIONS)), np.nan, dtype=float),
                np.full(len(xy), np.nan, dtype=float),
            )
        k_fetch = min(len(self.formation_values), self.spatial_k + 8)
        dist, idx = self.formation_tree.query(xy / self.formation_scale, k=k_fetch)
        if k_fetch == 1:
            dist = np.asarray(dist).reshape(len(xy), 1)
            idx = np.asarray(idx).reshape(len(xy), 1)
        else:
            dist = np.atleast_2d(dist)
            idx = np.atleast_2d(idx)
            if len(xy) == 1:
                dist = dist.reshape(1, -1)
                idx = idx.reshape(1, -1)
        if self_well is not None:
            dist = np.where(self.formation_wells[idx] == self_well, np.inf, dist)

        order = np.argsort(dist, axis=1)[:, : self.spatial_k]
        chosen_dist = np.take_along_axis(dist, order, axis=1)
        chosen_idx = np.take_along_axis(idx, order, axis=1)
        valid = np.isfinite(chosen_dist)
        if (valid.sum(axis=1) < 3).any():
            return self._impute_formations_row_loop(xy, dist, idx)
        any_valid = valid.any(axis=1)
        weights = np.where(valid, 1.0 / (chosen_dist + 1e-3), 0.0)

        xn = self.formation_xy[chosen_idx, 0]
        yn = self.formation_xy[chosen_idx, 1]
        ones = np.ones_like(xn)
        design = np.stack([xn, yn, ones], axis=2)
        values = self.formation_values[chosen_idx]
        normal = np.einsum("nki,nkj,nk->nij", design, design, weights)
        rhs = np.einsum("nki,nkf,nk->nif", design, values, weights)
        normal += np.eye(3, dtype=float)[None, :, :] * 1e-9

        try:
            coef = np.linalg.solve(normal, rhs)
        except np.linalg.LinAlgError:
            coef = np.einsum("nij,njf->nif", np.linalg.pinv(normal), rhs)

        query_design = np.column_stack([xy[:, 0], xy[:, 1], np.ones(len(xy))])
        pred = np.einsum("ni,nif->nf", query_design, coef)
        global_mean = np.nanmean(self.formation_values, axis=0)
        pred[~any_valid] = global_mean
        nearest_dist = np.full(len(xy), np.nan, dtype=float)
        if any_valid.any():
            nearest_dist[any_valid] = np.min(
                np.where(valid[any_valid], chosen_dist[any_valid], np.inf),
                axis=1,
            )
        return pred, nearest_dist

    def _impute_formations_row_loop(
        self,
        xy: np.ndarray,
        dist: np.ndarray,
        idx: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
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
            chosen_idx = idx[row, chosen]
            weights = 1.0 / (dist[row, chosen] + 1e-3)
            xn = self.formation_xy[chosen_idx, 0]
            yn = self.formation_xy[chosen_idx, 1]
            values = self.formation_values[chosen_idx]
            design = np.column_stack([xn, yn, np.ones_like(xn)])
            normal = design.T @ (design * weights[:, None])
            rhs = design.T @ (values * weights[:, None])
            normal += np.eye(3) * 1e-9
            try:
                coef = np.linalg.solve(normal, rhs)
            except np.linalg.LinAlgError:
                coef = np.linalg.pinv(normal) @ rhs
            pred[row] = xy[row, 0] * coef[0] + xy[row, 1] * coef[1] + coef[2]
            nearest_dist[row] = float(np.nanmin(dist[row, chosen]))
        return pred, nearest_dist

    def impute_dense_ancc(
        self, xy: np.ndarray, self_well: str | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.dense_tree is None or len(self.dense_ancc) == 0:
            return (
                np.full(len(xy), np.nan, dtype=float),
                np.full(len(xy), np.nan, dtype=float),
                np.full(len(xy), np.nan, dtype=float),
            )
        k_fetch = self._dense_fetch_count(self_well)
        pred = np.empty(len(xy), dtype=float)
        std = np.empty(len(xy), dtype=float)
        nearest_dist = np.empty(len(xy), dtype=float)
        global_mean = float(np.nanmean(self.dense_ancc))
        chunk_size = max(1, self.dense_query_chunk)
        for start in range(0, len(xy), chunk_size):
            stop = min(start + chunk_size, len(xy))
            chunk_xy = xy[start:stop]
            dist, idx = self.dense_tree.query(chunk_xy / self.dense_scale, k=k_fetch)
            if k_fetch == 1:
                dist = np.asarray(dist).reshape(len(chunk_xy), 1)
                idx = np.asarray(idx).reshape(len(chunk_xy), 1)
            else:
                dist = np.atleast_2d(dist)
                idx = np.atleast_2d(idx)
                if len(chunk_xy) == 1:
                    dist = dist.reshape(1, -1)
                    idx = idx.reshape(1, -1)
            if self_well is not None:
                dist = np.where(self.dense_wells[idx] == self_well, np.inf, dist)

            order = np.argsort(dist, axis=1)[:, : self.dense_k]
            chosen_dist = np.take_along_axis(dist, order, axis=1)
            chosen_idx = np.take_along_axis(idx, order, axis=1)
            valid = np.isfinite(chosen_dist)
            any_valid = valid.any(axis=1)
            values = self.dense_ancc[chosen_idx]
            raw_weights = np.where(valid, 1.0 / (chosen_dist + 1e-3), 0.0)
            weight_sums = raw_weights.sum(axis=1)
            weights = np.divide(
                raw_weights,
                weight_sums[:, None],
                out=np.zeros_like(raw_weights),
                where=weight_sums[:, None] > 0,
            )
            mean = np.sum(weights * values, axis=1)
            variance = np.sum(weights * (values - mean[:, None]) ** 2, axis=1)

            pred[start:stop] = np.where(any_valid, mean, global_mean)
            std[start:stop] = np.where(any_valid, np.sqrt(variance), np.nan)
            chunk_nearest = np.full(len(chunk_xy), np.nan, dtype=float)
            if any_valid.any():
                chunk_nearest[any_valid] = np.min(
                    np.where(valid[any_valid], chosen_dist[any_valid], np.inf),
                    axis=1,
                )
            nearest_dist[start:stop] = chunk_nearest
        return pred, std, nearest_dist
