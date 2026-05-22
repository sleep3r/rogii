from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .constants import FORMATIONS
from .io import horizontal_files, well_name


FORMATION_CANDIDATE_COLUMNS: tuple[str, ...] = tuple(
    f"tvtF_{formation}_{suffix}"
    for formation in FORMATIONS
    for suffix in ("full", "late", "wls")
)
EXTRA_CANDIDATE_COLUMNS: tuple[str, ...] = (
    "row_ancc_tvt",
    "dense_ancc_tvt",
    "formation_median_tvt",
    "formation_wls_median_tvt",
    "nearby_path_top1",
    "nearby_path_top3_median",
    "nearby_path_top5_median",
    "nearby_path_weighted_mean",
    "nearby_path_p10",
    "nearby_path_p90",
    "formation_sample_best_by_anchor",
    "formation_sample_mean",
    "formation_sample_median",
    "formation_sample_p10",
    "formation_sample_p90",
)
CANDIDATE_COLUMNS: tuple[str, ...] = (
    *FORMATION_CANDIDATE_COLUMNS,
    *EXTRA_CANDIDATE_COLUMNS,
)


@dataclass(frozen=True)
class FormationPlaneConfig:
    k_wells: int = 15
    sample_rows_per_well: int = 80
    min_points: int = 80
    dense_k: int = 120
    weight_power: float = 1.0
    eps: float = 1e-3
    bootstrap_samples: int = 64
    bootstrap_fraction: float = 0.75
    query_chunk: int = 2048
    seed: int = 42


@dataclass(frozen=True)
class SurfacePrediction:
    values: np.ndarray
    std: np.ndarray
    dist_mean: np.ndarray
    dist_min: np.ndarray
    residual: np.ndarray


@dataclass(frozen=True)
class NeighborPath:
    well: str
    xy: np.ndarray
    hidden_frac: np.ndarray
    hidden_drift: np.ndarray
    tail_slope: float
    last_known_tvt: float


def _numeric(frame: pd.DataFrame, column: str, default: float = np.nan) -> np.ndarray:
    if column not in frame.columns:
        return np.full(len(frame), default, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def _rmse(pred: np.ndarray | pd.Series, true: np.ndarray | pd.Series) -> float:
    pred_arr = np.asarray(pred, dtype=float)
    true_arr = np.asarray(true, dtype=float)
    mask = np.isfinite(pred_arr) & np.isfinite(true_arr)
    if not mask.any():
        return float("nan")
    err = pred_arr[mask] - true_arr[mask]
    return float(np.sqrt(np.mean(err * err)))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 3:
        return float("nan")
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def _slope(values: np.ndarray, idx: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    idx = np.asarray(idx, dtype=int)
    if len(idx) < 2:
        return float("nan")
    y = values[idx]
    mask = np.isfinite(y)
    if int(mask.sum()) < 2:
        return float("nan")
    x = np.linspace(0.0, 1.0, len(idx), dtype=float)[mask]
    y = y[mask]
    x = x - float(np.mean(x))
    y = y - float(np.mean(y))
    denom = float(np.dot(x, x))
    return float(np.dot(x, y) / denom) if denom > 1e-12 else 0.0


def _slope_error(pred: np.ndarray, true: np.ndarray, idx: np.ndarray) -> float:
    pred_slope = _slope(pred, idx)
    true_slope = _slope(true, idx)
    if not np.isfinite(pred_slope) or not np.isfinite(true_slope):
        return float("nan")
    return float(abs(pred_slope - true_slope))


def _roughness(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) < 3:
        return float("nan")
    second = np.diff(finite, n=2)
    if len(second) == 0:
        return 0.0
    return float(np.sqrt(np.mean(second * second)))


def _pctl_abs_error(pred: np.ndarray, true: np.ndarray, p: float) -> float:
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    mask = np.isfinite(pred) & np.isfinite(true)
    if not mask.any():
        return float("nan")
    return float(np.nanpercentile(np.abs(pred[mask] - true[mask]), p))


def _weighted_std(values: np.ndarray, weights: np.ndarray, mean: np.ndarray) -> np.ndarray:
    var = np.sum(weights[..., None] * (values - mean[:, None, :]) ** 2, axis=1)
    return np.sqrt(np.maximum(var, 0.0))


def _normalize_weights(dist: np.ndarray, power: float, eps: float) -> np.ndarray:
    weights = 1.0 / np.maximum(dist, eps) ** float(power)
    weights = np.where(np.isfinite(weights), weights, 0.0)
    sums = weights.sum(axis=1)
    return np.divide(
        weights,
        sums[:, None],
        out=np.zeros_like(weights),
        where=sums[:, None] > 0,
    )


def _hidden_entry(tvt_input: np.ndarray) -> tuple[int | None, np.ndarray]:
    known = np.flatnonzero(np.isfinite(tvt_input))
    if len(known) == 0:
        return None, np.zeros(0, dtype=int)
    last_idx = int(known[-1])
    hidden = np.flatnonzero(~np.isfinite(tvt_input) & (np.arange(len(tvt_input)) > last_idx))
    return last_idx, hidden.astype(int)


def _tail_slope(md: np.ndarray, tvt_input: np.ndarray, tail_rows: int = 200) -> float:
    known = np.flatnonzero(np.isfinite(tvt_input) & np.isfinite(md))
    if len(known) < 2:
        return 0.0
    tail = known[-int(tail_rows) :]
    x = md[tail] - float(np.nanmean(md[tail]))
    y = tvt_input[tail] - float(np.nanmean(tvt_input[tail]))
    denom = float(np.dot(x, x))
    return float(np.dot(x, y) / denom) if denom > 1e-12 else 0.0


def _hidden_frac(n_hidden: int) -> np.ndarray:
    if n_hidden <= 0:
        return np.zeros(0, dtype=float)
    return np.arange(n_hidden, dtype=float) / max(n_hidden - 1, 1)


def _stable_seed(text: str, seed: int) -> int:
    value = int(seed) & 0xFFFFFFFF
    for char in text:
        value = ((value * 131) + ord(char)) & 0xFFFFFFFF
    return value


class FormationPlaneKNN:
    """Fold-safe local formation top imputer.

    The model is deterministic and contains only training-fold wells. It predicts
    formation tops from row samples in XY space by IDW, weighted planes, and
    bootstrap plane realizations.
    """

    def __init__(
        self,
        *,
        sample_xy: np.ndarray,
        sample_values: np.ndarray,
        sample_wells: np.ndarray,
        well_xy: np.ndarray,
        well_ids: np.ndarray,
        config: FormationPlaneConfig,
    ) -> None:
        self.sample_xy = np.asarray(sample_xy, dtype=float)
        self.sample_values = np.asarray(sample_values, dtype=float)
        self.sample_wells = np.asarray(sample_wells, dtype=object)
        self.well_xy = np.asarray(well_xy, dtype=float)
        self.well_ids = np.asarray(well_ids, dtype=object)
        self.config = config

        if len(self.sample_xy):
            scale = np.nanstd(self.sample_xy, axis=0)
            self.xy_scale = np.where(scale > 1e-6, scale, 1.0)
            self.sample_tree = cKDTree(self.sample_xy / self.xy_scale)
        else:
            self.xy_scale = np.ones(2, dtype=float)
            self.sample_tree = None

        if len(self.well_xy):
            scale = np.nanstd(self.well_xy, axis=0)
            self.well_scale = np.where(scale > 1e-6, scale, 1.0)
            self.well_tree = cKDTree(self.well_xy / self.well_scale)
        else:
            self.well_scale = np.ones(2, dtype=float)
            self.well_tree = None

    @classmethod
    def from_paths(
        cls, paths: list[Path], config: FormationPlaneConfig
    ) -> "FormationPlaneKNN":
        sample_xy_parts: list[np.ndarray] = []
        sample_value_parts: list[np.ndarray] = []
        sample_well_parts: list[np.ndarray] = []
        well_xy_rows: list[np.ndarray] = []
        well_ids: list[str] = []

        usecols = ["X", "Y", *FORMATIONS]
        for path in paths:
            try:
                frame = pd.read_csv(path, usecols=lambda column: column in usecols)
            except Exception:
                continue
            frame = frame.dropna(subset=usecols)
            if frame.empty:
                continue

            well = well_name(path)
            well_xy_rows.append(frame[["X", "Y"]].median().to_numpy(dtype=float))
            well_ids.append(well)

            take_n = min(int(config.sample_rows_per_well), len(frame))
            take = np.linspace(0, len(frame) - 1, take_n, dtype=int)
            sample = frame.iloc[take]
            sample_xy_parts.append(sample[["X", "Y"]].to_numpy(dtype=float))
            sample_value_parts.append(sample[list(FORMATIONS)].to_numpy(dtype=float))
            sample_well_parts.append(np.full(len(sample), well, dtype=object))

        if sample_xy_parts:
            sample_xy = np.vstack(sample_xy_parts)
            sample_values = np.vstack(sample_value_parts)
            sample_wells = np.concatenate(sample_well_parts)
        else:
            sample_xy = np.empty((0, 2), dtype=float)
            sample_values = np.empty((0, len(FORMATIONS)), dtype=float)
            sample_wells = np.empty(0, dtype=object)

        well_xy = np.vstack(well_xy_rows) if well_xy_rows else np.empty((0, 2), dtype=float)
        return cls(
            sample_xy=sample_xy,
            sample_values=sample_values,
            sample_wells=sample_wells,
            well_xy=well_xy,
            well_ids=np.asarray(well_ids, dtype=object),
            config=config,
        )

    def _query_samples(
        self,
        xy: np.ndarray,
        *,
        k: int | None = None,
        exclude_well: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.sample_tree is None or len(self.sample_values) == 0:
            return np.empty((len(xy), 0), dtype=float), np.empty((len(xy), 0), dtype=int)

        k_eff = min(len(self.sample_values), max(1, int(k or self.config.min_points)))
        extra = 0
        if exclude_well is not None:
            extra = int(np.sum(self.sample_wells == exclude_well))
        k_fetch = min(len(self.sample_values), max(k_eff, k_eff + extra + 8))
        dist, idx = self.sample_tree.query(xy / self.xy_scale, k=k_fetch)
        dist = np.asarray(dist)
        idx = np.asarray(idx)
        if k_fetch == 1:
            dist = dist.reshape(len(xy), 1)
            idx = idx.reshape(len(xy), 1)
        elif len(xy) == 1:
            dist = dist.reshape(1, -1)
            idx = idx.reshape(1, -1)

        if exclude_well is not None and idx.size:
            dist = np.where(self.sample_wells[idx] == exclude_well, np.inf, dist)

        order = np.argsort(dist, axis=1)[:, :k_eff]
        chosen_dist = np.take_along_axis(dist, order, axis=1)
        chosen_idx = np.take_along_axis(idx, order, axis=1)
        return chosen_dist, chosen_idx

    def nearest_well_distances(
        self, xy: np.ndarray, exclude_well: str | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.well_tree is None or len(self.well_ids) == 0:
            empty = np.full((len(xy), 0), np.nan, dtype=float)
            return empty, np.empty((len(xy), 0), dtype=int)
        k_eff = min(len(self.well_ids), max(1, int(self.config.k_wells)))
        extra = 1 if exclude_well is not None else 0
        k_fetch = min(len(self.well_ids), k_eff + extra)
        dist, idx = self.well_tree.query(xy / self.well_scale, k=k_fetch)
        dist = np.asarray(dist)
        idx = np.asarray(idx)
        if k_fetch == 1:
            dist = dist.reshape(len(xy), 1)
            idx = idx.reshape(len(xy), 1)
        elif len(xy) == 1:
            dist = dist.reshape(1, -1)
            idx = idx.reshape(1, -1)
        if exclude_well is not None:
            dist = np.where(self.well_ids[idx] == exclude_well, np.inf, dist)
        order = np.argsort(dist, axis=1)[:, :k_eff]
        return np.take_along_axis(dist, order, axis=1), np.take_along_axis(idx, order, axis=1)

    def predict_idw(
        self,
        xy: np.ndarray,
        *,
        k: int | None = None,
        exclude_well: str | None = None,
    ) -> SurfacePrediction:
        xy = np.asarray(xy, dtype=float)
        out_shape = (len(xy), len(FORMATIONS))
        if len(xy) == 0 or self.sample_tree is None:
            nan = np.full(out_shape, np.nan, dtype=float)
            return SurfacePrediction(
                values=nan,
                std=nan.copy(),
                dist_mean=np.full(len(xy), np.nan),
                dist_min=np.full(len(xy), np.nan),
                residual=nan.copy(),
            )
        dist, idx = self._query_samples(xy, k=k, exclude_well=exclude_well)
        if idx.shape[1] == 0:
            nan = np.full(out_shape, np.nan, dtype=float)
            return SurfacePrediction(
                values=nan,
                std=nan.copy(),
                dist_mean=np.full(len(xy), np.nan),
                dist_min=np.full(len(xy), np.nan),
                residual=nan.copy(),
            )
        values = self.sample_values[idx]
        valid = np.isfinite(dist)
        weights = _normalize_weights(
            np.where(valid, dist, np.inf), self.config.weight_power, self.config.eps
        )
        pred = np.sum(weights[..., None] * values, axis=1)
        std = _weighted_std(values, weights, pred)
        dist_mean = np.divide(
            np.sum(np.where(valid, dist, 0.0), axis=1),
            valid.sum(axis=1),
            out=np.full(len(xy), np.nan, dtype=float),
            where=valid.sum(axis=1) > 0,
        )
        dist_min = np.min(np.where(valid, dist, np.inf), axis=1)
        dist_min = np.where(np.isfinite(dist_min), dist_min, np.nan)
        return SurfacePrediction(
            values=pred,
            std=std,
            dist_mean=dist_mean,
            dist_min=dist_min,
            residual=std.copy(),
        )

    def predict_plane(
        self,
        xy: np.ndarray,
        *,
        exclude_well: str | None = None,
    ) -> SurfacePrediction:
        xy = np.asarray(xy, dtype=float)
        pred_parts: list[np.ndarray] = []
        std_parts: list[np.ndarray] = []
        dist_mean_parts: list[np.ndarray] = []
        dist_min_parts: list[np.ndarray] = []
        residual_parts: list[np.ndarray] = []

        for start in range(0, len(xy), int(self.config.query_chunk)):
            stop = min(start + int(self.config.query_chunk), len(xy))
            chunk = xy[start:stop]
            dist, idx = self._query_samples(chunk, exclude_well=exclude_well)
            pred, std, residual = self._plane_from_query(chunk, dist, idx)
            valid = np.isfinite(dist)
            dist_mean = np.divide(
                np.sum(np.where(valid, dist, 0.0), axis=1),
                valid.sum(axis=1),
                out=np.full(len(chunk), np.nan, dtype=float),
                where=valid.sum(axis=1) > 0,
            )
            dist_min = np.min(np.where(valid, dist, np.inf), axis=1)
            dist_min = np.where(np.isfinite(dist_min), dist_min, np.nan)
            pred_parts.append(pred)
            std_parts.append(std)
            residual_parts.append(residual)
            dist_mean_parts.append(dist_mean)
            dist_min_parts.append(dist_min)

        if not pred_parts:
            empty = np.empty((0, len(FORMATIONS)), dtype=float)
            return SurfacePrediction(
                values=empty,
                std=empty.copy(),
                dist_mean=np.empty(0, dtype=float),
                dist_min=np.empty(0, dtype=float),
                residual=empty.copy(),
            )

        return SurfacePrediction(
            values=np.vstack(pred_parts),
            std=np.vstack(std_parts),
            dist_mean=np.concatenate(dist_mean_parts),
            dist_min=np.concatenate(dist_min_parts),
            residual=np.vstack(residual_parts),
        )

    def _plane_from_query(
        self,
        query_xy: np.ndarray,
        dist: np.ndarray,
        idx: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(query_xy)
        if idx.shape[1] < 3:
            idw = self.predict_idw(query_xy, k=max(1, idx.shape[1]))
            return idw.values, idw.std, idw.residual

        valid = np.isfinite(dist)
        values = self.sample_values[idx]
        coords = self.sample_xy[idx]
        weights = _normalize_weights(
            np.where(valid, dist, np.inf),
            self.config.weight_power,
            self.config.eps,
        )
        design = np.dstack([coords[:, :, 0], coords[:, :, 1], np.ones_like(dist)])
        normal = np.einsum("nki,nkj,nk->nij", design, design, weights)
        rhs = np.einsum("nki,nkf,nk->nif", design, values, weights)
        normal += np.eye(3, dtype=float)[None, :, :] * 1e-8
        try:
            coef = np.linalg.solve(normal, rhs)
        except np.linalg.LinAlgError:
            coef = np.einsum("nij,njf->nif", np.linalg.pinv(normal), rhs)

        query_design = np.column_stack([query_xy[:, 0], query_xy[:, 1], np.ones(n)])
        pred = np.einsum("ni,nif->nf", query_design, coef)
        fitted = np.einsum("nki,nif->nkf", design, coef)
        residual = np.sqrt(
            np.maximum(np.sum(weights[..., None] * (fitted - values) ** 2, axis=1), 0.0)
        )
        std = _weighted_std(values, weights, np.sum(weights[..., None] * values, axis=1))
        return pred, std, residual

    def sample_plane_predictions(
        self,
        xy: np.ndarray,
        *,
        samples: int,
        seed: int,
        exclude_well: str | None = None,
    ) -> np.ndarray:
        xy = np.asarray(xy, dtype=float)
        n_samples = max(0, int(samples))
        if n_samples == 0 or len(xy) == 0:
            return np.empty((0, len(xy), len(FORMATIONS)), dtype=float)
        result = np.empty((n_samples, len(xy), len(FORMATIONS)), dtype=float)
        rng = np.random.default_rng(int(seed))
        sub_k = max(3, int(round(self.config.min_points * self.config.bootstrap_fraction)))

        for start in range(0, len(xy), int(self.config.query_chunk)):
            stop = min(start + int(self.config.query_chunk), len(xy))
            chunk = xy[start:stop]
            dist, idx = self._query_samples(chunk, exclude_well=exclude_well)
            k = idx.shape[1]
            if k < 3:
                idw = self.predict_idw(chunk, k=max(k, 1), exclude_well=exclude_well).values
                result[:, start:stop, :] = idw[None, :, :]
                continue
            draw_k = min(k, sub_k)
            for sample_id in range(n_samples):
                draw_pos = rng.integers(0, k, size=(len(chunk), draw_k))
                rows = np.arange(len(chunk))[:, None]
                sample_idx = idx[rows, draw_pos]
                sample_dist = dist[rows, draw_pos]
                pred, _std, _residual = self._plane_from_query(chunk, sample_dist, sample_idx)
                result[sample_id, start:stop, :] = pred
        return result


class NearbyPathLibrary:
    def __init__(self, paths: list[NeighborPath], weight_power: float = 1.0) -> None:
        self.paths = paths
        self.weight_power = float(weight_power)
        if paths:
            self.xy = np.vstack([path.xy for path in paths])
            scale = np.nanstd(self.xy, axis=0)
            self.scale = np.where(scale > 1e-6, scale, 1.0)
            self.tree = cKDTree(self.xy / self.scale)
        else:
            self.xy = np.empty((0, 2), dtype=float)
            self.scale = np.ones(2, dtype=float)
            self.tree = None

    @classmethod
    def from_paths(cls, paths: list[Path], weight_power: float = 1.0) -> "NearbyPathLibrary":
        out: list[NeighborPath] = []
        for path in paths:
            try:
                frame = pd.read_csv(path)
            except Exception:
                continue
            md = _numeric(frame, "MD")
            x = _numeric(frame, "X")
            y = _numeric(frame, "Y")
            tvt = _numeric(frame, "TVT")
            tvt_input = _numeric(frame, "TVT_input")
            last_idx, hidden_idx = _hidden_entry(tvt_input)
            if last_idx is None or len(hidden_idx) < 2:
                continue
            valid_hidden = hidden_idx[np.isfinite(tvt[hidden_idx])]
            if len(valid_hidden) < 2 or not np.isfinite(tvt_input[last_idx]):
                continue
            out.append(
                NeighborPath(
                    well=well_name(path),
                    xy=np.array([np.nanmean(x), np.nanmean(y)], dtype=float),
                    hidden_frac=_hidden_frac(len(valid_hidden)),
                    hidden_drift=tvt[valid_hidden] - float(tvt_input[last_idx]),
                    tail_slope=_tail_slope(md, tvt_input),
                    last_known_tvt=float(tvt_input[last_idx]),
                )
            )
        return cls(out, weight_power=weight_power)

    def predict(
        self,
        *,
        frame: pd.DataFrame,
        hidden_idx: np.ndarray,
        last_idx: int,
        k: int,
        self_well: str,
        formation_reference: np.ndarray | None = None,
        slope_weight: float = 1.0,
        formation_weight: float = 0.25,
    ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        n_hidden = len(hidden_idx)
        tvt_input = _numeric(frame, "TVT_input")
        md = _numeric(frame, "MD")
        x = _numeric(frame, "X")
        y = _numeric(frame, "Y")
        fallback = np.full(n_hidden, float(tvt_input[last_idx]), dtype=float)
        empty = {
            "nearby_path_top1": fallback.copy(),
            "nearby_path_top3_median": fallback.copy(),
            "nearby_path_top5_median": fallback.copy(),
            "nearby_path_weighted_mean": fallback.copy(),
            "nearby_path_p10": fallback.copy(),
            "nearby_path_p90": fallback.copy(),
            "nearby_path_std": np.zeros(n_hidden, dtype=float),
            "nearby_path_entropy": np.zeros(n_hidden, dtype=float),
        }
        if self.tree is None or not self.paths or n_hidden == 0:
            return empty, {"nearby_k": 0.0, "nearby_dist_min": float("nan"), "nearby_dist_mean": float("nan")}

        target_xy = np.array([[np.nanmean(x), np.nanmean(y)]], dtype=float)
        k_fetch = min(len(self.paths), max(1, int(k) + 1))
        dist, idx = self.tree.query(target_xy / self.scale, k=k_fetch)
        dist = np.asarray(dist).reshape(-1)
        idx = np.asarray(idx).reshape(-1)
        chosen: list[NeighborPath] = []
        chosen_dist: list[float] = []
        for d, i in zip(dist, idx, strict=False):
            path = self.paths[int(i)]
            if path.well == self_well:
                continue
            chosen.append(path)
            chosen_dist.append(float(d))
            if len(chosen) >= int(k):
                break
        if not chosen:
            return empty, {"nearby_k": 0.0, "nearby_dist_min": float("nan"), "nearby_dist_mean": float("nan")}

        frac = _hidden_frac(n_hidden)
        last_tvt = float(tvt_input[last_idx])
        target_slope = _tail_slope(md, tvt_input)
        md_delta = md[hidden_idx] - float(md[last_idx])
        paths = []
        for neighbor in chosen:
            drift = np.interp(
                frac,
                neighbor.hidden_frac,
                neighbor.hidden_drift,
                left=neighbor.hidden_drift[0],
                right=neighbor.hidden_drift[-1],
            )
            path = last_tvt + drift + float(slope_weight) * (target_slope - neighbor.tail_slope) * md_delta
            if formation_reference is not None and len(formation_reference) == n_hidden:
                path = path + float(formation_weight) * (formation_reference - path)
            paths.append(path)

        matrix = np.vstack(paths)
        d = np.asarray(chosen_dist, dtype=float)
        weights = 1.0 / np.maximum(d, 1e-6) ** self.weight_power
        weights = weights / max(float(weights.sum()), 1e-12)
        entropy = float(-np.sum(weights * np.log(np.maximum(weights, 1e-12))))
        result = {
            "nearby_path_top1": matrix[0],
            "nearby_path_top3_median": np.nanmedian(matrix[: min(3, len(matrix))], axis=0),
            "nearby_path_top5_median": np.nanmedian(matrix[: min(5, len(matrix))], axis=0),
            "nearby_path_weighted_mean": weights @ matrix,
            "nearby_path_p10": np.nanpercentile(matrix, 10, axis=0),
            "nearby_path_p90": np.nanpercentile(matrix, 90, axis=0),
            "nearby_path_std": np.nanstd(matrix, axis=0),
            "nearby_path_entropy": np.full(n_hidden, entropy, dtype=float),
        }
        return result, {
            "nearby_k": float(len(chosen)),
            "nearby_dist_min": float(np.min(d)),
            "nearby_dist_mean": float(np.mean(d)),
        }


def calibrate_b_well(
    tvt_input: np.ndarray,
    z: np.ndarray,
    surfaces: np.ndarray,
) -> dict[str, dict[str, float]]:
    known = np.flatnonzero(np.isfinite(tvt_input) & np.isfinite(z))
    stats: dict[str, dict[str, float]] = {}
    if len(known) == 0:
        for formation in FORMATIONS:
            stats[formation] = {
                "full": 0.0,
                "late": 0.0,
                "early": 0.0,
                "mid": 0.0,
                "wls": 0.0,
                "std": float("nan"),
                "late_minus_full": float("nan"),
            }
        return stats

    for f_idx, formation in enumerate(FORMATIONS):
        b = tvt_input[known] + z[known] - surfaces[known, f_idx]
        finite = b[np.isfinite(b)]
        if len(finite) == 0:
            stats[formation] = {
                "full": 0.0,
                "late": 0.0,
                "early": 0.0,
                "mid": 0.0,
                "wls": 0.0,
                "std": float("nan"),
                "late_minus_full": float("nan"),
            }
            continue
        n = len(finite)
        t1, t2 = n // 3, 2 * n // 3
        full = float(np.nanmedian(finite))
        late = float(np.nanmedian(finite[-min(50, n):]))
        early = float(np.nanmedian(finite[: max(1, t1)])) if t1 > 0 else full
        mid = float(np.nanmedian(finite[t1:max(t1 + 1, t2)])) if t2 > t1 else full
        exp_w = np.exp(0.02 * np.arange(n, dtype=float))
        exp_w = exp_w / float(exp_w.sum())
        wls = float(np.dot(exp_w, finite))
        stats[formation] = {
            "full": full,
            "late": late,
            "early": early,
            "mid": mid,
            "wls": wls,
            "std": float(np.nanstd(finite)),
            "late_minus_full": late - full,
        }
    return stats


def _anchor_train_pseudo_split(
    anchor_idx: np.ndarray,
    train_fraction: float = 0.70,
) -> tuple[np.ndarray, np.ndarray]:
    anchor_idx = np.asarray(anchor_idx, dtype=int)
    if len(anchor_idx) < 4:
        return anchor_idx, np.empty(0, dtype=int)
    split = int(np.floor(len(anchor_idx) * float(train_fraction)))
    split = min(max(2, split), len(anchor_idx) - 1)
    return anchor_idx[:split], anchor_idx[split:]


def _candidate_anchor_metrics(
    candidates: dict[str, np.ndarray],
    anchor_idx: np.ndarray,
    tvt_input: np.ndarray,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, values in candidates.items():
        if name not in CANDIDATE_COLUMNS:
            continue
        anchor_values = np.asarray(values, dtype=float)[anchor_idx]
        out[f"anchor_fit_rmse__{name}"] = _rmse(anchor_values, tvt_input[anchor_idx])
        err = anchor_values - tvt_input[anchor_idx]
        finite = err[np.isfinite(err)]
        out[f"anchor_bias__{name}"] = float(np.nanmean(finite)) if len(finite) else float("nan")
        out[f"anchor_slope_error__{name}"] = _slope_error(
            np.asarray(values, dtype=float), tvt_input, anchor_idx
        )
        late = anchor_idx[-min(50, len(anchor_idx)):] if len(anchor_idx) else anchor_idx
        out[f"late_anchor_fit_rmse__{name}"] = _rmse(np.asarray(values)[late], tvt_input[late])
        candidate_values = np.asarray(values, dtype=float)
        out[f"roughness__{name}"] = _roughness(candidate_values)
        out[f"finite_frac__{name}"] = float(np.isfinite(candidate_values).mean())
    return out


def _candidate_pseudo_metrics(
    candidates: dict[str, np.ndarray],
    pseudo_idx: np.ndarray,
    tvt_input: np.ndarray,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, values in candidates.items():
        if name not in CANDIDATE_COLUMNS:
            continue
        values = np.asarray(values, dtype=float)
        pred = values[pseudo_idx] if len(pseudo_idx) else np.empty(0, dtype=float)
        true = tvt_input[pseudo_idx] if len(pseudo_idx) else np.empty(0, dtype=float)
        out[f"pseudo_hidden_rmse__{name}"] = _rmse(pred, true)
        err = pred - true
        finite = err[np.isfinite(err)]
        out[f"pseudo_hidden_bias__{name}"] = (
            float(np.nanmean(finite)) if len(finite) else float("nan")
        )
        out[f"pseudo_hidden_slope_error__{name}"] = _slope_error(
            values, tvt_input, pseudo_idx
        )
    return out


def _build_plane_candidate_values(
    *,
    z: np.ndarray,
    plane_values: np.ndarray,
    row_ancc_values: np.ndarray,
    dense_ancc_values: np.ndarray,
    b_stats: dict[str, dict[str, float]],
) -> dict[str, np.ndarray]:
    candidates: dict[str, np.ndarray] = {}
    formation_full: list[np.ndarray] = []
    formation_wls: list[np.ndarray] = []
    for f_idx, formation in enumerate(FORMATIONS):
        for suffix in ("full", "late", "wls"):
            name = f"tvtF_{formation}_{suffix}"
            values = -z + plane_values[:, f_idx] + b_stats[formation][suffix]
            candidates[name] = values
            if suffix == "full":
                formation_full.append(values)
            if suffix == "wls":
                formation_wls.append(values)

    ancc_idx = FORMATIONS.index("ANCC")
    candidates["row_ancc_tvt"] = -z + row_ancc_values[:, ancc_idx] + b_stats["ANCC"]["wls"]
    candidates["dense_ancc_tvt"] = -z + dense_ancc_values[:, ancc_idx] + b_stats["ANCC"]["wls"]
    candidates["formation_median_tvt"] = np.nanmedian(np.vstack(formation_full), axis=0)
    candidates["formation_wls_median_tvt"] = np.nanmedian(np.vstack(formation_wls), axis=0)
    return candidates


def _build_pseudo_candidate_values(
    *,
    frame: pd.DataFrame,
    solver: FormationPlaneKNN,
    nearby: NearbyPathLibrary,
    xy: np.ndarray,
    z: np.ndarray,
    tvt_input: np.ndarray,
    plane_values: np.ndarray,
    row_ancc_values: np.ndarray,
    dense_ancc_values: np.ndarray,
    anchor_train_idx: np.ndarray,
    pseudo_idx: np.ndarray,
    well: str,
    seed: int,
) -> dict[str, np.ndarray]:
    pseudo_input = tvt_input.copy()
    pseudo_input[pseudo_idx] = np.nan
    b_stats = calibrate_b_well(pseudo_input, z, plane_values)
    candidates = _build_plane_candidate_values(
        z=z,
        plane_values=plane_values,
        row_ancc_values=row_ancc_values,
        dense_ancc_values=dense_ancc_values,
        b_stats=b_stats,
    )
    if len(anchor_train_idx) == 0 or len(pseudo_idx) == 0:
        return candidates

    pseudo_frame = frame.copy()
    pseudo_frame["TVT_input"] = pseudo_input
    last_idx = int(anchor_train_idx[-1])
    nearby_candidates, _nearby_diag = nearby.predict(
        frame=pseudo_frame,
        hidden_idx=pseudo_idx,
        last_idx=last_idx,
        k=solver.config.k_wells,
        self_well=well,
        formation_reference=candidates["formation_wls_median_tvt"][pseudo_idx],
    )
    for name, values in nearby_candidates.items():
        full = np.full(len(frame), np.nan, dtype=float)
        full[pseudo_idx] = values
        candidates[name] = full

    sample_columns = _build_bootstrap_sample_candidates(
        solver,
        xy=xy,
        z=z,
        tvt_input=pseudo_input,
        anchor_idx=anchor_train_idx,
        hidden_idx=pseudo_idx,
        b_stats=b_stats,
        well=well,
        seed=seed + 1009,
    )
    for name, values in sample_columns.items():
        full = np.full(len(frame), np.nan, dtype=float)
        full[pseudo_idx] = values
        candidates[name] = full
    return candidates


def build_well_candidates(
    path: Path,
    solver: FormationPlaneKNN,
    nearby: NearbyPathLibrary,
    *,
    fold_id: int,
    seed: int,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    well = well_name(path)
    x = _numeric(frame, "X")
    y = _numeric(frame, "Y")
    z = _numeric(frame, "Z")
    gr = _numeric(frame, "GR")
    tvt = _numeric(frame, "TVT")
    tvt_input = _numeric(frame, "TVT_input")
    last_idx, hidden_idx = _hidden_entry(tvt_input)
    anchor_idx = np.flatnonzero(np.isfinite(tvt_input))
    if last_idx is None or len(hidden_idx) == 0 or len(anchor_idx) == 0:
        return pd.DataFrame()

    xy = np.column_stack([x, y])
    plane = solver.predict_plane(xy, exclude_well=well)
    row_idw = solver.predict_idw(xy, k=solver.config.min_points, exclude_well=well)
    dense_ancc = solver.predict_idw(xy, k=solver.config.dense_k, exclude_well=well)
    b_stats = calibrate_b_well(tvt_input, z, plane.values)
    all_candidates = _build_plane_candidate_values(
        z=z,
        plane_values=plane.values,
        row_ancc_values=row_idw.values,
        dense_ancc_values=dense_ancc.values,
        b_stats=b_stats,
    )

    nearby_candidates, nearby_diag = nearby.predict(
        frame=frame,
        hidden_idx=hidden_idx,
        last_idx=last_idx,
        k=solver.config.k_wells,
        self_well=well,
        formation_reference=all_candidates["formation_wls_median_tvt"][hidden_idx],
    )
    for name, values in nearby_candidates.items():
        full = np.full(len(frame), np.nan, dtype=float)
        full[hidden_idx] = values
        all_candidates[name] = full

    sample_columns = _build_bootstrap_sample_candidates(
        solver,
        xy=xy,
        z=z,
        tvt_input=tvt_input,
        anchor_idx=anchor_idx,
        hidden_idx=hidden_idx,
        b_stats=b_stats,
        well=well,
        seed=seed,
    )
    for name, values in sample_columns.items():
        full = np.full(len(frame), np.nan, dtype=float)
        full[hidden_idx] = values
        all_candidates[name] = full

    candidate_matrix = np.vstack(
        [all_candidates[name][hidden_idx] for name in CANDIDATE_COLUMNS if name in all_candidates]
    )
    surface_std = np.nanstd(candidate_matrix, axis=0)
    anchor_metrics = _candidate_anchor_metrics(all_candidates, anchor_idx, tvt_input)
    anchor_train_idx, pseudo_idx = _anchor_train_pseudo_split(anchor_idx)
    pseudo_candidates = _build_pseudo_candidate_values(
        frame=frame,
        solver=solver,
        nearby=nearby,
        xy=xy,
        z=z,
        tvt_input=tvt_input,
        plane_values=plane.values,
        row_ancc_values=row_idw.values,
        dense_ancc_values=dense_ancc.values,
        anchor_train_idx=anchor_train_idx,
        pseudo_idx=pseudo_idx,
        well=well,
        seed=seed,
    )
    pseudo_metrics = _candidate_pseudo_metrics(pseudo_candidates, pseudo_idx, tvt_input)

    output_columns: dict[str, Any] = {
        "id": [f"{well}_{int(i)}" for i in hidden_idx],
        "well_id": well,
        "fold": int(fold_id),
        "row_idx": hidden_idx.astype(int),
        "TVT": tvt[hidden_idx],
        "GR": gr[hidden_idx],
        "hidden_frac": _hidden_frac(len(hidden_idx)),
        "hidden_rows": float(len(hidden_idx)),
        "last_known_tvt": float(tvt_input[last_idx]),
        "anchor_rows": float(len(anchor_idx)),
        "pseudo_anchor_rows": float(len(pseudo_idx)),
        "surface_candidates_std": surface_std,
        "neighbor_dist_mean": plane.dist_mean[hidden_idx],
        "neighbor_dist_min": plane.dist_min[hidden_idx],
        "nearby_path_std": all_candidates["nearby_path_std"][hidden_idx],
        "nearby_path_entropy": all_candidates["nearby_path_entropy"][hidden_idx],
        **{key: value for key, value in nearby_diag.items()},
    }

    for f_idx, formation in enumerate(FORMATIONS):
        output_columns[f"S_hat_{formation}"] = plane.values[hidden_idx, f_idx]
        output_columns[f"S_hat_{formation}_std"] = plane.std[hidden_idx, f_idx]
        output_columns[f"S_hat_{formation}_plane_residual"] = plane.residual[
            hidden_idx, f_idx
        ]
        output_columns[f"true_{formation}"] = _numeric(frame, formation)[hidden_idx]
        output_columns[f"b_{formation}_full"] = b_stats[formation]["full"]
        output_columns[f"b_{formation}_late"] = b_stats[formation]["late"]
        output_columns[f"b_{formation}_wls"] = b_stats[formation]["wls"]
        output_columns[f"b_{formation}_std"] = b_stats[formation]["std"]
        output_columns[f"b_{formation}_late_minus_full"] = b_stats[formation][
            "late_minus_full"
        ]

    for name in CANDIDATE_COLUMNS:
        if name in all_candidates:
            output_columns[name] = all_candidates[name][hidden_idx]
    for key, value in anchor_metrics.items():
        output_columns[key] = value
    for key, value in pseudo_metrics.items():
        output_columns[key] = value
    return pd.DataFrame(output_columns)


def _build_bootstrap_sample_candidates(
    solver: FormationPlaneKNN,
    *,
    xy: np.ndarray,
    z: np.ndarray,
    tvt_input: np.ndarray,
    anchor_idx: np.ndarray,
    hidden_idx: np.ndarray,
    b_stats: dict[str, dict[str, float]],
    well: str,
    seed: int,
) -> dict[str, np.ndarray]:
    n_samples = int(solver.config.bootstrap_samples)
    n_hidden = len(hidden_idx)
    fallback = np.full(n_hidden, np.nan, dtype=float)
    if n_samples <= 0 or n_hidden == 0:
        return {
            "formation_sample_best_by_anchor": fallback.copy(),
            "formation_sample_mean": fallback.copy(),
            "formation_sample_median": fallback.copy(),
            "formation_sample_p10": fallback.copy(),
            "formation_sample_p90": fallback.copy(),
            "formation_sample_std": fallback.copy(),
            "formation_sample_top2_gap": fallback.copy(),
        }

    sample_seed = _stable_seed(well, seed)
    hidden_samples = solver.sample_plane_predictions(
        xy[hidden_idx],
        samples=n_samples,
        seed=sample_seed,
        exclude_well=well,
    )
    anchor_samples = solver.sample_plane_predictions(
        xy[anchor_idx],
        samples=n_samples,
        seed=sample_seed + 17,
        exclude_well=well,
    )
    rng = np.random.default_rng(sample_seed + 101)
    paths = np.empty((n_samples, n_hidden), dtype=float)
    anchor_rmse = np.empty(n_samples, dtype=float)

    b_anchor = {
        f_idx: tvt_input[anchor_idx] + z[anchor_idx] - anchor_samples[0, :, f_idx]
        for f_idx in range(len(FORMATIONS))
    }
    for sample_id in range(n_samples):
        b_values = np.zeros(len(FORMATIONS), dtype=float)
        for f_idx, formation in enumerate(FORMATIONS):
            b_source = b_anchor[f_idx]
            finite = b_source[np.isfinite(b_source)]
            if len(finite):
                b_values[f_idx] = float(rng.choice(finite))
            else:
                b_values[f_idx] = b_stats[formation]["wls"]
        hidden_by_form = -z[hidden_idx, None] + hidden_samples[sample_id] + b_values[None, :]
        anchor_by_form = -z[anchor_idx, None] + anchor_samples[sample_id] + b_values[None, :]
        paths[sample_id] = np.nanmedian(hidden_by_form, axis=1)
        anchor_path = np.nanmedian(anchor_by_form, axis=1)
        anchor_rmse[sample_id] = _rmse(anchor_path, tvt_input[anchor_idx])

    best_idx = int(np.nanargmin(anchor_rmse)) if np.isfinite(anchor_rmse).any() else 0
    median_path = np.nanmedian(paths, axis=0)
    central_order = np.argsort(np.abs(paths - median_path[None, :]), axis=0)
    if n_samples >= 2:
        first = np.take_along_axis(paths, central_order[:1], axis=0)[0]
        second = np.take_along_axis(paths, central_order[1:2], axis=0)[0]
        top2_gap = np.abs(second - first)
    else:
        top2_gap = np.zeros(n_hidden, dtype=float)
    return {
        "formation_sample_best_by_anchor": paths[best_idx],
        "formation_sample_mean": np.nanmean(paths, axis=0),
        "formation_sample_median": median_path,
        "formation_sample_p10": np.nanpercentile(paths, 10, axis=0),
        "formation_sample_p90": np.nanpercentile(paths, 90, axis=0),
        "formation_sample_std": np.nanstd(paths, axis=0),
        "formation_sample_top2_gap": top2_gap,
    }


def shuffled_path_folds(
    paths: list[Path], n_splits: int, seed: int
) -> list[tuple[int, list[Path], list[Path]]]:
    ordered = sorted(paths, key=well_name)
    wells = np.array([well_name(path) for path in ordered])
    rng = np.random.default_rng(seed)
    unique = np.array(sorted(set(wells)))
    rng.shuffle(unique)
    n_splits = min(max(2, int(n_splits)), len(unique))
    folds = []
    for fold in range(n_splits):
        valid_wells = set(unique[fold::n_splits])
        train_paths = [path for path in ordered if well_name(path) not in valid_wells]
        valid_paths = [path for path in ordered if well_name(path) in valid_wells]
        folds.append((fold + 1, train_paths, valid_paths))
    return folds


def score_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    y = frame["TVT"].to_numpy(dtype=float)
    hidden_counts = frame.groupby("well_id")["row_idx"].transform("count").to_numpy(dtype=float)
    long_mask = hidden_counts > np.nanmedian(hidden_counts)
    short_mask = ~long_mask
    gr_nan = ~np.isfinite(frame["GR"].to_numpy(dtype=float))
    for column in CANDIDATE_COLUMNS:
        if column not in frame.columns:
            continue
        pred = frame[column].to_numpy(dtype=float)
        score_frame = pd.DataFrame(
            {
                "well_id": frame["well_id"].to_numpy(),
                "TVT": frame["TVT"].to_numpy(dtype=float),
                "_pred": pred,
            }
        )
        well_scores = score_frame.groupby("well_id").apply(
            lambda g: _rmse(g["_pred"].to_numpy(float), g["TVT"].to_numpy(float)),
            include_groups=False,
        )
        rows.append(
            {
                "candidate": column,
                "rmse": _rmse(pred, y),
                "mean_well_rmse": float(np.nanmean(well_scores.to_numpy(float))),
                "p90_well_rmse": float(np.nanpercentile(well_scores, 90)),
                "p95_well_rmse": float(np.nanpercentile(well_scores, 95)),
                "worst_well_rmse": float(np.nanmax(well_scores)),
                "worst_well": str(well_scores.idxmax()) if len(well_scores) else "",
                "long_hidden_rmse": _rmse(pred[long_mask], y[long_mask]) if long_mask.any() else float("nan"),
                "short_hidden_rmse": _rmse(pred[short_mask], y[short_mask]) if short_mask.any() else float("nan"),
                "gr_nan_rmse": _rmse(pred[gr_nan], y[gr_nan]) if gr_nan.any() else float("nan"),
                "finite_frac": float(np.isfinite(pred).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)


def surface_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for formation in FORMATIONS:
        pred = frame[f"S_hat_{formation}"].to_numpy(dtype=float)
        true = frame[f"true_{formation}"].to_numpy(dtype=float)
        rows.append(
            {
                "formation": formation,
                "rmse": _rmse(pred, true),
                "bias": float(np.nanmean(pred - true)),
                "p90_abs_error": _pctl_abs_error(pred, true, 90),
                "p95_abs_error": _pctl_abs_error(pred, true, 95),
                "mean_pred_std": float(np.nanmean(frame[f"S_hat_{formation}_std"].to_numpy(dtype=float))),
                "mean_plane_residual": float(np.nanmean(frame[f"S_hat_{formation}_plane_residual"].to_numpy(dtype=float))),
            }
        )
    return pd.DataFrame(rows)


def anchor_hidden_correlations(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in CANDIDATE_COLUMNS:
        if column not in frame.columns:
            continue
        anchor_col = f"anchor_fit_rmse__{column}"
        if anchor_col not in frame.columns:
            continue
        per_well = []
        for well, group in frame.groupby("well_id"):
            per_well.append(
                {
                    "well_id": well,
                    "anchor_rmse": float(group[anchor_col].iloc[0]),
                    "hidden_rmse": _rmse(group[column], group["TVT"]),
                }
            )
        well_frame = pd.DataFrame(per_well)
        rows.append(
            {
                "candidate": column,
                "corr_anchor_hidden_rmse": _corr(
                    well_frame["anchor_rmse"].to_numpy(float),
                    well_frame["hidden_rmse"].to_numpy(float),
                ),
                "n_wells": int(len(well_frame)),
            }
        )
    return pd.DataFrame(rows)


def oracle_scores(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        column
        for column in CANDIDATE_COLUMNS
        if column in frame.columns and np.isfinite(frame[column].to_numpy(float)).any()
    ]
    if not columns:
        return pd.DataFrame(), pd.DataFrame()

    y = frame["TVT"].to_numpy(dtype=float)
    matrix = frame[columns].to_numpy(dtype=float)
    err = np.abs(matrix - y[:, None])
    err = np.where(np.isfinite(err), err, np.inf)
    row_choice_idx = np.argmin(err, axis=1)
    row_pred = matrix[np.arange(len(frame)), row_choice_idx]
    rows = [
        {
            "oracle": "row_oracle",
            "rmse": _rmse(row_pred, y),
        }
    ]
    winners = [
        {
            "oracle": "row_oracle",
            "candidate": columns[int(i)],
            "count": int(count),
        }
        for i, count in zip(*np.unique(row_choice_idx, return_counts=True), strict=False)
    ]

    well_pred = np.full(len(frame), np.nan, dtype=float)
    thirds_pred = np.full(len(frame), np.nan, dtype=float)
    whole_winners: dict[str, int] = {}
    thirds_winners: dict[str, int] = {}
    for _well, group in frame.groupby("well_id", sort=False):
        idx = group.index.to_numpy(dtype=int)
        group_y = frame.loc[idx, "TVT"].to_numpy(dtype=float)
        group_matrix = frame.loc[idx, columns].to_numpy(dtype=float)
        scores = np.array([_rmse(group_matrix[:, j], group_y) for j in range(len(columns))])
        best = int(np.nanargmin(scores))
        well_pred[idx] = group_matrix[:, best]
        whole_winners[columns[best]] = whole_winners.get(columns[best], 0) + 1

        local_order = np.arange(len(idx))
        segments = np.array_split(local_order, 3)
        for segment in segments:
            if len(segment) == 0:
                continue
            seg_idx = idx[segment]
            seg_y = frame.loc[seg_idx, "TVT"].to_numpy(dtype=float)
            seg_matrix = frame.loc[seg_idx, columns].to_numpy(dtype=float)
            seg_scores = np.array(
                [_rmse(seg_matrix[:, j], seg_y) for j in range(len(columns))]
            )
            seg_best = int(np.nanargmin(seg_scores))
            thirds_pred[seg_idx] = seg_matrix[:, seg_best]
            thirds_winners[columns[seg_best]] = thirds_winners.get(columns[seg_best], 0) + 1

    rows.extend(
        [
            {"oracle": "thirds_segment_oracle", "rmse": _rmse(thirds_pred, y)},
            {"oracle": "whole_well_oracle", "rmse": _rmse(well_pred, y)},
        ]
    )
    winners.extend(
        {
            "oracle": "whole_well_oracle",
            "candidate": candidate,
            "count": count,
        }
        for candidate, count in sorted(whole_winners.items())
    )
    winners.extend(
        {
            "oracle": "thirds_segment_oracle",
            "candidate": candidate,
            "count": count,
        }
        for candidate, count in sorted(thirds_winners.items())
    )
    return pd.DataFrame(rows), pd.DataFrame(winners)


def write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        try:
            frame.to_parquet(path, index=False)
            return
        except Exception:
            csv_path = path.with_suffix(".csv")
            frame.to_csv(csv_path, index=False)
            return
    frame.to_csv(path, index=False)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def markdown_table(frame: pd.DataFrame, max_rows: int = 30) -> str:
    visible = frame.head(max_rows)
    if visible.empty:
        return "_empty_"
    lines = [
        "| " + " | ".join(str(c) for c in visible.columns) + " |",
        "| " + " | ".join(["---"] * len(visible.columns)) + " |",
    ]
    for row in visible.itertuples(index=False, name=None):
        cells = []
        for value in row:
            if isinstance(value, float):
                cells.append("" if not np.isfinite(value) else f"{value:.6f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = Path(args.data_dir)
    train_dir = Path(args.train_dir) if args.train_dir else data_dir / "train"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = horizontal_files(train_dir, args.max_wells)
    cfg = FormationPlaneConfig(
        k_wells=args.k_wells,
        sample_rows_per_well=args.sample_rows_per_well,
        min_points=args.min_points,
        dense_k=args.dense_k,
        weight_power=args.weight_power,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_fraction=args.bootstrap_fraction,
        query_chunk=args.query_chunk,
        seed=args.seed,
    )

    frames: list[pd.DataFrame] = []
    folds = shuffled_path_folds(paths, args.n_splits, args.seed)
    for fold_id, train_paths, valid_paths in folds:
        print(
            f"fold={fold_id} train_wells={len(train_paths)} valid_wells={len(valid_paths)}",
            flush=True,
        )
        solver = FormationPlaneKNN.from_paths(train_paths, cfg)
        nearby = NearbyPathLibrary.from_paths(train_paths, weight_power=cfg.weight_power)
        for i, path in enumerate(valid_paths, start=1):
            print(
                f"  well {i}/{len(valid_paths)} {well_name(path)} samples={len(solver.sample_xy)}",
                flush=True,
            )
            frame = build_well_candidates(
                path,
                solver,
                nearby,
                fold_id=fold_id,
                seed=args.seed,
            )
            if not frame.empty:
                frames.append(frame)

    if not frames:
        raise RuntimeError("No OOF candidate rows were produced.")

    oof = pd.concat(frames, ignore_index=True)
    candidate_scores = score_candidates(oof)
    surface_scores = surface_diagnostics(oof)
    anchor_corr = anchor_hidden_correlations(oof)
    oracle_frame, oracle_winners = oracle_scores(oof)
    metrics = {
        "config": cfg.__dict__,
        "rows": int(len(oof)),
        "wells": int(oof["well_id"].nunique()),
        "folds": int(len(folds)),
        "best_candidate": candidate_scores.iloc[0].to_dict() if not candidate_scores.empty else {},
        "oracles": oracle_frame.to_dict("records"),
        "surface_rmse_mean": float(surface_scores["rmse"].mean()),
        "best_anchor_hidden_corr": (
            anchor_corr.sort_values("corr_anchor_hidden_rmse", ascending=False)
            .head(1)
            .to_dict("records")
        ),
    }

    write_frame(oof, output_dir / "oof_candidates.parquet")
    candidate_scores.to_csv(output_dir / "candidate_scores.csv", index=False)
    surface_scores.to_csv(output_dir / "surface_scores.csv", index=False)
    anchor_corr.to_csv(output_dir / "anchor_hidden_correlation.csv", index=False)
    oracle_frame.to_csv(output_dir / "oracle_scores.csv", index=False)
    oracle_winners.to_csv(output_dir / "oracle_winners.csv", index=False)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(json_safe(metrics), file, indent=2)
    report = "\n\n".join(
        [
            "# Formation Plane KNN Report",
            f"Rows: `{len(oof)}`",
            f"Wells: `{oof['well_id'].nunique()}`",
            "## Candidate Scores",
            markdown_table(candidate_scores),
            "## Surface Diagnostics",
            markdown_table(surface_scores),
            "## Oracles",
            markdown_table(oracle_frame),
            "## Anchor-Hidden Correlation",
            markdown_table(anchor_corr.sort_values("corr_anchor_hidden_rmse", ascending=False)),
        ]
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fold-safe FormationPlaneKNN solver experiment.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--train-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/formation_plane_knn"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--max-wells", type=int, default=None)
    parser.add_argument("--k-wells", type=int, default=15)
    parser.add_argument("--sample-rows-per-well", type=int, default=80)
    parser.add_argument("--min-points", type=int, default=80)
    parser.add_argument("--dense-k", type=int, default=120)
    parser.add_argument("--weight-power", type=float, default=1.0)
    parser.add_argument("--bootstrap-samples", type=int, default=64)
    parser.add_argument("--bootstrap-fraction", type=float, default=0.75)
    parser.add_argument("--query-chunk", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    metrics = run_experiment(parse_args(argv))
    print(json.dumps(json_safe(metrics), indent=2), flush=True)


if __name__ == "__main__":
    main()
