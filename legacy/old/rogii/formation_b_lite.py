from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .formation_plane_knn import CANDIDATE_COLUMNS, _rmse, json_safe, markdown_table, oracle_scores, write_frame
from .formation_selector import attach_schema10, make_score_context, score_prediction


SHIFT_GRID = np.array([-80, -60, -40, -20, -10, 0, 10, 20, 40, 60, 80], dtype=float)
NCC_HALFWIDTHS: tuple[int, ...] = (8, 15, 25)
SOFT_TEMPERATURES: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0)
SAFE_ALPHAS: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.6)
SAFE_CLIPS: tuple[float, ...] = (10.0, 15.0, 20.0, 30.0, 40.0)
B_SCORE_WEIGHTS: dict[str, float] = {
    "cost_path_corr": 1.00,
    "cost_dgr_corr": 0.80,
    "cost_ncc15_mean": 0.60,
    "cost_ncc25_mean": 0.40,
    "cost_path_mad": 0.50,
    "cost_ncc_bad_frac": 0.30,
    "cost_visible_shift_gap": 0.20,
    "cost_geology_switch_count": 0.15,
    "cost_surface_std": 0.10,
    "cost_roughness": 0.10,
}
GR_ONLY_WEIGHTS: dict[str, float] = {
    "cost_path_corr": 1.00,
    "cost_dgr_corr": 0.80,
    "cost_ncc15_mean": 0.60,
    "cost_ncc25_mean": 0.40,
    "cost_path_mad": 0.50,
    "cost_ncc_bad_frac": 0.30,
}


@dataclass(frozen=True)
class TypewellData:
    tvt: np.ndarray
    gr: np.ndarray
    geology: np.ndarray | None
    boundary_tvt: np.ndarray


def _numeric(frame: pd.DataFrame, column: str, default: float = np.nan) -> np.ndarray:
    if column not in frame.columns:
        return np.full(len(frame), default, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _horizontal_path(train_dir: Path, well: str) -> Path:
    return train_dir / f"{well}__horizontal_well.csv"


def _typewell_path(train_dir: Path, well: str) -> Path:
    return train_dir / f"{well}__typewell.csv"


def _find_typewell_path(data_dir: Path, train_dir: Path, well: str) -> Path:
    candidates = [
        train_dir / f"{well}__typewell.csv",
        data_dir / "train" / f"{well}__typewell.csv",
        data_dir / "test" / f"{well}__typewell.csv",
        data_dir / "public_train" / f"{well}__typewell.csv",
        data_dir / "public_test" / f"{well}__typewell.csv",
        data_dir / f"{well}__typewell.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def read_typewell(path: Path) -> TypewellData | None:
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path)
    except Exception:
        return None
    if "TVT" not in frame.columns or "GR" not in frame.columns:
        return None
    tvt = _numeric(frame, "TVT")
    gr = _numeric(frame, "GR")
    mask = np.isfinite(tvt) & np.isfinite(gr)
    if int(mask.sum()) < 16:
        return None
    geology_raw = None
    if "Geology" in frame.columns:
        geology_raw = frame["Geology"].astype(str).to_numpy(dtype=object)
    tvt = tvt[mask]
    gr = gr[mask]
    geology = geology_raw[mask] if geology_raw is not None else None
    order = np.argsort(tvt)
    tvt = tvt[order]
    gr = gr[order]
    geology = geology[order] if geology is not None else None
    unique_tvt, unique_idx = np.unique(tvt, return_index=True)
    tvt = unique_tvt.astype(float)
    gr = gr[unique_idx].astype(float)
    geology = geology[unique_idx] if geology is not None else None
    boundary_tvt = np.empty(0, dtype=float)
    if geology is not None and len(geology) > 1:
        changes = np.flatnonzero(geology[1:] != geology[:-1]) + 1
        if len(changes):
            boundary_tvt = tvt[changes].astype(float)
    return TypewellData(tvt=tvt, gr=gr, geology=geology, boundary_tvt=boundary_tvt)


def interp_typewell_gr(typewell: TypewellData | None, tvt_path: np.ndarray) -> np.ndarray:
    if typewell is None or len(typewell.tvt) < 2:
        return np.full(len(tvt_path), np.nan, dtype=float)
    return np.interp(
        np.asarray(tvt_path, dtype=float),
        typewell.tvt,
        typewell.gr,
        left=typewell.gr[0],
        right=typewell.gr[-1],
    ).astype(float)


def _robust_z(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return np.full(len(arr), np.nan, dtype=float)
    med = float(np.nanmedian(finite))
    q25, q75 = np.nanpercentile(finite, [25, 75])
    scale = float(q75 - q25)
    if not np.isfinite(scale) or scale < 1e-9:
        mad = float(np.nanmedian(np.abs(finite - med)))
        scale = 1.4826 * mad if mad > 1e-9 else 1.0
    return (arr - med) / scale


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 5:
        return float("nan")
    if float(np.nanstd(a[mask])) < 1e-12 or float(np.nanstd(b[mask])) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def _smooth(values: np.ndarray, window: int = 7) -> np.ndarray:
    series = pd.Series(values, dtype=float).interpolate(limit_direction="both")
    return (
        series.rolling(max(3, int(window)), center=True, min_periods=1).mean().to_numpy(float)
    )


def _rolling_corr(a: np.ndarray, b: np.ndarray, halfwidth: int) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(a)
    if n == 0:
        return np.empty(0, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    aa = np.where(valid, a, 0.0)
    bb = np.where(valid, b, 0.0)
    one = valid.astype(float)

    def cs(x: np.ndarray) -> np.ndarray:
        return np.r_[0.0, np.cumsum(x)]

    c1, ca, cb, caa, cbb, cab = (
        cs(one),
        cs(aa),
        cs(bb),
        cs(aa * aa),
        cs(bb * bb),
        cs(aa * bb),
    )
    idx = np.arange(n)
    start = np.maximum(0, idx - int(halfwidth))
    stop = np.minimum(n, idx + int(halfwidth) + 1)
    count = c1[stop] - c1[start]
    sum_a = ca[stop] - ca[start]
    sum_b = cb[stop] - cb[start]
    sum_aa = caa[stop] - caa[start]
    sum_bb = cbb[stop] - cbb[start]
    sum_ab = cab[stop] - cab[start]
    cov = sum_ab - (sum_a * sum_b / np.maximum(count, 1.0))
    var_a = sum_aa - (sum_a * sum_a / np.maximum(count, 1.0))
    var_b = sum_bb - (sum_b * sum_b / np.maximum(count, 1.0))
    denom = np.sqrt(np.maximum(var_a, 0.0) * np.maximum(var_b, 0.0))
    corr = np.divide(cov, denom, out=np.full(n, np.nan), where=denom > 1e-12)
    min_periods = max(6, int(halfwidth))
    corr[count < min_periods] = np.nan
    return corr


def _safe_mean(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.nanmean(finite)) if len(finite) else float("nan")


def _safe_median(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.nanmedian(finite)) if len(finite) else float("nan")


def _safe_percentile(values: np.ndarray, q: float) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.nanpercentile(finite, q)) if len(finite) else float("nan")


def _linear_anchor_extrapolation(horizontal: pd.DataFrame, row_idx: np.ndarray) -> np.ndarray:
    tvt_input = _numeric(horizontal, "TVT_input")
    known = np.flatnonzero(np.isfinite(tvt_input))
    if len(known) == 0:
        return np.full(len(row_idx), np.nan, dtype=float)
    last = int(known[-1])
    tail = known[-min(200, len(known)) :]
    if len(tail) >= 2:
        x = tail.astype(float) - float(np.mean(tail))
        y = tvt_input[tail] - float(np.nanmean(tvt_input[tail]))
        denom = float(np.dot(x, x))
        slope = float(np.dot(x, y) / denom) if denom > 1e-12 else 0.0
    else:
        slope = 0.0
    return float(tvt_input[last]) + slope * (row_idx.astype(float) - float(last))


def visible_shift_metrics(horizontal: pd.DataFrame, typewell: TypewellData | None) -> dict[str, float]:
    if typewell is None:
        return {
            "best_visible_shift": float("nan"),
            "best_visible_corr": float("nan"),
            "visible_shift_curve_width": float("nan"),
        }
    tvt_input = _numeric(horizontal, "TVT_input")
    gr = _numeric(horizontal, "GR")
    mask = np.isfinite(tvt_input) & np.isfinite(gr)
    if int(mask.sum()) < 16:
        return {
            "best_visible_shift": float("nan"),
            "best_visible_corr": float("nan"),
            "visible_shift_curve_width": float("nan"),
        }
    h_z = _robust_z(gr[mask])
    corrs = []
    for shift in SHIFT_GRID:
        tw_gr = interp_typewell_gr(typewell, tvt_input[mask] + float(shift))
        corrs.append(_corr(h_z, _robust_z(tw_gr)))
    corr_arr = np.asarray(corrs, dtype=float)
    if not np.isfinite(corr_arr).any():
        return {
            "best_visible_shift": float("nan"),
            "best_visible_corr": float("nan"),
            "visible_shift_curve_width": float("nan"),
        }
    best_idx = int(np.nanargmax(corr_arr))
    best_corr = float(corr_arr[best_idx])
    width = float(np.sum(corr_arr >= best_corr - 0.05))
    return {
        "best_visible_shift": float(SHIFT_GRID[best_idx]),
        "best_visible_corr": best_corr,
        "visible_shift_curve_width": width,
    }


def geology_metrics(typewell: TypewellData | None, path: np.ndarray) -> dict[str, float]:
    if typewell is None or typewell.geology is None or len(typewell.geology) == 0:
        return {
            "geology_switch_count": float("nan"),
            "distance_to_boundary_mean": float("nan"),
            "distance_to_boundary_min": float("nan"),
            "share_near_boundary": float("nan"),
        }
    path = np.asarray(path, dtype=float)
    idx = np.searchsorted(typewell.tvt, path, side="left")
    idx = np.clip(idx, 0, len(typewell.tvt) - 1)
    codes = typewell.geology[idx]
    valid = np.isfinite(path)
    finite_codes = codes[valid]
    switch_count = (
        float(np.sum(finite_codes[1:] != finite_codes[:-1])) if len(finite_codes) > 1 else 0.0
    )
    if len(typewell.boundary_tvt) == 0 or not valid.any():
        return {
            "geology_switch_count": switch_count,
            "distance_to_boundary_mean": float("nan"),
            "distance_to_boundary_min": float("nan"),
            "share_near_boundary": 0.0,
        }
    dist = np.min(np.abs(path[valid, None] - typewell.boundary_tvt[None, :]), axis=1)
    return {
        "geology_switch_count": switch_count,
        "distance_to_boundary_mean": _safe_mean(dist),
        "distance_to_boundary_min": float(np.nanmin(dist)) if len(dist) else float("nan"),
        "share_near_boundary": float(np.mean(dist < 5.0)) if len(dist) else float("nan"),
    }


def path_gr_metrics(
    *,
    path: np.ndarray,
    horizontal_gr: np.ndarray,
    typewell: TypewellData | None,
) -> dict[str, float]:
    tw_gr = interp_typewell_gr(typewell, path)
    h_z = _robust_z(horizontal_gr)
    tw_z = _robust_z(tw_gr)
    mask = np.isfinite(h_z) & np.isfinite(tw_z)
    diff = h_z - tw_z
    path_corr = _corr(h_z, tw_z)
    path_mad = _safe_median(np.abs(diff[mask])) if mask.any() else float("nan")
    path_rmse = (
        float(np.sqrt(np.nanmean(diff[mask] * diff[mask]))) if mask.any() else float("nan")
    )
    finite_frac = float(mask.mean()) if len(mask) else 0.0

    h_dgr = np.gradient(_smooth(h_z, window=7))
    tw_dgr = np.gradient(_smooth(tw_z, window=7))
    dmask = np.isfinite(h_dgr) & np.isfinite(tw_dgr)
    dgr_diff = h_dgr - tw_dgr
    sign_mask = dmask & (np.abs(h_dgr) > 1e-6) & (np.abs(tw_dgr) > 1e-6)
    out = {
        "b_path_corr": path_corr,
        "b_path_mad": path_mad,
        "b_path_rmse": path_rmse,
        "b_finite_frac": finite_frac,
        "b_dgr_corr": _corr(h_dgr, tw_dgr),
        "b_dgr_mad": _safe_median(np.abs(dgr_diff[dmask])) if dmask.any() else float("nan"),
        "b_dgr_sign_agreement": (
            float(np.mean(np.sign(h_dgr[sign_mask]) == np.sign(tw_dgr[sign_mask])))
            if sign_mask.any()
            else float("nan")
        ),
    }
    ncc_means = []
    bad_fracs = []
    for halfwidth in NCC_HALFWIDTHS:
        corr = _rolling_corr(h_z, tw_z, halfwidth)
        out[f"b_ncc{halfwidth}_mean"] = _safe_mean(corr)
        out[f"b_ncc{halfwidth}_median"] = _safe_median(corr)
        out[f"b_ncc{halfwidth}_p10"] = _safe_percentile(corr, 10)
        out[f"b_ncc{halfwidth}_p90"] = _safe_percentile(corr, 90)
        out[f"b_ncc{halfwidth}_bad_frac"] = (
            float(np.mean(corr[np.isfinite(corr)] < 0.0))
            if np.isfinite(corr).any()
            else float("nan")
        )
        ncc_means.append(out[f"b_ncc{halfwidth}_mean"])
        bad_fracs.append(out[f"b_ncc{halfwidth}_bad_frac"])
    out["b_ncc_multiscale_mean"] = _safe_mean(np.asarray(ncc_means, dtype=float))
    out["b_ncc_bad_frac"] = _safe_mean(np.asarray(bad_fracs, dtype=float))
    out.update(geology_metrics(typewell, path))
    return out


def _candidate_context_stats(group: pd.DataFrame, candidate: str) -> dict[str, float]:
    surface_cols = [c for c in group.columns if c.startswith("S_hat_") and c.endswith("_std")]
    residual_cols = [
        c for c in group.columns if c.startswith("S_hat_") and c.endswith("_plane_residual")
    ]
    surface_std = _safe_mean(group[surface_cols].to_numpy(float).ravel()) if surface_cols else float("nan")
    residual = _safe_mean(group[residual_cols].to_numpy(float).ravel()) if residual_cols else float("nan")
    roughness_col = f"roughness__{candidate}"
    roughness = (
        float(group[roughness_col].iloc[0])
        if roughness_col in group.columns and np.isfinite(float(group[roughness_col].iloc[0]))
        else _path_roughness(_numeric(group, candidate))
    )
    return {
        "surface_std": surface_std,
        "plane_fit_residual": residual,
        "roughness": roughness,
    }


def _path_roughness(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 3:
        return float("nan")
    second = np.diff(finite, n=2)
    return float(np.sqrt(np.mean(second * second))) if len(second) else 0.0


def hidden_rmse_by_candidate(group: pd.DataFrame, candidate: str) -> float:
    return _rmse(_numeric(group, candidate), _numeric(group, "TVT"))


def compute_raw_scores(
    frame: pd.DataFrame,
    *,
    data_dir: Path,
    train_dir: Path | None,
    progress_interval: int = 25,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    actual_train_dir = train_dir or data_dir / "train"
    candidates = [column for column in CANDIDATE_COLUMNS if column in frame.columns]
    rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    gr_finite: list[float] = []
    tw_finite: list[float] = []

    grouped = list(frame.groupby("well_id", sort=False))
    total_wells = len(grouped)
    started = pd.Timestamp.now()
    for well_num, (well, group) in enumerate(grouped, start=1):
        if progress_interval > 0 and (
            well_num == 1 or well_num % int(progress_interval) == 0 or well_num == total_wells
        ):
            elapsed = (pd.Timestamp.now() - started).total_seconds()
            rate = well_num / max(elapsed, 1e-9)
            remaining = (total_wells - well_num) / max(rate, 1e-9)
            print(
                "b-lite raw scores "
                f"well={well_num}/{total_wells} "
                f"name={well} "
                f"rows={len(group)} "
                f"elapsed={elapsed:.1f}s "
                f"eta={remaining:.1f}s",
                flush=True,
            )
        horizontal_path = _horizontal_path(actual_train_dir, str(well))
        horizontal = pd.read_csv(horizontal_path) if horizontal_path.exists() else pd.DataFrame()
        typewell = read_typewell(_find_typewell_path(data_dir, actual_train_dir, str(well)))
        visible = visible_shift_metrics(horizontal, typewell)
        row_idx = _numeric(group, "row_idx").astype(int)
        gr = _numeric(group, "GR")
        true_tvt = _numeric(group, "TVT")
        linear_hidden = (
            _linear_anchor_extrapolation(horizontal, row_idx)
            if not horizontal.empty
            else np.full(len(group), np.nan, dtype=float)
        )
        gr_finite.append(float(np.isfinite(gr).mean()) if len(gr) else 0.0)
        if typewell is not None:
            tw_finite.append(float(np.isfinite(interp_typewell_gr(typewell, true_tvt)).mean()))

        for candidate in candidates:
            path = _numeric(group, candidate)
            metrics = path_gr_metrics(path=path, horizontal_gr=gr, typewell=typewell)
            offset = _safe_median(path - linear_hidden)
            metrics.update(
                {
                    "b_visible_shift_gap": (
                        abs(offset - visible["best_visible_shift"])
                        if np.isfinite(offset) and np.isfinite(visible["best_visible_shift"])
                        else float("nan")
                    ),
                    **visible,
                    **_candidate_context_stats(group, candidate),
                }
            )
            rows.append(
                {
                    "well_id": well,
                    "candidate_name": candidate,
                    "diagnostic": False,
                    "hidden_rmse": hidden_rmse_by_candidate(group, candidate),
                    **metrics,
                }
            )

        diagnostic_paths = {
            "true_tvt_path": true_tvt,
            "random_shift_path_+40": true_tvt + 40.0,
            "random_shift_path_-40": true_tvt - 40.0,
        }
        if "schema10_oof_raw" in group.columns and np.isfinite(_numeric(group, "schema10_oof_raw")).any():
            diagnostic_paths["schema10_path"] = _numeric(group, "schema10_oof_raw")
        for name, path in diagnostic_paths.items():
            metrics = path_gr_metrics(path=path, horizontal_gr=gr, typewell=typewell)
            offset = _safe_median(path - linear_hidden)
            metrics.update(
                {
                    "b_visible_shift_gap": (
                        abs(offset - visible["best_visible_shift"])
                        if np.isfinite(offset) and np.isfinite(visible["best_visible_shift"])
                        else float("nan")
                    ),
                    **visible,
                    "surface_std": float("nan"),
                    "plane_fit_residual": float("nan"),
                    "roughness": _path_roughness(path),
                }
            )
            diag_rows.append(
                {
                    "well_id": well,
                    "candidate_name": name,
                    "diagnostic": True,
                    "hidden_rmse": _rmse(path, true_tvt),
                    **metrics,
                }
            )

    info = {
        "a_candidates_count": int(len(candidates)),
        "gr_finite_fraction": float(np.nanmean(gr_finite)) if gr_finite else float("nan"),
        "typewell_interpolation_finite_fraction": (
            float(np.nanmean(tw_finite)) if tw_finite else float("nan")
        ),
    }
    return pd.DataFrame(rows), pd.DataFrame(diag_rows), info


def _normalize_cost(values: pd.Series) -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return np.zeros(len(arr), dtype=float)
    med = float(np.nanmedian(finite))
    q25, q75 = np.nanpercentile(finite, [25, 75])
    scale = float(q75 - q25)
    if not np.isfinite(scale) or scale < 1e-9:
        mad = float(np.nanmedian(np.abs(finite - med)))
        scale = mad if mad > 1e-9 else 1.0
    norm = (arr - med) / scale
    finite_norm = norm[np.isfinite(norm)]
    missing = float(np.nanmax(finite_norm) + 5.0) if len(finite_norm) else 5.0
    return np.where(np.isfinite(norm), norm, missing)


def add_b_scores(scores: pd.DataFrame) -> pd.DataFrame:
    out = scores.copy()
    out["cost_path_corr"] = -_numeric(out, "b_path_corr")
    out["cost_dgr_corr"] = -_numeric(out, "b_dgr_corr")
    out["cost_ncc15_mean"] = -_numeric(out, "b_ncc15_mean")
    out["cost_ncc25_mean"] = -_numeric(out, "b_ncc25_mean")
    out["cost_path_mad"] = _numeric(out, "b_path_mad")
    out["cost_ncc_bad_frac"] = _numeric(out, "b_ncc_bad_frac")
    out["cost_visible_shift_gap"] = _numeric(out, "b_visible_shift_gap")
    out["cost_geology_switch_count"] = _numeric(out, "geology_switch_count")
    out["cost_surface_std"] = _numeric(out, "surface_std")
    out["cost_roughness"] = _numeric(out, "roughness")
    for name in set(B_SCORE_WEIGHTS) | set(GR_ONLY_WEIGHTS):
        out[f"norm_{name}"] = (
            out.groupby("well_id", group_keys=False)[name]
            .transform(lambda values: pd.Series(_normalize_cost(values), index=values.index))
            .astype(float)
        )

    def weighted_score(weights: dict[str, float]) -> np.ndarray:
        result = np.zeros(len(out), dtype=float)
        for name, weight in weights.items():
            result += float(weight) * out[f"norm_{name}"].to_numpy(dtype=float)
        return result

    no_surface = {
        key: value
        for key, value in B_SCORE_WEIGHTS.items()
        if key not in {"cost_surface_std", "cost_roughness"}
    }
    out["b_combined_score"] = weighted_score(B_SCORE_WEIGHTS)
    out["b_combined_without_surface_terms"] = weighted_score(no_surface)
    out["b_combined_gr_only"] = weighted_score(GR_ONLY_WEIGHTS)
    out = out.sort_values(["well_id", "b_combined_score", "candidate_name"]).reset_index(drop=True)
    out["b_rank"] = out.groupby("well_id").cumcount() + 1
    gap: dict[Any, float] = {}
    for well, group in out.groupby("well_id", sort=False):
        vals = group["b_combined_score"].to_numpy(dtype=float)
        gap[well] = float(vals[1] - vals[0]) if len(vals) > 1 else float("nan")
    out["b_score_gap"] = out["well_id"].map(gap).astype(float)
    return out


def _select_by_metric(scores: pd.DataFrame, metric: str, *, ascending: bool) -> pd.DataFrame:
    rows = []
    for _well, group in scores.groupby("well_id", sort=False):
        vals = pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=float)
        penalty = np.where(np.isfinite(vals), vals, np.inf if ascending else -np.inf)
        order = np.lexsort((group["candidate_name"].astype(str).to_numpy(), penalty))
        chosen_pos = int(order[0] if ascending else order[-1])
        rows.append(group.iloc[chosen_pos])
    return pd.DataFrame(rows).reset_index(drop=True)


def _weighted_nanmean(matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    finite = np.isfinite(matrix)
    weighted = np.where(finite, matrix * weights[None, :], 0.0)
    denom = np.where(finite, weights[None, :], 0.0).sum(axis=1)
    return np.divide(weighted.sum(axis=1), denom, out=np.full(matrix.shape[0], np.nan), where=denom > 0)


def _prediction_from_candidate_map(
    frame: pd.DataFrame, candidate_by_well: dict[Any, str]
) -> np.ndarray:
    row_candidates = frame["well_id"].map(candidate_by_well).to_numpy(dtype=object)
    out = np.full(len(frame), np.nan, dtype=float)
    for candidate in sorted({str(item) for item in row_candidates if item == item}):
        if candidate not in frame.columns:
            continue
        mask = row_candidates == candidate
        values = frame[candidate].to_numpy(dtype=float)
        out[mask] = values[mask]
    return out


def _prediction_from_weight_map(
    frame: pd.DataFrame, weights_by_well: dict[Any, dict[str, float]]
) -> np.ndarray:
    out = np.zeros(len(frame), dtype=float)
    denom = np.zeros(len(frame), dtype=float)
    wells = frame["well_id"]
    candidates = sorted({candidate for weights in weights_by_well.values() for candidate in weights})
    for candidate in candidates:
        if candidate not in frame.columns:
            continue
        per_well = {
            well: float(weights[candidate])
            for well, weights in weights_by_well.items()
            if candidate in weights and float(weights[candidate]) > 0.0
        }
        if not per_well:
            continue
        row_weights = wells.map(per_well).fillna(0.0).to_numpy(dtype=float)
        values = frame[candidate].to_numpy(dtype=float)
        mask = (row_weights > 0.0) & np.isfinite(values)
        out[mask] += values[mask] * row_weights[mask]
        denom[mask] += row_weights[mask]
    return np.divide(out, denom, out=np.full(len(frame), np.nan), where=denom > 0)


def build_selector_predictions(
    frame: pd.DataFrame,
    scores: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    base_cols = [
        col
        for col in ("id", "well_id", "fold", "row_idx", "TVT", "GR", "hidden_frac", "hidden_rows", "schema10_oof_raw")
        if col in frame.columns
    ]
    pred = frame[base_cols].copy()
    choices: list[dict[str, Any]] = []
    selector_names: list[str] = []
    selectors = {
        "B_path_corr_max": ("b_path_corr", False),
        "B_dgr_corr_max": ("b_dgr_corr", False),
        "B_ncc15_mean_max": ("b_ncc15_mean", False),
        "B_ncc_multiscale_mean_max": ("b_ncc_multiscale_mean", False),
        "B_combined_score_min": ("b_combined_score", True),
        "B_combined_without_surface_terms": ("b_combined_without_surface_terms", True),
        "B_combined_GR_only": ("b_combined_gr_only", True),
    }
    for name, (metric, ascending) in selectors.items():
        selected = _select_by_metric(scores, metric, ascending=ascending)
        pred[name] = _prediction_from_candidate_map(
            frame,
            {
                row.well_id: str(row.candidate_name)
                for row in selected.itertuples(index=False)
            },
        )
        selector_names.append(name)
        for row in selected.itertuples(index=False):
            choices.append(
                {
                    "selector_name": name,
                    "well_id": row.well_id,
                    "selected_candidate": row.candidate_name,
                    "blend_candidates": row.candidate_name,
                    "blend_weights": "1.0",
                    "selector_metric": metric,
                    "selector_metric_value": getattr(row, metric),
                    "b_combined_score": row.b_combined_score,
                    "b_score_gap": row.b_score_gap,
                }
            )

    for top_k in (3, 5):
        for temp in SOFT_TEMPERATURES:
            name = f"B_top{top_k}_soft_t{temp:g}"
            selector_names.append(name)
            weights_by_well: dict[Any, dict[str, float]] = {}
            for well, group in scores.groupby("well_id", sort=False):
                ranked = group.sort_values(["b_combined_score", "candidate_name"]).head(top_k)
                raw = ranked["b_combined_score"].to_numpy(dtype=float)
                finite = raw[np.isfinite(raw)]
                fill = float(np.nanmax(finite)) if len(finite) else 0.0
                safe = np.where(np.isfinite(raw), raw, fill + 5.0)
                weights = np.exp(-(safe - float(np.nanmin(safe))) / max(float(temp), 1e-6))
                weights = weights / max(float(weights.sum()), 1e-12)
                candidates = ranked["candidate_name"].astype(str).tolist()
                weights_by_well[well] = {
                    candidate: float(weight)
                    for candidate, weight in zip(candidates, weights, strict=False)
                }
                top = ranked.iloc[0]
                choices.append(
                    {
                        "selector_name": name,
                        "well_id": well,
                        "selected_candidate": str(top["candidate_name"]),
                        "blend_candidates": ",".join(candidates),
                        "blend_weights": json.dumps([float(w) for w in weights]),
                        "selector_metric": "b_combined_score_softblend",
                        "selector_metric_value": float(top["b_combined_score"]),
                        "b_combined_score": float(top["b_combined_score"]),
                        "b_score_gap": float(top["b_score_gap"]),
                    }
                )
            pred[name] = _prediction_from_weight_map(frame, weights_by_well)
    return pred, pd.DataFrame(choices), selector_names


def score_selectors(predictions: pd.DataFrame, selector_names: list[str]) -> pd.DataFrame:
    ctx = make_score_context(predictions)
    rows = [
        score_prediction(predictions, predictions[name].to_numpy(float), name, ctx)
        for name in selector_names
    ]
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)


def score_safe_blends(predictions: pd.DataFrame) -> pd.DataFrame:
    if "schema10_oof_raw" not in predictions.columns:
        return pd.DataFrame()
    schema = _numeric(predictions, "schema10_oof_raw")
    if not np.isfinite(schema).any() or "B_combined_score_min" not in predictions.columns:
        return pd.DataFrame()
    selected = _numeric(predictions, "B_combined_score_min")
    ctx = make_score_context(predictions)
    rows = []
    for alpha in SAFE_ALPHAS:
        for clip in SAFE_CLIPS:
            delta = selected - schema
            delta = np.where(
                np.isfinite(delta),
                np.clip(delta, -float(clip), float(clip)),
                0.0,
            )
            blended = schema + float(alpha) * delta
            row = score_prediction(
                predictions,
                blended,
                f"schema10_safe_blend_B_combined_a{alpha:g}_clip{clip:g}",
                ctx,
            )
            shift = np.abs(blended - schema)
            row.update(
                {
                    "alpha": float(alpha),
                    "clip": float(clip),
                    "median_shift_vs_schema10": _safe_median(shift),
                    "p95_shift_vs_schema10": _safe_percentile(shift, 95),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)


def rank_diagnostics(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for well, group in scores.groupby("well_id", sort=False):
        corr = pd.Series(group["b_combined_score"]).corr(
            pd.Series(group["hidden_rmse"]), method="spearman"
        )
        order = group.sort_values(["hidden_rmse", "candidate_name"]).reset_index(drop=True)
        oracle_best = str(order.iloc[0]["candidate_name"])
        b_order = group.sort_values(["b_combined_score", "candidate_name"]).reset_index(drop=True)
        ranks = {name: idx + 1 for idx, name in enumerate(b_order["candidate_name"].astype(str))}
        oracle_rank = int(ranks.get(oracle_best, len(group) + 1))
        rows.append(
            {
                "well_id": well,
                "spearman_b_score_hidden_rmse": float(corr) if np.isfinite(corr) else float("nan"),
                "oracle_best_candidate": oracle_best,
                "oracle_best_rank_by_b_score": oracle_rank,
                "oracle_best_hidden_rmse": float(order.iloc[0]["hidden_rmse"]),
                "b_best_candidate": str(b_order.iloc[0]["candidate_name"]),
                "b_best_hidden_rmse": float(b_order.iloc[0]["hidden_rmse"]),
            }
        )
    rank_frame = pd.DataFrame(rows)
    finite = rank_frame["spearman_b_score_hidden_rmse"].to_numpy(float)
    summary = pd.DataFrame(
        [
            {
                "median_spearman": _safe_median(finite),
                "mean_spearman": _safe_mean(finite),
                "share_positive_spearman": float(np.nanmean(finite > 0.0)),
                "share_spearman_gt_0_3": float(np.nanmean(finite > 0.3)),
                "oracle_top1_rate": float(np.mean(rank_frame["oracle_best_rank_by_b_score"] <= 1)),
                "oracle_top3_rate": float(np.mean(rank_frame["oracle_best_rank_by_b_score"] <= 3)),
                "oracle_top5_rate": float(np.mean(rank_frame["oracle_best_rank_by_b_score"] <= 5)),
                "oracle_top10_rate": float(np.mean(rank_frame["oracle_best_rank_by_b_score"] <= 10)),
            }
        ]
    )
    return rank_frame, summary


def b0_sanity(scores: pd.DataFrame, diagnostics: pd.DataFrame) -> pd.DataFrame:
    combined = add_b_scores(pd.concat([scores, diagnostics], ignore_index=True))
    rows = []
    for well, group in combined.groupby("well_id", sort=False):
        sanity_score = "b_combined_without_surface_terms"
        ordered = group.sort_values([sanity_score, "candidate_name"]).reset_index(drop=True)
        ranks = {name: idx + 1 for idx, name in enumerate(ordered["candidate_name"].astype(str))}
        true_score = group.loc[group["candidate_name"] == "true_tvt_path", sanity_score]
        plus_score = group.loc[group["candidate_name"] == "random_shift_path_+40", sanity_score]
        minus_score = group.loc[group["candidate_name"] == "random_shift_path_-40", sanity_score]
        a_only = group[~group["diagnostic"].astype(bool)]
        oracle_best = str(a_only.sort_values(["hidden_rmse", "candidate_name"]).iloc[0]["candidate_name"])
        rows.append(
            {
                "well_id": well,
                "true_path_rank": int(ranks.get("true_tvt_path", len(group) + 1)),
                "true_better_than_plus40": bool(
                    len(true_score)
                    and len(plus_score)
                    and float(true_score.iloc[0]) < float(plus_score.iloc[0])
                ),
                "true_better_than_minus40": bool(
                    len(true_score)
                    and len(minus_score)
                    and float(true_score.iloc[0]) < float(minus_score.iloc[0])
                ),
                "oracle_best_a_candidate": oracle_best,
                "oracle_best_a_rank_by_b_score": int(ranks.get(oracle_best, len(group) + 1)),
            }
        )
    return pd.DataFrame(rows)


def b0_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "true_path_mean_rank": float(np.nanmean(frame["true_path_rank"])),
                "true_path_median_rank": _safe_median(frame["true_path_rank"].to_numpy(float)),
                "true_path_top1_rate": float(np.mean(frame["true_path_rank"] <= 1)),
                "true_path_top3_rate": float(np.mean(frame["true_path_rank"] <= 3)),
                "true_path_top5_rate": float(np.mean(frame["true_path_rank"] <= 5)),
                "true_better_than_plus40_rate": float(np.mean(frame["true_better_than_plus40"])),
                "true_better_than_minus40_rate": float(np.mean(frame["true_better_than_minus40"])),
                "oracle_best_a_top1_rate": float(np.mean(frame["oracle_best_a_rank_by_b_score"] <= 1)),
                "oracle_best_a_top3_rate": float(np.mean(frame["oracle_best_a_rank_by_b_score"] <= 3)),
                "oracle_best_a_top5_rate": float(np.mean(frame["oracle_best_a_rank_by_b_score"] <= 5)),
                "oracle_best_a_top10_rate": float(np.mean(frame["oracle_best_a_rank_by_b_score"] <= 10)),
            }
        ]
    )


def bad_wells_report(
    predictions: pd.DataFrame,
    selector_names: list[str],
    rank_frame: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for well, group in predictions.groupby("well_id", sort=False):
        y = _numeric(group, "TVT")
        selector_rmse = {
            name: _rmse(_numeric(group, name), y)
            for name in selector_names
            if name in group.columns
        }
        best_selector = min(
            selector_rmse,
            key=lambda name: np.inf if not np.isfinite(selector_rmse[name]) else selector_rmse[name],
        )
        b_rmse = selector_rmse.get("B_combined_score_min", float("nan"))
        rank_row = rank_frame.loc[rank_frame["well_id"] == well]
        oracle_rmse = float(rank_row["oracle_best_hidden_rmse"].iloc[0]) if len(rank_row) else float("nan")
        oracle_rank = int(rank_row["oracle_best_rank_by_b_score"].iloc[0]) if len(rank_row) else -1
        schema_rmse = _rmse(_numeric(group, "schema10_oof_raw"), y)
        rows.append(
            {
                "well_id": well,
                "schema10_rmse": schema_rmse,
                "a_oracle_rmse": oracle_rmse,
                "a_oracle_rank_by_b": oracle_rank,
                "best_b_selector": best_selector,
                "best_b_selector_rmse": selector_rmse[best_selector],
                "b_combined_rmse": b_rmse,
                "schema10_bad_b_improves": bool(
                    np.isfinite(schema_rmse) and np.isfinite(b_rmse) and schema_rmse > 20.0 and b_rmse + 5.0 < schema_rmse
                ),
                "b_catastrophic": bool(
                    np.isfinite(b_rmse)
                    and ((np.isfinite(schema_rmse) and b_rmse > schema_rmse + 10.0) or b_rmse > 30.0)
                ),
                "oracle_good_b_missed": bool(np.isfinite(oracle_rmse) and oracle_rmse < 10.0 and oracle_rank > 10),
                "gr_nan_failure": bool(
                    "GR" in group.columns
                    and not np.isfinite(_numeric(group, "GR")).any()
                    and np.isfinite(b_rmse)
                    and b_rmse > 20.0
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["b_catastrophic", "oracle_good_b_missed", "b_combined_rmse"],
        ascending=[False, False, False],
    )


def write_report(
    *,
    output_dir: Path,
    info: dict[str, Any],
    rows: int,
    wells: int,
    b0: pd.DataFrame,
    rank_summary: pd.DataFrame,
    selector_scores: pd.DataFrame,
    safe_scores: pd.DataFrame,
    oracle_frame: pd.DataFrame,
    bad_wells: pd.DataFrame,
) -> None:
    best = selector_scores.iloc[0].to_dict() if not selector_scores.empty else {}
    oracle_whole = oracle_frame.loc[oracle_frame["oracle"] == "whole_well_oracle", "rmse"]
    oracle_thirds = oracle_frame.loc[oracle_frame["oracle"] == "thirds_segment_oracle", "rmse"]
    whole = float(oracle_whole.iloc[0]) if len(oracle_whole) else float("nan")
    thirds = float(oracle_thirds.iloc[0]) if len(oracle_thirds) else float("nan")
    b_rmse = float(best.get("rmse", np.nan))
    lines = [
        "# B Lite GR/NCC Scorer Report",
        "## Inputs",
        f"A candidates count: `{info.get('a_candidates_count')}`",
        f"Rows: `{rows}`",
        f"Wells: `{wells}`",
        f"GR finite fraction: `{info.get('gr_finite_fraction')}`",
        f"Typewell interpolation finite fraction: `{info.get('typewell_interpolation_finite_fraction')}`",
        "## B0 Sanity",
        markdown_table(b0),
        "## Rank Correlation",
        markdown_table(rank_summary),
        "## Selectors",
        markdown_table(selector_scores),
        "## Safe Blends",
        markdown_table(safe_scores) if not safe_scores.empty else "_schema10 unavailable_",
        "## Oracle Comparison",
        f"A whole oracle: `{whole}`",
        f"A thirds oracle: `{thirds}`",
        f"Best B selector: `{best.get('selector_name', '')}` RMSE `{b_rmse}`",
        f"Regret vs whole: `{b_rmse - whole if np.isfinite(b_rmse) and np.isfinite(whole) else np.nan}`",
        f"Regret vs thirds: `{b_rmse - thirds if np.isfinite(b_rmse) and np.isfinite(thirds) else np.nan}`",
        "## Bad Wells",
        markdown_table(bad_wells),
        "## Decision",
        "_Use GO/NO-GO gates from the experiment brief._",
    ]
    (output_dir / "B_LITE_REPORT.md").write_text("\n\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = _read_frame(Path(args.input))
    frame, schema_available = attach_schema10(
        frame,
        Path(args.schema10_oof) if args.schema10_oof else None,
        schema10_column=args.schema10_column,
    )
    raw_scores, diagnostic_scores, info = compute_raw_scores(
        frame,
        data_dir=Path(args.data_dir),
        train_dir=Path(args.train_dir) if args.train_dir else None,
        progress_interval=int(args.progress_interval),
    )
    print("b-lite raw scores complete", flush=True)
    scores = add_b_scores(raw_scores)
    print("b-lite selectors scoring", flush=True)
    predictions, choices, selector_names = build_selector_predictions(frame, scores)
    selector_scores = score_selectors(predictions, selector_names)
    safe_scores = score_safe_blends(predictions)
    oracle_frame, oracle_winners = oracle_scores(frame)
    rank_frame, rank_summary = rank_diagnostics(scores)
    b0_frame = b0_sanity(scores, diagnostic_scores)
    b0 = b0_summary(b0_frame)
    bad_wells = bad_wells_report(predictions, selector_names, rank_frame)

    write_frame(scores, output_dir / "b_candidate_scores.parquet")
    write_frame(diagnostic_scores, output_dir / "b_diagnostic_raw_scores.parquet")
    write_frame(predictions, output_dir / "b_selector_predictions.parquet")
    write_frame(choices, output_dir / "b_selector_choices.parquet")
    rank_frame.to_csv(output_dir / "b_rank_diagnostics.csv", index=False)
    rank_summary.to_csv(output_dir / "b_rank_summary.csv", index=False)
    b0_frame.to_csv(output_dir / "b0_true_path_sanity_by_well.csv", index=False)
    b0.to_csv(output_dir / "b0_true_path_sanity.csv", index=False)
    selector_scores.to_csv(output_dir / "b_selector_scores.csv", index=False)
    safe_scores.to_csv(output_dir / "b_safe_blend_scores.csv", index=False)
    oracle_frame.to_csv(output_dir / "b_oracle_scores.csv", index=False)
    oracle_winners.to_csv(output_dir / "b_oracle_winners.csv", index=False)
    bad_wells.to_csv(output_dir / "b_bad_wells.csv", index=False)
    write_report(
        output_dir=output_dir,
        info=info,
        rows=len(frame),
        wells=int(frame["well_id"].nunique()),
        b0=b0,
        rank_summary=rank_summary,
        selector_scores=selector_scores,
        safe_scores=safe_scores,
        oracle_frame=oracle_frame,
        bad_wells=bad_wells,
    )
    metrics = {
        "input": str(args.input),
        "rows": int(len(frame)),
        "wells": int(frame["well_id"].nunique()),
        "schema10_available": bool(schema_available),
        "inputs": info,
        "b0": b0.iloc[0].to_dict() if not b0.empty else {},
        "rank_summary": rank_summary.iloc[0].to_dict() if not rank_summary.empty else {},
        "best_selector": selector_scores.iloc[0].to_dict() if not selector_scores.empty else {},
        "best_safe_blend": safe_scores.iloc[0].to_dict() if not safe_scores.empty else {},
        "oracles": oracle_frame.to_dict("records"),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(json_safe(metrics), file, indent=2)
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B-lite GR/NCC scorer for FormationPlaneKNN A candidates.")
    parser.add_argument("--input", type=Path, default=Path("artifacts/formation_plane_knn/oof_candidates.parquet"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--train-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/formation_b_lite"))
    parser.add_argument("--schema10-oof", type=Path, default=None)
    parser.add_argument("--schema10-column", type=str, default=None)
    parser.add_argument("--progress-interval", type=int, default=25)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    metrics = run(parse_args(argv))
    print(json.dumps(json_safe(metrics), indent=2), flush=True)


if __name__ == "__main__":
    main()
