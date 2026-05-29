from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import VALID_COORD_SOURCES

FORBIDDEN_INFERENCE_COLUMNS = {
    "TVT",
    "true_TVT",
    "Geology",
    "ANCC",
    "ASTNU",
    "ASTNL",
    "EGFDU",
    "EGFDL",
    "BUDA",
}


@dataclass(frozen=True)
class WellFiles:
    well_id: str
    horizontal_path: Path
    typewell_path: Path


def assert_schema_safe_columns(columns: Iterable[str]) -> None:
    forbidden = FORBIDDEN_INFERENCE_COLUMNS.intersection(set(columns))
    if forbidden:
        raise ValueError(f"template-envelope inference columns contain forbidden columns: {sorted(forbidden)}")


def _well_id_from_horizontal(path: Path) -> str:
    suffix = "__horizontal_well.csv"
    if not path.name.endswith(suffix):
        raise ValueError(f"Not a horizontal well file: {path}")
    return path.name[: -len(suffix)]


def discover_wells(data_dir: Path, *, k_wells: int = -1) -> list[WellFiles]:
    horizontal_paths = sorted(Path(data_dir).glob("*__horizontal_well.csv"), key=_well_id_from_horizontal)
    if k_wells > 0:
        horizontal_paths = horizontal_paths[:k_wells]
    wells: list[WellFiles] = []
    for horizontal_path in horizontal_paths:
        well_id = _well_id_from_horizontal(horizontal_path)
        typewell_path = horizontal_path.with_name(f"{well_id}__typewell.csv")
        if typewell_path.exists():
            wells.append(WellFiles(well_id, horizontal_path, typewell_path))
    if not wells:
        raise FileNotFoundError(f"No paired horizontal/typewell CSVs found in {data_dir}")
    return wells


def load_well_pair(well: WellFiles) -> tuple[pd.DataFrame, pd.DataFrame]:
    horizontal = pd.read_csv(well.horizontal_path)
    typewell = pd.read_csv(well.typewell_path)
    horizontal["well_id"] = well.well_id
    if "row_idx" not in horizontal.columns:
        horizontal["row_idx"] = np.arange(len(horizontal), dtype=np.int64)
    if "id" not in horizontal.columns:
        horizontal["id"] = [f"{well.well_id}_{idx}" for idx in horizontal["row_idx"]]
    for column in ("MD", "X", "Y", "Z", "GR", "TVT_input"):
        if column not in horizontal.columns:
            horizontal[column] = np.nan
    if "TVT" not in typewell.columns or "GR" not in typewell.columns:
        raise ValueError(f"{well.typewell_path} must contain TVT and GR")
    return horizontal, typewell


def _finite_interp(values: np.ndarray) -> np.ndarray:
    out = values.astype(np.float64, copy=True)
    finite = np.isfinite(out)
    if finite.all():
        return out
    if not finite.any():
        return np.zeros_like(out, dtype=np.float64)
    x = np.arange(out.size, dtype=np.float64)
    out[~finite] = np.interp(x[~finite], x[finite], out[finite])
    return out


def compress_horizontal(
    horizontal: pd.DataFrame,
    *,
    rows_per_step: int,
    coord_source: str = "bridge_slope",
) -> pd.DataFrame:
    rows_per_step = max(int(rows_per_step), 1)
    records: list[dict[str, float | int | str | bool]] = []
    ordered = horizontal.sort_values("row_idx").reset_index(drop=True)
    for step, start in enumerate(range(0, len(ordered), rows_per_step)):
        chunk = ordered.iloc[start : start + rows_per_step]
        tvt_input = pd.to_numeric(chunk["TVT_input"], errors="coerce")
        record: dict[str, float | int | str | bool] = {
            "well_id": str(chunk["well_id"].iloc[0]),
            "step": int(step),
            "row_start": int(pd.to_numeric(chunk["row_idx"], errors="coerce").min()),
            "row_end": int(pd.to_numeric(chunk["row_idx"], errors="coerce").max()),
            "row_count": int(len(chunk)),
            "known_frac": float(tvt_input.notna().mean()),
            "known_tvt": float(tvt_input.mean()) if tvt_input.notna().any() else np.nan,
            "hidden_mask": bool(tvt_input.isna().mean() > 0.5),
        }
        for column in ("MD", "X", "Y", "Z", "GR"):
            values = pd.to_numeric(chunk[column], errors="coerce")
            record[column] = float(values.mean()) if values.notna().any() else np.nan
        if "TVT" in chunk.columns:
            true_tvt = pd.to_numeric(chunk["TVT"], errors="coerce")
            record["true_tvt"] = float(true_tvt.mean()) if true_tvt.notna().any() else np.nan
        records.append(record)
    comp = pd.DataFrame(records)
    comp["GR_filled"] = _finite_interp(pd.to_numeric(comp["GR"], errors="coerce").to_numpy(dtype=np.float64))
    comp["bridge_TVT"] = build_known_tvt_bridge(comp)
    comp["coord"] = build_coord_proxy(comp, source=coord_source)
    comp.attrs["coord_source"] = coord_source
    return comp


def build_known_tvt_bridge(compressed: pd.DataFrame) -> np.ndarray:
    """Linear interp between known steps; clip-extrapolate beyond endpoints.

    Kept for backward compatibility / `coord_source='bridge'`. Note: this
    collapses to a constant on heel-only-known wells (typical ROGII layout).
    Prefer `build_coord_proxy(..., source='bridge_slope')` for the template
    envelope coordinate.
    """
    known = pd.to_numeric(compressed["known_tvt"], errors="coerce").to_numpy(dtype=np.float64)
    steps = pd.to_numeric(compressed["step"], errors="coerce").to_numpy(dtype=np.float64)
    finite = np.isfinite(known)
    if finite.sum() >= 2:
        return np.interp(steps, steps[finite], known[finite])
    if finite.sum() == 1:
        return np.full_like(steps, known[finite][0], dtype=np.float64)
    z = pd.to_numeric(compressed.get("Z", pd.Series(np.nan, index=compressed.index)), errors="coerce").to_numpy(dtype=np.float64)
    finite_z = np.isfinite(z)
    if finite_z.any():
        return np.full_like(steps, float(np.nanmean(np.abs(z[finite_z]))), dtype=np.float64)
    return np.zeros_like(steps, dtype=np.float64)


def build_coord_proxy(compressed: pd.DataFrame, *, source: str) -> np.ndarray:
    """Build a per-step coordinate proxy used as `tvt` in the template path.

    The envelope template is `path_tvt = a * coord + offset` (centered).
    `coord` must vary across the well, including the hidden region. In real
    ROGII wells `TVT_input` is heel-only, so the legacy `bridge` proxy is
    constant on the hidden region and yields degenerate (constant) templates.

    Supported sources:
        - `bridge_slope`: linear least-squares fit through known TVT_input
          steps, evaluated at every step. Test-safe (uses only TVT_input,
          never hidden TVT). Default.
        - `bridge`: legacy linear interp + clip-extrapolate (kept for ablation).
        - `z_aligned`: `-Z` shifted so its mean matches mean known TVT_input.
          Test-safe geometric proxy (Z is observed everywhere).
        - `md_aligned`: MD shifted to match mean known TVT_input. Last resort.
    """
    if source not in VALID_COORD_SOURCES:
        raise ValueError(f"Unknown coord_source={source!r}; allowed: {VALID_COORD_SOURCES}")

    steps = pd.to_numeric(compressed["step"], errors="coerce").to_numpy(dtype=np.float64)
    known = pd.to_numeric(compressed["known_tvt"], errors="coerce").to_numpy(dtype=np.float64)
    finite_known = np.isfinite(known)
    mean_known = float(np.nanmean(known[finite_known])) if finite_known.any() else 0.0

    if source == "bridge":
        return build_known_tvt_bridge(compressed)

    if source == "bridge_slope":
        if finite_known.sum() >= 2:
            slope, intercept = np.polyfit(steps[finite_known], known[finite_known], 1)
            return slope * steps + intercept
        if finite_known.sum() == 1:
            # Single anchor — fall back to z_aligned-style geometric proxy
            # rather than a constant.
            z = pd.to_numeric(
                compressed.get("Z", pd.Series(np.nan, index=compressed.index)),
                errors="coerce",
            ).to_numpy(dtype=np.float64)
            if np.isfinite(z).sum() >= 2:
                neg_z = -z
                shift = float(np.nanmean(known[finite_known])) - float(np.nanmean(neg_z[np.isfinite(neg_z)]))
                return neg_z + shift
            return np.full_like(steps, known[finite_known][0], dtype=np.float64)
        return np.zeros_like(steps, dtype=np.float64)

    if source == "bridge_slope_z_capped":
        # Bridge slope only while Z is still changing (kick-off / building
        # section). Once Z plateaus (horizontal section), freeze the coord at
        # its last bridge value. Test-safe: uses only Z and TVT_input.
        z = pd.to_numeric(
            compressed.get("Z", pd.Series(np.nan, index=compressed.index)),
            errors="coerce",
        ).to_numpy(dtype=np.float64)
        if finite_known.sum() >= 2:
            slope, intercept = np.polyfit(steps[finite_known], known[finite_known], 1)
            raw = slope * steps + intercept
        elif finite_known.sum() == 1:
            raw = np.full_like(steps, known[finite_known][0], dtype=np.float64)
        else:
            raw = np.zeros_like(steps, dtype=np.float64)
        if np.isfinite(z).sum() < 3:
            return raw
        # dZ per step; absolute mean serves as a Z-activity scale.
        dz = np.gradient(np.where(np.isfinite(z), z, np.nan))
        # Z plateau detected when |dZ| < 10% of the heel-region mean |dZ|.
        last_known_step = int(np.max(steps[finite_known])) if finite_known.any() else 0
        heel_mask = steps <= max(last_known_step, 1)
        heel_dz = dz[heel_mask & np.isfinite(dz)]
        heel_scale = float(np.nanmean(np.abs(heel_dz))) if heel_dz.size else 0.0
        plateau_threshold = 0.1 * heel_scale if heel_scale > 0 else 0.0
        plateau = np.abs(dz) < max(plateau_threshold, 1e-6)
        # Cap: once we are past the last known step AND Z is on a plateau,
        # freeze coord at the value reached at plateau onset.
        out = raw.copy()
        if last_known_step > 0:
            past_known = steps > last_known_step
            plateau_in_hidden = plateau & past_known
            if plateau_in_hidden.any():
                first_plateau_idx = int(np.argmax(plateau_in_hidden))
                cap_value = float(raw[first_plateau_idx])
                out[first_plateau_idx:] = cap_value
        return out

    if source == "z_aligned":
        z = pd.to_numeric(
            compressed.get("Z", pd.Series(np.nan, index=compressed.index)),
            errors="coerce",
        ).to_numpy(dtype=np.float64)
        finite_z = np.isfinite(z)
        if not finite_z.any():
            return np.zeros_like(steps, dtype=np.float64)
        neg_z = -z
        shift = mean_known - float(np.nanmean(neg_z[finite_z])) if finite_known.any() else 0.0
        return neg_z + shift

    if source == "md_aligned":
        md = pd.to_numeric(
            compressed.get("MD", pd.Series(np.nan, index=compressed.index)),
            errors="coerce",
        ).to_numpy(dtype=np.float64)
        finite_md = np.isfinite(md)
        if not finite_md.any():
            return np.zeros_like(steps, dtype=np.float64)
        shift = mean_known - float(np.nanmean(md[finite_md])) if finite_known.any() else 0.0
        return md + shift

    return np.zeros_like(steps, dtype=np.float64)


def hidden_rows_with_steps(horizontal: pd.DataFrame, *, rows_per_step: int) -> pd.DataFrame:
    ordered = horizontal.copy()
    ordered["well_id"] = ordered["well_id"].astype(str)
    if "row_idx" not in ordered.columns:
        ordered["row_idx"] = np.arange(len(ordered), dtype=np.int64)
    if "id" not in ordered.columns:
        ordered["id"] = [f"{well_id}_{idx}" for well_id, idx in zip(ordered["well_id"], ordered["row_idx"], strict=False)]
    ordered["step"] = (pd.to_numeric(ordered["row_idx"], errors="coerce") // max(int(rows_per_step), 1)).astype(int)
    tvt_input = pd.to_numeric(ordered.get("TVT_input", pd.Series(np.nan, index=ordered.index)), errors="coerce")
    keep_cols = ["id", "well_id", "row_idx", "step"]
    if "TVT" in ordered.columns:
        keep_cols.append("TVT")
    return ordered.loc[tvt_input.isna(), keep_cols].copy()

