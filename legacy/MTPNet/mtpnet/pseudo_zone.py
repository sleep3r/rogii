from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DataConfig
from .correlation_panel import _compress_nanmean, _json_clean
from .io import discover_wells, load_well


@dataclass(frozen=True)
class PseudoZoneTemplate:
    labels_by_bin: np.ndarray
    n_bins: int

    def filled(self) -> "PseudoZoneTemplate":
        labels = self.labels_by_bin.astype(object).copy()
        known = np.flatnonzero(labels != "__unknown__")
        if known.size == 0:
            return self
        for idx, label in enumerate(labels):
            if label != "__unknown__":
                continue
            nearest = known[np.argmin(np.abs(known - idx))]
            labels[idx] = labels[nearest]
        return PseudoZoneTemplate(labels_by_bin=labels, n_bins=self.n_bins)


def _relative_tvt_bins(tvt: np.ndarray, n_bins: int) -> np.ndarray:
    finite = np.isfinite(tvt)
    if not finite.any():
        return np.zeros_like(tvt, dtype=np.int32)
    lo = float(np.nanmin(tvt[finite]))
    hi = float(np.nanmax(tvt[finite]))
    denom = max(hi - lo, 1e-6)
    rel = np.clip((tvt.astype(np.float32) - lo) / denom, 0.0, 1.0)
    return np.clip(np.floor(rel * n_bins), 0, n_bins - 1).astype(np.int32)


def fit_pseudo_zone_template(
    typewells: list[pd.DataFrame], *, n_bins: int = 128
) -> PseudoZoneTemplate:
    counts: list[dict[str, int]] = [dict() for _ in range(n_bins)]
    for typewell in typewells:
        if "Geology" not in typewell.columns:
            continue
        tvt = pd.to_numeric(typewell["TVT"], errors="coerce").to_numpy(dtype=np.float32)
        labels = typewell["Geology"].fillna("__unknown__").astype(str).to_numpy(dtype=object)
        bins = _relative_tvt_bins(tvt, n_bins)
        for bin_idx, label in zip(bins, labels):
            if str(label) == "__unknown__":
                continue
            bucket = counts[int(bin_idx)]
            bucket[str(label)] = bucket.get(str(label), 0) + 1
    labels_by_bin = np.full(n_bins, "__unknown__", dtype=object)
    for idx, bucket in enumerate(counts):
        if bucket:
            labels_by_bin[idx] = max(bucket, key=bucket.get)
    return PseudoZoneTemplate(labels_by_bin=labels_by_bin, n_bins=n_bins).filled()


def predict_typewell_geology(typewell: pd.DataFrame, model: PseudoZoneTemplate) -> np.ndarray:
    tvt = pd.to_numeric(typewell["TVT"], errors="coerce").to_numpy(dtype=np.float32)
    bins = _relative_tvt_bins(tvt, model.n_bins)
    return model.labels_by_bin[bins].astype(object)


def _true_geology_for_tvt(typewell: pd.DataFrame, tvt_values: np.ndarray) -> np.ndarray:
    tvt_grid = pd.to_numeric(typewell["TVT"], errors="coerce").to_numpy(dtype=np.float32)
    labels = typewell["Geology"].fillna("__unknown__").astype(str).to_numpy(dtype=object)
    if tvt_grid.size == 0:
        return np.full(len(tvt_values), "__unknown__", dtype=object)
    nearest = np.searchsorted(tvt_grid, tvt_values, side="left")
    nearest = np.clip(nearest, 0, len(tvt_grid) - 1)
    prev = np.clip(nearest - 1, 0, len(tvt_grid) - 1)
    choose_prev = np.abs(tvt_values - tvt_grid[prev]) <= np.abs(tvt_values - tvt_grid[nearest])
    idx = np.where(choose_prev, prev, nearest)
    return labels[idx].astype(object)


def _pred_geology_for_tvt(
    typewell: pd.DataFrame, pred_typewell_labels: np.ndarray, tvt_values: np.ndarray
) -> np.ndarray:
    tvt_grid = pd.to_numeric(typewell["TVT"], errors="coerce").to_numpy(dtype=np.float32)
    if tvt_grid.size == 0:
        return np.full(len(tvt_values), "__unknown__", dtype=object)
    nearest = np.searchsorted(tvt_grid, tvt_values, side="left")
    nearest = np.clip(nearest, 0, len(tvt_grid) - 1)
    prev = np.clip(nearest - 1, 0, len(tvt_grid) - 1)
    choose_prev = np.abs(tvt_values - tvt_grid[prev]) <= np.abs(tvt_values - tvt_grid[nearest])
    idx = np.where(choose_prev, prev, nearest)
    return pred_typewell_labels[idx].astype(object)


def _split_folds(n_items: int, n_folds: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(n_items)
    rng.shuffle(indices)
    return [fold.astype(np.int32) for fold in np.array_split(indices, max(1, n_folds))]


def _metrics_from_steps(steps: pd.DataFrame) -> dict[str, float | int]:
    if steps.empty:
        return {"num_steps": 0}
    match = steps["zone_match"].to_numpy(np.float32)
    unknown = steps["pred_geology"].astype(str).eq("__unknown__").to_numpy(np.float32)
    return {
        "num_steps": int(len(steps)),
        "hidden_zone_match_rate": float(np.nanmean(match)),
        "pred_unknown_rate": float(np.nanmean(unknown)),
        "num_wells": int(steps["well_id"].nunique()),
    }


def _write_report(output_dir: Path, metrics: dict[str, object]) -> None:
    aggregate = metrics.get("aggregate", {})
    if not isinstance(aggregate, dict):
        aggregate = {}
    lines = [
        "# PSEUDO_ZONE_TEMPLATE_V0",
        "",
        "## Aggregate",
        f"- wells: {aggregate.get('num_wells')}",
        f"- hidden steps: {aggregate.get('num_steps')}",
        f"- hidden_zone_match_rate: {aggregate.get('hidden_zone_match_rate')}",
        f"- pred_unknown_rate: {aggregate.get('pred_unknown_rate')}",
        "",
        "## Interpretation",
        "This is a test-schema-safe baseline: train uses train typewell Geology labels, but held-out prediction uses only the typewell TVT coordinate. It estimates whether a deployable pseudo-zone model can recover enough stratigraphic state to make formation-constrained correlation practical.",
    ]
    (output_dir / "pseudo_zone_report.md").write_text("\n".join(lines) + "\n")


def run_pseudo_zone_audit(
    *,
    data_dir: Path,
    output_dir: Path,
    rows_per_step: int = 32,
    n_bins: int = 128,
    n_folds: int = 5,
    k_wells: int = -1,
    seed: int = 42,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    wells = discover_wells(DataConfig(data_dir=data_dir, k_wells=k_wells))
    loaded = [(well, *load_well(well)) for well in wells]
    folds = _split_folds(len(loaded), n_folds, seed)
    all_rows: list[dict[str, object]] = []
    for fold_idx, valid_idx in enumerate(folds):
        valid_set = set(int(idx) for idx in valid_idx)
        train_typewells = [typewell for idx, (_, _, typewell) in enumerate(loaded) if idx not in valid_set]
        model = fit_pseudo_zone_template(train_typewells, n_bins=n_bins)
        for idx in valid_idx:
            well, horizontal, typewell = loaded[int(idx)]
            if "Geology" not in typewell.columns:
                continue
            pred_typewell_labels = predict_typewell_geology(typewell, model)
            tvt = pd.to_numeric(horizontal["TVT"], errors="coerce").to_numpy(dtype=np.float32)
            tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(
                dtype=np.float32
            )
            comp_tvt = _compress_nanmean(tvt, rows_per_step)
            comp_tvt_input = _compress_nanmean(tvt_input, rows_per_step)
            hidden_steps = np.flatnonzero(~np.isfinite(comp_tvt_input) & np.isfinite(comp_tvt))
            true_geo = _true_geology_for_tvt(typewell, comp_tvt[hidden_steps])
            pred_geo = _pred_geology_for_tvt(typewell, pred_typewell_labels, comp_tvt[hidden_steps])
            for step, true_label, pred_label in zip(hidden_steps, true_geo, pred_geo):
                all_rows.append(
                    {
                        "well_id": well.well_id,
                        "fold": int(fold_idx),
                        "step": int(step),
                        "true_geology": str(true_label),
                        "pred_geology": str(pred_label),
                        "zone_match": int(str(true_label) == str(pred_label)),
                    }
                )
    steps = pd.DataFrame(all_rows)
    steps.to_parquet(output_dir / "pseudo_zone_steps.parquet", index=False)
    by_well = (
        steps.groupby("well_id", as_index=False)
        .agg(
            num_steps=("zone_match", "size"),
            hidden_zone_match_rate=("zone_match", "mean"),
        )
        if not steps.empty
        else pd.DataFrame(columns=["well_id", "num_steps", "hidden_zone_match_rate"])
    )
    by_well.to_csv(output_dir / "pseudo_zone_by_well.csv", index=False)
    metrics = {
        "aggregate": _metrics_from_steps(steps),
        "config": {"rows_per_step": rows_per_step, "n_bins": n_bins, "n_folds": n_folds},
    }
    (output_dir / "pseudo_zone_metrics.json").write_text(
        json.dumps(_json_clean(metrics), indent=2) + "\n"
    )
    _write_report(output_dir, _json_clean(metrics))
    print(json.dumps(_json_clean(metrics), indent=2), flush=True)
    return metrics
