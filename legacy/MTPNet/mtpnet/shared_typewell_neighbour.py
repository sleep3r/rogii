from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .correlation_panel import _compress_nanmean, known_tail_linear_anchor
from .heatmap import fill_nan


@dataclass(frozen=True)
class SharedTypewellNeighbourConfig:
    data_dir: Path = Path("data/train")
    output_dir: Path = Path("artifacts/shared_typewell_neighbour_v0")
    rows_per_step: int = 32
    n_folds: int = 5
    seed: int = 42
    k_wells: int = -1
    n_signature_bins: int = 256
    max_shift_bins: int = 32
    top_k_neighbours: int = 8


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _zscore(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(arr)
    if finite.sum() < 2:
        return np.zeros_like(arr, dtype=np.float32)
    mean = float(np.nanmean(arr[finite]))
    std = float(np.nanstd(arr[finite]))
    if not np.isfinite(std) or std < 1e-6:
        std = 1.0
    out = (arr - mean) / std
    return np.where(np.isfinite(out), out, 0.0).astype(np.float32)


def best_shifted_corr(
    left: np.ndarray,
    right: np.ndarray,
    *,
    max_shift_bins: int = 32,
    min_overlap: int = 8,
) -> tuple[float, int]:
    """Return best normalized correlation and lag between two equal-length signatures."""
    a = _zscore(np.asarray(left, dtype=np.float32))
    b = _zscore(np.asarray(right, dtype=np.float32))
    if a.size == 0 or b.size == 0:
        return float("nan"), 0
    n = min(a.size, b.size)
    a = a[:n]
    b = b[:n]
    best_corr = -np.inf
    best_shift = 0
    max_shift = min(abs(int(max_shift_bins)), max(n - 2, 0))
    a = _zscore(a)
    b = _zscore(b)
    for shift in range(-max_shift, max_shift + 1):
        # Use a circular lag for the fingerprint audit. The goal is detecting
        # repeated/offset-copy typewell shapes, not estimating a deployable path
        # shift directly from the wrapped edges.
        x = a
        y = np.roll(b, shift)
        if x.size < min_overlap or y.size < min_overlap:
            continue
        corr = float(np.mean(x * y))
        if corr > best_corr:
            best_corr = corr
            best_shift = shift
    return float(best_corr), int(best_shift)


def typewell_signature(typewell: pd.DataFrame, *, n_bins: int = 256) -> np.ndarray:
    tvt = pd.to_numeric(typewell["TVT"], errors="coerce").to_numpy(np.float32)
    gr_raw = pd.to_numeric(typewell["GR"], errors="coerce").to_numpy(np.float32)
    gr, _ = fill_nan(gr_raw)
    finite = np.isfinite(tvt) & np.isfinite(gr)
    if finite.sum() < 2:
        return np.zeros(int(n_bins), dtype=np.float32)
    order = np.argsort(tvt[finite])
    tvt_sorted = tvt[finite][order]
    gr_sorted = gr[finite][order]
    lo = float(tvt_sorted[0])
    hi = float(tvt_sorted[-1])
    if hi <= lo:
        return _zscore(np.resize(gr_sorted, int(n_bins))).astype(np.float32)
    x = (tvt_sorted - lo) / (hi - lo)
    grid = np.linspace(0.0, 1.0, int(n_bins), dtype=np.float32)
    sig = np.interp(grid, x, gr_sorted).astype(np.float32)
    return _zscore(sig)


def _well_xy(frame: pd.DataFrame) -> tuple[float, float, float]:
    coords = []
    for column in ("X", "Y", "Z"):
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)
            finite = values[np.isfinite(values)]
            coords.append(float(finite[0]) if finite.size else 0.0)
        else:
            coords.append(0.0)
    return tuple(coords)  # type: ignore[return-value]


def _hidden_mask(frame: pd.DataFrame) -> pd.Series:
    if "TVT_input" not in frame.columns:
        return pd.Series(True, index=frame.index)
    return pd.to_numeric(frame["TVT_input"], errors="coerce").isna()


def _ensure_ids(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["well_id"] = out["well_id"].astype(str)
    if "row_idx" not in out.columns:
        out["row_idx"] = out.groupby("well_id").cumcount()
    if "id" not in out.columns:
        out["id"] = [
            f"{well}_{int(row)}"
            for well, row in zip(out["well_id"], out["row_idx"], strict=False)
        ]
    out["id"] = out["id"].astype(str)
    if "step" not in out.columns:
        out["step"] = 0
    return out


def _compressed_paths(well: pd.DataFrame, rows_per_step: int) -> dict[str, np.ndarray]:
    sorted_well = well.sort_values("row_idx")
    tvt = (
        pd.to_numeric(sorted_well["TVT"], errors="coerce").to_numpy(np.float32)
        if "TVT" in sorted_well.columns
        else np.full(len(sorted_well), np.nan, dtype=np.float32)
    )
    tvt_input = (
        pd.to_numeric(sorted_well["TVT_input"], errors="coerce").to_numpy(np.float32)
        if "TVT_input" in sorted_well.columns
        else np.full(len(sorted_well), np.nan, dtype=np.float32)
    )
    comp_tvt = _compress_nanmean(tvt, rows_per_step)
    comp_input = _compress_nanmean(tvt_input, rows_per_step)
    anchor = known_tail_linear_anchor(comp_input)
    hidden_steps = np.flatnonzero(~np.isfinite(comp_input))
    return {
        "tvt": comp_tvt,
        "input": comp_input,
        "anchor": anchor,
        "hidden_steps": hidden_steps.astype(np.int32),
    }


def _residual_curve(well: pd.DataFrame, rows_per_step: int) -> tuple[np.ndarray, np.ndarray] | None:
    paths = _compressed_paths(well, rows_per_step)
    hidden_steps = paths["hidden_steps"]
    tvt = paths["tvt"]
    anchor = paths["anchor"]
    if hidden_steps.size < 2:
        return None
    valid = hidden_steps[(hidden_steps < tvt.size) & (hidden_steps < anchor.size)]
    valid = valid[np.isfinite(tvt[valid]) & np.isfinite(anchor[valid])]
    if valid.size < 2:
        return None
    denom = max(float(valid[-1] - valid[0]), 1.0)
    progress = ((valid - valid[0]) / denom).astype(np.float32)
    residual = (tvt[valid] - anchor[valid]).astype(np.float32)
    return progress, residual


def _interp_residual(curve: tuple[np.ndarray, np.ndarray], progress: np.ndarray) -> np.ndarray:
    x, y = curve
    if x.size == 0:
        return np.zeros_like(progress, dtype=np.float32)
    order = np.argsort(x)
    return np.interp(progress, x[order], y[order], left=y[order][0], right=y[order][-1]).astype(np.float32)


def _make_folds(well_ids: list[str], n_folds: int, seed: int) -> list[list[str]]:
    rng = np.random.default_rng(seed)
    ids = np.asarray(sorted(map(str, well_ids)), dtype=object)
    rng.shuffle(ids)
    n = max(2, min(int(n_folds), len(ids)))
    return [list(part.astype(str)) for part in np.array_split(ids, n) if len(part)]


def _candidate_rows(
    hidden_rows: pd.DataFrame,
    *,
    candidate: str,
    values_by_step: dict[int, float],
) -> pd.DataFrame:
    cols = ["id", "well_id", "row_idx", "step"]
    out = hidden_rows[cols].copy()
    out["candidate"] = candidate
    out["pred_tvt"] = [
        float(values_by_step.get(int(step), np.nan)) for step in out["step"].to_numpy()
    ]
    return out[np.isfinite(out["pred_tvt"].to_numpy(np.float64))].copy()


def _neighbour_table(
    *,
    valid_wells: Iterable[str],
    train_wells: Iterable[str],
    signatures: dict[str, np.ndarray],
    coords: dict[str, tuple[float, float, float]],
    cfg: SharedTypewellNeighbourConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for query in sorted(map(str, valid_wells)):
        qsig = signatures.get(query)
        if qsig is None:
            continue
        qxyz = np.asarray(coords.get(query, (0.0, 0.0, 0.0)), dtype=np.float64)
        for neighbour in sorted(map(str, train_wells)):
            nsig = signatures.get(neighbour)
            if nsig is None:
                continue
            corr, shift = best_shifted_corr(
                qsig,
                nsig,
                max_shift_bins=cfg.max_shift_bins,
                min_overlap=max(8, cfg.n_signature_bins // 12),
            )
            nxyz = np.asarray(coords.get(neighbour, (0.0, 0.0, 0.0)), dtype=np.float64)
            distance = float(np.linalg.norm(qxyz - nxyz))
            rows.append(
                {
                    "query_well_id": query,
                    "neighbour_well_id": neighbour,
                    "typewell_corr": corr,
                    "typewell_shift_bins": shift,
                    "xyz_distance": distance,
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "query_well_id",
                "neighbour_well_id",
                "typewell_corr",
                "typewell_shift_bins",
                "xyz_distance",
                "rank",
            ]
        )
    table = pd.DataFrame(rows)
    table = table.sort_values(
        ["query_well_id", "typewell_corr", "xyz_distance"],
        ascending=[True, False, True],
    )
    table["rank"] = table.groupby("query_well_id").cumcount() + 1
    return table[table["rank"] <= int(cfg.top_k_neighbours)].reset_index(drop=True)


def build_neighbour_candidates_from_frames(
    frame: pd.DataFrame,
    typewells: dict[str, pd.DataFrame],
    *,
    train_wells: Iterable[str],
    valid_wells: Iterable[str],
    cfg: SharedTypewellNeighbourConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = _ensure_ids(frame)
    raw["step"] = (
        pd.to_numeric(raw["row_idx"], errors="coerce") // int(cfg.rows_per_step)
    ).astype(int)
    well_groups = {str(well): group.copy() for well, group in raw.groupby("well_id", sort=False)}
    signatures = {
        str(well): typewell_signature(tw, n_bins=cfg.n_signature_bins)
        for well, tw in typewells.items()
    }
    coords = {
        str(well): _well_xy(group)
        for well, group in well_groups.items()
    }
    neighbours = _neighbour_table(
        valid_wells=valid_wells,
        train_wells=train_wells,
        signatures=signatures,
        coords=coords,
        cfg=cfg,
    )
    residuals = {
        str(well): curve
        for well in train_wells
        if (curve := _residual_curve(well_groups[str(well)], cfg.rows_per_step)) is not None
    }

    parts: list[pd.DataFrame] = []
    for well_id in sorted(map(str, valid_wells)):
        if well_id not in well_groups:
            continue
        well = well_groups[well_id].sort_values("row_idx")
        hidden = well.loc[_hidden_mask(well)].copy()
        if hidden.empty:
            continue
        paths = _compressed_paths(well, cfg.rows_per_step)
        hidden_steps = paths["hidden_steps"]
        hidden_steps = hidden_steps[hidden_steps < paths["anchor"].size]
        if hidden_steps.size == 0:
            continue
        denom = max(float(hidden_steps[-1] - hidden_steps[0]), 1.0)
        progress = ((hidden_steps - hidden_steps[0]) / denom).astype(np.float32)
        anchor_values = {
            int(step): float(paths["anchor"][step])
            for step in hidden_steps
            if np.isfinite(paths["anchor"][step])
        }
        parts.append(_candidate_rows(hidden, candidate="known_tail_anchor", values_by_step=anchor_values))

        source_ids = neighbours.loc[
            neighbours["query_well_id"].astype(str) == well_id, "neighbour_well_id"
        ].astype(str).tolist()
        curves = [residuals[source] for source in source_ids if source in residuals]
        if not curves:
            continue
        transferred: list[np.ndarray] = []
        for source, curve in zip(
            [source for source in source_ids if source in residuals],
            curves,
            strict=False,
        ):
            residual = _interp_residual(curve, progress)
            pred = paths["anchor"][hidden_steps] + residual
            values = {
                int(step): float(value)
                for step, value in zip(hidden_steps, pred, strict=False)
                if np.isfinite(value)
            }
            transferred.append(pred.astype(np.float32))
            parts.append(_candidate_rows(hidden, candidate=f"neighbour_{source}", values_by_step=values))
        stack = np.vstack(transferred)
        for k in (1, 3, 5):
            take = stack[: min(k, stack.shape[0])]
            if take.size == 0:
                continue
            mean_pred = np.nanmean(take, axis=0)
            median_pred = np.nanmedian(take, axis=0)
            parts.append(
                _candidate_rows(
                    hidden,
                    candidate=f"neighbour_mean_top{k}",
                    values_by_step={
                        int(step): float(value)
                        for step, value in zip(hidden_steps, mean_pred, strict=False)
                        if np.isfinite(value)
                    },
                )
            )
            parts.append(
                _candidate_rows(
                    hidden,
                    candidate=f"neighbour_median_top{k}",
                    values_by_step={
                        int(step): float(value)
                        for step, value in zip(hidden_steps, median_pred, strict=False)
                        if np.isfinite(value)
                    },
                )
            )
        parts.append(
            _candidate_rows(
                hidden,
                candidate="neighbour_top1",
                values_by_step={
                    int(step): float(value)
                    for step, value in zip(hidden_steps, stack[0], strict=False)
                    if np.isfinite(value)
                },
            )
        )

    if not parts:
        return pd.DataFrame(), neighbours
    candidates = pd.concat(parts, ignore_index=True)
    return candidates.reset_index(drop=True), neighbours.reset_index(drop=True)


def _rmse(pred: np.ndarray | pd.Series, true: np.ndarray | pd.Series) -> float:
    pred_arr = np.asarray(pred, dtype=np.float64)
    true_arr = np.asarray(true, dtype=np.float64)
    mask = np.isfinite(pred_arr) & np.isfinite(true_arr)
    if not mask.any():
        return float("nan")
    return float(np.sqrt(np.mean(np.square(pred_arr[mask] - true_arr[mask]))))


def _metric_summary(values: pd.Series) -> dict[str, float]:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "p50": float("nan"), "p90": float("nan"), "p95": float("nan"), "worst": float("nan")}
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
        "p95": float(np.quantile(arr, 0.95)),
        "worst": float(np.max(arr)),
    }


def evaluate_candidate_predictions(candidates: pd.DataFrame, frame: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    truth_cols = ["id", "well_id", "TVT"]
    if "tail_class" in frame.columns:
        truth_cols.append("tail_class")
    truth = _ensure_ids(frame)[truth_cols].copy()
    merged = candidates.merge(truth, on=["id", "well_id"], how="left")
    merged = merged[np.isfinite(pd.to_numeric(merged["TVT"], errors="coerce"))].copy()
    rows: list[dict[str, Any]] = []
    for (well_id, candidate), group in merged.groupby(["well_id", "candidate"], sort=True):
        rows.append(
            {
                "well_id": well_id,
                "candidate": candidate,
                "rmse": _rmse(group["pred_tvt"], group["TVT"]),
                "rows": int(len(group)),
                "tail_class": str(group["tail_class"].iloc[0]) if "tail_class" in group.columns else "unknown",
            }
        )
    per_candidate = pd.DataFrame(rows)
    if per_candidate.empty:
        return {"wells": 0, "candidates": 0}, per_candidate, pd.DataFrame()

    wide = per_candidate.pivot(index="well_id", columns="candidate", values="rmse")
    wide.columns = [f"rmse__{col}" for col in wide.columns]
    wide = wide.reset_index()
    candidate_cols = [col for col in wide.columns if col.startswith("rmse__")]
    values = wide[candidate_cols].to_numpy(np.float64)
    best_idx = np.nanargmin(values, axis=1)
    wide["oracle_rmse"] = values[np.arange(values.shape[0]), best_idx]
    wide["oracle_candidate"] = [candidate_cols[idx].removeprefix("rmse__") for idx in best_idx]
    if "rmse__known_tail_anchor" in wide.columns:
        wide["anchor_rmse"] = wide["rmse__known_tail_anchor"]
        wide["oracle_gain_vs_anchor"] = wide["anchor_rmse"] - wide["oracle_rmse"]
    else:
        wide["anchor_rmse"] = np.nan
        wide["oracle_gain_vs_anchor"] = np.nan
    tail = per_candidate[["well_id", "tail_class"]].drop_duplicates("well_id")
    wide = wide.merge(tail, on="well_id", how="left")

    candidate_summary = (
        per_candidate.groupby("candidate")
        .agg(
            mean_well_rmse=("rmse", "mean"),
            p95_well_rmse=("rmse", lambda x: float(np.quantile(x, 0.95))),
            worst_well_rmse=("rmse", "max"),
            wells=("well_id", "nunique"),
        )
        .reset_index()
        .sort_values(["mean_well_rmse", "candidate"])
    )
    metrics: dict[str, Any] = {
        "wells": int(wide["well_id"].nunique()),
        "candidates": int(per_candidate["candidate"].nunique()),
        "anchor": _metric_summary(wide["anchor_rmse"]),
        "oracle": _metric_summary(wide["oracle_rmse"]),
        "oracle_gain_mean": float(np.nanmean(wide["oracle_gain_vs_anchor"])),
        "oracle_beats_anchor_wells": int((wide["oracle_gain_vs_anchor"] > 0).sum()),
        "oracle_not_better_anchor_wells": int((wide["oracle_gain_vs_anchor"] <= 0).sum()),
    }
    if "tail_class" in wide.columns:
        metrics["tail_class"] = {
            str(cls): {
                "wells": int(len(group)),
                "anchor_mean": float(np.nanmean(group["anchor_rmse"])),
                "oracle_mean": float(np.nanmean(group["oracle_rmse"])),
                "gain_mean": float(np.nanmean(group["oracle_gain_vs_anchor"])),
            }
            for cls, group in wide.groupby("tail_class")
        }
    return metrics, wide, candidate_summary


def run_shared_typewell_neighbour_audit_from_frames(
    frame: pd.DataFrame,
    typewells: dict[str, pd.DataFrame],
    *,
    output_dir: str | Path,
    cfg: SharedTypewellNeighbourConfig,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw = _ensure_ids(frame)
    well_ids = sorted(raw["well_id"].astype(str).unique().tolist())
    folds = _make_folds(well_ids, cfg.n_folds, cfg.seed)
    all_candidates: list[pd.DataFrame] = []
    all_neighbours: list[pd.DataFrame] = []
    for fold_idx, valid in enumerate(folds):
        train = [well for well in well_ids if well not in set(valid)]
        print(
            f"[shared-typewell] fold {fold_idx + 1}/{len(folds)} train={len(train)} valid={len(valid)}",
            flush=True,
        )
        candidates, neighbours = build_neighbour_candidates_from_frames(
            raw,
            typewells,
            train_wells=train,
            valid_wells=valid,
            cfg=cfg,
        )
        if not candidates.empty:
            candidates["fold"] = fold_idx
            all_candidates.append(candidates)
        if not neighbours.empty:
            neighbours["fold"] = fold_idx
            all_neighbours.append(neighbours)
    candidates = pd.concat(all_candidates, ignore_index=True) if all_candidates else pd.DataFrame()
    neighbours = pd.concat(all_neighbours, ignore_index=True) if all_neighbours else pd.DataFrame()
    metrics, well_oracle, candidate_summary = evaluate_candidate_predictions(candidates, raw)
    metrics["config"] = asdict(cfg)
    metrics["folds"] = len(folds)

    candidates.to_parquet(out / "neighbour_candidate_predictions.parquet", index=False)
    neighbours.to_parquet(out / "typewell_neighbours.parquet", index=False)
    well_oracle.to_csv(out / "shared_typewell_neighbour_wells.csv", index=False)
    candidate_summary.to_csv(out / "shared_typewell_neighbour_candidate_summary.csv", index=False)
    (out / "shared_typewell_neighbour_metrics.json").write_text(
        json.dumps(_json_safe(metrics), indent=2) + "\n",
        encoding="utf-8",
    )
    _write_report(out, metrics, well_oracle, candidate_summary, neighbours)
    print(json.dumps(_json_safe(metrics), indent=2), flush=True)
    return metrics


def _format(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "nan"
        return f"{float(value):.4f}"
    return str(value)


def _table(frame: pd.DataFrame, columns: list[str], max_rows: int = 20) -> str:
    if frame.empty:
        return "(empty)"
    view = frame[[col for col in columns if col in frame.columns]].head(max_rows)
    lines = [
        "| " + " | ".join(view.columns) + " |",
        "| " + " | ".join(["---"] * len(view.columns)) + " |",
    ]
    for row in view.itertuples(index=False):
        lines.append("| " + " | ".join(_format(value) for value in row) + " |")
    return "\n".join(lines)


def _write_report(
    output_dir: Path,
    metrics: dict[str, Any],
    well_oracle: pd.DataFrame,
    candidate_summary: pd.DataFrame,
    neighbours: pd.DataFrame,
) -> None:
    neighbour_stats = {}
    if not neighbours.empty:
        neighbour_stats = {
            "mean_typewell_corr": float(pd.to_numeric(neighbours["typewell_corr"], errors="coerce").mean()),
            "p95_typewell_corr": float(pd.to_numeric(neighbours["typewell_corr"], errors="coerce").quantile(0.95)),
            "mean_abs_shift_bins": float(pd.to_numeric(neighbours["typewell_shift_bins"], errors="coerce").abs().mean()),
        }
    lines = [
        "# SHARED_TYPEWELL_NEIGHBOUR_AUDIT_V0",
        "",
        "## Purpose",
        "Test the Kaggle-thread hypothesis that shared/offset typewells and nearby wells can provide a TVT range/path family beyond raw GR matching.",
        "",
        "## Summary",
        "```json",
        json.dumps(_json_safe({k: v for k, v in metrics.items() if k != "config"}), indent=2),
        "```",
        "",
        "## Typewell Neighbour Stats",
        "```json",
        json.dumps(_json_safe(neighbour_stats), indent=2),
        "```",
        "",
        "## Candidate Summary",
        _table(candidate_summary, ["candidate", "mean_well_rmse", "p95_well_rmse", "worst_well_rmse", "wells"], max_rows=40),
        "",
        "## Best Oracle Wells",
        _table(
            well_oracle.sort_values("oracle_gain_vs_anchor", ascending=False),
            ["well_id", "anchor_rmse", "oracle_rmse", "oracle_gain_vs_anchor", "oracle_candidate", "tail_class"],
            max_rows=30,
        ),
        "",
        "## Worst Non-Gain Wells",
        _table(
            well_oracle.sort_values("oracle_gain_vs_anchor", ascending=True),
            ["well_id", "anchor_rmse", "oracle_rmse", "oracle_gain_vs_anchor", "oracle_candidate", "tail_class"],
            max_rows=30,
        ),
        "",
        "## Interpretation",
        "If the neighbour oracle materially beats the known-tail anchor and improves bad tail classes, this is a candidate-space lever. If not, shared typewell similarity alone is not enough.",
    ]
    (output_dir / "shared_typewell_neighbour_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_frames_from_data_dir(data_dir: Path, *, k_wells: int = -1) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    paths = sorted(Path(data_dir).glob("*__horizontal_well.csv"))
    if k_wells > 0:
        paths = paths[: int(k_wells)]
    frames: list[pd.DataFrame] = []
    typewells: dict[str, pd.DataFrame] = {}
    for path in paths:
        well_id = path.name.replace("__horizontal_well.csv", "")
        h = pd.read_csv(path)
        h["well_id"] = well_id
        h["row_idx"] = np.arange(len(h), dtype=np.int32)
        h["id"] = [f"{well_id}_{idx}" for idx in h["row_idx"]]
        for column in ("MD", "X", "Y", "Z", "GR", "TVT", "TVT_input"):
            if column not in h.columns:
                h[column] = np.nan
        frames.append(h)
        tw_path = path.with_name(f"{well_id}__typewell.csv")
        if tw_path.exists():
            typewells[well_id] = pd.read_csv(tw_path)
    if not frames:
        raise FileNotFoundError(f"No horizontal wells found in {data_dir}")
    return pd.concat(frames, ignore_index=True), typewells


def run_shared_typewell_neighbour_audit(cfg: SharedTypewellNeighbourConfig) -> dict[str, Any]:
    frame, typewells = load_frames_from_data_dir(cfg.data_dir, k_wells=cfg.k_wells)
    return run_shared_typewell_neighbour_audit_from_frames(
        frame,
        typewells,
        output_dir=cfg.output_dir,
        cfg=cfg,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fold-safe shared typewell/neighbour candidate audit.")
    parser.add_argument("--data-dir", type=Path, default=SharedTypewellNeighbourConfig.data_dir)
    parser.add_argument("--output-dir", type=Path, default=SharedTypewellNeighbourConfig.output_dir)
    parser.add_argument("--rows-per-step", type=int, default=SharedTypewellNeighbourConfig.rows_per_step)
    parser.add_argument("--n-folds", type=int, default=SharedTypewellNeighbourConfig.n_folds)
    parser.add_argument("--seed", type=int, default=SharedTypewellNeighbourConfig.seed)
    parser.add_argument("--k-wells", type=int, default=SharedTypewellNeighbourConfig.k_wells)
    parser.add_argument("--n-signature-bins", type=int, default=SharedTypewellNeighbourConfig.n_signature_bins)
    parser.add_argument("--max-shift-bins", type=int, default=SharedTypewellNeighbourConfig.max_shift_bins)
    parser.add_argument("--top-k-neighbours", type=int, default=SharedTypewellNeighbourConfig.top_k_neighbours)
    args = parser.parse_args()
    cfg = SharedTypewellNeighbourConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        rows_per_step=args.rows_per_step,
        n_folds=args.n_folds,
        seed=args.seed,
        k_wells=args.k_wells,
        n_signature_bins=args.n_signature_bins,
        max_shift_bins=args.max_shift_bins,
        top_k_neighbours=args.top_k_neighbours,
    )
    run_shared_typewell_neighbour_audit(cfg)


if __name__ == "__main__":
    main()
