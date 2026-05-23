from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import load_config
from .io import discover_wells
from .stitch import run_stitch, write_mode_windows
from .track import run_tracker
from .train import train_from_config


@dataclass(frozen=True)
class OOFFold:
    fold: int
    train_wells: tuple[str, ...]
    valid_wells: tuple[str, ...]


def make_oof_folds(
    well_ids: list[str], *, n_folds: int, seed: int, max_folds: int | None = None
) -> list[OOFFold]:
    ordered = np.asarray(sorted(str(well_id) for well_id in well_ids), dtype=object)
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    if len(ordered) < n_folds:
        raise ValueError("n_folds cannot exceed the number of wells")
    rng = np.random.default_rng(seed)
    shuffled = ordered.copy()
    rng.shuffle(shuffled)
    parts = np.array_split(shuffled, n_folds)
    folds: list[OOFFold] = []
    for fold_index, valid_part in enumerate(parts):
        valid = tuple(sorted(str(item) for item in valid_part.tolist()))
        valid_set = set(valid)
        train = tuple(sorted(str(item) for item in ordered.tolist() if item not in valid_set))
        folds.append(OOFFold(fold=fold_index, train_wells=train, valid_wells=valid))
    if max_folds is not None:
        if max_folds < 1:
            raise ValueError("max_folds must be positive")
        folds = folds[:max_folds]
    return folds


def _metric_weighted_rmse(items: list[dict[str, Any]], *, rows_key: str = "rows") -> float:
    rows = np.asarray([float(item.get(rows_key, 0.0)) for item in items], dtype=np.float64)
    rmse = np.asarray([float(item.get("rmse", np.nan)) for item in items], dtype=np.float64)
    mask = (rows > 0) & np.isfinite(rmse)
    if not mask.any():
        return float("nan")
    return float(np.sqrt(np.sum((rmse[mask] ** 2) * rows[mask]) / np.sum(rows[mask])))


def _metric_sort_key(item: dict[str, Any]) -> float:
    value = float(item.get("rmse", float("nan")))
    return value if np.isfinite(value) else float("inf")


def _aggregate_named_metrics(
    fold_items: list[dict[str, Any]], *, name_key: str = "candidate"
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in fold_items:
        name = str(item[name_key])
        grouped.setdefault(name, []).append(item)
    rows: list[dict[str, Any]] = []
    for name, items in sorted(grouped.items()):
        total_rows = int(sum(int(item.get("rows", 0)) for item in items))
        row: dict[str, Any] = {
            name_key: name,
            "folds": len(items),
            "rows": total_rows,
            "rmse": _metric_weighted_rmse(items),
            "fold_rmse_min": float(min(float(item["rmse"]) for item in items)),
            "fold_rmse_max": float(max(float(item["rmse"]) for item in items)),
        }
        if all("covered_rows" in item and "covered_rmse" in item for item in items):
            covered_items = [
                {
                    "rmse": float(item["covered_rmse"]),
                    "rows": int(item["covered_rows"]),
                }
                for item in items
            ]
            row["covered_rows"] = int(sum(item["rows"] for item in covered_items))
            row["covered_rmse"] = _metric_weighted_rmse(covered_items)
        if any("p95_abs_shift_vs_b2" in item for item in items):
            row["p95_abs_shift_vs_b2_max"] = float(
                max(float(item.get("p95_abs_shift_vs_b2", 0.0)) for item in items)
            )
        if any("worst_well_rmse" in item for item in items):
            row["worst_well_rmse_max"] = float(
                max(float(item.get("worst_well_rmse", 0.0)) for item in items)
            )
        rows.append(row)
    return sorted(rows, key=_metric_sort_key)


def _write_oof_report(output_dir: Path, summary: dict[str, Any]) -> Path:
    best = summary["aggregate"]["best_candidate"]
    b2 = summary["aggregate"]["b2_guarded_submit"]
    fold_gains = [item["best_gain_vs_b2"] for item in summary["folds"]]
    lines = [
        "MTP_OOF_TRACK_REPORT",
        "",
        "setup:",
        f"  config: {summary['config_path']}",
        f"  folds_requested: {summary['n_folds']}",
        f"  folds_run: {summary['folds_run']}",
        f"  seed: {summary['seed']}",
        f"  logit_source: {summary['tracker']['logit_source']}",
        "",
        "aggregate:",
        f"  B2_rmse: {b2['rmse']}",
        f"  best_candidate: {best['candidate']}",
        f"  best_rmse: {best['rmse']}",
        f"  gain_vs_B2: {b2['rmse'] - best['rmse']}",
        f"  fold_gain_min: {min(fold_gains) if fold_gains else 'n/a'}",
        f"  fold_gain_max: {max(fold_gains) if fold_gains else 'n/a'}",
        f"  negative_folds: {sum(1 for gain in fold_gains if gain < 0.0)}",
        "",
        "folds:",
    ]
    for item in summary["folds"]:
        lines.extend(
            [
                f"  fold {item['fold']}:",
                f"    train_wells: {item['train_wells']}",
                f"    valid_wells: {item['valid_wells']}",
                f"    B2_rmse: {item['b2_rmse']}",
                f"    best_candidate: {item['best_candidate']}",
                f"    best_rmse: {item['best_rmse']}",
                f"    gain_vs_B2: {item['best_gain_vs_b2']}",
            ]
        )
    path = output_dir / "oof_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def run_oof(
    config_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    n_folds: int = 5,
    max_folds: int | None = None,
    seed: int = 42,
    logit_source: str = "nn",
    n_realizations: int = 32,
    keep_top: int = 32,
    merge_tolerance_ft: float = 3.0,
    overlap_penalty: float = 0.10,
    max_modes_per_window: int = 8,
    full_stitch: bool = False,
    resume: bool = True,
) -> dict[str, Any]:
    config_path = Path(config_path)
    cfg = load_config(config_path)
    root = Path(output_dir) if output_dir is not None else (
        cfg.run.output_dir.parent / f"{cfg.run.name}_oof"
    )
    root.mkdir(parents=True, exist_ok=True)
    well_ids = [well.well_id for well in discover_wells(cfg.data)]
    folds = make_oof_folds(well_ids, n_folds=n_folds, seed=seed, max_folds=max_folds)

    fold_summaries: list[dict[str, Any]] = []
    candidate_items: list[dict[str, Any]] = []
    baseline_items: dict[str, list[dict[str, Any]]] = {}
    row_prediction_paths: list[Path] = []

    for fold in folds:
        fold_dir = root / f"fold_{fold.fold}"
        fold_name = f"{cfg.run.name}_oof_fold{fold.fold}"
        metrics_path = fold_dir / "track_metrics.json"
        if resume and metrics_path.exists():
            print(
                json.dumps(
                    {
                        "event": "oof_fold_resume",
                        "fold": fold.fold,
                        "run_dir": str(fold_dir),
                    }
                ),
                flush=True,
            )
            track_summary = json.loads(metrics_path.read_text(encoding="utf-8"))
        else:
            print(
                json.dumps(
                    {
                        "event": "oof_fold_train_start",
                        "fold": fold.fold,
                        "train_wells": len(fold.train_wells),
                        "valid_wells": len(fold.valid_wells),
                        "run_dir": str(fold_dir),
                    }
                ),
                flush=True,
            )
            train_from_config(
                config_path,
                output_dir=fold_dir,
                run_name=fold_name,
                train_wells=fold.train_wells,
                valid_wells=fold.valid_wells,
            )
            print(
                json.dumps({"event": "oof_fold_train_done", "fold": fold.fold}),
                flush=True,
            )
            print(
                json.dumps(
                    {
                        "event": "oof_fold_modes_start",
                        "fold": fold.fold,
                        "full_stitch": full_stitch,
                    }
                ),
                flush=True,
            )
            if full_stitch:
                stitch_summary = run_stitch(fold_dir)
                print(
                    json.dumps(
                        {
                            "event": "oof_fold_modes_done",
                            "fold": fold.fold,
                            "windows": int(stitch_summary["window"]["num_windows"]),
                        }
                    ),
                    flush=True,
                )
            else:
                mode_windows = write_mode_windows(fold_dir)
                print(
                    json.dumps(
                        {
                            "event": "oof_fold_modes_done",
                            "fold": fold.fold,
                            "windows": int(len(mode_windows)),
                        }
                    ),
                    flush=True,
                )
            print(
                json.dumps({"event": "oof_fold_track_start", "fold": fold.fold}),
                flush=True,
            )
            track_summary = run_tracker(
                fold_dir,
                n_realizations=n_realizations,
                keep_top=keep_top,
                merge_tolerance_ft=merge_tolerance_ft,
                overlap_penalty=overlap_penalty,
                max_modes_per_window=max_modes_per_window,
                logit_source=logit_source,
            )
        for name, item in track_summary["baselines"].items():
            baseline_items.setdefault(name, []).append(item)
        candidate_items.extend(track_summary["candidates"])
        b2 = track_summary["baselines"]["b2_guarded_submit"]
        best = min(
            track_summary["candidates"],
            key=_metric_sort_key,
        )
        fold_summary = {
            "fold": fold.fold,
            "run_dir": fold_dir,
            "train_wells": len(fold.train_wells),
            "valid_wells": len(fold.valid_wells),
            "b2_rmse": float(b2["rmse"]),
            "best_candidate": best["candidate"],
            "best_rmse": float(best["rmse"]),
            "best_gain_vs_b2": float(b2["rmse"] - best["rmse"]),
        }
        fold_summaries.append(fold_summary)
        row_path = fold_dir / "track_row_predictions.parquet"
        if row_path.exists():
            row_prediction_paths.append(row_path)
        print(
            json.dumps({"event": "oof_fold_track_done", **_json_safe(fold_summary)}),
            flush=True,
        )

    aggregate_candidates = _aggregate_named_metrics(candidate_items)
    aggregate_baselines = {
        name: _aggregate_named_metrics(items, name_key="candidate")[0]
        for name, items in baseline_items.items()
    }
    b2_aggregate = aggregate_baselines["b2_guarded_submit"]
    best_candidate = min(
        aggregate_candidates, key=_metric_sort_key
    )
    candidates_frame = pd.DataFrame(aggregate_candidates)
    candidates_frame.to_csv(root / "oof_candidates.csv", index=False)
    if row_prediction_paths:
        pd.concat(
            [pd.read_parquet(path) for path in row_prediction_paths],
            ignore_index=True,
        ).to_parquet(root / "oof_track_row_predictions.parquet", index=False)
    summary = {
        "config_path": config_path,
        "output_dir": root,
        "n_folds": n_folds,
        "folds_run": len(folds),
        "seed": seed,
        "tracker": {
            "logit_source": logit_source,
            "n_realizations": n_realizations,
            "keep_top": keep_top,
            "merge_tolerance_ft": merge_tolerance_ft,
            "overlap_penalty": overlap_penalty,
            "max_modes_per_window": max_modes_per_window,
        },
        "folds": fold_summaries,
        "aggregate": {
            **aggregate_baselines,
            "candidates": aggregate_candidates,
            "best_candidate": best_candidate,
            "gain_vs_b2": float(b2_aggregate["rmse"] - best_candidate["rmse"]),
            "fold_gain_min": float(
                min(item["best_gain_vs_b2"] for item in fold_summaries)
            ),
            "fold_gain_max": float(
                max(item["best_gain_vs_b2"] for item in fold_summaries)
            ),
            "negative_folds": int(
                sum(1 for item in fold_summaries if item["best_gain_vs_b2"] < 0.0)
            ),
        },
    }
    (root / "oof_metrics.json").write_text(
        json.dumps(_json_safe(summary), indent=2), encoding="utf-8"
    )
    _write_oof_report(root, _json_safe(summary))
    print(json.dumps(_json_safe(summary["aggregate"]["best_candidate"]), indent=2), flush=True)
    return summary
