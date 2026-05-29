"""Evaluation harness for the plane solver.

Picks a sample of labelled training wells, masks their TVT to simulate the
public test split (i.e. only TVT_input is visible), runs every solver mode,
and reports per-well + aggregate RMSE on the hidden tail.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .solver import PlaneSolverConfig, solve_well
from .typewell_align import build_typewell_ref


DATA_DIR = Path("data/train")


@dataclass
class EvalConfig:
    data_dir: Path = DATA_DIR
    n_wells: int = 100
    seed: int = 42
    output_dir: Path | None = None
    modes: tuple[str, ...] = ("constant", "linear", "linear_typewell")
    slope_window: int = 600
    slope_min_window: int = 100
    offset_search_radius_ft: float = 30.0
    offset_grid_size: int = 301
    dp_n_chunks: int = 8
    dp_smooth_lambda: float = 1.0
    dp_search_radius_ft: float = 30.0
    dp_grid_size: int = 201


def _hidden_rmse(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    """RMSE evaluated on rows in ``mask`` only."""

    mask = mask & np.isfinite(pred) & np.isfinite(target)
    if not mask.any():
        return float("nan")
    err = pred[mask] - target[mask]
    return float(np.sqrt(np.mean(err * err)))


def _list_train_wells(data_dir: Path) -> list[str]:
    ids = set()
    for p in data_dir.glob("*__horizontal_well.csv"):
        wid = p.name.split("__")[0]
        if (data_dir / f"{wid}__typewell.csv").exists():
            ids.add(wid)
    return sorted(ids)


def _load_well(data_dir: Path, well_id: str):
    h = pd.read_csv(data_dir / f"{well_id}__horizontal_well.csv")
    t = pd.read_csv(data_dir / f"{well_id}__typewell.csv")
    return h, t


def evaluate(cfg: EvalConfig) -> dict:
    well_ids = _list_train_wells(cfg.data_dir)
    rng = np.random.default_rng(cfg.seed)
    if cfg.n_wells > 0 and cfg.n_wells < len(well_ids):
        choice = rng.choice(len(well_ids), size=cfg.n_wells, replace=False)
        well_ids = [well_ids[i] for i in sorted(choice)]

    solver_cfg_kwargs = dict(
        slope_window=cfg.slope_window,
        slope_min_window=cfg.slope_min_window,
        offset_search_radius_ft=cfg.offset_search_radius_ft,
        offset_grid_size=cfg.offset_grid_size,
        dp_n_chunks=cfg.dp_n_chunks,
        dp_smooth_lambda=cfg.dp_smooth_lambda,
        dp_search_radius_ft=cfg.dp_search_radius_ft,
        dp_grid_size=cfg.dp_grid_size,
    )

    per_well = []
    for wid in well_ids:
        h, t = _load_well(cfg.data_dir, wid)
        if "TVT" not in h.columns or h["TVT"].isna().all():
            continue
        md = h["MD"].to_numpy(dtype=np.float64)
        z = h["Z"].to_numpy(dtype=np.float64)
        tvt_input = h["TVT_input"].to_numpy(dtype=np.float64)
        tvt_true = h["TVT"].to_numpy(dtype=np.float64)
        gr = h["GR"].to_numpy(dtype=np.float64) if "GR" in h.columns else None

        hidden_mask = ~np.isfinite(tvt_input) & np.isfinite(tvt_true)
        if hidden_mask.sum() < 10:
            continue

        try:
            ref = build_typewell_ref(t)
        except Exception:
            ref = None

        well_record = {"well_id": wid, "n": int(len(h)), "hidden": int(hidden_mask.sum())}
        for mode in cfg.modes:
            res = solve_well(
                md=md,
                z=z,
                tvt_input=tvt_input,
                gr=gr,
                typewell=ref,
                config=PlaneSolverConfig(mode=mode, **solver_cfg_kwargs),
            )
            rmse = _hidden_rmse(res["tvt"], tvt_true, hidden_mask)
            well_record[f"rmse_{mode}"] = rmse
            well_record[f"mode_used_{mode}"] = res["mode_used"]
            well_record[f"offset_{mode}"] = res["offset_applied_ft"]
        per_well.append(well_record)

    df = pd.DataFrame(per_well)
    summary = {"n_wells_evaluated": int(len(df))}
    for mode in cfg.modes:
        col = f"rmse_{mode}"
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        summary[f"mean_rmse_{mode}"] = float(vals.mean())
        summary[f"median_rmse_{mode}"] = float(vals.median())
        summary[f"p90_rmse_{mode}"] = float(vals.quantile(0.90))
        summary[f"p95_rmse_{mode}"] = float(vals.quantile(0.95))
        summary[f"worst_rmse_{mode}"] = float(vals.max())

    # Pairwise gains versus the simplest baseline (constant).
    if "rmse_constant" in df.columns:
        for mode in cfg.modes:
            if mode == "constant":
                continue
            col = f"rmse_{mode}"
            if col not in df.columns:
                continue
            gain = (df["rmse_constant"] - df[col]).dropna()
            summary[f"mean_gain_{mode}_vs_constant"] = float(gain.mean())
            summary[f"p50_gain_{mode}_vs_constant"] = float(gain.median())
            summary[f"frac_wells_improved_{mode}_vs_constant"] = float((gain > 0).mean())

    if cfg.output_dir is not None:
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(cfg.output_dir / "per_well_metrics.csv", index=False)
        (cfg.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True)
        )
        (cfg.output_dir / "config.json").write_text(
            json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()}, indent=2)
        )
    return {"per_well": df, "summary": summary}


def _parse_args() -> EvalConfig:
    p = argparse.ArgumentParser(description="Evaluate plane_solver on train holdout")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--n-wells", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/plane_solver_eval"))
    p.add_argument(
        "--modes",
        nargs="+",
        default=["constant", "linear", "linear_typewell"],
    )
    p.add_argument("--slope-window", type=int, default=600)
    p.add_argument("--slope-min-window", type=int, default=100)
    p.add_argument("--offset-radius", type=float, default=30.0)
    p.add_argument("--offset-grid", type=int, default=301)
    args = p.parse_args()
    return EvalConfig(
        data_dir=args.data_dir,
        n_wells=args.n_wells,
        seed=args.seed,
        output_dir=args.output_dir,
        modes=tuple(args.modes),
        slope_window=args.slope_window,
        slope_min_window=args.slope_min_window,
        offset_search_radius_ft=args.offset_radius,
        offset_grid_size=args.offset_grid,
    )


def _print_summary(summary: dict) -> None:
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    cfg = _parse_args()
    out = evaluate(cfg)
    _print_summary(out["summary"])
