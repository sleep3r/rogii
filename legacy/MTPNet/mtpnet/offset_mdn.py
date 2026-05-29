"""Discrete offset posterior / MDN-style diagnostic.

The Kaggle "dTVT ~= -dZ + offset" observation makes the path formula simple,
but our fixed-grid experiment showed the deployable bottleneck is posterior
selection:

* fine offset-grid oracle: strong (~7.6 ft on the full train hidden audit);
* hard fold-safe selector: much weaker (~14-15 ft).

This module tests the next claim: do not choose one offset too early.  It
trains a fold-safe model that assigns a probability to every ``well × offset``
candidate, normalizes those scores into a per-well posterior, and evaluates
both hard top-1 and fuzzy posterior-mean paths.

It is intentionally discrete rather than a full PyTorch MDN: the goal is a
quick, leak-safe answer to "does a posterior over offset states help?"
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from .discrete_offset import (
    DiscreteOffsetConfig,
    _evaluate_selected_offsets,
    _load_wells,
    _materialise_predictions,
    _offset_oracle_from_rows,
    _select_offsets_from_candidate_rows,
    _train_oof_cost_selector,
    build_discrete_offset_dataset,
    parse_offset_grid,
)
from .residual_stack import make_group_folds


@dataclass(frozen=True)
class OffsetMDNConfig:
    data_dir: Path = Path("data/train")
    output_dir: Path = Path("artifacts/offset_mdn_v0")
    offset_grid: str = "-0.16:0.16:0.001"
    temperatures: str = "0.25,0.5,1.0,2.0"
    n_folds: int = 5
    seed: int = 42
    k_wells: int = -1
    iterations: int = 300
    learning_rate: float = 0.04
    depth: int = 5
    l2_leaf_reg: float = 8.0
    include_cost_posterior: bool = True


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def parse_temperatures(text: str) -> list[float]:
    temps = [float(part) for part in str(text).split(",") if part.strip()]
    if not temps:
        raise ValueError("temperature grid is empty")
    if any(temp <= 0 for temp in temps):
        raise ValueError("temperatures must be positive")
    return temps


def _stable_softmax(scores: np.ndarray) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.full(arr.shape, 1.0 / max(arr.size, 1), dtype=np.float64)
    safe = np.where(finite, arr, np.nanmin(arr[finite]) - 100.0)
    safe = safe - float(np.max(safe))
    exp = np.exp(np.clip(safe, -80.0, 80.0))
    denom = float(np.sum(exp))
    if denom <= 0.0 or not np.isfinite(denom):
        return np.full(arr.shape, 1.0 / max(arr.size, 1), dtype=np.float64)
    return exp / denom


def add_softmax_posterior(
    rows: pd.DataFrame,
    *,
    score_column: str,
    prob_column: str,
    temperature: float,
    lower_is_better: bool,
) -> pd.DataFrame:
    """Normalize candidate scores into ``P(offset | well)``."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if score_column not in rows.columns:
        raise ValueError(f"missing score column: {score_column}")
    out_parts: list[pd.DataFrame] = []
    for _, group in rows.groupby("well_id", sort=False):
        g = group.copy()
        score = pd.to_numeric(g[score_column], errors="coerce").to_numpy(dtype=np.float64)
        logits = (-score if lower_is_better else score) / float(temperature)
        g[prob_column] = _stable_softmax(logits)
        out_parts.append(g)
    if not out_parts:
        return rows.copy()
    return pd.concat(out_parts, ignore_index=True)


def selected_offsets_from_posterior(
    rows: pd.DataFrame,
    *,
    prob_column: str,
    mode: str,
) -> dict[str, float]:
    """Select a hard/fuzzy offset from a per-well posterior."""
    selected: dict[str, float] = {}
    for well_id, group in rows.groupby("well_id", sort=False):
        offsets = pd.to_numeric(group["offset"], errors="coerce").to_numpy(dtype=np.float64)
        probs = pd.to_numeric(group[prob_column], errors="coerce").to_numpy(dtype=np.float64)
        mask = np.isfinite(offsets) & np.isfinite(probs)
        if not mask.any():
            continue
        offsets = offsets[mask]
        probs = probs[mask]
        probs = probs / max(float(probs.sum()), 1e-12)
        if mode == "top1":
            selected[str(well_id)] = float(offsets[int(np.argmax(probs))])
        elif mode == "mean":
            selected[str(well_id)] = float(np.sum(offsets * probs))
        else:
            raise ValueError(f"unknown posterior mode: {mode}")
    return selected


def _train_oof_best_classifier(
    candidates: pd.DataFrame,
    feature_columns: list[str],
    *,
    config: OffsetMDNConfig,
) -> pd.DataFrame:
    pred_frames: list[pd.DataFrame] = []
    folds = sorted(candidates["fold"].astype(int).unique().tolist())
    for fold in folds:
        train_mask = candidates["fold"].to_numpy(dtype=int) != int(fold)
        valid_mask = candidates["fold"].to_numpy(dtype=int) == int(fold)
        if not train_mask.any() or not valid_mask.any():
            continue
        print(
            f"[offset-mdn] fold {fold + 1}/{len(folds)} "
            f"train_rows={int(train_mask.sum())} valid_rows={int(valid_mask.sum())}",
            file=sys.stderr,
            flush=True,
        )
        y_train = candidates.loc[train_mask, "is_best_offset"].astype(int)
        model = CatBoostClassifier(
            loss_function="Logloss",
            iterations=config.iterations,
            learning_rate=config.learning_rate,
            depth=config.depth,
            l2_leaf_reg=config.l2_leaf_reg,
            random_seed=config.seed,
            auto_class_weights="SqrtBalanced",
            allow_writing_files=False,
            verbose=False,
        )
        model.fit(candidates.loc[train_mask, feature_columns], y_train)
        valid = candidates.loc[valid_mask].copy()
        proba = model.predict_proba(valid[feature_columns])
        classes = [int(c) for c in model.classes_]
        pos_idx = classes.index(1) if 1 in classes else int(np.argmax(classes))
        valid["pred_best_prob"] = proba[:, pos_idx].astype(np.float64)
        pred_frames.append(valid)
    if not pred_frames:
        raise ValueError("No OOF MDN probability predictions were produced")
    return pd.concat(pred_frames, ignore_index=True)


def _posterior_rank_metrics(rows: pd.DataFrame, *, prob_column: str, label: str) -> dict[str, Any]:
    top_hits = {1: [], 3: [], 5: [], 10: []}
    oracle_ranks: list[int] = []
    oracle_probs: list[float] = []
    for _, group in rows.groupby("well_id", sort=False):
        if group.empty:
            continue
        ordered = group.sort_values(prob_column, ascending=False).reset_index(drop=True)
        oracle_idx = int(pd.to_numeric(ordered["candidate_mse"], errors="coerce").idxmin())
        # idxmin returns the dataframe index label; after reset_index this is rank-0.
        rank0 = int(oracle_idx)
        oracle_ranks.append(rank0 + 1)
        oracle_probs.append(float(ordered.loc[rank0, prob_column]))
        for k in top_hits:
            top_hits[k].append(float(rank0 < k))
    return {
        "label": label,
        "oracle_rank_mean": float(np.mean(oracle_ranks)) if oracle_ranks else float("nan"),
        "oracle_rank_median": float(np.median(oracle_ranks)) if oracle_ranks else float("nan"),
        "oracle_prob_mean": float(np.mean(oracle_probs)) if oracle_probs else float("nan"),
        "top1_oracle_rate": float(np.mean(top_hits[1])) if top_hits[1] else float("nan"),
        "top3_oracle_rate": float(np.mean(top_hits[3])) if top_hits[3] else float("nan"),
        "top5_oracle_rate": float(np.mean(top_hits[5])) if top_hits[5] else float("nan"),
        "top10_oracle_rate": float(np.mean(top_hits[10])) if top_hits[10] else float("nan"),
    }


def _posterior_topk_oracle_metrics(
    rows: pd.DataFrame,
    *,
    score_column: str,
    label: str,
    lower_is_better: bool,
    ks: tuple[int, ...] = (1, 3, 5, 10, 20),
) -> list[dict[str, Any]]:
    """Train-only ceiling if a decoder can choose among posterior top-k states."""
    out: list[dict[str, Any]] = []
    for k in ks:
        sse = 0.0
        n_rows = 0
        well_rmse: list[float] = []
        for _, group in rows.groupby("well_id", sort=False):
            ordered = group.sort_values(score_column, ascending=lower_is_better).head(k)
            if ordered.empty:
                continue
            best = ordered.loc[pd.to_numeric(ordered["candidate_mse"], errors="coerce").idxmin()]
            mse = float(best["candidate_mse"])
            rows_n = int(best.get("rows", 0))
            if rows_n <= 0 or not np.isfinite(mse):
                continue
            sse += mse * rows_n
            n_rows += rows_n
            well_rmse.append(float(np.sqrt(mse)))
        out.append(
            {
                "label": label,
                "k": int(k),
                "row_rmse": float(np.sqrt(sse / n_rows)) if n_rows else float("nan"),
                "mean_well_rmse": float(np.mean(well_rmse)) if well_rmse else float("nan"),
                "wells": int(len(well_rmse)),
                "rows": int(n_rows),
            }
        )
    return out


def _candidate_name(source: str, mode: str, temperature: float) -> str:
    temp = f"{temperature:g}".replace(".", "p")
    return f"{source}_{mode}_t{temp}"


def run_offset_mdn(config: OffsetMDNConfig) -> dict[str, Any]:
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(config.data_dir)
    paths = sorted(data_dir.glob("*__horizontal_well.csv"))
    if config.k_wells > 0:
        paths = paths[: config.k_wells]
    well_ids = [path.name.replace("__horizontal_well.csv", "") for path in paths]
    folds = make_group_folds(well_ids, n_folds=config.n_folds, seed=config.seed)
    fold_of_well = {str(w): idx for idx, (_, valid) in enumerate(folds) for w in valid}
    offset_grid = parse_offset_grid(config.offset_grid)
    temperatures = parse_temperatures(config.temperatures)
    wells = _load_wells(data_dir, k_wells=config.k_wells, fold_of_well=fold_of_well)
    print(
        f"[offset-mdn] wells={len(wells)} offsets={len(offset_grid)} "
        f"temps={temperatures}",
        file=sys.stderr,
        flush=True,
    )
    candidates, feature_columns = build_discrete_offset_dataset(
        wells, offset_grid=offset_grid, seed=config.seed, shuffled=False
    )
    candidates.to_parquet(out_dir / "offset_mdn_candidates.parquet", index=False)
    prob_scores = _train_oof_best_classifier(candidates, feature_columns, config=config)

    score_frames = [prob_scores]
    if config.include_cost_posterior:
        cost_config = DiscreteOffsetConfig(
            data_dir=config.data_dir,
            output_dir=config.output_dir,
            offset_grid=config.offset_grid,
            n_folds=config.n_folds,
            seed=config.seed,
            k_wells=config.k_wells,
            iterations=config.iterations,
            learning_rate=config.learning_rate,
            depth=config.depth,
            l2_leaf_reg=config.l2_leaf_reg,
            include_shuffled=False,
        )
        _, _, cost_scores = _train_oof_cost_selector(
            candidates, feature_columns, config=cost_config, shuffled_candidates=None
        )
        cost_scores = cost_scores[["well_id", "offset", "pred_log_mse"]].copy()
        prob_scores = prob_scores.merge(cost_scores, on=["well_id", "offset"], how="left")
    prob_scores.to_parquet(out_dir / "offset_mdn_oof_candidate_scores.parquet", index=False)

    oracle = _offset_oracle_from_rows(candidates)
    candidate_metrics: list[dict[str, Any]] = [
        _evaluate_selected_offsets(wells, oracle, candidate="grid_oracle"),
        _evaluate_selected_offsets(
            wells,
            _select_offsets_from_candidate_rows(
                prob_scores, score_column="pred_best_prob", minimize=False
            ),
            candidate="mdn_prob_top1_raw",
        ),
    ]
    rank_metrics: list[dict[str, Any]] = []
    topk_oracle_metrics: list[dict[str, Any]] = [
        *_posterior_topk_oracle_metrics(
            prob_scores,
            score_column="pred_best_prob",
            label="mdn_prob",
            lower_is_better=False,
        )
    ]
    if config.include_cost_posterior and "pred_log_mse" in prob_scores.columns:
        topk_oracle_metrics.extend(
            _posterior_topk_oracle_metrics(
                prob_scores,
                score_column="pred_log_mse",
                label="cost_posterior",
                lower_is_better=True,
            )
        )
    pred_frames: list[pd.DataFrame] = []

    for temp in temperatures:
        post = add_softmax_posterior(
            prob_scores,
            score_column="pred_best_prob",
            prob_column="prob_mdn",
            temperature=temp,
            lower_is_better=False,
        )
        rank_metrics.append(_posterior_rank_metrics(post, prob_column="prob_mdn", label=f"mdn_prob_t{temp:g}"))
        for mode in ("top1", "mean"):
            name = _candidate_name("mdn_prob", mode, temp)
            selected = selected_offsets_from_posterior(post, prob_column="prob_mdn", mode=mode)
            candidate_metrics.append(_evaluate_selected_offsets(wells, selected, candidate=name))
            pred_frames.append(_materialise_predictions(wells, selected, candidate=name))

        if config.include_cost_posterior and "pred_log_mse" in prob_scores.columns:
            cost_post = add_softmax_posterior(
                prob_scores,
                score_column="pred_log_mse",
                prob_column="prob_cost",
                temperature=temp,
                lower_is_better=True,
            )
            rank_metrics.append(
                _posterior_rank_metrics(cost_post, prob_column="prob_cost", label=f"cost_t{temp:g}")
            )
            for mode in ("top1", "mean"):
                name = _candidate_name("cost_posterior", mode, temp)
                selected = selected_offsets_from_posterior(cost_post, prob_column="prob_cost", mode=mode)
                candidate_metrics.append(_evaluate_selected_offsets(wells, selected, candidate=name))
                pred_frames.append(_materialise_predictions(wells, selected, candidate=name))

    if pred_frames:
        pd.concat(pred_frames, ignore_index=True).to_parquet(
            out_dir / "offset_mdn_oof_predictions.parquet", index=False
        )
    else:
        pd.DataFrame().to_parquet(out_dir / "offset_mdn_oof_predictions.parquet", index=False)

    deployable = [m for m in candidate_metrics if "oracle" not in str(m["candidate"])]
    metrics: dict[str, Any] = {
        "experiment": "offset_mdn_v0",
        "config": asdict(config),
        "wells": int(len(wells)),
        "candidate_rows": int(len(candidates)),
        "feature_columns": feature_columns,
        "temperatures": temperatures,
        "candidates": candidate_metrics,
        "posterior_rank_metrics": rank_metrics,
        "posterior_topk_oracle_metrics": topk_oracle_metrics,
        "grid_oracle": candidate_metrics[0],
        "best_deployable": min(deployable, key=lambda item: item.get("row_rmse", float("inf")))
        if deployable
        else {},
    }
    (out_dir / "offset_mdn_metrics.json").write_text(
        json.dumps(_json_safe(metrics), indent=2), encoding="utf-8"
    )
    _write_report(out_dir, metrics)
    return metrics


def _write_report(output_dir: Path, metrics: dict[str, Any]) -> None:
    lines = [
        "# OFFSET_MDN_V0",
        "",
        "Discrete posterior diagnostic for the `cumsum(-dZ + offset)` idea.",
        "This is a fuzzy/MDN-style test: every offset gets a fold-safe score,",
        "scores are normalized into `P(offset | well)`, then both top-1 and",
        "posterior-mean paths are evaluated.",
        "",
        "## candidate metrics",
        "",
        "| candidate | row RMSE | mean well | p95 | worst |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in metrics["candidates"]:
        lines.append(
            f"| {item['candidate']} | {item.get('row_rmse', float('nan')):.4f} | "
            f"{item.get('mean_well_rmse', float('nan')):.4f} | "
            f"{item.get('p95_well_rmse', float('nan')):.4f} | "
            f"{item.get('worst_well_rmse', float('nan')):.4f} |"
        )
    lines.extend(
        [
            "",
            "## posterior rank diagnostics",
            "",
            "| posterior | oracle top1 | top3 | top5 | top10 | mean oracle rank |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in metrics["posterior_rank_metrics"]:
        lines.append(
            f"| {item['label']} | {item.get('top1_oracle_rate', float('nan')):.3f} | "
            f"{item.get('top3_oracle_rate', float('nan')):.3f} | "
            f"{item.get('top5_oracle_rate', float('nan')):.3f} | "
            f"{item.get('top10_oracle_rate', float('nan')):.3f} | "
            f"{item.get('oracle_rank_mean', float('nan')):.2f} |"
        )
    lines.extend(
        [
            "",
            "## posterior top-k oracle ceiling",
            "",
            "Train-only diagnostic: if a future beam/DP decoder could choose the",
            "best true offset among the posterior top-k candidates, this is the",
            "ceiling available inside that posterior shortlist.",
            "",
            "| posterior | k | row RMSE | mean well |",
            "|---|---:|---:|---:|",
        ]
    )
    for item in metrics.get("posterior_topk_oracle_metrics", []):
        lines.append(
            f"| {item['label']} | {item['k']} | {item.get('row_rmse', float('nan')):.4f} | "
            f"{item.get('mean_well_rmse', float('nan')):.4f} |"
        )
    lines.extend(
        [
            "",
            "## decision hints",
            "",
            "- If posterior top-k captures oracle often but top-1 RMSE is poor,",
            "  a beam/DP RACFormer-style decoder is justified.",
            "- If posterior top-k is also poor, the available features do not",
            "  identify the offset state and a richer state model is needed.",
            "- `grid_oracle` remains train-only ceiling; deployable rows use only",
            "  `MD/X/Y/Z/GR/TVT_input` and typewell `TVT/GR` features.",
        ]
    )
    (output_dir / "offset_mdn_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run discrete offset posterior / MDN diagnostic")
    parser.add_argument("--data-dir", type=Path, default=OffsetMDNConfig.data_dir)
    parser.add_argument("--output-dir", type=Path, default=OffsetMDNConfig.output_dir)
    parser.add_argument("--offset-grid", type=str, default=OffsetMDNConfig.offset_grid)
    parser.add_argument("--temperatures", type=str, default=OffsetMDNConfig.temperatures)
    parser.add_argument("--n-folds", type=int, default=OffsetMDNConfig.n_folds)
    parser.add_argument("--seed", type=int, default=OffsetMDNConfig.seed)
    parser.add_argument("--k-wells", type=int, default=OffsetMDNConfig.k_wells)
    parser.add_argument("--iterations", type=int, default=OffsetMDNConfig.iterations)
    parser.add_argument("--learning-rate", type=float, default=OffsetMDNConfig.learning_rate)
    parser.add_argument("--depth", type=int, default=OffsetMDNConfig.depth)
    parser.add_argument("--l2-leaf-reg", type=float, default=OffsetMDNConfig.l2_leaf_reg)
    parser.add_argument("--no-cost-posterior", action="store_true")
    args = parser.parse_args(argv)
    metrics = run_offset_mdn(
        OffsetMDNConfig(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            offset_grid=args.offset_grid,
            temperatures=args.temperatures,
            n_folds=args.n_folds,
            seed=args.seed,
            k_wells=args.k_wells,
            iterations=args.iterations,
            learning_rate=args.learning_rate,
            depth=args.depth,
            l2_leaf_reg=args.l2_leaf_reg,
            include_cost_posterior=not args.no_cost_posterior,
        )
    )
    print(
        json.dumps(
            {
                "experiment": metrics["experiment"],
                "best_deployable": metrics.get("best_deployable", {}),
                "grid_oracle": metrics.get("grid_oracle", {}),
                "report": str(Path(args.output_dir) / "offset_mdn_report.md"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
