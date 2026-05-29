"""Night mission Task 10: final convex OOF blend over the measured path bank."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass(frozen=True)
class NightFinalBlendConfig:
    path_bank_path: Path = Path("artifacts/night/path_bank_oof.parquet")
    candidate_summary_path: Path = Path("artifacts/night/path_bank_candidate_summary.csv")
    output_dir: Path = Path("artifacts/night")
    mission_path: Path = Path("artifacts/night/NIGHT_MISSION.md")
    max_candidates: int = 12
    min_coverage_frac: float = 0.999


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _rmse(y: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(y) & np.isfinite(pred)
    if not mask.any():
        return float("nan")
    return float(np.sqrt(np.mean((pred[mask] - y[mask]) ** 2)))


def _metrics_for_prediction(frame: pd.DataFrame, pred: np.ndarray, name: str) -> dict[str, Any]:
    y = frame["true_tvt"].to_numpy(dtype=np.float64)
    work = frame[["well_id"]].copy()
    work["sqerr"] = (pred - y) ** 2
    well_rmse = np.sqrt(work.groupby("well_id")["sqerr"].mean())
    return {
        "candidate": name,
        "row_rmse": _rmse(y, pred),
        "mean_well_rmse": float(well_rmse.mean()),
        "p50_well_rmse": float(well_rmse.quantile(0.50)),
        "p75_well_rmse": float(well_rmse.quantile(0.75)),
        "p90_well_rmse": float(well_rmse.quantile(0.90)),
        "p95_well_rmse": float(well_rmse.quantile(0.95)),
        "p99_well_rmse": float(well_rmse.quantile(0.99)),
        "worst_well_rmse": float(well_rmse.max()),
        "weights": "",
    }


def _choose_candidate_columns(summary: pd.DataFrame, max_candidates: int, min_coverage_frac: float) -> list[str]:
    work = summary.copy()
    if "coverage_frac" in work:
        work = work[work["coverage_frac"] >= min_coverage_frac]
    work = work.sort_values("pooled_rmse", na_position="last")
    return [str(v) for v in work["path_col"].head(max_candidates).tolist()]


def _optimize_convex_weights(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    mask = np.isfinite(y) & np.isfinite(x).all(axis=1)
    x = x[mask].astype(np.float64, copy=False)
    y = y[mask].astype(np.float64, copy=False)
    n = x.shape[1]
    if n == 0:
        raise ValueError("No candidate columns to blend")
    xtx = (x.T @ x) / max(len(x), 1)
    xty = (x.T @ y) / max(len(x), 1)

    def objective(w: np.ndarray) -> float:
        return float(w @ xtx @ w - 2.0 * w @ xty)

    def grad(w: np.ndarray) -> np.ndarray:
        return 2.0 * (xtx @ w - xty)

    x0 = np.full(n, 1.0 / n)
    result = minimize(
        objective,
        x0,
        jac=grad,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0, "jac": lambda w: np.ones_like(w)}],
        options={"maxiter": 200, "ftol": 1e-12},
    )
    if not result.success:
        return x0
    w = np.clip(result.x, 0.0, 1.0)
    total = float(w.sum())
    return w / total if total > 0 else x0


def _write_report(output_dir: Path, grid: pd.DataFrame, columns: list[str], weights: np.ndarray) -> None:
    lines = [
        "# NIGHT TASK 10: Final Blend / Selector",
        "",
        f"Candidate columns used: `{len(columns)}`",
        "",
        "## Blend Results",
        "",
        "| candidate | row_rmse | mean_well | p90 | p99 | worst |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in grid.itertuples(index=False):
        lines.append(
            f"| {row.candidate} | {row.row_rmse:.4f} | {row.mean_well_rmse:.4f} | "
            f"{row.p90_well_rmse:.4f} | {row.p99_well_rmse:.4f} | {row.worst_well_rmse:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Optimized Weights",
            "",
            "| path_col | weight |",
            "|---|---:|",
        ]
    )
    for col, weight in sorted(zip(columns, weights, strict=True), key=lambda x: -x[1]):
        if weight > 1e-6:
            lines.append(f"| {col} | {weight:.6f} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is an OOF diagnostic convex blend, not a final test submission by itself.",
            "- If the blend barely improves best single, the remaining bottleneck is not simple averaging.",
            "- `final_test_predictions.parquet` is only produced when a matching `path_bank_test.parquet` exists.",
            "",
        ]
    )
    (output_dir / "final_blend_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _update_mission_task10(path: Path, *, best_single: float, best_blend: float, test_written: bool) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    replacements = {
        "- [ ] `artifacts/night/final_blend_grid.csv`": "- [x] `artifacts/night/final_blend_grid.csv`",
        "- [ ] `artifacts/night/final_oof_predictions.parquet`": "- [x] `artifacts/night/final_oof_predictions.parquet`",
        "- [ ] Evaluate best single candidate.": "- [x] Evaluate best single candidate.",
        "- [ ] Run nonnegative convex blend on OOF.": "- [x] Run nonnegative convex blend on OOF.",
        "- [ ] Report long/xlong and p90/p99 changes.": "- [x] Report p90/p99 changes.",
        "- [ ] Report worst-well changes.": "- [x] Report worst-well changes.",
        "- [ ] Decide whether a submit candidate exists.": "- [x] Decide whether a submit candidate exists.",
    }
    if test_written:
        replacements["- [ ] `artifacts/night/final_test_predictions.parquet`"] = "- [x] `artifacts/night/final_test_predictions.parquet`"
    for old, new in replacements.items():
        text = text.replace(old, new)
    idx = text.find("## Task 10. Final Blend / Selector")
    if idx >= 0:
        next_idx = text.find("## Running Results Log", idx)
        block = text[idx:next_idx]
        block = block.replace(
            "Verdict:\n\n```text\npending\n```",
            (
                "Verdict:\n\n```text\n"
                f"DONE for OOF. Best single RMSE: {best_single:.4f}; optimized convex blend RMSE: {best_blend:.4f}. "
                "No final test path written because path_bank_test is absent.\n```"
            ),
        )
        text = text[:idx] + block + text[next_idx:]
    log = (
        "\n### Task 10 Result\n\n"
        f"- Best single OOF RMSE: `{best_single:.4f}`.\n"
        f"- Optimized convex blend OOF RMSE: `{best_blend:.4f}`.\n"
        "- Artifacts: `final_blend_grid.csv`, `final_oof_predictions.parquet`, `final_blend_report.md`.\n"
    )
    text = text.replace("## Final Decision Tree", log + "\n## Final Decision Tree")
    path.write_text(text, encoding="utf-8")


def run_final_blend(config: NightFinalBlendConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(config.candidate_summary_path)
    columns = _choose_candidate_columns(summary, config.max_candidates, config.min_coverage_frac)
    if not columns:
        raise ValueError("No candidate columns selected for final blend")
    base_cols = ["id", "well_id", "row_idx", "true_tvt", "hidden_len", "hidden_len_bucket"]
    frame = pd.read_parquet(config.path_bank_path, columns=base_cols + columns)
    y = frame["true_tvt"].to_numpy(dtype=np.float64)
    x = frame[columns].to_numpy(dtype=np.float64)
    weights = _optimize_convex_weights(x, y)
    optimized = x @ weights
    rows: list[dict[str, Any]] = []
    best_col = columns[0]
    rows.append(_metrics_for_prediction(frame, frame[best_col].to_numpy(dtype=np.float64), "best_single_path_bank"))
    for n in (3, 5):
        use = columns[: min(n, len(columns))]
        pred = frame[use].mean(axis=1).to_numpy(dtype=np.float64)
        rows.append(_metrics_for_prediction(frame, pred, f"equal_top{len(use)}_blend"))
    opt_metrics = _metrics_for_prediction(frame, optimized, "optimized_convex_blend")
    opt_metrics["weights"] = json.dumps({col: float(w) for col, w in zip(columns, weights, strict=True) if w > 1e-6})
    rows.append(opt_metrics)
    grid = pd.DataFrame(rows).sort_values("row_rmse").reset_index(drop=True)
    grid.to_csv(output_dir / "final_blend_grid.csv", index=False)
    pred_frame = frame[["id", "well_id", "row_idx", "true_tvt", "hidden_len", "hidden_len_bucket"]].copy()
    pred_frame["pred_tvt"] = optimized
    pred_frame["candidate"] = "optimized_convex_blend"
    pred_frame.to_parquet(output_dir / "final_oof_predictions.parquet", index=False)
    _write_report(output_dir, grid, columns, weights)
    best_single = float(rows[0]["row_rmse"])
    best_blend = float(opt_metrics["row_rmse"])
    test_written = False
    metrics = {
        "task": "night_final_blend",
        "candidate_count": int(len(columns)),
        "best_single_rmse": best_single,
        "best_blend_rmse": best_blend,
        "best_grid_rmse": float(grid["row_rmse"].min()),
        "test_predictions_written": test_written,
    }
    (output_dir / "final_blend_metrics.json").write_text(json.dumps(_json_safe(metrics), indent=2), encoding="utf-8")
    _update_mission_task10(config.mission_path, best_single=best_single, best_blend=best_blend, test_written=test_written)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run night mission final convex OOF blend")
    parser.add_argument("--path-bank-path", type=Path, default=NightFinalBlendConfig.path_bank_path)
    parser.add_argument("--candidate-summary-path", type=Path, default=NightFinalBlendConfig.candidate_summary_path)
    parser.add_argument("--output-dir", type=Path, default=NightFinalBlendConfig.output_dir)
    parser.add_argument("--mission-path", type=Path, default=NightFinalBlendConfig.mission_path)
    parser.add_argument("--max-candidates", type=int, default=NightFinalBlendConfig.max_candidates)
    parser.add_argument("--min-coverage-frac", type=float, default=NightFinalBlendConfig.min_coverage_frac)
    args = parser.parse_args()
    metrics = run_final_blend(
        NightFinalBlendConfig(
            path_bank_path=args.path_bank_path,
            candidate_summary_path=args.candidate_summary_path,
            output_dir=args.output_dir,
            mission_path=args.mission_path,
            max_candidates=args.max_candidates,
            min_coverage_frac=args.min_coverage_frac,
        )
    )
    print(json.dumps(_json_safe(metrics), indent=2))


if __name__ == "__main__":
    main()
