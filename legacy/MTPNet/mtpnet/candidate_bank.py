from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .residual_stack import ResidualStackConfig, _add_missing_prior_columns, _ensure_ids, load_training_frame
from .tail_audit import run_tail_audit_from_frames

LEVEL_SHIFTS: tuple[float, ...] = (-80.0, -60.0, -40.0, -20.0, 20.0, 40.0, 60.0, 80.0)
SLOPE_DELTAS: tuple[float, ...] = (-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0)
ROWS_PER_STEP = 32


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _hidden_only(frame: pd.DataFrame) -> pd.DataFrame:
    if "TVT_input" not in frame.columns:
        return frame.copy()
    return frame[pd.to_numeric(frame["TVT_input"], errors="coerce").isna()].copy()


def _candidate_rows(base: pd.DataFrame, *, candidate: str, values: pd.Series | np.ndarray) -> pd.DataFrame:
    out_cols = ["id", "well_id", "row_idx"]
    if "step" in base.columns:
        out_cols.append("step")
    for col in (
        "MD",
        "X",
        "Y",
        "Z",
        "TVT",
        "TVT_input",
        "GR",
        "b2_tvt",
        "base_tvt",
        "a_p50_tvt",
        "tail_class",
    ):
        if col in base.columns:
            out_cols.append(col)
    out = base[out_cols].copy()
    out["candidate"] = candidate
    out["pred_tvt"] = np.asarray(values, dtype=np.float64)
    return out


def _normalise_step_predictions(step_predictions: pd.DataFrame | None) -> pd.DataFrame:
    if step_predictions is None or step_predictions.empty:
        return pd.DataFrame()
    frame = step_predictions.copy()
    if "well_id" not in frame.columns:
        return pd.DataFrame()
    if "step" not in frame.columns and "compressed_step" in frame.columns:
        frame = frame.rename(columns={"compressed_step": "step"})
    if "step" not in frame.columns:
        return pd.DataFrame()
    keep = ["well_id", "step"]
    for column in ("top1_tvt", "dp_tvt"):
        if column in frame.columns:
            keep.append(column)
    out = frame[keep].copy()
    out["well_id"] = out["well_id"].astype(str)
    out["step"] = pd.to_numeric(out["step"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["step"])
    out["step"] = out["step"].astype(int)
    return out.drop_duplicates(["well_id", "step"])


def _normalise_traceback_candidates(traceback_candidates: pd.DataFrame | None) -> pd.DataFrame:
    if traceback_candidates is None or traceback_candidates.empty:
        return pd.DataFrame()
    frame = traceback_candidates.copy()
    required = {"id", "candidate", "pred_tvt"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    keep = ["id", "candidate", "pred_tvt"]
    for column in ("well_id", "row_idx", "step"):
        if column in frame.columns:
            keep.append(column)
    out = frame[keep].copy()
    out["id"] = out["id"].astype(str)
    out["candidate"] = out["candidate"].astype(str)
    out["pred_tvt"] = pd.to_numeric(out["pred_tvt"], errors="coerce")
    out = out.dropna(subset=["id", "candidate", "pred_tvt"])
    return out.drop_duplicates(["id", "candidate"]).reset_index(drop=True)


def _well_known_state(well: pd.DataFrame) -> tuple[float, int, float]:
    if "TVT_input" not in well.columns:
        anchor = pd.to_numeric(well["anchor_tvt"], errors="coerce")
        first = anchor.dropna().iloc[0] if anchor.notna().any() else 0.0
        return float(first), int(well["row_idx"].min()), 0.0
    known = well[pd.to_numeric(well["TVT_input"], errors="coerce").notna()]
    if known.empty:
        anchor = pd.to_numeric(well["anchor_tvt"], errors="coerce")
        first = anchor.dropna().iloc[0] if anchor.notna().any() else 0.0
        return float(first), int(well["row_idx"].min()), 0.0
    last = known.iloc[-1]
    last_tvt = float(last["TVT_input"])
    last_row = int(last["row_idx"])
    tail = known.tail(5)
    if len(tail) >= 2:
        slope = float(
            np.polyfit(
                pd.to_numeric(tail["row_idx"], errors="coerce").to_numpy(dtype=np.float64),
                pd.to_numeric(tail["TVT_input"], errors="coerce").to_numpy(dtype=np.float64),
                1,
            )[0]
        )
    else:
        slope = 0.0
    return last_tvt, last_row, slope


def build_candidate_bank_from_frames(
    hidden_rows: pd.DataFrame,
    *,
    residual_predictions: pd.DataFrame | None = None,
    step_predictions: pd.DataFrame | None = None,
    traceback_candidates: pd.DataFrame | None = None,
    k_offset_predictions: pd.DataFrame | None = None,
    dtvt_state_predictions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    raw = _add_missing_prior_columns(_ensure_ids(hidden_rows))
    if "step" not in raw.columns:
        raw["step"] = (pd.to_numeric(raw["row_idx"], errors="coerce") // ROWS_PER_STEP).astype(int)
    hidden = _hidden_only(raw)
    parts: list[pd.DataFrame] = []
    anchors = {
        "b2": "b2_tvt",
        "base": "base_tvt",
        "a_p50": "a_p50_tvt",
    }
    for name, column in anchors.items():
        if column not in hidden.columns:
            continue
        values = pd.to_numeric(hidden[column], errors="coerce")
        if values.notna().any():
            parts.append(_candidate_rows(hidden, candidate=name, values=values))
            for shift in LEVEL_SHIFTS:
                parts.append(
                    _candidate_rows(
                        hidden,
                        candidate=f"{name}_shift{shift:+.0f}",
                        values=values + shift,
                    )
                )
                progress = hidden.groupby("well_id").cumcount() / hidden.groupby("well_id")["id"].transform("count").clip(lower=1)
                parts.append(
                    _candidate_rows(
                        hidden,
                        candidate=f"{name}_drift_end{shift:+.0f}",
                        values=values + shift * progress.to_numpy(dtype=np.float64),
                    )
                )
    if residual_predictions is not None and not residual_predictions.empty:
        residual = residual_predictions[["id", "pred_tvt"]].rename(
            columns={"pred_tvt": "_residual_stack_pred_tvt"}
        )
        merged = hidden.merge(residual, on="id", how="left")
        if merged["_residual_stack_pred_tvt"].notna().any():
            parts.append(
                _candidate_rows(
                    merged,
                    candidate="residual_stack_v0",
                    values=merged["_residual_stack_pred_tvt"],
                )
            )
    if k_offset_predictions is not None and not k_offset_predictions.empty:
        # K-segment per-well constant-offset candidate. Schema mirrors
        # residual_stack OOF (id, pred_tvt) so plumbing stays uniform.
        k_off = k_offset_predictions[["id", "pred_tvt"]].rename(
            columns={"pred_tvt": "_k_offset_pred_tvt"}
        )
        merged = hidden.merge(k_off, on="id", how="left")
        if merged["_k_offset_pred_tvt"].notna().any():
            parts.append(
                _candidate_rows(
                    merged,
                    candidate="k_segment_offset_v0",
                    values=merged["_k_offset_pred_tvt"],
                )
            )
    if dtvt_state_predictions is not None and not dtvt_state_predictions.empty:
        # dTVT state-model candidate: per-row NN predictor trained with
        # local r-MSE + global cumsum-TVT MSE. Same (id, pred_tvt) schema.
        dtvt = dtvt_state_predictions[["id", "pred_tvt"]].rename(
            columns={"pred_tvt": "_dtvt_state_pred_tvt"}
        )
        merged = hidden.merge(dtvt, on="id", how="left")
        if merged["_dtvt_state_pred_tvt"].notna().any():
            parts.append(
                _candidate_rows(
                    merged,
                    candidate="dtvt_state_model_v0",
                    values=merged["_dtvt_state_pred_tvt"],
                )
            )
    step_preds = _normalise_step_predictions(step_predictions)
    traceback_preds = _normalise_traceback_candidates(traceback_candidates)
    if not step_preds.empty:
        merged_steps = hidden.merge(step_preds, on=["well_id", "step"], how="left")
        for column, candidate in (("top1_tvt", "softseg_top1"), ("dp_tvt", "softseg_dp")):
            if column in merged_steps.columns:
                values = pd.to_numeric(merged_steps[column], errors="coerce")
                if values.notna().any():
                    parts.append(
                        _candidate_rows(
                            merged_steps.loc[values.notna()],
                            candidate=candidate,
                            values=values.loc[values.notna()],
                        )
                    )
    if not traceback_preds.empty:
        merged_tb = hidden.merge(
            traceback_preds[["id", "candidate", "pred_tvt"]],
            on="id",
            how="inner",
            suffixes=("", "_traceback"),
        )
        if not merged_tb.empty:
            for candidate, group in merged_tb.groupby("candidate", sort=True):
                values = pd.to_numeric(group["pred_tvt"], errors="coerce")
                valid = values.notna()
                if valid.any():
                    parts.append(
                        _candidate_rows(
                            group.loc[valid],
                            candidate=str(candidate),
                            values=values.loc[valid],
                        )
                    )
    for well_id, well in raw.groupby("well_id", sort=True):
        well_hidden = hidden[hidden["well_id"].astype(str) == str(well_id)]
        if well_hidden.empty:
            continue
        last_tvt, last_row, slope = _well_known_state(well)
        row_delta = pd.to_numeric(well_hidden["row_idx"], errors="coerce") - last_row
        for delta in SLOPE_DELTAS:
            slope_per_row = slope + delta / ROWS_PER_STEP
            values = last_tvt + row_delta.to_numpy(dtype=np.float64) * slope_per_row
            parts.append(_candidate_rows(well_hidden, candidate=f"slope{delta:+.0f}", values=values))
    if not parts:
        raise ValueError("Candidate bank has no available candidate paths")
    bank = pd.concat(parts, ignore_index=True)
    return bank.reset_index(drop=True)


def _rmse_pair(pred: pd.Series | np.ndarray, true: pd.Series | np.ndarray) -> float:
    pred_arr = np.asarray(pred, dtype=np.float64)
    true_arr = np.asarray(true, dtype=np.float64)
    mask = np.isfinite(pred_arr) & np.isfinite(true_arr)
    if not mask.any():
        return float("nan")
    return float(np.sqrt(np.mean(np.square(pred_arr[mask] - true_arr[mask]))))


def _format_float(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "nan"
        return f"{float(value):.4f}"
    return str(value)


def _markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 30) -> str:
    if frame.empty:
        return "(empty)"
    clipped = frame.loc[:, [col for col in columns if col in frame.columns]].head(max_rows)
    lines = [
        "| " + " | ".join(clipped.columns) + " |",
        "| " + " | ".join(["---"] * len(clipped.columns)) + " |",
    ]
    for row in clipped.itertuples(index=False):
        lines.append("| " + " | ".join(_format_float(value) for value in row) + " |")
    return "\n".join(lines)


def _candidate_summary_from_wide(well_oracle: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in sorted(col for col in well_oracle.columns if col.startswith("rmse__")):
        candidate = column.removeprefix("rmse__")
        values = pd.to_numeric(well_oracle[column], errors="coerce")
        rows.append(
            {
                "candidate": candidate,
                "mean_well_rmse": float(values.mean()),
                "p50_well_rmse": float(values.quantile(0.50)),
                "p90_well_rmse": float(values.quantile(0.90)),
                "p95_well_rmse": float(values.quantile(0.95)),
                "worst_well_rmse": float(values.max()),
                "wells_best_oracle": int(
                    (well_oracle["best_candidate_oracle_name"].astype(str) == candidate).sum()
                ),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["mean_well_rmse", "candidate"])


def _candidate_bank_report(
    output_dir: Path,
    *,
    metrics: dict[str, Any],
    well_oracle: pd.DataFrame,
    candidate_summary: pd.DataFrame,
) -> None:
    worst_cols = [
        "well_id",
        "rmse_b2",
        "rmse_primary_candidate",
        "best_candidate_oracle_rmse",
        "best_candidate_oracle_name",
        "tail_class",
        "hidden_rows",
        "all_candidates_bad",
    ]
    candidate_cols = [
        "candidate",
        "mean_well_rmse",
        "p95_well_rmse",
        "worst_well_rmse",
        "wells_best_oracle",
    ]
    lines = [
        "CANDIDATE_BANK_STREAMING_ORACLE_REPORT",
        "",
        "summary:",
        json.dumps(_json_safe(metrics), indent=2),
        "",
        "top worst wells:",
        _markdown_table(
            well_oracle.sort_values("rmse_b2", ascending=False),
            worst_cols,
            max_rows=40,
        ),
        "",
        "candidate summary:",
        _markdown_table(candidate_summary, candidate_cols, max_rows=80),
    ]
    (output_dir / "candidate_bank_oracle_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run_candidate_bank_oracle_from_frames(
    hidden_rows: pd.DataFrame,
    *,
    output_dir: str | Path,
    residual_predictions: pd.DataFrame | None = None,
    traceback_candidates: pd.DataFrame | None = None,
    primary_candidate: str | None = None,
) -> dict[str, Any]:
    """Evaluate expanded candidate-bank oracle without materializing a full long table.

    This keeps only one well's long candidate table in memory at a time, then writes
    wide per-well and per-candidate summaries. Candidate generation itself remains
    target-free; true TVT is used only inside this diagnostic oracle evaluator.
    """
    if "TVT" not in hidden_rows.columns:
        raise ValueError("streaming oracle requires TVT for diagnostic evaluation")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw = _add_missing_prior_columns(_ensure_ids(hidden_rows))
    if "step" not in raw.columns:
        raw["step"] = (pd.to_numeric(raw["row_idx"], errors="coerce") // ROWS_PER_STEP).astype(int)
    residual = residual_predictions.copy() if residual_predictions is not None else None
    if residual is not None and not residual.empty:
        residual["well_id"] = residual["well_id"].astype(str)
    traceback = _normalise_traceback_candidates(traceback_candidates)
    if not traceback.empty and "well_id" not in traceback.columns:
        traceback = traceback.merge(raw[["id", "well_id"]].drop_duplicates("id"), on="id", how="left")
    if not traceback.empty:
        traceback["well_id"] = traceback["well_id"].astype(str)

    well_records: list[dict[str, Any]] = []
    all_candidates: set[str] = set()
    for well_id, well in raw.groupby("well_id", sort=True):
        well_residual = None
        if residual is not None and not residual.empty and "well_id" in residual.columns:
            well_residual = residual[residual["well_id"].astype(str) == str(well_id)]
        well_traceback = (
            traceback[traceback["well_id"].astype(str) == str(well_id)]
            if not traceback.empty and "well_id" in traceback.columns
            else None
        )
        bank = build_candidate_bank_from_frames(
            well,
            residual_predictions=well_residual,
            traceback_candidates=well_traceback,
        )
        hidden = _hidden_only(well)
        true_by_id = hidden.set_index("id")["TVT"]
        b2_rmse = _rmse_pair(hidden.get("b2_tvt", pd.Series(np.nan, index=hidden.index)), hidden["TVT"])
        base_rmse = _rmse_pair(hidden.get("base_tvt", pd.Series(np.nan, index=hidden.index)), hidden["TVT"])
        record: dict[str, Any] = {
            "well_id": str(well_id),
            "hidden_rows": int(len(hidden)),
            "tail_class": str(hidden["tail_class"].iloc[0]) if "tail_class" in hidden.columns else "unknown",
            "rmse_b2": b2_rmse,
            "rmse_base_schema10": base_rmse,
        }
        best_name: str | None = None
        best_rmse = float("inf")
        for candidate, group in bank.groupby("candidate", sort=True):
            all_candidates.add(str(candidate))
            aligned_true = true_by_id.reindex(group["id"].astype(str)).to_numpy(dtype=np.float64)
            rmse = _rmse_pair(group["pred_tvt"], aligned_true)
            record[f"rmse__{candidate}"] = rmse
            if np.isfinite(rmse) and rmse < best_rmse:
                best_name = str(candidate)
                best_rmse = float(rmse)
        record["best_candidate_oracle_name"] = best_name
        record["best_candidate_oracle_rmse"] = best_rmse
        if primary_candidate is not None:
            record["primary_candidate"] = primary_candidate
            record["rmse_primary_candidate"] = record.get(
                f"rmse__{primary_candidate}", float("nan")
            )
        else:
            record["primary_candidate"] = best_name
            record["rmse_primary_candidate"] = best_rmse
        record["B2_bad_and_oracle_good"] = bool(
            np.isfinite(b2_rmse) and np.isfinite(best_rmse) and b2_rmse >= 5.0 and best_rmse <= b2_rmse - 2.0
        )
        record["all_candidates_bad"] = bool(
            np.isfinite(best_rmse)
            and best_rmse >= 10.0
            and (not np.isfinite(b2_rmse) or best_rmse >= 0.75 * b2_rmse)
        )
        primary_rmse = float(record["rmse_primary_candidate"])
        record["candidate_exists_selector_fails"] = bool(
            record["B2_bad_and_oracle_good"]
            and np.isfinite(primary_rmse)
            and primary_rmse >= best_rmse + 2.0
        )
        well_records.append(record)

    well_oracle = pd.DataFrame(well_records)
    candidate_summary = _candidate_summary_from_wide(well_oracle)
    total_rows = pd.to_numeric(well_oracle["hidden_rows"], errors="coerce").fillna(0.0)
    b2_sq = np.square(pd.to_numeric(well_oracle["rmse_b2"], errors="coerce")) * total_rows
    oracle_sq = (
        np.square(pd.to_numeric(well_oracle["best_candidate_oracle_rmse"], errors="coerce"))
        * total_rows
    )
    primary_sq = (
        np.square(pd.to_numeric(well_oracle["rmse_primary_candidate"], errors="coerce"))
        * total_rows
    )
    total = max(float(total_rows.sum()), 1.0)
    tail_metrics: list[dict[str, Any]] = []
    for tail_class, group in well_oracle.groupby("tail_class", sort=True):
        tail_metrics.append(
            {
                "tail_class": str(tail_class),
                "wells": int(len(group)),
                "mean_b2_rmse": float(pd.to_numeric(group["rmse_b2"], errors="coerce").mean()),
                "mean_oracle_rmse": float(
                    pd.to_numeric(group["best_candidate_oracle_rmse"], errors="coerce").mean()
                ),
                "mean_primary_rmse": float(
                    pd.to_numeric(group["rmse_primary_candidate"], errors="coerce").mean()
                ),
                "mean_oracle_gain_vs_b2": float(
                    pd.to_numeric(group["rmse_b2"], errors="coerce").mean()
                    - pd.to_numeric(group["best_candidate_oracle_rmse"], errors="coerce").mean()
                ),
            }
        )
    metrics: dict[str, Any] = {
        "wells": int(well_oracle["well_id"].nunique()),
        "rows": int(total_rows.sum()),
        "candidates": int(len(all_candidates)),
        "primary_candidate": primary_candidate,
        "b2": {
            "row_rmse": float(np.sqrt(np.nansum(b2_sq) / total)),
            "mean_well_rmse": float(pd.to_numeric(well_oracle["rmse_b2"], errors="coerce").mean()),
            "p95_well_rmse": float(pd.to_numeric(well_oracle["rmse_b2"], errors="coerce").quantile(0.95)),
            "worst_well_rmse": float(pd.to_numeric(well_oracle["rmse_b2"], errors="coerce").max()),
        },
        "primary": {
            "row_rmse": float(np.sqrt(np.nansum(primary_sq) / total)),
            "mean_well_rmse": float(
                pd.to_numeric(well_oracle["rmse_primary_candidate"], errors="coerce").mean()
            ),
        },
        "oracle": {
            "row_rmse": float(np.sqrt(np.nansum(oracle_sq) / total)),
            "mean_well_rmse": float(
                pd.to_numeric(well_oracle["best_candidate_oracle_rmse"], errors="coerce").mean()
            ),
            "p95_well_rmse": float(
                pd.to_numeric(well_oracle["best_candidate_oracle_rmse"], errors="coerce").quantile(0.95)
            ),
            "worst_well_rmse": float(
                pd.to_numeric(well_oracle["best_candidate_oracle_rmse"], errors="coerce").max()
            ),
            "row_gain_vs_b2": float(np.sqrt(np.nansum(b2_sq) / total) - np.sqrt(np.nansum(oracle_sq) / total)),
            "mean_well_gain_vs_b2": float(
                pd.to_numeric(well_oracle["rmse_b2"], errors="coerce").mean()
                - pd.to_numeric(well_oracle["best_candidate_oracle_rmse"], errors="coerce").mean()
            ),
        },
        "diagnostics": {
            "b2_bad_and_oracle_good_wells": int(well_oracle["B2_bad_and_oracle_good"].sum()),
            "all_candidates_bad_wells": int(well_oracle["all_candidates_bad"].sum()),
            "selector_fail_wells": int(well_oracle["candidate_exists_selector_fails"].sum()),
        },
        "tail_class_metrics": tail_metrics,
    }
    well_oracle.to_csv(out / "candidate_bank_oracle_wells.csv", index=False)
    candidate_summary.to_csv(out / "candidate_bank_oracle_candidate_summary.csv", index=False)
    (out / "candidate_bank_oracle_metrics.json").write_text(
        json.dumps(_json_safe(metrics), indent=2), encoding="utf-8"
    )
    _candidate_bank_report(
        out,
        metrics=metrics,
        well_oracle=well_oracle,
        candidate_summary=candidate_summary,
    )
    return metrics


def run_candidate_bank_from_frames(
    hidden_rows: pd.DataFrame,
    *,
    output_dir: str | Path,
    residual_predictions: pd.DataFrame | None = None,
    traceback_candidates: pd.DataFrame | None = None,
    primary_candidate: str | None = None,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    bank = build_candidate_bank_from_frames(
        hidden_rows,
        residual_predictions=residual_predictions,
        traceback_candidates=traceback_candidates,
    )
    bank.to_parquet(out / "candidate_bank.parquet", index=False)
    if "TVT" in hidden_rows.columns and "b2_tvt" in hidden_rows.columns:
        metrics = run_tail_audit_from_frames(
            hidden_rows=_hidden_only(_add_missing_prior_columns(_ensure_ids(hidden_rows))),
            candidate_rows=bank,
            output_dir=out,
            primary_candidate=primary_candidate,
        )
    else:
        metrics = {
            "rows": int(len(bank)),
            "candidates": int(bank["candidate"].nunique()),
            "primary_candidate": primary_candidate,
        }
        (out / "candidate_bank_metrics.json").write_text(
            json.dumps(_json_safe(metrics), indent=2), encoding="utf-8"
        )
    return metrics


def run_candidate_bank(
    *,
    data_dir: Path,
    output_dir: Path,
    residual_predictions_path: Path | None,
    traceback_candidates_path: Path | None,
    primary_candidate: str | None,
    k_wells: int,
    streaming_oracle: bool = False,
) -> dict[str, Any]:
    frame = load_training_frame(
        ResidualStackConfig(data_dir=data_dir, k_wells=k_wells)
    )
    residual = (
        pd.read_parquet(residual_predictions_path)
        if residual_predictions_path is not None and Path(residual_predictions_path).exists()
        else None
    )
    traceback = (
        pd.read_parquet(traceback_candidates_path)
        if traceback_candidates_path is not None and Path(traceback_candidates_path).exists()
        else None
    )
    if streaming_oracle:
        metrics = run_candidate_bank_oracle_from_frames(
            frame,
            output_dir=output_dir,
            residual_predictions=residual,
            traceback_candidates=traceback,
            primary_candidate=primary_candidate,
        )
    else:
        metrics = run_candidate_bank_from_frames(
            frame,
            output_dir=output_dir,
            residual_predictions=residual,
            traceback_candidates=traceback,
            primary_candidate=primary_candidate,
        )
    print(json.dumps(_json_safe(metrics), indent=2))
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build expanded schema-safe candidate bank")
    parser.add_argument("--data-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/candidate_bank_v2"))
    parser.add_argument("--residual-predictions", type=Path)
    parser.add_argument("--traceback-candidates", type=Path)
    parser.add_argument("--primary-candidate", default="residual_stack_v0")
    parser.add_argument("--k-wells", type=int, default=-1)
    parser.add_argument(
        "--streaming-oracle",
        action="store_true",
        help="Evaluate full candidate-bank oracle without writing long candidate_bank.parquet",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_candidate_bank(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        residual_predictions_path=args.residual_predictions,
        traceback_candidates_path=args.traceback_candidates,
        primary_candidate=args.primary_candidate,
        k_wells=args.k_wells,
        streaming_oracle=args.streaming_oracle,
    )


if __name__ == "__main__":
    main()
