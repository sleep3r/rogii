from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd


DIRECT_SOLVER_CANDIDATES: dict[str, str] = {
    "stage1_raw": "submission_direct_stage1_raw.csv",
    "stage12_raw": "submission_direct_stage12_raw.csv",
    "cem_raw": "submission_direct_cem_raw.csv",
    "cem_top_median": "submission_direct_cem_top_median.csv",
    "geo_consensus": "submission_direct_geo_consensus.csv",
    "geo_tailfit": "submission_direct_geo_tailfit.csv",
    "linear_tailfit": "submission_direct_linear_tailfit.csv",
    "crosswell_md_raw": "submission_direct_crosswell_md_raw.csv",
    "crosswell_z_raw": "submission_direct_crosswell_z_raw.csv",
    "crosswell_median": "submission_direct_crosswell_median.csv",
    "cem_over_crosswell_raw": "submission_direct_cem_over_crosswell_raw.csv",
    "cem_over_crosswell_top_median": "submission_direct_cem_over_crosswell_top_median.csv",
}
VIRTUAL_GEOLOGIST_CANDIDATES: tuple[str, ...] = (
    "vg_best_tvt",
    "vg_mean_tvt",
    "vg_p10_tvt",
    "vg_p50_tvt",
    "vg_p90_tvt",
    "vg_top1_tvt",
    "vg_top2_tvt",
    "vg_top3_tvt",
    "vg_top4_tvt",
    "vg_top5_tvt",
)

PRIVILEGED_CANDIDATES: set[str] = {
    "geo_teacher_tvt",
    "stage1_raw",
    "stage12_raw",
    "geo_consensus",
    "geo_tailfit",
    "cem_raw",
    "cem_top_median",
    "crosswell_md_raw",
    "crosswell_z_raw",
    "crosswell_median",
    "cem_over_crosswell_raw",
    "cem_over_crosswell_top_median",
}
ABSOLUTE_CANDIDATE_COLUMNS: set[str] = {
    "schema10_oof_raw",
    "schema10_plus_student_raw",
    "geo_student_v0_tvt",
    "last_known_tvt",
    "last_known",
    "flat_last_known",
    "flat_tvt",
    "tail_slope_baseline",
    "geo_teacher_tvt",
    *VIRTUAL_GEOLOGIST_CANDIDATES,
    *DIRECT_SOLVER_CANDIDATES.keys(),
}

REQUESTED_BUT_MISSING: tuple[str, ...] = (
    "schema10_oof_pp",
    "schema15_oof_raw",
    "schema15_oof_pp",
    "kg_pf_ancc_tvt",
    "kg_pf_z_tvt",
    "kg_dtw_best_tvt",
    "kg_dtw_mean_tvt",
    "kg_dwt_best_tvt",
    "kg_dwt_mean_tvt",
    "kg_beam_tvt",
    "kg_ncc_tvt",
    "public_family_path",
)


def _rmse(pred: np.ndarray | pd.Series, true: np.ndarray | pd.Series) -> float:
    pred_arr = np.asarray(pred, dtype=float)
    true_arr = np.asarray(true, dtype=float)
    mask = np.isfinite(pred_arr) & np.isfinite(true_arr)
    if not np.any(mask):
        return float("nan")
    err = pred_arr[mask] - true_arr[mask]
    return float(np.sqrt(np.mean(err * err)))


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes:02d}:{sec:02d}"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{sec:02d}"


def _log(message: str, **fields: Any) -> None:
    suffix = " ".join(f"{key}={value}" for key, value in fields.items())
    print(f"{message}{' | ' + suffix if suffix else ''}", flush=True)


def _read_submission_candidate(path: Path, column_name: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "id" not in frame.columns or "tvt" not in frame.columns:
        raise ValueError(f"{path} must contain id,tvt columns")
    return frame[["id", "tvt"]].rename(columns={"tvt": column_name})


def _split_id(ids: pd.Series) -> tuple[pd.Series, pd.Series]:
    well = ids.astype(str).str.rsplit("_", n=1).str[0]
    row = ids.astype(str).str.rsplit("_", n=1).str[1].astype(int)
    return well, row


def _markdown_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return "" if not np.isfinite(value) else f"{value:.6f}"
    if isinstance(value, np.floating):
        value = float(value)
        return "" if not np.isfinite(value) else f"{value:.6f}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def _markdown_table(frame: pd.DataFrame, *, max_rows: int | None = None) -> str:
    visible = frame.head(max_rows) if max_rows is not None else frame
    if visible.empty:
        return "_empty_"
    headers = [str(column) for column in visible.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in visible.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_markdown_cell(value) for value in row) + " |")
    return "\n".join(lines)


def load_base_candidates(
    *,
    integration_oof: Path,
    surface_student_oof: Path,
    direct_solver_dir: Path,
    virtual_geologist_oof: Path | None,
) -> pd.DataFrame:
    integ = pd.read_parquet(integration_oof)
    base = integ[
        [
            "id",
            "well",
            "row_index",
            "tvt_true_schema10",
            "schema10_oof_raw",
            "geo_student_tvt",
            "schema10_plus_student_raw",
        ]
    ].copy()
    base = base.rename(
        columns={
            "well": "well_id",
            "row_index": "row_idx",
            "tvt_true_schema10": "TVT",
            "geo_student_tvt": "geo_student_v0_tvt",
        }
    )
    student = pd.read_parquet(surface_student_oof)
    student_keep = [
        "id",
        "last_known_tvt",
        "flat_tvt",
        "geo_teacher_tvt",
        "geo_student_delta_last",
        "geo_student_minus_flat",
    ]
    missing_student = [column for column in student_keep if column not in student.columns]
    if missing_student:
        raise ValueError(f"Surface student OOF missing required columns: {missing_student}")
    base = base.merge(student[student_keep], on="id", how="left")
    base["last_known"] = base["last_known_tvt"]
    base["flat_last_known"] = base["last_known_tvt"]
    base["geo_student_minus_schema10"] = base["geo_student_v0_tvt"] - base["schema10_oof_raw"]

    for candidate, filename in DIRECT_SOLVER_CANDIDATES.items():
        path = direct_solver_dir / filename
        if path.exists():
            base = base.merge(_read_submission_candidate(path, candidate), on="id", how="left")
    if virtual_geologist_oof is not None and virtual_geologist_oof.exists():
        if virtual_geologist_oof.suffix == ".parquet":
            vg = pd.read_parquet(virtual_geologist_oof)
        else:
            vg = pd.read_csv(virtual_geologist_oof)
        keep = ["id", *[column for column in VIRTUAL_GEOLOGIST_CANDIDATES if column in vg.columns]]
        if len(keep) > 1:
            base = base.merge(vg[keep], on="id", how="left")
    return base


def add_tail_slope_baseline(frame: pd.DataFrame, data_dir: Path, tail_rows: int = 384) -> pd.DataFrame:
    out = frame.copy()
    out["tail_slope_baseline"] = np.nan
    train_dir = data_dir / "train"
    by_well = {well: idx for well, idx in out.groupby("well_id").groups.items()}
    for path in sorted(train_dir.glob("*__horizontal_well.csv")):
        well = path.name.split("__", 1)[0]
        if well not in by_well:
            continue
        df = pd.read_csv(path, usecols=lambda column: column in {"MD", "TVT_input"})
        md = pd.to_numeric(df["MD"], errors="coerce").to_numpy(float)
        tvt_input = pd.to_numeric(df["TVT_input"], errors="coerce").to_numpy(float)
        hidden_idx = out.loc[by_well[well], "row_idx"].to_numpy(int)
        known = np.flatnonzero(np.isfinite(tvt_input))
        if len(known) == 0 or len(hidden_idx) == 0:
            continue
        first_hidden = int(np.min(hidden_idx))
        known_before = known[known < first_hidden]
        if len(known_before) == 0:
            known_before = known
        last_idx = int(known_before[-1])
        tail = known_before[-int(tail_rows) :]
        if len(tail) >= 3:
            x = md[tail] - float(np.nanmedian(md[tail]))
            y = tvt_input[tail] - float(np.nanmedian(tvt_input[tail]))
            denom = float(np.dot(x, x))
            slope = float(np.dot(x, y) / denom) if denom > 1e-9 else 0.0
        else:
            slope = 0.0
        pred = float(tvt_input[last_idx]) + slope * (md[hidden_idx] - md[last_idx])
        out.loc[by_well[well], "tail_slope_baseline"] = pred
    return out


def candidate_columns(frame: pd.DataFrame, *, include_privileged: bool) -> list[str]:
    columns = [
        column
        for column in frame.columns
        if column in ABSOLUTE_CANDIDATE_COLUMNS
        and pd.api.types.is_numeric_dtype(frame[column])
        and np.isfinite(frame[column].to_numpy(float)).any()
    ]
    if not include_privileged:
        columns = [column for column in columns if column not in PRIVILEGED_CANDIDATES]
    return columns


def individual_scores(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    y = frame["TVT"].to_numpy(float)
    for column in columns:
        pred = frame[column].to_numpy(float)
        valid = np.isfinite(pred) & np.isfinite(y)
        if not valid.any():
            continue
        rows.append(
            {
                "candidate": column,
                "coverage": float(valid.mean()),
                "rmse": _rmse(pred, y),
                "mae": float(np.nanmean(np.abs(pred[valid] - y[valid]))),
                "bias": float(np.nanmean(pred[valid] - y[valid])),
            }
        )
    return pd.DataFrame(rows).sort_values(["rmse", "coverage"], ascending=[True, False])


def row_oracle(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    y = frame["TVT"].to_numpy(float)
    matrix = frame[columns].to_numpy(float)
    err = np.abs(matrix - y[:, None])
    err[~np.isfinite(err)] = np.inf
    best_idx = np.argmin(err, axis=1)
    pred = matrix[np.arange(len(frame)), best_idx]
    pred[~np.isfinite(pred)] = np.nan
    return pred


def _segment_ids(row_idx: np.ndarray, mode: str) -> np.ndarray:
    n = len(row_idx)
    if n == 0:
        return np.zeros(0, dtype=int)
    if mode == "thirds":
        return np.floor(np.linspace(0, 3, n, endpoint=False)).astype(int)
    match = re.fullmatch(r"chunk(\d+)", mode)
    if match:
        chunk = max(1, int(match.group(1)))
        order = np.argsort(row_idx)
        seg = np.zeros(n, dtype=int)
        seg[order] = np.arange(n) // chunk
        return seg
    if mode == "whole":
        return np.zeros(n, dtype=int)
    raise ValueError(f"Unknown segment mode: {mode}")


def segment_oracle(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    mode: str,
    top_k_average: int = 1,
) -> tuple[np.ndarray, pd.DataFrame]:
    pred = np.full(len(frame), np.nan, dtype=float)
    choices: list[dict[str, Any]] = []
    y_all = frame["TVT"].to_numpy(float)
    for well, idx in frame.groupby("well_id").groups.items():
        idx_arr = np.asarray(idx, dtype=int)
        row_idx = frame.loc[idx_arr, "row_idx"].to_numpy(int)
        seg_ids = _segment_ids(row_idx, mode)
        for seg in sorted(set(seg_ids.tolist())):
            seg_idx = idx_arr[seg_ids == seg]
            if len(seg_idx) == 0:
                continue
            scores = []
            for column in columns:
                score = _rmse(frame.loc[seg_idx, column], y_all[seg_idx])
                if np.isfinite(score):
                    scores.append((score, column))
            if not scores:
                continue
            scores.sort(key=lambda item: item[0])
            selected = [column for _score, column in scores[: max(1, int(top_k_average))]]
            values = frame.loc[seg_idx, selected].to_numpy(float)
            pred[seg_idx] = np.nanmean(values, axis=1)
            choices.append(
                {
                    "well_id": well,
                    "segment": int(seg),
                    "mode": mode,
                    "top_k_average": int(top_k_average),
                    "rows": int(len(seg_idx)),
                    "best_candidate": selected[0],
                    "selected": ",".join(selected),
                    "best_rmse": float(scores[0][0]),
                }
            )
    return pred, pd.DataFrame(choices)


def oracle_report(frame: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    choice_parts: list[pd.DataFrame] = []
    y = frame["TVT"].to_numpy(float)
    oracle_specs = [
        ("row_oracle", row_oracle(frame, columns), "row", 1),
    ]
    for mode in ("whole", "thirds", "chunk1024", "chunk512"):
        pred, choices = segment_oracle(frame, columns, mode=mode, top_k_average=1)
        oracle_specs.append((f"{mode}_oracle", pred, mode, 1))
        choice_parts.append(choices.assign(oracle=f"{mode}_oracle"))
    for top_k in (2, 3):
        pred, choices = segment_oracle(frame, columns, mode="thirds", top_k_average=top_k)
        oracle_specs.append((f"smooth_top{top_k}_thirds_oracle", pred, "thirds", top_k))
        choice_parts.append(choices.assign(oracle=f"smooth_top{top_k}_thirds_oracle"))
    for name, pred, mode, top_k in oracle_specs:
        rows.append(
            {
                "oracle": name,
                "mode": mode,
                "top_k_average": top_k,
                "rmse": _rmse(pred, y),
                "coverage": float((np.isfinite(pred) & np.isfinite(y)).mean()),
            }
        )
    choices = pd.concat(choice_parts, ignore_index=True) if choice_parts else pd.DataFrame()
    return pd.DataFrame(rows).sort_values("rmse"), choices


def decision(realistic_oracle_rmse: float) -> str:
    if realistic_oracle_rmse <= 9.0:
        return "STRONG GO: router may have a path to gold"
    if realistic_oracle_rmse <= 9.3:
        return "GO: router + richer validation may reach low-9"
    if realistic_oracle_rmse <= 9.5:
        return "WEAK GO: expand candidate space before NN"
    return "NO-GO: current production-like candidate space is not gold-capable"


def write_report(
    *,
    output: Path,
    frame: pd.DataFrame,
    scores: pd.DataFrame,
    oracle: pd.DataFrame,
    oracle_privileged: pd.DataFrame,
    columns: list[str],
    privileged_columns: list[str],
) -> None:
    realistic_row = oracle.loc[oracle["oracle"].eq("smooth_top2_thirds_oracle")]
    realistic = float(realistic_row.iloc[0]["rmse"]) if len(realistic_row) else float("nan")
    lines = [
        "# Candidate Bank + Oracle v0",
        "",
        f"- Rows: `{len(frame)}`",
        f"- Wells: `{frame['well_id'].nunique()}`",
        f"- Production-like candidates: `{len(columns)}`",
        f"- Privileged diagnostic candidates: `{len(privileged_columns)}`",
        f"- Realistic oracle used for gate: `smooth_top2_thirds_oracle = {realistic:.6f}`",
        f"- Decision: **{decision(realistic)}**",
        "",
        "## Missing Requested Row-Level Candidates",
        "",
        "These were requested but no full row-level OOF parquet exists yet. They need a feature-table candidate export pass before the oracle is complete:",
        "",
        ", ".join(f"`{name}`" for name in REQUESTED_BUT_MISSING),
        "",
        "## Candidate RMSE",
        "",
        _markdown_table(scores, max_rows=30),
        "",
        "## Production-Like Oracles",
        "",
        _markdown_table(oracle),
        "",
        "## Oracles Including Privileged Diagnostics",
        "",
        "Includes `geo_teacher_tvt` and current `direct_solver_tier1/train_eval` paths that use train-only surfaces or surface-derived energy in train-eval. This is diagnostic only.",
        "",
        _markdown_table(oracle_privileged),
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")


def build_candidate_bank(
    *,
    integration_oof: Path,
    surface_student_oof: Path,
    direct_solver_dir: Path,
    virtual_geologist_oof: Path | None,
    data_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    started = perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    _log(
        "Candidate bank start",
        integration_oof=integration_oof,
        surface_student_oof=surface_student_oof,
        direct_solver_dir=direct_solver_dir,
        virtual_geologist_oof=virtual_geologist_oof,
    )
    stage = perf_counter()
    frame = load_base_candidates(
        integration_oof=integration_oof,
        surface_student_oof=surface_student_oof,
        direct_solver_dir=direct_solver_dir,
        virtual_geologist_oof=virtual_geologist_oof,
    )
    _log(
        "Candidate bank loaded base candidates",
        rows=len(frame),
        wells=frame["well_id"].nunique(),
        columns=len(frame.columns),
        duration=_format_duration(perf_counter() - stage),
    )
    stage = perf_counter()
    frame = add_tail_slope_baseline(frame, data_dir)
    _log(
        "Candidate bank added tail slope baseline",
        duration=_format_duration(perf_counter() - stage),
    )
    stage = perf_counter()
    frame.to_parquet(output_dir / "oof_candidates.parquet", index=False)
    _log(
        "Candidate bank wrote oof candidates",
        path=output_dir / "oof_candidates.parquet",
        duration=_format_duration(perf_counter() - stage),
    )

    prod_columns = candidate_columns(frame, include_privileged=False)
    all_columns = candidate_columns(frame, include_privileged=True)
    privileged_columns = [column for column in all_columns if column not in prod_columns]
    _log(
        "Candidate bank selected columns",
        production=len(prod_columns),
        privileged=len(privileged_columns),
        total=len(all_columns),
    )
    stage = perf_counter()
    scores = individual_scores(frame, all_columns)
    _log(
        "Candidate bank scored individual candidates",
        candidates=len(scores),
        best=scores.iloc[0]["candidate"] if len(scores) else None,
        best_rmse=f"{float(scores.iloc[0]['rmse']):.6f}" if len(scores) else None,
        duration=_format_duration(perf_counter() - stage),
    )
    stage = perf_counter()
    oracle, choices = oracle_report(frame, prod_columns)
    _log(
        "Candidate bank scored production-like oracles",
        oracles=len(oracle),
        best=oracle.iloc[0]["oracle"] if len(oracle) else None,
        best_rmse=f"{float(oracle.iloc[0]['rmse']):.6f}" if len(oracle) else None,
        duration=_format_duration(perf_counter() - stage),
    )
    stage = perf_counter()
    oracle_priv, choices_priv = oracle_report(frame, all_columns)
    _log(
        "Candidate bank scored privileged oracles",
        oracles=len(oracle_priv),
        best=oracle_priv.iloc[0]["oracle"] if len(oracle_priv) else None,
        best_rmse=f"{float(oracle_priv.iloc[0]['rmse']):.6f}" if len(oracle_priv) else None,
        duration=_format_duration(perf_counter() - stage),
    )
    stage = perf_counter()
    scores.to_csv(output_dir / "candidate_scores.csv", index=False)
    oracle.to_csv(output_dir / "oracle_scores.csv", index=False)
    oracle_priv.to_csv(output_dir / "oracle_scores_with_privileged.csv", index=False)
    choices.to_parquet(output_dir / "oracle_choices.parquet", index=False)
    choices_priv.to_parquet(output_dir / "oracle_choices_with_privileged.parquet", index=False)
    metrics = {
        "rows": int(len(frame)),
        "wells": int(frame["well_id"].nunique()),
        "production_candidates": prod_columns,
        "privileged_candidates": privileged_columns,
        "requested_missing": list(REQUESTED_BUT_MISSING),
        "best_candidate": scores.iloc[0].to_dict() if len(scores) else None,
        "oracle": oracle.to_dict(orient="records"),
        "oracle_with_privileged": oracle_priv.to_dict(orient="records"),
    }
    (output_dir / "candidate_bank_metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str),
        encoding="utf-8",
    )
    write_report(
        output=output_dir / "candidate_bank_report.md",
        frame=frame,
        scores=scores,
        oracle=oracle,
        oracle_privileged=oracle_priv,
        columns=prod_columns,
        privileged_columns=privileged_columns,
    )
    _log(
        "Candidate bank wrote reports",
        output_dir=output_dir,
        duration=_format_duration(perf_counter() - stage),
        total_duration=_format_duration(perf_counter() - started),
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build row-level candidate bank and oracle report.")
    parser.add_argument(
        "--integration-oof",
        type=Path,
        default=Path("artifacts/surface_student_integration/integration_oof_predictions.parquet"),
    )
    parser.add_argument(
        "--surface-student-oof",
        type=Path,
        default=Path("artifacts/surface_student/oof_predictions.parquet"),
    )
    parser.add_argument(
        "--direct-solver-dir",
        type=Path,
        default=Path("artifacts/direct_solver_tier1/train_eval"),
    )
    parser.add_argument(
        "--virtual-geologist-oof",
        type=Path,
        default=Path("artifacts/virtual_geologist/virtual_geologist_oof.parquet"),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/candidate_bank"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = build_candidate_bank(
        integration_oof=args.integration_oof,
        surface_student_oof=args.surface_student_oof,
        direct_solver_dir=args.direct_solver_dir,
        virtual_geologist_oof=args.virtual_geologist_oof,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )
    realistic = next(
        item["rmse"]
        for item in metrics["oracle"]
        if item["oracle"] == "smooth_top2_thirds_oracle"
    )
    print(
        "Candidate bank complete | "
        f"rows={metrics['rows']} wells={metrics['wells']} "
        f"production_candidates={len(metrics['production_candidates'])} "
        f"realistic_oracle={realistic:.6f} decision={decision(float(realistic))}",
        flush=True,
    )


if __name__ == "__main__":
    main()
