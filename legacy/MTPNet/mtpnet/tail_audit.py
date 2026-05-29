from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .stitch import _is_oracle_candidate, _load_hidden_rows, _load_run_config


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _rmse(values: pd.Series | np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(arr))))


def _mae(values: pd.Series | np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(np.abs(arr)))


def _nan_p95(values: pd.Series | np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, 95.0))


def _ensure_ids(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "id" not in out.columns:
        out["id"] = [
            f"{well_id}_{int(row_idx)}"
            for well_id, row_idx in zip(out["well_id"], out["row_idx"], strict=False)
        ]
    out["id"] = out["id"].astype(str)
    out["well_id"] = out["well_id"].astype(str)
    return out


def _hidden_from_candidate_rows(candidate_rows: pd.DataFrame) -> pd.DataFrame:
    required = {"id", "well_id", "row_idx", "step", "TVT", "GR", "base_tvt", "b2_tvt"}
    missing = required.difference(candidate_rows.columns)
    if missing:
        raise ValueError(
            "Cannot infer hidden rows from candidates; missing columns: "
            f"{sorted(missing)}. Pass --run-dir with config_resolved.yml instead."
        )
    return (
        _ensure_ids(candidate_rows[list(required)].copy())
        .drop_duplicates("id")
        .reset_index(drop=True)
    )


def _scored_candidate_rows(
    hidden_rows: pd.DataFrame, candidate_rows: pd.DataFrame
) -> pd.DataFrame:
    needed = {"TVT", "b2_tvt"}
    candidates = candidate_rows.copy()
    if not needed.issubset(candidates.columns):
        candidates = candidates.merge(
            hidden_rows[["id", "well_id", "TVT", "b2_tvt"]],
            on="id",
            how="left",
            suffixes=("", "_hidden"),
        )
        if "well_id_hidden" in candidates.columns:
            candidates["well_id"] = candidates["well_id"].where(
                candidates["well_id"].notna(), candidates["well_id_hidden"]
            )
            candidates = candidates.drop(columns=["well_id_hidden"])
    candidates["candidate"] = candidates["candidate"].astype(str)
    candidates["well_id"] = candidates["well_id"].astype(str)
    candidates["_candidate_sq_err"] = np.square(
        pd.to_numeric(candidates["pred_tvt"], errors="coerce")
        - pd.to_numeric(candidates["TVT"], errors="coerce")
    )
    candidates["_covered_b2_sq_err"] = np.square(
        pd.to_numeric(candidates["b2_tvt"], errors="coerce")
        - pd.to_numeric(candidates["TVT"], errors="coerce")
    )
    return candidates


def _per_well_candidate_metrics(
    hidden_rows: pd.DataFrame, candidate_rows: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scored = _scored_candidate_rows(hidden_rows, candidate_rows)
    hidden = hidden_rows.copy()
    hidden["_b2_sq_err"] = np.square(
        pd.to_numeric(hidden["b2_tvt"], errors="coerce")
        - pd.to_numeric(hidden["TVT"], errors="coerce")
    )
    hidden_totals = hidden.groupby("well_id").agg(
        hidden_rows=("id", "size"),
        b2_sq_total=("_b2_sq_err", "sum"),
    )
    grouped = scored.groupby(["candidate", "well_id"], sort=True).agg(
        covered_rows=("id", "nunique"),
        candidate_sq_total=("_candidate_sq_err", "sum"),
        covered_b2_sq_total=("_covered_b2_sq_err", "sum"),
    )
    per_well: dict[str, dict[str, Any]] = {
        well_id: {"well_id": well_id}
        for well_id in sorted(hidden_rows["well_id"].astype(str).unique())
    }
    summary_rows: list[dict[str, Any]] = []
    for candidate, local in grouped.groupby(level=0, sort=True):
        local = local.droplevel(0)
        joined = hidden_totals.join(local, how="left").fillna(
            {
                "covered_rows": 0.0,
                "candidate_sq_total": 0.0,
                "covered_b2_sq_total": 0.0,
            }
        )
        full_sq = (
            joined["b2_sq_total"]
            - joined["covered_b2_sq_total"]
            + joined["candidate_sq_total"]
        )
        well_rmse = np.sqrt(full_sq / joined["hidden_rows"].clip(lower=1))
        well_coverage = joined["covered_rows"] / joined["hidden_rows"].clip(lower=1)
        for well_id, value in well_rmse.items():
            per_well[str(well_id)][f"rmse__{candidate}"] = float(value)
            per_well[str(well_id)][f"coverage__{candidate}"] = float(
                well_coverage.loc[well_id]
            )
        summary_rows.append(
            {
                "candidate": candidate,
                "is_oracle": bool(_is_oracle_candidate(candidate)),
                "rows": int(joined["hidden_rows"].sum()),
                "wells": int(len(joined)),
                "rmse": float(
                    np.sqrt(full_sq.sum() / max(float(joined["hidden_rows"].sum()), 1.0))
                ),
                "mean_well_rmse": float(well_rmse.mean()),
                "p95_well_rmse": float(well_rmse.quantile(0.95)),
                "worst_well_rmse": float(well_rmse.max()),
                "coverage_frac": float(
                    joined["covered_rows"].sum()
                    / max(float(joined["hidden_rows"].sum()), 1.0)
                ),
            }
        )
    return pd.DataFrame(per_well.values()), pd.DataFrame(summary_rows)


def _gr_volatility(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return 0.0
    return _mae(np.diff(arr))


def _candidate_disagreement(
    hidden_rows: pd.DataFrame, candidate_rows: pd.DataFrame
) -> pd.Series:
    deployable = candidate_rows[
        ~candidate_rows["candidate"].astype(str).map(_is_oracle_candidate)
    ].copy()
    if deployable.empty:
        return pd.Series(dtype=np.float64)
    spread = deployable.groupby("id")["pred_tvt"].std(ddof=0).rename("_spread")
    merged = hidden_rows[["id", "well_id"]].merge(spread, on="id", how="left")
    return merged.groupby("well_id")["_spread"].mean()


def classify_tail_row(row: dict[str, Any] | pd.Series) -> str:
    item = dict(row)
    if bool(item.get("all_candidates_bad", False)):
        return "G_all_candidates_fail"
    if bool(item.get("candidate_exists_selector_fails", False)):
        return "H_candidate_exists_selector_fails"
    if bool(item.get("high_gr_nan_flag", False)) or bool(
        item.get("high_gr_volatility_flag", False)
    ):
        return "C_GR_missing_or_noisy"
    if bool(item.get("long_hidden_flag", False)) and bool(
        item.get("high_endpoint_error_flag", False)
    ):
        return "B_long_well_drift"
    if bool(item.get("level_shift_flag", False)):
        return "A_base_b2_level_shift"
    if bool(item.get("sharp_curvature_flag", False)):
        return "F_sharp_structural_change"
    if bool(item.get("high_candidate_disagreement_flag", False)):
        return "D_alignment_ambiguity"
    if bool(item.get("formation_prior_fail_flag", False)):
        return "E_formation_prior_fail"
    return "OK_or_mixed"


def _best_candidate_columns(
    row: pd.Series, candidate_summary: pd.DataFrame, *, include_oracle: bool
) -> tuple[str | None, float]:
    best_name: str | None = None
    best_value = float("inf")
    for candidate in candidate_summary["candidate"].astype(str):
        if not include_oracle and _is_oracle_candidate(candidate):
            continue
        value = row.get(f"rmse__{candidate}", float("nan"))
        if np.isfinite(value) and float(value) < best_value:
            best_name = candidate
            best_value = float(value)
    return best_name, best_value


def build_tail_audit(
    *,
    hidden_rows: pd.DataFrame,
    candidate_rows: pd.DataFrame,
    primary_candidate: str | None = None,
    top_n: int = 30,
) -> pd.DataFrame:
    hidden = _ensure_ids(hidden_rows)
    candidates = _ensure_ids(candidate_rows)
    if "candidate" not in candidates.columns or "pred_tvt" not in candidates.columns:
        raise ValueError("candidate rows must contain candidate and pred_tvt columns")
    if "base_tvt" not in hidden.columns:
        hidden["base_tvt"] = np.nan
    if "b2_tvt" not in hidden.columns:
        hidden["b2_tvt"] = np.nan

    grouped = hidden.sort_values(["well_id", "row_idx"]).groupby("well_id", sort=True)
    rows: list[dict[str, Any]] = []
    for well_id, group in grouped:
        true = pd.to_numeric(group["TVT"], errors="coerce")
        base_err = pd.to_numeric(group["base_tvt"], errors="coerce") - true
        b2_err = pd.to_numeric(group["b2_tvt"], errors="coerce") - true
        sorted_true = true.to_numpy(dtype=np.float64)
        diffs = np.diff(sorted_true[np.isfinite(sorted_true)])
        curv = np.diff(diffs) if diffs.size >= 2 else np.asarray([], dtype=np.float64)
        rows.append(
            {
                "well_id": str(well_id),
                "hidden_len": int(len(group)),
                "known_len": int(group["TVT_input"].notna().sum())
                if "TVT_input" in group.columns
                else np.nan,
                "GR_nan_frac": float(group["GR"].isna().mean())
                if "GR" in group.columns
                else np.nan,
                "GR_volatility": _gr_volatility(group["GR"])
                if "GR" in group.columns
                else np.nan,
                "tail_slope": float(diffs[-1]) if diffs.size else np.nan,
                "tail_curvature": float(curv[-1]) if curv.size else np.nan,
                "curvature_abs_p95": _nan_p95(np.abs(curv)) if curv.size else 0.0,
                "rmse_base_schema10": _rmse(base_err),
                "rmse_b2": _rmse(b2_err),
                "base_bias": float(np.nanmean(base_err)) if np.isfinite(base_err).any() else np.nan,
                "b2_bias": float(np.nanmean(b2_err)) if np.isfinite(b2_err).any() else np.nan,
                "base_endpoint_abs_err": float(abs(base_err.iloc[-1]))
                if np.isfinite(base_err.iloc[-1])
                else np.nan,
                "b2_endpoint_abs_err": float(abs(b2_err.iloc[-1]))
                if np.isfinite(b2_err.iloc[-1])
                else np.nan,
            }
        )
    audit = pd.DataFrame(rows)
    candidate_well, candidate_summary = _per_well_candidate_metrics(hidden, candidates)
    audit = audit.merge(candidate_well, on="well_id", how="left")

    disagreement = _candidate_disagreement(hidden, candidates)
    audit = audit.merge(
        disagreement.rename("candidate_disagreement").reset_index(),
        on="well_id",
        how="left",
    )
    audit["candidate_disagreement"] = audit["candidate_disagreement"].fillna(0.0)

    best_all: list[tuple[str | None, float]] = []
    best_deployable: list[tuple[str | None, float]] = []
    for _, row in audit.iterrows():
        best_all.append(_best_candidate_columns(row, candidate_summary, include_oracle=True))
        best_deployable.append(
            _best_candidate_columns(row, candidate_summary, include_oracle=False)
        )
    audit["best_candidate_oracle_name"] = [name for name, _ in best_all]
    audit["best_candidate_oracle_rmse"] = [value for _, value in best_all]
    audit["best_deployable_candidate"] = [name for name, _ in best_deployable]
    audit["best_deployable_rmse"] = [value for _, value in best_deployable]

    if primary_candidate is None:
        primary_candidate = (
            str(candidate_summary.loc[~candidate_summary["is_oracle"], "candidate"].iloc[0])
            if (~candidate_summary["is_oracle"]).any()
            else str(candidate_summary["candidate"].iloc[0])
        )
    primary_column = f"rmse__{primary_candidate}"
    audit["primary_candidate"] = primary_candidate
    audit["rmse_primary_candidate"] = audit.get(
        primary_column, audit["best_deployable_rmse"]
    )

    hidden_len_q75 = float(audit["hidden_len"].quantile(0.75))
    gr_nan_q75 = float(audit["GR_nan_frac"].fillna(0.0).quantile(0.75))
    gr_vol_q75 = float(audit["GR_volatility"].fillna(0.0).quantile(0.75))
    disagreement_q75 = float(audit["candidate_disagreement"].fillna(0.0).quantile(0.75))
    endpoint_q75 = float(audit["b2_endpoint_abs_err"].fillna(0.0).quantile(0.75))
    curvature_q90 = float(audit["curvature_abs_p95"].fillna(0.0).quantile(0.90))

    audit["long_hidden_flag"] = audit["hidden_len"] >= max(1.0, hidden_len_q75)
    audit["high_gr_nan_flag"] = (audit["GR_nan_frac"].fillna(0.0) > 0.0) & (
        audit["GR_nan_frac"].fillna(0.0) >= gr_nan_q75
    )
    audit["high_gr_volatility_flag"] = (audit["GR_volatility"].fillna(0.0) > 0.0) & (
        audit["GR_volatility"].fillna(0.0) >= gr_vol_q75
    )
    audit["high_candidate_disagreement_flag"] = (
        audit["candidate_disagreement"].fillna(0.0) > 0.0
    ) & (audit["candidate_disagreement"].fillna(0.0) >= disagreement_q75)
    audit["high_endpoint_error_flag"] = audit["b2_endpoint_abs_err"].fillna(0.0) >= max(
        5.0, endpoint_q75
    )
    audit["level_shift_flag"] = (
        audit["b2_bias"].abs().fillna(0.0) >= 5.0
    ) & (audit["base_bias"].abs().fillna(0.0) >= 5.0)
    audit["sharp_curvature_flag"] = (
        audit["curvature_abs_p95"].fillna(0.0) > 0.0
    ) & (audit["curvature_abs_p95"].fillna(0.0) >= max(5.0, curvature_q90))
    audit["formation_prior_fail_flag"] = (
        audit["rmse_base_schema10"].fillna(0.0) >= 12.0
    ) & (audit["rmse_b2"].fillna(0.0) >= 12.0)

    audit["B2_bad_and_oracle_good"] = (
        audit["rmse_b2"].fillna(np.inf) >= 5.0
    ) & (
        audit["best_candidate_oracle_rmse"].fillna(np.inf)
        <= audit["rmse_b2"].fillna(np.inf) - 2.0
    )
    audit["all_candidates_bad"] = (
        audit["best_candidate_oracle_rmse"].fillna(np.inf) >= 10.0
    ) & (
        audit["best_candidate_oracle_rmse"].fillna(np.inf)
        >= 0.75 * audit["rmse_b2"].fillna(np.inf)
    )
    audit["MTP_improves"] = (
        audit["best_deployable_rmse"].fillna(np.inf)
        <= audit["rmse_b2"].fillna(np.inf) - 0.5
    )
    audit["MTP_worsens"] = (
        audit["best_deployable_rmse"].fillna(np.inf)
        >= audit["rmse_b2"].fillna(np.inf) + 0.5
    )
    audit["candidate_exists_selector_fails"] = audit["B2_bad_and_oracle_good"] & (
        audit["rmse_primary_candidate"].fillna(np.inf)
        >= audit["best_candidate_oracle_rmse"].fillna(np.inf) + 2.0
    )
    audit["tail_class"] = audit.apply(classify_tail_row, axis=1)
    audit["rank_by_b2_rmse"] = audit["rmse_b2"].rank(
        method="first", ascending=False
    ).astype(int)
    audit["rank_by_oracle_headroom"] = (
        audit["rmse_b2"].fillna(np.inf)
        - audit["best_candidate_oracle_rmse"].fillna(np.inf)
    ).rank(method="first", ascending=False)
    audit = audit.sort_values(["rmse_b2", "hidden_len"], ascending=[False, False])
    if top_n > 0:
        audit["top_worst_flag"] = audit["rank_by_b2_rmse"] <= int(top_n)
    else:
        audit["top_worst_flag"] = False
    for column in audit.columns:
        if audit[column].dtype == bool:
            audit[column] = audit[column].astype(object)
    return audit.reset_index(drop=True)


def _candidate_summary_from_audit(
    audit: pd.DataFrame, candidate_rows: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate in sorted(candidate_rows["candidate"].astype(str).unique()):
        column = f"rmse__{candidate}"
        if column not in audit.columns:
            continue
        values = pd.to_numeric(audit[column], errors="coerce")
        rows.append(
            {
                "candidate": candidate,
                "is_oracle": bool(_is_oracle_candidate(candidate)),
                "mean_well_rmse": float(values.mean()),
                "p50_well_rmse": float(values.quantile(0.50)),
                "p90_well_rmse": float(values.quantile(0.90)),
                "p95_well_rmse": float(values.quantile(0.95)),
                "worst_well_rmse": float(values.max()),
                "wells_best_deployable": int(
                    (audit["best_deployable_candidate"].astype(str) == candidate).sum()
                ),
                "wells_best_oracle": int(
                    (audit["best_candidate_oracle_name"].astype(str) == candidate).sum()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["is_oracle", "mean_well_rmse"])


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


def _write_report(
    output_dir: Path,
    *,
    summary: dict[str, Any],
    audit: pd.DataFrame,
    candidate_summary: pd.DataFrame,
) -> Path:
    worst_cols = [
        "well_id",
        "rmse_b2",
        "rmse_base_schema10",
        "best_deployable_rmse",
        "best_candidate_oracle_rmse",
        "tail_class",
        "hidden_len",
        "GR_nan_frac",
        "candidate_disagreement",
    ]
    candidate_cols = [
        "candidate",
        "is_oracle",
        "mean_well_rmse",
        "p95_well_rmse",
        "worst_well_rmse",
        "wells_best_deployable",
        "wells_best_oracle",
    ]
    lines = [
        "GEOMTP_TAIL_AUDIT",
        "",
        "summary:",
        json.dumps(
            _json_safe(
                {key: value for key, value in summary.items() if key != "class_counts_frame"}
            ),
            indent=2,
        ),
        "",
        "class counts:",
        _markdown_table(
            summary["class_counts_frame"],
            ["tail_class", "wells"],
            max_rows=20,
        ),
        "",
        "top worst wells:",
        _markdown_table(audit, worst_cols, max_rows=30),
        "",
        "candidate summary:",
        _markdown_table(candidate_summary, candidate_cols, max_rows=50),
    ]
    path = output_dir / "tail_audit_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_tail_audit_from_frames(
    *,
    hidden_rows: pd.DataFrame,
    candidate_rows: pd.DataFrame,
    output_dir: str | Path,
    primary_candidate: str | None = None,
    top_n: int = 30,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    hidden = _ensure_ids(hidden_rows)
    candidates = _ensure_ids(candidate_rows)
    audit = build_tail_audit(
        hidden_rows=hidden,
        candidate_rows=candidates,
        primary_candidate=primary_candidate,
        top_n=top_n,
    )
    candidate_summary = _candidate_summary_from_audit(audit, candidates)
    class_counts = (
        audit.groupby("tail_class", as_index=False)
        .size()
        .rename(columns={"size": "wells"})
        .sort_values("wells", ascending=False)
    )
    top_worst = audit.head(top_n if top_n > 0 else len(audit))
    summary: dict[str, Any] = {
        "wells": int(audit["well_id"].nunique()),
        "rows": int(len(hidden)),
        "candidates": int(candidates["candidate"].nunique()),
        "primary_candidate": primary_candidate,
        "b2": {
            "mean_well_rmse": float(audit["rmse_b2"].mean()),
            "p50_well_rmse": float(audit["rmse_b2"].quantile(0.50)),
            "p90_well_rmse": float(audit["rmse_b2"].quantile(0.90)),
            "p95_well_rmse": float(audit["rmse_b2"].quantile(0.95)),
            "worst_well_rmse": float(audit["rmse_b2"].max()),
        },
        "diagnostics": {
            "b2_bad_and_oracle_good_wells": int(audit["B2_bad_and_oracle_good"].sum()),
            "all_candidates_bad_wells": int(audit["all_candidates_bad"].sum()),
            "selector_fail_wells": int(audit["candidate_exists_selector_fails"].sum()),
            "mtp_improves_wells": int(audit["MTP_improves"].sum()),
            "mtp_worsens_wells": int(audit["MTP_worsens"].sum()),
        },
        "top_worst_wells": top_worst[
            [
                "well_id",
                "rmse_b2",
                "best_deployable_rmse",
                "best_candidate_oracle_rmse",
                "tail_class",
            ]
        ].to_dict(orient="records"),
        "class_counts": class_counts.to_dict(orient="records"),
        "class_counts_frame": class_counts,
    }
    audit.to_csv(output_path / "well_tail_audit.csv", index=False)
    candidate_summary.to_csv(output_path / "tail_candidate_summary.csv", index=False)
    metrics = {key: value for key, value in summary.items() if key != "class_counts_frame"}
    (output_path / "tail_audit_metrics.json").write_text(
        json.dumps(_json_safe(metrics), indent=2), encoding="utf-8"
    )
    _write_report(output_path, summary=summary, audit=audit, candidate_summary=candidate_summary)
    return metrics


def _default_prediction_path(run_dir: Path) -> Path:
    for name in (
        "track_row_predictions.parquet",
        "stitch_row_predictions.parquet",
        "oof_track_row_predictions.parquet",
        "ranker_row_predictions.parquet",
    ):
        path = run_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No row prediction artifact found in {run_dir}; checked track/stitch/oof/ranker files"
    )


def run_tail_audit(
    *,
    run_dir: str | Path | None = None,
    candidates_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    primary_candidate: str | None = None,
    top_n: int = 30,
) -> dict[str, Any]:
    run_path = Path(run_dir) if run_dir is not None else None
    if run_path is None and candidates_path is None:
        raise ValueError("tail-audit requires --run-dir or --candidates")
    prediction_path = (
        Path(candidates_path)
        if candidates_path is not None
        else _default_prediction_path(run_path)
    )
    candidate_rows = pd.read_parquet(prediction_path)
    if run_path is not None and (run_path / "config_resolved.yml").exists():
        cfg = _load_run_config(run_path)
        well_ids = set(candidate_rows["well_id"].astype(str))
        hidden_rows = _load_hidden_rows(cfg, well_ids)
    else:
        hidden_rows = _hidden_from_candidate_rows(candidate_rows)
    out_path = (
        Path(output_dir)
        if output_dir is not None
        else (run_path / "tail_audit" if run_path is not None else prediction_path.parent / "tail_audit")
    )
    summary = run_tail_audit_from_frames(
        hidden_rows=hidden_rows,
        candidate_rows=candidate_rows,
        output_dir=out_path,
        primary_candidate=primary_candidate,
        top_n=top_n,
    )
    print(json.dumps(_json_safe(summary["diagnostics"]), indent=2), flush=True)
    return summary
