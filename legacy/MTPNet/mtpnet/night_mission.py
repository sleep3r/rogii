"""Night mission utilities for leakage-safe path-bank diagnostics."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PREDICTION_COLUMN_CANDIDATES = (
    "pred_tvt",
    "prediction",
    "tvt_pred",
    "pred",
)


@dataclass(frozen=True)
class NightScoreboardConfig:
    data_dir: Path = Path("data/train")
    artifacts_dir: Path = Path("artifacts")
    output_dir: Path = Path("artifacts/night")
    max_artifacts: int = 0
    max_file_mb: float = 512.0
    top_n_worst: int = 30


@dataclass(frozen=True)
class NightPathBankConfig:
    data_dir: Path = Path("data/train")
    output_dir: Path = Path("artifacts/night")
    min_rows_fraction: float = 0.95
    min_rows: int = 0
    max_candidates: int = 24
    exclude_oracle_candidates: bool = True


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


def _slug(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")
    return value[:180] or "artifact"


def _ensure_id(frame: pd.DataFrame, well_id: str | None = None) -> pd.Series:
    if "id" in frame.columns:
        return frame["id"].astype(str)
    if {"well_id", "row_idx"}.issubset(frame.columns):
        return frame["well_id"].astype(str) + "_" + pd.to_numeric(frame["row_idx"], errors="coerce").astype("Int64").astype(str)
    if well_id is None:
        raise ValueError("cannot build id without well_id/row_idx")
    return pd.Series([f"{well_id}_{idx}" for idx in range(len(frame))], index=frame.index, dtype=str)


def _hidden_len_bucket(hidden_len: int) -> str:
    h = int(hidden_len)
    if h < 2000:
        return "short"
    if h < 5000:
        return "medium"
    if h < 8000:
        return "long"
    return "xlong"


def load_hidden_truth(data_dir: Path) -> pd.DataFrame:
    """Load train hidden rows with truth and test-available context columns."""
    parts: list[pd.DataFrame] = []
    for path in sorted(Path(data_dir).glob("*__horizontal_well.csv")):
        well_id = path.name.replace("__horizontal_well.csv", "")
        frame = pd.read_csv(path)
        if "row_idx" not in frame.columns:
            frame["row_idx"] = np.arange(len(frame), dtype=np.int64)
        frame["well_id"] = well_id
        frame["id"] = _ensure_id(frame, well_id=well_id)
        tvt = pd.to_numeric(frame.get("TVT"), errors="coerce")
        tvt_input = pd.to_numeric(frame.get("TVT_input"), errors="coerce")
        hidden_mask = tvt.notna() & tvt_input.isna()
        hidden = frame.loc[hidden_mask].copy()
        if hidden.empty:
            continue
        hidden_len = int(len(hidden))
        hidden["TVT"] = tvt.loc[hidden.index].to_numpy(dtype=np.float64)
        hidden["hidden_len"] = hidden_len
        hidden["hidden_len_bucket"] = _hidden_len_bucket(hidden_len)
        hidden["gr_valid_frac"] = float(pd.to_numeric(hidden.get("GR"), errors="coerce").notna().mean()) if "GR" in hidden else np.nan
        keep = [
            "id",
            "well_id",
            "row_idx",
            "TVT",
            "hidden_len",
            "hidden_len_bucket",
            "gr_valid_frac",
        ]
        for col in ("MD", "X", "Y", "Z", "GR", "TVT_input"):
            if col in hidden.columns:
                keep.append(col)
        parts.append(hidden[keep])
    if not parts:
        raise ValueError(f"no hidden truth rows found under {data_dir}")
    truth = pd.concat(parts, ignore_index=True)
    truth["id"] = truth["id"].astype(str)
    truth["well_id"] = truth["well_id"].astype(str)
    truth["row_idx"] = pd.to_numeric(truth["row_idx"], errors="coerce").astype(int)
    return truth


def _prediction_columns(frame: pd.DataFrame) -> list[str]:
    return [col for col in PREDICTION_COLUMN_CANDIDATES if col in frame.columns]


def _experiment_name(path: Path, artifact_root: Path) -> str:
    try:
        rel = path.relative_to(artifact_root)
    except ValueError:
        rel = path
    parts = rel.parts
    if len(parts) >= 2:
        return "/".join(parts[:-1])
    return path.stem


def normalize_prediction_artifact(path: Path, *, truth: pd.DataFrame, artifact_root: Path) -> pd.DataFrame:
    """Normalize one row-wise prediction parquet and join train truth."""
    frame = pd.read_parquet(path)
    pred_cols = _prediction_columns(frame)
    if not pred_cols:
        return pd.DataFrame()
    if "id" not in frame.columns and not {"well_id", "row_idx"}.issubset(frame.columns):
        return pd.DataFrame()
    base = frame.copy()
    if "id" not in base.columns:
        base["id"] = _ensure_id(base)
    base["id"] = base["id"].astype(str)
    if "well_id" in base.columns:
        base["well_id"] = base["well_id"].astype(str)
    if "row_idx" in base.columns:
        base["row_idx"] = pd.to_numeric(base["row_idx"], errors="coerce")

    experiment = _experiment_name(path, artifact_root)
    outputs: list[pd.DataFrame] = []
    for pred_col in pred_cols:
        cols = ["id", pred_col]
        for col in ("well_id", "row_idx", "candidate", "fold"):
            if col in base.columns and col not in cols:
                cols.append(col)
        pred = base[cols].copy()
        pred = pred.rename(columns={pred_col: "pred_tvt"})
        pred["pred_tvt"] = pd.to_numeric(pred["pred_tvt"], errors="coerce")
        pred = pred.dropna(subset=["pred_tvt"])
        if pred.empty:
            continue
        if "candidate" not in pred.columns:
            suffix = "" if pred_col == "pred_tvt" else f":{pred_col}"
            pred["candidate"] = Path(path).stem + suffix
        pred["candidate"] = pred["candidate"].astype(str)
        pred["experiment"] = experiment
        pred["artifact_path"] = str(path)
        joined = pred.merge(
            truth[
                [
                    "id",
                    "well_id",
                    "row_idx",
                    "TVT",
                    "hidden_len",
                    "hidden_len_bucket",
                    "gr_valid_frac",
                ]
            ].rename(columns={"TVT": "true_tvt"}),
            on="id",
            how="inner",
            suffixes=("", "_truth"),
        )
        if joined.empty:
            continue
        if "well_id_truth" in joined.columns:
            joined["well_id"] = joined["well_id_truth"]
            joined = joined.drop(columns=["well_id_truth"])
        if "row_idx_truth" in joined.columns:
            joined["row_idx"] = joined["row_idx_truth"]
            joined = joined.drop(columns=["row_idx_truth"])
        outputs.append(
            joined[
                [
                    "experiment",
                    "candidate",
                    "artifact_path",
                    "id",
                    "well_id",
                    "row_idx",
                    "pred_tvt",
                    "true_tvt",
                    "hidden_len",
                    "hidden_len_bucket",
                    "gr_valid_frac",
                ]
                + (["fold"] if "fold" in joined.columns else [])
            ]
        )
    if not outputs:
        return pd.DataFrame()
    out = pd.concat(outputs, ignore_index=True)
    out["well_id"] = out["well_id"].astype(str)
    out["row_idx"] = pd.to_numeric(out["row_idx"], errors="coerce").astype(int)
    return out


def _rmse(err: pd.Series | np.ndarray) -> float:
    arr = np.asarray(err, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(arr * arr)))


def _markdown_table(frame: pd.DataFrame, *, floatfmt: str = ".4f") -> str:
    if frame.empty:
        return ""
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in frame.iterrows():
        values: list[str] = []
        for col in cols:
            value = row[col]
            if isinstance(value, float) or isinstance(value, np.floating):
                values.append(format(float(value), floatfmt))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def score_predictions(frame: pd.DataFrame, *, top_n: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame()
    rows: list[dict[str, Any]] = []
    worst_parts: list[pd.DataFrame] = []
    group_cols = ["experiment", "candidate"]
    for (experiment, candidate), group in frame.groupby(group_cols, sort=True):
        g = group.copy()
        if "hidden_len" not in g.columns:
            g["hidden_len"] = g.groupby("well_id")["well_id"].transform("size")
        if "hidden_len_bucket" not in g.columns:
            g["hidden_len_bucket"] = g["hidden_len"].map(_hidden_len_bucket)
        if "gr_valid_frac" not in g.columns:
            g["gr_valid_frac"] = np.nan
        g["err"] = pd.to_numeric(g["pred_tvt"], errors="coerce") - pd.to_numeric(g["true_tvt"], errors="coerce")
        g = g.dropna(subset=["err"])
        if g.empty:
            continue
        well = (
            g.groupby("well_id")
            .agg(
                rows=("err", "size"),
                rmse=("err", _rmse),
                hidden_len=("hidden_len", "max"),
                bucket=("hidden_len_bucket", "first"),
                gr_valid_frac=("gr_valid_frac", "first"),
            )
            .reset_index()
        )
        wr = well["rmse"].to_numpy(dtype=np.float64)
        bucket_metrics = {}
        for bucket, bg in g.groupby("hidden_len_bucket", sort=True):
            bucket_metrics[f"rmse_bucket_{bucket}"] = _rmse(bg["err"])
            bucket_metrics[f"rows_bucket_{bucket}"] = int(len(bg))
        row = {
            "experiment": experiment,
            "candidate": candidate,
            "rows": int(len(g)),
            "wells": int(well["well_id"].nunique()),
            "pooled_rmse": _rmse(g["err"]),
            "mean_well_rmse": float(np.mean(wr)),
            "p50_well_rmse": float(np.quantile(wr, 0.50)),
            "p75_well_rmse": float(np.quantile(wr, 0.75)),
            "p90_well_rmse": float(np.quantile(wr, 0.90)),
            "p99_well_rmse": float(np.quantile(wr, 0.99)),
            "worst_well_rmse": float(np.max(wr)),
            **bucket_metrics,
        }
        rows.append(row)
        worst = well.sort_values("rmse", ascending=False).head(top_n).copy()
        worst.insert(0, "candidate", candidate)
        worst.insert(0, "experiment", experiment)
        worst_parts.append(worst)
    scoreboard = pd.DataFrame(rows).sort_values("pooled_rmse", na_position="last").reset_index(drop=True)
    worst_df = pd.concat(worst_parts, ignore_index=True) if worst_parts else pd.DataFrame()
    return scoreboard, worst_df


def _scan_prediction_paths(artifacts_dir: Path) -> list[Path]:
    skip_parts = {"night"}
    paths: list[Path] = []
    for path in sorted(Path(artifacts_dir).glob("**/*.parquet")):
        if any(part in skip_parts for part in path.parts):
            continue
        name = path.name.lower()
        if any(token in name for token in ("chunk_predictions", "mode_dataset", "candidate_scores", "features")):
            continue
        paths.append(path)
    return paths


def _write_markdown_report(output_dir: Path, scoreboard: pd.DataFrame, worst: pd.DataFrame, *, scanned: int, loaded: int) -> None:
    def _markdown_table(frame: pd.DataFrame, *, floatfmt: str = ".4f") -> str:
        if frame.empty:
            return ""
        cols = list(frame.columns)
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
        for _, row in frame.iterrows():
            values: list[str] = []
            for col in cols:
                value = row[col]
                if isinstance(value, float) or isinstance(value, np.floating):
                    values.append(format(float(value), floatfmt))
                else:
                    values.append(str(value))
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    lines = [
        "# NIGHT TASK 1: Unified OOF Scoreboard",
        "",
        f"Scanned parquet artifacts: `{scanned}`",
        f"Loaded row-wise prediction artifacts: `{loaded}`",
        "",
        "## Top Candidates By Pooled RMSE",
        "",
    ]
    if scoreboard.empty:
        lines.append("No candidates loaded.")
    else:
        keep = [
            "experiment",
            "candidate",
            "rows",
            "wells",
            "pooled_rmse",
            "mean_well_rmse",
            "p90_well_rmse",
            "p99_well_rmse",
            "worst_well_rmse",
        ]
        lines.append(_markdown_table(scoreboard[keep].head(40), floatfmt=".4f"))
    lines.extend(["", "## Worst Wells", ""])
    if worst.empty:
        lines.append("No worst-well table.")
    else:
        lines.append(_markdown_table(worst.head(120), floatfmt=".4f"))
    (output_dir / "oof_scoreboard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _update_mission_task1(path: Path, *, loaded: int, best_rmse: float) -> None:
    text = path.read_text(encoding="utf-8")
    replacements = {
        "- [ ] `artifacts/night/oof_scoreboard.csv`": "- [x] `artifacts/night/oof_scoreboard.csv`",
        "- [ ] `artifacts/night/oof_scoreboard.md`": "- [x] `artifacts/night/oof_scoreboard.md`",
        "- [ ] `artifacts/night/predictions/`": "- [x] `artifacts/night/predictions/`",
        "- [ ] `artifacts/night/worst_wells/`": "- [x] `artifacts/night/worst_wells/`",
        "- [ ] Scan existing artifacts for row-wise predictions.": "- [x] Scan existing artifacts for row-wise predictions.",
        "- [ ] Normalize prediction schemas to `experiment/fold/well_id/row_id/pred_tvt/true_tvt`.": "- [x] Normalize prediction schemas to `experiment/fold/well_id/row_id/pred_tvt/true_tvt`.",
        "- [ ] Compute pooled hidden-row RMSE.": "- [x] Compute pooled hidden-row RMSE.",
        "- [ ] Compute per-well RMSE quantiles.": "- [x] Compute per-well RMSE quantiles.",
        "- [ ] Compute hidden-length buckets: short, medium, long, xlong.": "- [x] Compute hidden-length buckets: short, medium, long, xlong.",
        "- [ ] Save worst 30 wells per important candidate.": "- [x] Save worst 30 wells per important candidate.",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    old_verdict = "## Task 1. Unified OOF Scoreboard"
    idx = text.find(old_verdict)
    if idx >= 0:
        next_idx = text.find("## Task 2.", idx)
        block = text[idx:next_idx]
        block = re.sub(
            r"Verdict:\n\n```text\n.*?\n```",
            f"Verdict:\n\n```text\nDONE. Loaded {loaded} row-wise artifacts. Best pooled RMSE in scoreboard: {best_rmse:.4f}. See artifacts/night/oof_scoreboard.md.\n```",
            block,
            flags=re.S,
        )
        text = text[:idx] + block + text[next_idx:]
    log = (
        "\n### Task 1 Result\n\n"
        f"- Loaded row-wise artifacts: `{loaded}`.\n"
        f"- Best pooled RMSE in current scoreboard: `{best_rmse:.4f}`.\n"
        "- Artifacts: `oof_scoreboard.csv`, `oof_scoreboard.md`, `predictions/`, `worst_wells/`.\n"
    )
    text = text.replace("## Final Decision Tree", log + "\n## Final Decision Tree")
    path.write_text(text, encoding="utf-8")


def run_scoreboard(config: NightScoreboardConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    pred_dir = output_dir / "predictions"
    worst_dir = output_dir / "worst_wells"
    if pred_dir.exists():
        shutil.rmtree(pred_dir)
    if worst_dir.exists():
        shutil.rmtree(worst_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)
    worst_dir.mkdir(parents=True, exist_ok=True)
    truth = load_hidden_truth(config.data_dir)
    paths = _scan_prediction_paths(config.artifacts_dir)
    if config.max_artifacts > 0:
        paths = paths[: config.max_artifacts]
    scoreboard_parts: list[pd.DataFrame] = []
    worst_parts: list[pd.DataFrame] = []
    loaded = 0
    prediction_rows = 0
    errors: list[dict[str, str]] = []
    skipped_large: list[dict[str, Any]] = []
    for idx, path in enumerate(paths):
        if idx == 0 or (idx + 1) % 25 == 0 or idx + 1 == len(paths):
            print(
                f"[night-scoreboard] artifact {idx + 1}/{len(paths)} loaded={loaded} path={path}",
                flush=True,
            )
        size_mb = path.stat().st_size / (1024.0 * 1024.0)
        if config.max_file_mb > 0 and size_mb > float(config.max_file_mb):
            skipped_large.append({"path": str(path), "size_mb": float(size_mb)})
            continue
        try:
            norm = normalize_prediction_artifact(path, truth=truth, artifact_root=config.artifacts_dir)
        except Exception as exc:  # noqa: BLE001 - diagnostics should continue scanning.
            errors.append({"path": str(path), "error": str(exc)})
            continue
        if norm.empty:
            continue
        loaded += 1
        slug = _slug(str(path.relative_to(config.artifacts_dir)))
        norm.to_parquet(pred_dir / f"{slug}.parquet", index=False)
        prediction_rows += int(len(norm))
        sb, ww = score_predictions(norm, top_n=config.top_n_worst)
        if not sb.empty:
            scoreboard_parts.append(sb)
        if not ww.empty:
            worst_parts.append(ww)
    scoreboard = (
        pd.concat(scoreboard_parts, ignore_index=True)
        .sort_values("pooled_rmse", na_position="last")
        .reset_index(drop=True)
        if scoreboard_parts
        else pd.DataFrame()
    )
    worst = pd.concat(worst_parts, ignore_index=True) if worst_parts else pd.DataFrame()
    scoreboard.to_csv(output_dir / "oof_scoreboard.csv", index=False)
    worst.to_csv(output_dir / "worst_wells" / "worst_wells_all.csv", index=False)
    if not worst.empty:
        for (experiment, candidate), group in worst.groupby(["experiment", "candidate"], sort=False):
            name = _slug(f"{experiment}__{candidate}")
            group.to_csv(worst_dir / f"{name}.csv", index=False)
    _write_markdown_report(output_dir, scoreboard, worst, scanned=len(paths), loaded=loaded)
    metrics = {
        "task": "night_scoreboard",
        "scanned_parquets": int(len(paths)),
        "loaded_rowwise_artifacts": int(loaded),
        "skipped_large_artifacts": skipped_large,
        "prediction_rows": int(prediction_rows),
        "scoreboard_rows": int(len(scoreboard)),
        "best_pooled_rmse": float(scoreboard["pooled_rmse"].min()) if not scoreboard.empty else float("nan"),
        "errors": errors[:100],
    }
    (output_dir / "oof_scoreboard_metrics.json").write_text(
        json.dumps(_json_safe(metrics), indent=2),
        encoding="utf-8",
    )
    mission = output_dir / "NIGHT_MISSION.md"
    if mission.exists() and not scoreboard.empty:
        _update_mission_task1(mission, loaded=loaded, best_rmse=float(scoreboard["pooled_rmse"].min()))
    return metrics


def _select_path_bank_manifest(scoreboard: pd.DataFrame, *, truth_rows: int, config: NightPathBankConfig) -> pd.DataFrame:
    required = {"experiment", "candidate", "rows", "pooled_rmse"}
    missing = sorted(required - set(scoreboard.columns))
    if missing:
        raise ValueError(f"scoreboard missing required columns: {missing}")
    clean = scoreboard.copy()
    if config.exclude_oracle_candidates:
        marker = (clean["experiment"].astype(str) + " " + clean["candidate"].astype(str)).str.lower()
        oracle_tokens = ("oracle", "target", "truth")
        clean = clean.loc[~marker.str.contains("|".join(oracle_tokens), regex=True)].copy()
    clean["rows"] = pd.to_numeric(clean["rows"], errors="coerce").fillna(0).astype(int)
    clean["pooled_rmse"] = pd.to_numeric(clean["pooled_rmse"], errors="coerce")
    min_rows = max(int(config.min_rows), int(np.ceil(float(config.min_rows_fraction) * float(truth_rows))))
    clean = clean.loc[clean["rows"] >= min_rows].copy()
    if clean.empty:
        raise ValueError(f"no candidates cover min_rows={min_rows}; lower min_rows_fraction")
    clean = clean.sort_values(["pooled_rmse", "rows"], ascending=[True, False], na_position="last")
    clean = clean.drop_duplicates(["experiment", "candidate"], keep="first").head(int(config.max_candidates)).copy()
    clean = clean.reset_index(drop=True)
    clean["path_col"] = [
        f"p{idx:03d}__{_slug(row.experiment + '__' + row.candidate)}" for idx, row in clean.iterrows()
    ]
    clean["selected_rank"] = np.arange(len(clean), dtype=np.int64)
    out_cols = [
        "selected_rank",
        "path_col",
        "experiment",
        "candidate",
        "rows",
        "wells",
        "pooled_rmse",
        "mean_well_rmse",
        "p90_well_rmse",
        "worst_well_rmse",
    ]
    for col in out_cols:
        if col not in clean.columns:
            clean[col] = np.nan
    return clean[out_cols]


def _candidate_metrics(bank: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    true = pd.to_numeric(bank["true_tvt"], errors="coerce")
    rows: list[dict[str, Any]] = []
    for record in manifest.itertuples(index=False):
        col = record.path_col
        pred = pd.to_numeric(bank[col], errors="coerce")
        valid = pred.notna() & true.notna()
        if not valid.any():
            continue
        err = pred[valid] - true[valid]
        by_well = (
            pd.DataFrame({"well_id": bank.loc[valid, "well_id"].to_numpy(), "err": err.to_numpy()})
            .groupby("well_id")["err"]
            .agg(_rmse)
        )
        smoothness_vals: list[float] = []
        offset_vals: list[float] = []
        if "Z" in bank.columns:
            path = bank.loc[valid, ["well_id", "row_idx", col, "Z"]].copy()
            for _, group in path.groupby("well_id", sort=False):
                g = group.sort_values("row_idx")
                tvt = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=np.float64)
                z = pd.to_numeric(g["Z"], errors="coerce").to_numpy(dtype=np.float64)
                if tvt.size >= 2:
                    offset = np.diff(tvt) + np.diff(z)
                    offset_vals.extend(np.abs(offset[np.isfinite(offset)]).tolist())
                if tvt.size >= 3:
                    second = np.diff(tvt, n=2)
                    smoothness_vals.extend(np.abs(second[np.isfinite(second)]).tolist())
        rows.append(
            {
                "path_col": col,
                "experiment": record.experiment,
                "candidate": record.candidate,
                "rows": int(valid.sum()),
                "wells": int(by_well.size),
                "coverage_frac": float(valid.mean()),
                "pooled_rmse": _rmse(err),
                "mean_well_rmse": float(by_well.mean()),
                "p90_well_rmse": float(by_well.quantile(0.90)),
                "worst_well_rmse": float(by_well.max()),
                "mean_abs_offset_dtvt_plus_dz": float(np.mean(offset_vals)) if offset_vals else float("nan"),
                "mean_abs_second_diff_tvt": float(np.mean(smoothness_vals)) if smoothness_vals else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values("pooled_rmse", na_position="last").reset_index(drop=True)


def _write_path_bank_report(
    output_dir: Path,
    *,
    manifest: pd.DataFrame,
    summary: pd.DataFrame,
    oracle_metrics: dict[str, Any],
    distribution: pd.DataFrame,
    no_good: pd.DataFrame,
) -> None:
    lines = [
        "# NIGHT TASK 2: Path Bank v0",
        "",
        "## Selection",
        "",
        f"Selected candidates: `{len(manifest)}`",
        f"Oracle rows covered: `{oracle_metrics['oracle_rows']}`",
        f"Best-of-bank oracle pooled RMSE: `{oracle_metrics['oracle_pooled_rmse']:.4f}`",
        f"Best single selected candidate pooled RMSE: `{oracle_metrics['best_single_pooled_rmse']:.4f}`",
        "",
        "## Candidate Manifest",
        "",
        _markdown_table(
            manifest[["selected_rank", "path_col", "experiment", "candidate", "rows", "pooled_rmse"]].head(80),
            floatfmt=".4f",
        ),
        "",
        "## Candidate Diagnostics",
        "",
        _markdown_table(
            summary[
                [
                    "path_col",
                    "rows",
                    "wells",
                    "pooled_rmse",
                    "mean_well_rmse",
                    "p90_well_rmse",
                    "mean_abs_offset_dtvt_plus_dz",
                    "mean_abs_second_diff_tvt",
                ]
            ].head(80),
            floatfmt=".4f",
        ),
        "",
        "## Oracle Chosen-Path Distribution",
        "",
        _markdown_table(distribution.head(80), floatfmt=".4f"),
        "",
        "## Worst Wells Under Bank Oracle",
        "",
        _markdown_table(no_good.head(40), floatfmt=".4f"),
        "",
    ]
    (output_dir / "best_of_bank_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _update_mission_task2(path: Path, *, oracle_rmse: float, selected: int, test_written: bool) -> None:
    text = path.read_text(encoding="utf-8")
    replacements = {
        "- [ ] `artifacts/night/path_bank_oof.parquet`": "- [x] `artifacts/night/path_bank_oof.parquet`",
        "- [ ] `artifacts/night/best_of_bank_report.md`": "- [x] `artifacts/night/best_of_bank_report.md`",
        "- [ ] TVT path": "- [x] TVT path",
        "- [ ] `C = TVT + Z`": "- [x] `C = TVT + Z`",
        "- [ ] `offset = dTVT + dZ`": "- [x] `offset = dTVT + dZ`",
        "- [ ] path smoothness": "- [x] path smoothness",
        "- [ ] best-of-bank oracle pooled RMSE": "- [x] best-of-bank oracle pooled RMSE",
        "- [ ] oracle chosen-path distribution": "- [x] oracle chosen-path distribution",
        "- [ ] wells with no good candidate": "- [x] wells with no good candidate",
        "- [ ] whether oracle `< 8`, `8-10`, or `> 10`": "- [x] whether oracle `< 8`, `8-10`, or `> 10`",
    }
    if test_written:
        replacements["- [ ] `artifacts/night/path_bank_test.parquet`"] = "- [x] `artifacts/night/path_bank_test.parquet`"
    for old, new in replacements.items():
        text = text.replace(old, new)
    idx = text.find("## Task 2. Path Bank v0")
    if idx >= 0:
        next_idx = text.find("## Task 3.", idx)
        block = text[idx:next_idx]
        verdict_band = "< 8" if oracle_rmse < 8 else ("8-10" if oracle_rmse <= 10 else "> 10")
        block = re.sub(
            r"Verdict:\n\n```text\n.*?\n```",
            (
                "Verdict:\n\n```text\n"
                f"OOF DONE. Selected {selected} broad-coverage candidates. "
                f"Best-of-bank oracle RMSE: {oracle_rmse:.4f} ({verdict_band}). "
                "Test bank not generated in this pass unless explicitly marked above.\n```"
            ),
            block,
            flags=re.S,
        )
        text = text[:idx] + block + text[next_idx:]
    log = (
        "\n### Task 2 Result\n\n"
        f"- Selected broad-coverage candidates: `{selected}`.\n"
        f"- Best-of-bank oracle pooled RMSE: `{oracle_rmse:.4f}`.\n"
        "- Artifacts: `path_bank_oof.parquet`, `path_bank_manifest.csv`, `best_of_bank_report.md`.\n"
    )
    text = text.replace("## Final Decision Tree", log + "\n## Final Decision Tree")
    path.write_text(text, encoding="utf-8")


def run_path_bank(config: NightPathBankConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    pred_dir = output_dir / "predictions"
    scoreboard_path = output_dir / "oof_scoreboard.csv"
    if not pred_dir.exists():
        raise FileNotFoundError(f"prediction directory not found: {pred_dir}")
    if not scoreboard_path.exists():
        raise FileNotFoundError(f"scoreboard not found: {scoreboard_path}")
    truth = load_hidden_truth(config.data_dir)
    truth = truth.rename(columns={"TVT": "true_tvt"})
    scoreboard = pd.read_csv(scoreboard_path)
    manifest = _select_path_bank_manifest(scoreboard, truth_rows=len(truth), config=config)
    manifest.to_csv(output_dir / "path_bank_manifest.csv", index=False)
    bank_cols = [
        col
        for col in ("id", "well_id", "row_idx", "true_tvt", "hidden_len", "hidden_len_bucket", "gr_valid_frac", "MD", "X", "Y", "Z", "GR")
        if col in truth.columns
    ]
    bank = truth[bank_cols].copy().set_index("id", drop=False)
    selected = {(row.experiment, row.candidate): row.path_col for row in manifest.itertuples(index=False)}
    for col in selected.values():
        bank[col] = np.nan
    paths = sorted(pred_dir.glob("*.parquet"))
    loaded_files = 0
    for idx, path in enumerate(paths):
        if idx == 0 or (idx + 1) % 10 == 0 or idx + 1 == len(paths):
            print(f"[night-path-bank] prediction artifact {idx + 1}/{len(paths)} path={path}", flush=True)
        try:
            frame = pd.read_parquet(path, columns=["experiment", "candidate", "id", "pred_tvt"])
        except Exception:
            continue
        frame["experiment"] = frame["experiment"].astype(str)
        frame["candidate"] = frame["candidate"].astype(str)
        mask = pd.Series(False, index=frame.index)
        for experiment, candidate in selected:
            mask |= (frame["experiment"] == experiment) & (frame["candidate"] == candidate)
        if not mask.any():
            continue
        loaded_files += 1
        sub = frame.loc[mask, ["experiment", "candidate", "id", "pred_tvt"]].copy()
        sub["id"] = sub["id"].astype(str)
        sub["pred_tvt"] = pd.to_numeric(sub["pred_tvt"], errors="coerce")
        sub = sub.dropna(subset=["pred_tvt"])
        for (experiment, candidate), group in sub.groupby(["experiment", "candidate"], sort=False):
            col = selected[(experiment, candidate)]
            pred = group.groupby("id", sort=False)["pred_tvt"].mean()
            idx_common = pred.index.intersection(bank.index)
            if len(idx_common) == 0:
                continue
            bank.loc[idx_common, col] = pred.loc[idx_common].to_numpy(dtype=np.float64)
    bank_reset = bank.reset_index(drop=True)
    bank_reset.to_parquet(output_dir / "path_bank_oof.parquet", index=False)
    summary = _candidate_metrics(bank_reset, manifest)
    summary.to_csv(output_dir / "path_bank_candidate_summary.csv", index=False)
    true = pd.to_numeric(bank_reset["true_tvt"], errors="coerce").to_numpy(dtype=np.float64)
    best_sqerr = np.full(len(bank_reset), np.inf, dtype=np.float64)
    best_code = np.full(len(bank_reset), -1, dtype=np.int32)
    best_pred = np.full(len(bank_reset), np.nan, dtype=np.float64)
    path_cols = manifest["path_col"].tolist()
    for code, col in enumerate(path_cols):
        pred = pd.to_numeric(bank_reset[col], errors="coerce").to_numpy(dtype=np.float64)
        valid = np.isfinite(pred) & np.isfinite(true)
        sqerr = np.full(len(pred), np.inf, dtype=np.float64)
        sqerr[valid] = (pred[valid] - true[valid]) ** 2
        improve = sqerr < best_sqerr
        best_sqerr[improve] = sqerr[improve]
        best_code[improve] = code
        best_pred[improve] = pred[improve]
    covered = np.isfinite(best_sqerr)
    code_to_col = dict(enumerate(path_cols))
    oracle_rows = bank_reset.loc[covered, ["id", "well_id", "row_idx", "true_tvt"]].copy()
    oracle_rows["best_path_col"] = [code_to_col[int(code)] for code in best_code[covered]]
    oracle_rows["best_pred_tvt"] = best_pred[covered]
    oracle_rows["best_sqerr"] = best_sqerr[covered]
    oracle_rows.to_parquet(output_dir / "best_of_bank_oracle_rows.parquet", index=False)
    distribution = (
        oracle_rows.groupby("best_path_col")
        .agg(rows=("best_sqerr", "size"), oracle_rmse=("best_sqerr", lambda x: float(np.sqrt(np.mean(x)))))
        .reset_index()
        .sort_values("rows", ascending=False)
    )
    distribution = distribution.merge(manifest[["path_col", "experiment", "candidate"]], left_on="best_path_col", right_on="path_col", how="left")
    distribution = distribution.drop(columns=["path_col"])
    distribution.to_csv(output_dir / "best_of_bank_candidate_distribution.csv", index=False)
    oracle_well = (
        oracle_rows.groupby("well_id")
        .agg(rows=("best_sqerr", "size"), oracle_rmse=("best_sqerr", lambda x: float(np.sqrt(np.mean(x)))))
        .reset_index()
        .sort_values("oracle_rmse", ascending=False)
    )
    oracle_well.to_csv(output_dir / "best_of_bank_worst_wells.csv", index=False)
    oracle_rmse = float(np.sqrt(np.mean(best_sqerr[covered]))) if covered.any() else float("nan")
    best_single = float(summary["pooled_rmse"].min()) if not summary.empty else float("nan")
    metrics = {
        "task": "night_path_bank",
        "selected_candidates": int(len(manifest)),
        "loaded_prediction_files": int(loaded_files),
        "path_bank_rows": int(len(bank_reset)),
        "oracle_rows": int(covered.sum()),
        "oracle_coverage_frac": float(covered.mean()),
        "oracle_pooled_rmse": oracle_rmse,
        "best_single_pooled_rmse": best_single,
    }
    (output_dir / "path_bank_metrics.json").write_text(json.dumps(_json_safe(metrics), indent=2), encoding="utf-8")
    _write_path_bank_report(
        output_dir,
        manifest=manifest,
        summary=summary,
        oracle_metrics=metrics,
        distribution=distribution,
        no_good=oracle_well,
    )
    mission = output_dir / "NIGHT_MISSION.md"
    if mission.exists():
        _update_mission_task2(mission, oracle_rmse=oracle_rmse, selected=len(manifest), test_written=(output_dir / "path_bank_test.parquet").exists())
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build night mission diagnostics")
    parser.add_argument("--task", choices=("scoreboard", "path-bank", "all"), default="scoreboard")
    parser.add_argument("--data-dir", type=Path, default=NightScoreboardConfig.data_dir)
    parser.add_argument("--artifacts-dir", type=Path, default=NightScoreboardConfig.artifacts_dir)
    parser.add_argument("--output-dir", type=Path, default=NightScoreboardConfig.output_dir)
    parser.add_argument("--max-artifacts", type=int, default=NightScoreboardConfig.max_artifacts)
    parser.add_argument("--max-file-mb", type=float, default=NightScoreboardConfig.max_file_mb)
    parser.add_argument("--top-n-worst", type=int, default=NightScoreboardConfig.top_n_worst)
    parser.add_argument("--path-bank-min-rows-fraction", type=float, default=NightPathBankConfig.min_rows_fraction)
    parser.add_argument("--path-bank-min-rows", type=int, default=NightPathBankConfig.min_rows)
    parser.add_argument("--path-bank-max-candidates", type=int, default=NightPathBankConfig.max_candidates)
    parser.add_argument("--path-bank-include-oracle-candidates", action="store_true")
    args = parser.parse_args(argv)
    metrics: dict[str, Any] = {}
    if args.task in {"scoreboard", "all"}:
        metrics["scoreboard"] = run_scoreboard(
            NightScoreboardConfig(
                data_dir=args.data_dir,
                artifacts_dir=args.artifacts_dir,
                output_dir=args.output_dir,
                max_artifacts=args.max_artifacts,
                max_file_mb=args.max_file_mb,
                top_n_worst=args.top_n_worst,
            )
        )
    if args.task in {"path-bank", "all"}:
        metrics["path_bank"] = run_path_bank(
            NightPathBankConfig(
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                min_rows_fraction=args.path_bank_min_rows_fraction,
                min_rows=args.path_bank_min_rows,
                max_candidates=args.path_bank_max_candidates,
                exclude_oracle_candidates=not args.path_bank_include_oracle_candidates,
            )
        )
    print(json.dumps(_json_safe(metrics), indent=2))


if __name__ == "__main__":
    main()
