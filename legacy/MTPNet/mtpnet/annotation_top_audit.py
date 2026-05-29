from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

FORMATION_TOP_COLUMNS: tuple[str, ...] = (
    "ANCC",
    "ASTNU",
    "ASTNL",
    "EGFDU",
    "EGFDL",
    "BUDA",
)


@dataclass(frozen=True)
class AnnotationTopAuditConfig:
    data_dir: Path = Path("data/train")
    output_dir: Path = Path("artifacts/annotation_top_audit_v0")
    k_wells: int = -1
    control_spacing_rows: int = 323
    n_panel_wells: int = 8
    progress_every: int = 100


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
    if isinstance(value, Path):
        return str(value)
    return value


def _rmse(diff: np.ndarray | pd.Series) -> float:
    arr = np.asarray(diff, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(arr))))


def _safe_mean(values: pd.Series | np.ndarray) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def _load_horizontal_frame(data_dir: Path, k_wells: int = -1) -> pd.DataFrame:
    paths = sorted(Path(data_dir).glob("*__horizontal_well.csv"))
    if k_wells > 0:
        paths = paths[:k_wells]
    rows: list[pd.DataFrame] = []
    for path in paths:
        well_id = path.name.replace("__horizontal_well.csv", "")
        frame = pd.read_csv(path)
        frame["well_id"] = well_id
        frame["row_idx"] = np.arange(len(frame), dtype=np.int32)
        frame["id"] = [f"{well_id}_{idx}" for idx in frame["row_idx"]]
        rows.append(frame)
    if not rows:
        raise FileNotFoundError(f"No horizontal wells found in {data_dir}")
    return pd.concat(rows, ignore_index=True)


def uniform_piecewise_reconstruct(
    values: np.ndarray | pd.Series,
    *,
    spacing_rows: int,
) -> tuple[np.ndarray, int]:
    """Reconstruct a curve from sparse uniformly spaced control points."""
    arr = np.asarray(values, dtype=np.float64)
    n = arr.size
    if n == 0:
        return arr.copy(), 0
    finite = np.isfinite(arr)
    if finite.sum() == 0:
        return np.full(n, np.nan, dtype=np.float64), 0
    x = np.arange(n, dtype=np.float64)
    filled = arr.copy()
    if finite.sum() == 1:
        filled[~finite] = float(arr[finite][0])
    else:
        filled[~finite] = np.interp(x[~finite], x[finite], arr[finite])
    step = max(int(spacing_rows), 1)
    controls = list(range(0, n, step))
    if controls[-1] != n - 1:
        controls.append(n - 1)
    control_idx = np.asarray(sorted(set(controls)), dtype=np.int64)
    recon = np.interp(x, control_idx.astype(np.float64), filled[control_idx])
    return recon.astype(np.float64), int(control_idx.size)


def state_agreement(reference: np.ndarray | pd.Series, candidate: np.ndarray | pd.Series) -> dict[str, float]:
    """Compare derivative states of two sequences."""
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    n = min(ref.size, cand.size)
    if n < 2:
        return {"available_frac": 0.0, "corr": 0.0, "sign_agree": 0.0, "scaled_rmse": 1.0}
    ref = ref[:n]
    cand = cand[:n]
    valid = np.isfinite(ref[:-1]) & np.isfinite(ref[1:]) & np.isfinite(cand[:-1]) & np.isfinite(cand[1:])
    if not valid.any():
        return {"available_frac": 0.0, "corr": 0.0, "sign_agree": 0.0, "scaled_rmse": 1.0}
    d_ref = np.diff(ref)[valid]
    d_cand = np.diff(cand)[valid]
    ref_std = float(np.std(d_ref))
    cand_std = float(np.std(d_cand))
    if d_ref.size >= 2 and ref_std > 1e-9 and cand_std > 1e-9:
        corr = float(np.corrcoef(d_ref, d_cand)[0, 1])
        z_ref = (d_ref - float(np.mean(d_ref))) / ref_std
        z_cand = (d_cand - float(np.mean(d_cand))) / cand_std
        scaled_rmse = _rmse(z_ref - z_cand)
    else:
        corr = 0.0
        scaled_rmse = 1.0
    sign_ref = np.sign(d_ref)
    sign_cand = np.sign(d_cand)
    decisive = (np.abs(d_ref) > 1e-9) | (np.abs(d_cand) > 1e-9)
    sign_agree = float((sign_ref[decisive] == sign_cand[decisive]).mean()) if decisive.any() else 0.0
    return {
        "available_frac": float(valid.mean()),
        "corr": corr if np.isfinite(corr) else 0.0,
        "sign_agree": sign_agree,
        "scaled_rmse": scaled_rmse if np.isfinite(scaled_rmse) else 1.0,
    }


def _formation_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in FORMATION_TOP_COLUMNS if column in frame.columns]


def _well_top_metrics(well: pd.DataFrame, *, top_column: str, control_spacing_rows: int) -> dict[str, Any]:
    ordered = well.sort_values("row_idx")
    top = pd.to_numeric(ordered[top_column], errors="coerce").to_numpy(dtype=np.float64)
    tvt = pd.to_numeric(ordered["TVT"], errors="coerce").to_numpy(dtype=np.float64)
    z = pd.to_numeric(ordered["Z"], errors="coerce").to_numpy(dtype=np.float64)
    tvt_input = pd.to_numeric(ordered.get("TVT_input", pd.Series(np.nan, index=ordered.index)), errors="coerce")
    hidden_mask = tvt_input.isna().to_numpy()
    recon, controls = uniform_piecewise_reconstruct(top, spacing_rows=control_spacing_rows)
    all_top_tvt = state_agreement(top, tvt)
    all_top_negz = state_agreement(top, -z)
    hidden_top_tvt = state_agreement(top[hidden_mask], tvt[hidden_mask])
    hidden_top_negz = state_agreement(top[hidden_mask], -z[hidden_mask])
    return {
        "well_id": str(ordered["well_id"].iloc[0]),
        "top_column": top_column,
        "rows": int(len(ordered)),
        "hidden_rows": int(hidden_mask.sum()),
        "control_count": int(controls),
        "rows_per_control": float(len(ordered) / max(controls, 1)),
        "piecewise_rmse_ft": _rmse(recon - top),
        "piecewise_p95_abs_ft": float(np.nanpercentile(np.abs(recon - top), 95.0)),
        "sign_agree_top_vs_tvt_all": all_top_tvt["sign_agree"],
        "corr_top_vs_tvt_all": all_top_tvt["corr"],
        "sign_agree_top_vs_negz_all": all_top_negz["sign_agree"],
        "corr_top_vs_negz_all": all_top_negz["corr"],
        "sign_agree_top_vs_tvt_hidden": hidden_top_tvt["sign_agree"],
        "corr_top_vs_tvt_hidden": hidden_top_tvt["corr"],
        "sign_agree_top_vs_negz_hidden": hidden_top_negz["sign_agree"],
        "corr_top_vs_negz_hidden": hidden_top_negz["corr"],
    }


def _summary_by_top(well_metrics: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}
    for top, group in well_metrics.groupby("top_column", sort=True):
        summary[str(top)] = {
            "wells": int(group["well_id"].nunique()),
            "rows_per_control_mean": _safe_mean(group["rows_per_control"]),
            "control_count_mean": _safe_mean(group["control_count"]),
            "piecewise_rmse_ft_mean": _safe_mean(group["piecewise_rmse_ft"]),
            "piecewise_p95_abs_ft_mean": _safe_mean(group["piecewise_p95_abs_ft"]),
            "sign_agree_top_vs_tvt_all": _safe_mean(group["sign_agree_top_vs_tvt_all"]),
            "sign_agree_top_vs_tvt_hidden": _safe_mean(group["sign_agree_top_vs_tvt_hidden"]),
            "sign_agree_top_vs_negz_all": _safe_mean(group["sign_agree_top_vs_negz_all"]),
            "sign_agree_top_vs_negz_hidden": _safe_mean(group["sign_agree_top_vs_negz_hidden"]),
            "corr_top_vs_tvt_all": _safe_mean(group["corr_top_vs_tvt_all"]),
            "corr_top_vs_tvt_hidden": _safe_mean(group["corr_top_vs_tvt_hidden"]),
            "corr_top_vs_negz_all": _safe_mean(group["corr_top_vs_negz_all"]),
            "corr_top_vs_negz_hidden": _safe_mean(group["corr_top_vs_negz_hidden"]),
        }
    return summary


def _format_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 30) -> str:
    if frame.empty:
        return "(empty)"
    clipped = frame.loc[:, [col for col in columns if col in frame.columns]].head(max_rows)
    lines = ["| " + " | ".join(clipped.columns) + " |"]
    lines.append("| " + " | ".join(["---"] * len(clipped.columns)) + " |")
    for _, row in clipped.iterrows():
        vals = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                vals.append("nan" if not np.isfinite(value) else f"{float(value):.4f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _write_panel_figure(well: pd.DataFrame, *, top_column: str, output_path: Path, spacing_rows: int) -> None:
    import matplotlib.pyplot as plt

    ordered = well.sort_values("row_idx")
    row = pd.to_numeric(ordered["row_idx"], errors="coerce").to_numpy(dtype=np.float64)
    tvt = pd.to_numeric(ordered["TVT"], errors="coerce").to_numpy(dtype=np.float64)
    tvt_input = pd.to_numeric(ordered["TVT_input"], errors="coerce").to_numpy(dtype=np.float64)
    top = pd.to_numeric(ordered[top_column], errors="coerce").to_numpy(dtype=np.float64)
    z = pd.to_numeric(ordered["Z"], errors="coerce").to_numpy(dtype=np.float64)
    recon, controls = uniform_piecewise_reconstruct(top, spacing_rows=spacing_rows)
    dtvt = np.diff(tvt, prepend=tvt[0])
    dtop = np.diff(top, prepend=top[0])
    neg_dz = -np.diff(z, prepend=z[0])

    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    fig.suptitle(f"{ordered['well_id'].iloc[0]}: {top_column} sparse annotation audit", fontweight="bold")

    axes[0].plot(row, tvt, color="black", linewidth=1.5, label="true TVT")
    axes[0].plot(row, top, color="#1f77b4", linewidth=1.2, label=top_column)
    axes[0].plot(row, recon, color="#ff7f0e", linewidth=1.0, linestyle="--", label=f"{controls} ctrl interp")
    known = np.isfinite(tvt_input)
    if known.any():
        axes[0].scatter(row[known], tvt_input[known], s=10, color="#2ca02c", label="known TVT_input", zorder=3)
    axes[0].set_ylabel("TVT ft")
    axes[0].legend(loc="best", fontsize=8)
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(row, dtop, color="#1f77b4", linewidth=1.0, label=f"d{top_column}")
    axes[1].plot(row, dtvt, color="black", linewidth=1.0, alpha=0.75, label="dTVT")
    axes[1].plot(row, neg_dz, color="#d62728", linewidth=1.0, alpha=0.75, label="-dZ")
    axes[1].axhline(0.0, color="gray", linewidth=0.8)
    axes[1].set_ylabel("delta")
    axes[1].legend(loc="best", fontsize=8)
    axes[1].grid(True, alpha=0.25)

    state = np.sign(dtop)
    axes[2].fill_between(row, 0, np.maximum(state, 0), where=state >= 0, color="#d62728", alpha=0.55, step="mid", label="top up/positive")
    axes[2].fill_between(row, 0, np.minimum(state, 0), where=state < 0, color="#1f77b4", alpha=0.55, step="mid", label="top down/negative")
    hidden = ~np.isfinite(tvt_input)
    axes[2].fill_between(row, -1.2, 1.2, where=hidden, color="gray", alpha=0.08, step="mid", label="hidden")
    axes[2].set_ylim(-1.2, 1.2)
    axes[2].set_ylabel("state")
    axes[2].set_xlabel("row_idx")
    axes[2].legend(loc="best", fontsize=8)
    axes[2].grid(True, alpha=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _write_report(
    output_dir: Path,
    *,
    metrics: dict[str, Any],
    well_metrics: pd.DataFrame,
) -> None:
    summary_rows = []
    for top, values in metrics.get("summary", {}).items():
        row = {"top_column": top}
        row.update(values)
        summary_rows.append(row)
    summary_frame = pd.DataFrame(summary_rows)
    lines = [
        "# ANNOTATION_TOP_AUDIT_V0",
        "",
        "Diagnostic-only audit of the Kaggle discussion claim that formation top annotations",
        "look like sparse StarSteer dip/state control lines. Horizontal formation columns",
        "are absent from test, so this is **not deployable input**; it is a potential teacher",
        "for test-safe state models.",
        "",
        "## config",
        "```json",
        json.dumps(_json_safe(metrics.get("config", {})), indent=2),
        "```",
        "",
        "## summary by top",
        _format_table(
            summary_frame,
            [
                "top_column",
                "wells",
                "control_count_mean",
                "rows_per_control_mean",
                "piecewise_rmse_ft_mean",
                "sign_agree_top_vs_tvt_hidden",
                "sign_agree_top_vs_negz_hidden",
                "corr_top_vs_tvt_hidden",
                "corr_top_vs_negz_hidden",
            ],
            max_rows=20,
        ),
        "",
        "## top wells by ANCC piecewise error",
        _format_table(
            well_metrics[well_metrics["top_column"].eq("ANCC")].sort_values("piecewise_rmse_ft", ascending=False),
            [
                "well_id",
                "rows",
                "hidden_rows",
                "control_count",
                "piecewise_rmse_ft",
                "sign_agree_top_vs_tvt_hidden",
                "sign_agree_top_vs_negz_hidden",
            ],
            max_rows=20,
        ),
        "",
        "## interpretation",
        "",
        "- If `piecewise_rmse_ft_mean` is small with about 15 controls, the formation top behaves like a sparse annotation line.",
        "- If hidden `sign_agree_top_vs_tvt` is high, the top state is a strong train-only teacher for `dTVT` direction.",
        "- If hidden `sign_agree_top_vs_negz` is high, our new `dz_state_features_v0` is a plausible deployable proxy, but still not a direct TVT formula.",
        "- Next deployable step after GO: train `top_state_teacher_v0` from `MD/X/Y/Z/GR/TVT_input` to predict this state, then feed predicted state to chunk-policy.",
    ]
    (output_dir / "annotation_top_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_annotation_top_audit_from_frame(
    frame: pd.DataFrame,
    *,
    output_dir: Path,
    config: AnnotationTopAuditConfig,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    top_columns = _formation_columns(frame)
    if not top_columns:
        raise ValueError("No horizontal formation top columns found in frame")
    records: list[dict[str, Any]] = []
    grouped = list(frame.groupby("well_id", sort=True))
    for index, (well_id, well) in enumerate(grouped, start=1):
        if config.progress_every > 0 and (
            index == 1 or index % config.progress_every == 0 or index == len(grouped)
        ):
            print(
                f"[annotation-top] processed {index}/{len(grouped)} wells",
                file=sys.stderr,
                flush=True,
            )
        for top in top_columns:
            records.append(
                _well_top_metrics(
                    well,
                    top_column=top,
                    control_spacing_rows=config.control_spacing_rows,
                )
            )
    well_metrics = pd.DataFrame(records)
    summary = _summary_by_top(well_metrics)
    metrics = {
        "candidate": "annotation_top_audit_v0",
        "wells": int(frame["well_id"].nunique()),
        "rows": int(len(frame)),
        "top_columns": top_columns,
        "config": asdict(config),
        "summary": summary,
    }
    well_metrics.to_csv(out / "annotation_top_well_metrics.csv", index=False)
    (out / "annotation_top_metrics.json").write_text(
        json.dumps(_json_safe(metrics), indent=2),
        encoding="utf-8",
    )
    _write_report(out, metrics=metrics, well_metrics=well_metrics)

    panel_dir = out / "figures"
    for well_id, well in grouped[: max(int(config.n_panel_wells), 0)]:
        if "ANCC" in top_columns:
            _write_panel_figure(
                well,
                top_column="ANCC",
                output_path=panel_dir / f"{well_id}_ancc_state_audit.png",
                spacing_rows=config.control_spacing_rows,
            )
    return metrics


def run_annotation_top_audit(config: AnnotationTopAuditConfig) -> dict[str, Any]:
    frame = _load_horizontal_frame(config.data_dir, k_wells=config.k_wells)
    return run_annotation_top_audit_from_frame(frame, output_dir=config.output_dir, config=config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit train-only formation top annotation state")
    parser.add_argument("--data-dir", type=Path, default=AnnotationTopAuditConfig.data_dir)
    parser.add_argument("--output-dir", type=Path, default=AnnotationTopAuditConfig.output_dir)
    parser.add_argument("--k-wells", type=int, default=AnnotationTopAuditConfig.k_wells)
    parser.add_argument("--control-spacing-rows", type=int, default=AnnotationTopAuditConfig.control_spacing_rows)
    parser.add_argument("--n-panel-wells", type=int, default=AnnotationTopAuditConfig.n_panel_wells)
    parser.add_argument("--progress-every", type=int, default=AnnotationTopAuditConfig.progress_every)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metrics = run_annotation_top_audit(
        AnnotationTopAuditConfig(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            k_wells=args.k_wells,
            control_spacing_rows=args.control_spacing_rows,
            n_panel_wells=args.n_panel_wells,
            progress_every=args.progress_every,
        )
    )
    compact = {
        "candidate": metrics["candidate"],
        "wells": metrics["wells"],
        "rows": metrics["rows"],
        "summary": metrics["summary"].get("ANCC", {}),
        "report": str(Path(args.output_dir) / "annotation_top_report.md"),
    }
    print(json.dumps(_json_safe(compact), indent=2))


if __name__ == "__main__":
    main()
