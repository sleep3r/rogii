from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class WellShift:
    well: str
    rows: int
    median_shift: float
    median_abs_shift: float
    p95_abs_shift: float
    max_abs_shift: float
    same_sign_fraction: float
    slope_ratio: float
    curvature_ratio: float
    endpoint_shift: float | None
    endpoint_abs_shift: float | None
    tail_continuity_error: float | None
    status: str
    reasons: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a candidate submission against an anchor before submit."
    )
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("artifacts/prediction_guard.md"))
    parser.add_argument("--json", type=Path, default=Path("artifacts/prediction_guard.json"))
    parser.add_argument(
        "--mode",
        choices=["strict", "bold"],
        default="strict",
        help=(
            "strict: hard-fail at HMM-level shifts (p95>25 ft, median>12 ft). "
            "bold: allow honest 5-20 ft per-well shifts, hard-fail only at "
            "HMM-level p95>35 ft or median>20 ft, warn in the 5-12 ft band."
        ),
    )
    parser.add_argument("--max-well-p95-shift", type=float, default=None)
    parser.add_argument("--max-well-median-abs-shift", type=float, default=None)
    parser.add_argument("--max-slope-ratio", type=float, default=None)
    parser.add_argument("--max-curvature-ratio", type=float, default=None)
    parser.add_argument("--max-endpoint-abs-shift", type=float, default=None)
    parser.add_argument("--warn-one-sided-frac", type=float, default=0.95)
    parser.add_argument("--warn-one-sided-median-abs", type=float, default=None)
    parser.add_argument("--warn-well-median-abs-shift", type=float, default=None,
        help="If set, candidate is warned (not failed) when per-well median_abs_shift exceeds this.")
    parser.add_argument("--warn-well-p95-shift", type=float, default=None,
        help="If set, candidate is warned (not failed) when per-well p95_abs_shift exceeds this.")
    parser.add_argument("--max-tail-continuity-error", type=float, default=120.0)
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args()
    _apply_mode_defaults(args)
    return args


def _apply_mode_defaults(args: argparse.Namespace) -> None:
    if args.mode == "strict":
        presets = {
            "max_well_p95_shift": 25.0,
            "max_well_median_abs_shift": 12.0,
            "max_slope_ratio": 3.0,
            "max_curvature_ratio": 3.0,
            "max_endpoint_abs_shift": 25.0,
            "warn_one_sided_median_abs": 5.0,
            "warn_well_median_abs_shift": 8.0,
            "warn_well_p95_shift": 18.0,
        }
    else:
        presets = {
            "max_well_p95_shift": 35.0,
            "max_well_median_abs_shift": 20.0,
            "max_slope_ratio": 4.5,
            "max_curvature_ratio": 4.5,
            "max_endpoint_abs_shift": 35.0,
            "warn_one_sided_median_abs": 5.0,
            "warn_well_median_abs_shift": 5.0,
            "warn_well_p95_shift": 12.0,
        }
    for key, value in presets.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)


def read_submission(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    id_col = "id" if "id" in frame.columns else "ID"
    tvt_col = "tvt" if "tvt" in frame.columns else "TVT"
    if id_col not in frame.columns or tvt_col not in frame.columns:
        raise ValueError(f"Submission must contain id/tvt columns: {path}")
    parsed = frame[id_col].astype(str).str.rsplit("_", n=1, expand=True)
    if parsed.shape[1] != 2:
        raise ValueError(f"Malformed submission ids in {path}")
    out = pd.DataFrame(
        {
            "id": frame[id_col].astype(str),
            "well": parsed[0].astype(str),
            "row_index": pd.to_numeric(parsed[1], errors="raise").astype(int),
            "tvt": pd.to_numeric(frame[tvt_col], errors="coerce").astype(float),
        }
    )
    if out["tvt"].isna().any():
        raise ValueError(f"Submission contains NaN predictions: {path}")
    return out


def last_known_tvt_by_well(data_dir: Path | None) -> dict[str, float]:
    if data_dir is None:
        return {}
    test_dir = data_dir / "test"
    if not test_dir.is_dir():
        return {}
    result: dict[str, float] = {}
    for path in sorted(test_dir.glob("*__horizontal_well.csv")):
        well = path.name.split("__", 1)[0]
        try:
            tvt_input = pd.to_numeric(
                pd.read_csv(path, usecols=["TVT_input"])["TVT_input"],
                errors="coerce",
            ).to_numpy(dtype=float)
        except Exception:
            continue
        known = tvt_input[np.isfinite(tvt_input)]
        if len(known):
            result[well] = float(known[-1])
    return result


def ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator):
        return np.nan
    if not np.isfinite(denominator) or abs(denominator) < 1e-9:
        return np.inf if abs(numerator) > 1e-9 else 1.0
    return float(numerator / denominator)


def roughness(values: np.ndarray) -> tuple[float, float]:
    if len(values) < 2:
        return 0.0, 0.0
    slope = np.diff(values)
    slope_abs = float(np.nanmean(np.abs(slope))) if len(slope) else 0.0
    if len(slope) < 2:
        return slope_abs, 0.0
    curvature = np.diff(slope)
    curvature_abs = float(np.nanmean(np.abs(curvature))) if len(curvature) else 0.0
    return slope_abs, curvature_abs


def evaluate_well(
    well: str,
    frame: pd.DataFrame,
    thresholds: argparse.Namespace,
    last_known: dict[str, float],
) -> WellShift:
    frame = frame.sort_values("row_index")
    candidate = frame["candidate"].to_numpy(dtype=float)
    anchor = frame["anchor"].to_numpy(dtype=float)
    shift = candidate - anchor
    abs_shift = np.abs(shift)
    cand_slope, cand_curv = roughness(candidate)
    anchor_slope, anchor_curv = roughness(anchor)
    reasons: list[str] = []
    status = "pass"
    p95_abs_shift = float(np.nanpercentile(abs_shift, 95))
    median_abs_shift = float(np.nanmedian(abs_shift))
    slope_ratio = ratio(cand_slope, anchor_slope)
    curvature_ratio = ratio(cand_curv, anchor_curv)
    same_sign_fraction = float(max(np.mean(shift >= 0), np.mean(shift <= 0)))
    endpoint_shift: float | None = None
    endpoint_abs_shift: float | None = None
    tail_error: float | None = None

    if p95_abs_shift > thresholds.max_well_p95_shift:
        status = "fail"
        reasons.append(
            f"p95_abs_shift={p95_abs_shift:.3f}>{thresholds.max_well_p95_shift:.3f}"
        )
    if median_abs_shift > thresholds.max_well_median_abs_shift:
        status = "fail"
        reasons.append(
            "median_abs_shift="
            f"{median_abs_shift:.3f}>{thresholds.max_well_median_abs_shift:.3f}"
        )
    if slope_ratio > thresholds.max_slope_ratio:
        status = "fail"
        reasons.append(f"slope_ratio={slope_ratio:.3f}>{thresholds.max_slope_ratio:.3f}")
    if curvature_ratio > thresholds.max_curvature_ratio:
        status = "fail"
        reasons.append(
            f"curvature_ratio={curvature_ratio:.3f}>{thresholds.max_curvature_ratio:.3f}"
        )
    if len(candidate):
        endpoint_shift = float(candidate[-1] - anchor[-1])
        endpoint_abs_shift = abs(endpoint_shift)
        if endpoint_abs_shift > thresholds.max_endpoint_abs_shift:
            status = "fail"
            reasons.append(
                "endpoint_abs_shift="
                f"{endpoint_abs_shift:.3f}>{thresholds.max_endpoint_abs_shift:.3f}"
            )
    if (
        status == "pass"
        and same_sign_fraction > thresholds.warn_one_sided_frac
        and median_abs_shift > thresholds.warn_one_sided_median_abs
    ):
        status = "warn"
        reasons.append(
            f"one_sided_shift={same_sign_fraction:.3f}, "
            f"median_abs_shift={median_abs_shift:.3f}"
        )
    warn_median = getattr(thresholds, "warn_well_median_abs_shift", None)
    warn_p95 = getattr(thresholds, "warn_well_p95_shift", None)
    if status == "pass" and warn_median is not None and median_abs_shift > warn_median:
        status = "warn"
        reasons.append(
            f"median_abs_shift={median_abs_shift:.3f}>warn={warn_median:.3f}"
        )
    if status == "pass" and warn_p95 is not None and p95_abs_shift > warn_p95:
        status = "warn"
        reasons.append(
            f"p95_abs_shift={p95_abs_shift:.3f}>warn={warn_p95:.3f}"
        )
    if well in last_known and len(candidate):
        tail_error = float(candidate[0] - last_known[well])
        if abs(tail_error) > thresholds.max_tail_continuity_error:
            status = "fail"
            reasons.append(
                "tail_continuity_error="
                f"{tail_error:.3f}>{thresholds.max_tail_continuity_error:.3f}"
            )

    return WellShift(
        well=well,
        rows=len(frame),
        median_shift=float(np.nanmedian(shift)),
        median_abs_shift=median_abs_shift,
        p95_abs_shift=p95_abs_shift,
        max_abs_shift=float(np.nanmax(abs_shift)),
        same_sign_fraction=same_sign_fraction,
        slope_ratio=slope_ratio,
        curvature_ratio=curvature_ratio,
        endpoint_shift=endpoint_shift,
        endpoint_abs_shift=endpoint_abs_shift,
        tail_continuity_error=tail_error,
        status=status,
        reasons=reasons,
    )


def compare_predictions(
    candidate_path: Path,
    anchor_path: Path,
    data_dir: Path | None,
    thresholds: argparse.Namespace,
) -> dict[str, Any]:
    candidate = read_submission(candidate_path).rename(columns={"tvt": "candidate"})
    anchor = read_submission(anchor_path).rename(columns={"tvt": "anchor"})
    merged = candidate.merge(
        anchor[["id", "anchor"]],
        on="id",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(candidate) or len(merged) != len(anchor):
        raise ValueError(
            "Candidate and anchor submissions do not contain the same ids: "
            f"candidate={len(candidate)} anchor={len(anchor)} overlap={len(merged)}"
        )
    last_known = last_known_tvt_by_well(data_dir)
    wells = [
        evaluate_well(str(well), group, thresholds, last_known)
        for well, group in merged.groupby("well", sort=True)
    ]
    shift = merged["candidate"].to_numpy(dtype=float) - merged["anchor"].to_numpy(
        dtype=float
    )
    failed = [item for item in wells if item.status == "fail"]
    warned = [item for item in wells if item.status == "warn"]
    return {
        "status": "fail" if failed else ("warn" if warned else "pass"),
        "candidate": str(candidate_path),
        "anchor": str(anchor_path),
        "rows": int(len(merged)),
        "wells": int(len(wells)),
        "global": {
            "median_abs_shift": float(np.nanmedian(np.abs(shift))),
            "mean_abs_shift": float(np.nanmean(np.abs(shift))),
            "p95_abs_shift": float(np.nanpercentile(np.abs(shift), 95)),
            "max_abs_shift": float(np.nanmax(np.abs(shift))),
            "median_shift": float(np.nanmedian(shift)),
        },
        "well_failures": [asdict(item) for item in failed],
        "well_warnings": [asdict(item) for item in warned],
        "wells_detail": [asdict(item) for item in wells],
    }


def write_markdown(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Prediction Shift Guard",
        "",
        f"- Status: **{report['status'].upper()}**",
        f"- Candidate: `{report['candidate']}`",
        f"- Anchor: `{report['anchor']}`",
        f"- Rows: `{report['rows']}`",
        f"- Wells: `{report['wells']}`",
        "",
        "## Global",
        "",
    ]
    for key, value in report["global"].items():
        lines.append(f"- `{key}`: `{value:.6f}`")
    lines.extend(["", "## Wells", ""])
    table = pd.DataFrame(report["wells_detail"]).sort_values(
        ["status", "p95_abs_shift"], ascending=[True, False]
    )
    if not table.empty:
        columns = [
            "well",
            "status",
            "rows",
            "median_abs_shift",
            "p95_abs_shift",
            "max_abs_shift",
            "endpoint_abs_shift",
            "slope_ratio",
            "curvature_ratio",
            "same_sign_fraction",
            "reasons",
        ]
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
        for row in table[columns].to_dict(orient="records"):
            rendered = []
            for column in columns:
                value = row[column]
                if isinstance(value, float):
                    rendered.append(f"{value:.3f}")
                elif isinstance(value, list):
                    rendered.append("<br>".join(str(item) for item in value))
                else:
                    rendered.append(str(value))
            lines.append("| " + " | ".join(rendered) + " |")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    report = compare_predictions(args.candidate, args.anchor, args.data_dir, args)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, args.output)
    print(f"Prediction guard status: {report['status']}")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.json}")
    if report["status"] == "fail" and not args.allow_fail:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
