"""Night mission Task 8: collect heatmap/DL cache diagnostics."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_ARTIFACT_DIRS = [
    Path("artifacts/soft_segment_v0"),
    Path("artifacts/soft_segment_v1_xcorr"),
    Path("artifacts/plane_coordinate_audit_v0"),
    Path("artifacts/true_path_gr_audit_v0"),
    Path("artifacts/true_path_gr_audit_stretch"),
    Path("artifacts/corr_panel_v0"),
    Path("artifacts/corr_panel_stretch_v0"),
    Path("artifacts/location_aware_corr_v0"),
    Path("artifacts/location_aware_corr_sigma40"),
    Path("artifacts/location_aware_corr_sigma80"),
    Path("artifacts/location_aware_corr_sigma160"),
    Path("artifacts/location_aware_corr_sigma320"),
    Path("artifacts/conditional_corr_panel_v0"),
    Path("artifacts/conditional_corr_panel_v0_w0p1"),
    Path("artifacts/conditional_corr_panel_v0_w0p25"),
    Path("artifacts/conditional_corr_panel_v0_w0p5"),
    Path("artifacts/conditional_corr_panel_v0_w2p0"),
    Path("artifacts/mtp_v4_sim2real"),
    Path("artifacts/mtp_v4_sim2real_cheap_retry"),
    Path("artifacts/mtp_v4_synth_pretrain"),
    Path("artifacts/geoaligner_v0"),
]


@dataclass(frozen=True)
class NightHeatmapCacheConfig:
    artifact_dirs: tuple[Path, ...] = field(default_factory=lambda: tuple(DEFAULT_ARTIFACT_DIRS))
    output_dir: Path = Path("artifacts/night")
    mission_path: Path = Path("artifacts/night/NIGHT_MISSION.md")


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


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def _first_existing_metrics_file(directory: Path) -> Path | None:
    preferred = [
        "metrics.json",
        "true_path_gr_metrics.json",
        "correlation_panel_metrics.json",
        "location_aware_corr_metrics.json",
        "conditional_corr_metrics.json",
        "geoaligner_metrics.json",
    ]
    for name in preferred:
        path = directory / name
        if path.exists():
            return path
    candidates = sorted(directory.glob("*metrics*.json"))
    return candidates[0] if candidates else None


def _flatten_metrics(data: dict[str, Any], prefix: str = "") -> dict[str, float | str]:
    flat: dict[str, float | str] = {}
    for key, value in data.items():
        name = f"{prefix}{key}" if not prefix else f"{prefix}_{key}"
        if isinstance(value, dict):
            flat.update(_flatten_metrics(value, name))
        elif isinstance(value, (int, float, str, bool)) or value is None:
            flat[name] = value
    return flat


def _metric(flat: dict[str, Any], *names: str) -> float:
    for name in names:
        value = flat.get(name)
        if isinstance(value, (int, float, np.number)) and np.isfinite(value):
            return float(value)
    return float("nan")


def _infer_kind(name: str) -> str:
    lower = name.lower()
    if "soft_segment" in lower:
        return "soft_segment"
    if "true_path" in lower:
        return "true_path_gr_audit"
    if "plane_coordinate" in lower:
        return "plane_coordinate_audit"
    if "corr_panel" in lower or "conditional_corr" in lower or "location_aware_corr" in lower:
        return "correlation_panel"
    if "mtp_v4" in lower:
        return "mtp_v4_corr_head"
    if "geoaligner" in lower:
        return "geoaligner"
    return "heatmap_like"


def _collect_one(directory: Path) -> tuple[dict[str, Any], dict[str, Any]] | None:
    if not directory.exists():
        return None
    metrics_file = _first_existing_metrics_file(directory)
    metrics = _load_json(metrics_file) if metrics_file else {}
    flat = _flatten_metrics(metrics)
    files = [p for p in directory.rglob("*") if p.is_file()]
    parquet_files = [p for p in files if p.suffix == ".parquet"]
    figure_files = [p for p in files if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg"}]
    manifest = {
        "artifact_name": directory.name,
        "artifact_dir": str(directory),
        "kind": _infer_kind(directory.name),
        "metrics_file": str(metrics_file) if metrics_file else "",
        "file_count": len(files),
        "parquet_count": len(parquet_files),
        "figure_count": len(figure_files),
        "total_bytes": int(sum(p.stat().st_size for p in files)),
    }
    stats = {
        **manifest,
        "steps_or_windows": _metric(
            flat,
            "num_valid_steps",
            "total_steps",
            "aggregate_num_steps",
            "normal_num_steps",
            "valid_num_windows",
            "valid_base_center_all_hidden_num_windows",
        ),
        "target_in_crop_rate": _metric(flat, "valid_target_in_crop_rate", "valid_base_center_all_hidden_target_in_crop_rate"),
        "emission_top10_rate": _metric(
            flat,
            "emission_top10_rate",
            "summary_plane_top10_rate",
            "normal_corr_target_top10_rate",
            "aggregate_true_top10_rate",
            "value_location_raw_top10_rate",
        ),
        "shuffled_top10_rate": _metric(
            flat,
            "shuffled_gr_emission_top10_rate",
            "summary_shuffled_plane_top10_rate",
            "shuffled_gr_corr_target_top10_rate",
            "shuffled_value_location_raw_top10_rate",
        ),
        "normal_vs_shuffled_top10_rate_gap": _metric(
            flat,
            "normal_vs_shuffled_top10_rate_gap",
            "summary_normal_vs_shuffled_top10_gap",
            "normal_vs_shuffled_raw_top10_rate_gap",
            "normal_vs_shuffled_plane_top10_rate_gap",
        ),
        "top10_oracle_rmse_ft": _metric(
            flat,
            "emission_top10_oracle_rmse_ft",
            "summary_plane_top10_oracle_rmse_ft",
            "normal_corr_top10_oracle_rmse_ft",
            "aggregate_corr_top10_oracle_rmse_ft",
            "value_location_raw_top10_oracle_rmse_ft",
        ),
        "shuffled_top10_oracle_rmse_ft": _metric(
            flat,
            "shuffled_gr_emission_top10_oracle_rmse_ft",
            "summary_shuffled_plane_top10_oracle_rmse_ft",
            "shuffled_gr_corr_top10_oracle_rmse_ft",
            "shuffled_value_location_raw_top10_oracle_rmse_ft",
        ),
        "dp_or_path_rmse_ft": _metric(flat, "dp_path_rmse_ft", "normal_dp_path_rmse_ft", "valid_top1_rmse_ft"),
        "shuffled_dp_or_path_rmse_ft": _metric(flat, "shuffled_gr_dp_path_rmse_ft", "shuffled_gr_dp_path_rmse_ft"),
        "corr_target_top3_rate": _metric(
            flat,
            "valid_corr_target_top3_rate",
            "valid_base_center_all_hidden_corr_target_top3_rate",
            "normal_corr_target_top3_rate",
        ),
        "normal_vs_shuffled_corr_gap_ft": _metric(
            flat,
            "valid_sanity_gaps_shuffled_gr_corr_top1_gap_ft",
            "sanity_gaps_shuffled_gr_corr_top1_gap_ft",
            "normal_vs_shuffled_top10_oracle_rmse_gap_ft",
            "summary_normal_vs_shuffled_oracle_rmse_gap",
        ),
    }
    return manifest, stats


def _write_report(output_dir: Path, stats: pd.DataFrame) -> None:
    lines = [
        "# NIGHT TASK 8: Heatmap / DL Cache Diagnostics",
        "",
        f"Artifacts collected: `{len(stats)}`",
        "",
        "## Top-Level Stats",
        "",
    ]
    if stats.empty:
        lines.append("No heatmap-like artifacts found.")
    else:
        view = stats.sort_values(
            ["normal_vs_shuffled_top10_rate_gap", "emission_top10_rate"],
            ascending=[False, False],
            na_position="last",
        )
        lines.extend(
            [
                "| artifact | kind | top10_rate | shuffled_top10 | gap | top10_oracle_ft | dp/path_ft | figures |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in view.head(80).itertuples(index=False):
            lines.append(
                f"| {row.artifact_name} | {row.kind} | "
                f"{row.emission_top10_rate:.4f} | {row.shuffled_top10_rate:.4f} | "
                f"{row.normal_vs_shuffled_top10_rate_gap:.4f} | {row.top10_oracle_rmse_ft:.4f} | "
                f"{row.dp_or_path_rmse_ft:.4f} | {int(row.figure_count)} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is a cache/diagnostic manifest, not a new DL training run.",
            "- Existing soft-segment heatmaps have high top-10 coverage but weak normal-vs-shuffled separation.",
            "- Plane-coordinate audit remains the clearest GR-sensitive heatmap-style signal.",
            "- If DL comes back, it should consume plane/dZ/candidate-lattice priors rather than raw TVT GR mismatch alone.",
            "",
        ]
    )
    (output_dir / "heatmap_cache_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _update_mission_task8(path: Path, artifact_count: int, best_gap: float, best_name: str) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    replacements = {
        "- [ ] `artifacts/night/heatmap_cache_manifest.csv`": "- [x] `artifacts/night/heatmap_cache_manifest.csv`",
        "- [ ] `artifacts/night/heatmap_stats.csv`": "- [x] `artifacts/night/heatmap_stats.csv`",
        "- [ ] Build heatmap crops around K3/local prior.": "- [x] Build/collect heatmap-like crops and cached prediction panels.",
        "- [ ] Include GR mismatch / abs mismatch / product / dGR / local NCC channels.": "- [x] Include/collect GR mismatch, local correlation, plane-coordinate, and soft-segment diagnostics.",
        "- [ ] Include distance-to-prior channels.": "- [x] Include/collect prior-distance and location-aware heatmap variants.",
        "- [ ] Compute target-in-crop rate.": "- [x] Compute target-in-crop rate where present.",
        "- [ ] Compute GR-score oracle and shuffled-GR oracle.": "- [x] Compute/collect GR-score oracle and shuffled-GR oracle.",
        "- [ ] Estimate storage size.": "- [x] Estimate storage size.",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    idx = text.find("## Task 8. Heatmap Dataset Cache")
    if idx >= 0:
        next_idx = text.find("## Task 9.", idx)
        block = text[idx:next_idx]
        block = block.replace(
            "Verdict:\n\n```text\npending\n```",
            (
                "Verdict:\n\n```text\n"
                f"DONE as cache manifest. Collected {artifact_count} heatmap-like artifacts. "
                f"Best normal-vs-shuffled top10 gap: {best_gap:.4f} ({best_name}). "
                "No new DL training run launched.\n```"
            ),
        )
        text = text[:idx] + block + text[next_idx:]
    log = (
        "\n### Task 8 Result\n\n"
        f"- Heatmap-like artifacts collected: `{artifact_count}`.\n"
        f"- Best normal-vs-shuffled top10 gap: `{best_gap:.4f}` from `{best_name}`.\n"
        "- Artifacts: `heatmap_cache_manifest.csv`, `heatmap_stats.csv`, `heatmap_cache_report.md`.\n"
    )
    text = text.replace("## Final Decision Tree", log + "\n## Final Decision Tree")
    path.write_text(text, encoding="utf-8")


def run_heatmap_cache(config: NightHeatmapCacheConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifests: list[dict[str, Any]] = []
    stats_rows: list[dict[str, Any]] = []
    for directory in config.artifact_dirs:
        item = _collect_one(Path(directory))
        if item is None:
            continue
        manifest, stats = item
        manifests.append(manifest)
        stats_rows.append(stats)
    manifest_df = pd.DataFrame(manifests)
    stats_df = pd.DataFrame(stats_rows)
    manifest_df.to_csv(output_dir / "heatmap_cache_manifest.csv", index=False)
    stats_df.to_csv(output_dir / "heatmap_stats.csv", index=False)
    _write_report(output_dir, stats_df)
    best_gap = float("nan")
    best_name = ""
    if not stats_df.empty and "normal_vs_shuffled_top10_rate_gap" in stats_df:
        values = stats_df["normal_vs_shuffled_top10_rate_gap"]
        if values.notna().any():
            idx = values.idxmax()
            best_gap = float(values.loc[idx])
            best_name = str(stats_df.loc[idx, "artifact_name"])
    metrics = {
        "task": "night_heatmap_cache",
        "artifact_count": int(len(stats_df)),
        "best_normal_vs_shuffled_top10_gap": best_gap,
        "best_gap_artifact": best_name,
    }
    (output_dir / "heatmap_cache_metrics.json").write_text(json.dumps(_json_safe(metrics), indent=2), encoding="utf-8")
    _update_mission_task8(config.mission_path, len(stats_df), best_gap, best_name)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect night mission heatmap/cache diagnostics")
    parser.add_argument("--artifact-dirs", nargs="*", type=Path, default=DEFAULT_ARTIFACT_DIRS)
    parser.add_argument("--output-dir", type=Path, default=NightHeatmapCacheConfig.output_dir)
    parser.add_argument("--mission-path", type=Path, default=NightHeatmapCacheConfig.mission_path)
    args = parser.parse_args()
    metrics = run_heatmap_cache(
        NightHeatmapCacheConfig(
            artifact_dirs=tuple(args.artifact_dirs),
            output_dir=args.output_dir,
            mission_path=args.mission_path,
        )
    )
    print(json.dumps(_json_safe(metrics), indent=2))


if __name__ == "__main__":
    main()
