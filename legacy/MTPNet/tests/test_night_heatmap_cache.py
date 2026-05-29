import json
from pathlib import Path

import pandas as pd

from mtpnet.night_heatmap_cache import NightHeatmapCacheConfig, run_heatmap_cache


def test_run_heatmap_cache_collects_metrics_manifest_and_stats(tmp_path: Path) -> None:
    soft_dir = tmp_path / "soft_segment_v1_xcorr"
    soft_dir.mkdir()
    (soft_dir / "metrics.json").write_text(
        json.dumps(
            {
                "emission_top10_rate": 0.6,
                "emission_top10_oracle_rmse_ft": 31.2,
                "shuffled_gr_emission_top10_rate": 0.59,
                "normal_vs_shuffled_top10_rate_gap": 0.01,
                "num_valid_steps": 123,
            }
        )
    )
    (soft_dir / "window_predictions.parquet").write_bytes(b"placeholder")

    plane_dir = tmp_path / "plane_coordinate_audit_v0"
    plane_dir.mkdir()
    (plane_dir / "metrics.json").write_text(
        json.dumps(
            {
                "summary": {
                    "plane_top10_rate": 0.42,
                    "shuffled_plane_top10_rate": 0.30,
                    "normal_vs_shuffled_top10_gap": 0.12,
                    "plane_top10_oracle_rmse_ft": 12.0,
                },
                "total_steps": 77,
            }
        )
    )

    output_dir = tmp_path / "night"
    metrics = run_heatmap_cache(
        NightHeatmapCacheConfig(
            artifact_dirs=(soft_dir, plane_dir),
            output_dir=output_dir,
            mission_path=tmp_path / "missing.md",
        )
    )

    assert metrics["artifact_count"] == 2
    manifest = pd.read_csv(output_dir / "heatmap_cache_manifest.csv")
    stats = pd.read_csv(output_dir / "heatmap_stats.csv")
    assert set(manifest["artifact_name"]) == {"soft_segment_v1_xcorr", "plane_coordinate_audit_v0"}
    assert "normal_vs_shuffled_top10_rate_gap" in stats.columns
    assert stats["normal_vs_shuffled_top10_rate_gap"].notna().any()
    assert (output_dir / "heatmap_cache_report.md").exists()
