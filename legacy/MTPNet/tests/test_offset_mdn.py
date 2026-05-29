from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _write_tiny_offset_data(root: Path, n_wells: int = 8, n_rows: int = 72) -> None:
    rng = np.random.default_rng(91)
    offsets = np.asarray([-0.02, 0.0, 0.02], dtype=float)
    for well_idx in range(n_wells):
        well_id = f"w{well_idx:02d}"
        offset = float(offsets[well_idx % offsets.size])
        z = -100.0 - np.linspace(0.0, 55.0, n_rows) + rng.normal(scale=0.02, size=n_rows)
        tvt = np.zeros(n_rows, dtype=float)
        tvt[0] = 500.0 + well_idx
        for i in range(1, n_rows):
            tvt[i] = tvt[i - 1] - (z[i] - z[i - 1]) + offset
        gr = 90.0 + 20.0 * np.sin(tvt / 17.0)
        hidden_start = int(0.65 * n_rows)
        horizontal = pd.DataFrame(
            {
                "MD": 1000.0 + np.arange(n_rows) * 10.0,
                "X": np.arange(n_rows, dtype=float),
                "Y": np.full(n_rows, well_idx, dtype=float),
                "Z": z,
                "GR": gr,
                "TVT": tvt,
                "TVT_input": np.where(np.arange(n_rows) < hidden_start, tvt, np.nan),
                "ANCC": tvt + z,
            }
        )
        tw_tvt = np.linspace(float(tvt.min()) - 15.0, float(tvt.max()) + 15.0, 160)
        pd.DataFrame({"TVT": tw_tvt, "GR": 90.0 + 20.0 * np.sin(tw_tvt / 17.0)}).to_csv(
            root / f"{well_id}__typewell.csv", index=False
        )
        horizontal.to_csv(root / f"{well_id}__horizontal_well.csv", index=False)


def test_offset_posterior_normalizes_and_prefers_low_cost() -> None:
    from mtpnet.offset_mdn import add_softmax_posterior

    rows = pd.DataFrame(
        {
            "well_id": ["w0", "w0", "w0", "w1", "w1", "w1"],
            "offset": [-0.02, 0.0, 0.02, -0.02, 0.0, 0.02],
            "pred_log_mse": [3.0, 0.0, 3.0, 1.0, 2.0, 0.0],
        }
    )
    out = add_softmax_posterior(
        rows,
        score_column="pred_log_mse",
        prob_column="prob",
        temperature=1.0,
        lower_is_better=True,
    )

    assert np.allclose(out.groupby("well_id")["prob"].sum().to_numpy(), 1.0)
    assert float(out[(out["well_id"] == "w0") & (out["offset"] == 0.0)]["prob"].iloc[0]) > 0.8
    assert float(out[(out["well_id"] == "w1") & (out["offset"] == 0.02)]["prob"].iloc[0]) > 0.55


def test_posterior_mean_offset_is_probability_weighted() -> None:
    from mtpnet.offset_mdn import selected_offsets_from_posterior

    rows = pd.DataFrame(
        {
            "well_id": ["w0", "w0", "w0"],
            "offset": [-0.02, 0.0, 0.02],
            "prob": [0.25, 0.25, 0.50],
        }
    )
    mean = selected_offsets_from_posterior(rows, prob_column="prob", mode="mean")
    top1 = selected_offsets_from_posterior(rows, prob_column="prob", mode="top1")

    assert np.isclose(mean["w0"], 0.005)
    assert np.isclose(top1["w0"], 0.02)


def test_offset_mdn_smoke_writes_artifacts(tmp_path: Path) -> None:
    from mtpnet.offset_mdn import OffsetMDNConfig, run_offset_mdn

    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    _write_tiny_offset_data(data_dir, n_wells=8, n_rows=72)

    metrics = run_offset_mdn(
        OffsetMDNConfig(
            data_dir=data_dir,
            output_dir=out_dir,
            offset_grid="-0.02:0.02:0.02",
            n_folds=2,
            iterations=20,
            learning_rate=0.2,
            depth=3,
            seed=9,
            temperatures="0.5,1.0",
            include_cost_posterior=True,
        )
    )

    assert metrics["experiment"] == "offset_mdn_v0"
    assert (out_dir / "offset_mdn_metrics.json").exists()
    assert (out_dir / "offset_mdn_report.md").exists()
    assert (out_dir / "offset_mdn_oof_candidate_scores.parquet").exists()
    preds = pd.read_parquet(out_dir / "offset_mdn_oof_predictions.parquet")
    assert {"id", "well_id", "row_idx", "pred_tvt", "candidate", "selected_offset"}.issubset(
        preds.columns
    )
    assert preds["pred_tvt"].notna().all()
    assert any(item["candidate"].startswith("mdn_prob") for item in metrics["candidates"])
