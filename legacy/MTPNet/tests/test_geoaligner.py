from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from geoaligner.config import GAConfig, GADataConfig, GAModelConfig, GARunConfig, GATrainConfig
from geoaligner.dataset import (
    N_LATERAL_FEATURES,
    N_TYPEWELL_FEATURES,
    build_alignment_sample,
    collate_alignment_samples,
    regular_typewell_grid,
)
from geoaligner.dp import viterbi_decode
from geoaligner.evaluate import evaluate_model
from geoaligner.loss import alignment_ce_loss
from geoaligner.model import GeoAligner
from geoaligner.train import train


def _horizontal_frame() -> pd.DataFrame:
    tvt = np.arange(8, dtype=np.float32) * 10.0
    tvt_input = tvt.copy()
    tvt_input[4:] = np.nan
    return pd.DataFrame(
        {
            "id": [f"well_a_{i}" for i in range(8)],
            "MD": np.arange(8, dtype=np.float32),
            "X": np.linspace(0.0, 1.0, 8, dtype=np.float32),
            "Y": np.linspace(2.0, 3.0, 8, dtype=np.float32),
            "Z": np.linspace(100.0, 96.0, 8, dtype=np.float32),
            "GR": np.array([2, 4, 8, 16, 32, 16, 8, 4], dtype=np.float32),
            "TVT_input": tvt_input,
            "TVT": tvt,
        }
    )


def _typewell_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TVT": np.arange(8, dtype=np.float32) * 10.0,
            "GR": np.array([2, 4, 8, 16, 32, 16, 8, 4], dtype=np.float32),
        }
    )


def test_dataset_uses_test_available_inputs_and_train_only_tvt_target() -> None:
    horizontal = _horizontal_frame()
    typewell = _typewell_frame()

    sample = build_alignment_sample(
        "well_a",
        horizontal,
        typewell,
        GADataConfig(rows_per_step=1, vertical_step_ft=10.0, max_horizontal_steps=32),
    )

    assert sample is not None
    assert sample.lateral_features.shape == (8, N_LATERAL_FEATURES)
    assert sample.typewell_features.shape == (8, N_TYPEWELL_FEATURES)
    assert sample.hidden_mask.tolist() == [False, False, False, False, True, True, True, True]
    assert sample.target_bins.tolist() == pytest.approx(np.arange(8, dtype=np.float32))
    assert "ANCC" not in sample.feature_columns
    assert "Geology" not in sample.feature_columns
    assert "schema10" not in " ".join(sample.feature_columns)


def test_regular_typewell_grid_is_regular_and_finite() -> None:
    grid, gr = regular_typewell_grid(_typewell_frame(), vertical_step_ft=5.0)

    assert np.diff(grid).tolist() == pytest.approx([5.0] * (len(grid) - 1))
    assert np.isfinite(gr).all()
    assert grid[0] == pytest.approx(0.0)
    assert grid[-1] == pytest.approx(70.0)


def test_model_returns_emission_logits_for_lateral_by_typewell_bins() -> None:
    model = GeoAligner(
        GAModelConfig(
            d_model=32,
            n_heads=4,
            lateral_layers=1,
            typewell_layers=1,
            ffn_dim=64,
            dropout=0.0,
        )
    )
    batch = {
        "lateral_features": torch.zeros(2, 5, N_LATERAL_FEATURES),
        "typewell_features": torch.zeros(2, 7, N_TYPEWELL_FEATURES),
        "lateral_pad_mask": torch.zeros(2, 5, dtype=torch.bool),
        "typewell_pad_mask": torch.zeros(2, 7, dtype=torch.bool),
    }

    logits = model(**batch)

    assert logits.shape == (2, 5, 7)


def test_alignment_loss_is_lower_when_logits_peak_at_target_bin() -> None:
    target_bins = torch.tensor([[0, 1, 2]], dtype=torch.long)
    hidden_mask = torch.tensor([[False, True, True]])
    lat_pad = torch.zeros(1, 3, dtype=torch.bool)
    type_pad = torch.zeros(1, 4, dtype=torch.bool)
    bad = torch.zeros(1, 3, 4)
    good = bad.clone()
    good[0, 1, 1] = 8.0
    good[0, 2, 2] = 8.0

    assert alignment_ce_loss(good, target_bins, hidden_mask, lat_pad, type_pad) < alignment_ce_loss(
        bad, target_bins, hidden_mask, lat_pad, type_pad
    )


def test_dp_recovers_diagonal_alignment_from_perfect_emissions() -> None:
    log_probs = np.full((5, 7), -20.0, dtype=np.float32)
    for t in range(5):
        log_probs[t, t + 1] = 0.0

    decoded = viterbi_decode(log_probs, max_jump_bins=2, jump_penalty=0.0)

    assert decoded.tolist() == [1, 2, 3, 4, 5]


def test_dp_respects_max_jump_transition_penalty() -> None:
    log_probs = np.full((3, 8), -10.0, dtype=np.float32)
    log_probs[0, 1] = 0.0
    log_probs[1, 7] = 0.0
    log_probs[1, 2] = -1.0
    log_probs[2, 3] = 0.0

    decoded = viterbi_decode(log_probs, max_jump_bins=1, jump_penalty=0.0)

    assert decoded.tolist() == [1, 2, 3]


def test_shuffled_gr_diagnostic_runs_without_train_only_columns() -> None:
    sample = build_alignment_sample(
        "well_a",
        _horizontal_frame().drop(columns=["MD"]),
        _typewell_frame(),
        GADataConfig(rows_per_step=1, vertical_step_ft=10.0, max_horizontal_steps=32),
    )
    assert sample is not None
    batch = collate_alignment_samples([sample])
    model = GeoAligner(GAModelConfig(d_model=32, n_heads=4, lateral_layers=1, typewell_layers=1, ffn_dim=64))

    metrics, rows, steps = evaluate_model(
        model,
        [sample],
        torch.device("cpu"),
        variants=("normal", "shuffled_gr"),
        max_jump_bins=2,
        jump_penalty=0.01,
    )

    assert metrics["normal"]["hidden_steps"] == 4
    assert metrics["shuffled_gr"]["hidden_steps"] == 4
    assert not rows.empty
    assert not steps.empty
    assert batch["target_bins"].shape == (1, 8)


def test_tiny_smoke_train_writes_geoaligner_artifacts(tmp_path: Path) -> None:
    train_dir = tmp_path / "data" / "train"
    train_dir.mkdir(parents=True)
    for i in range(4):
        well_id = f"well_{i}"
        h = _horizontal_frame()
        h["id"] = [f"{well_id}_{j}" for j in range(len(h))]
        h["GR"] = h["GR"] + i
        h.to_csv(train_dir / f"{well_id}__horizontal_well.csv", index=False)
        _typewell_frame().to_csv(train_dir / f"{well_id}__typewell.csv", index=False)

    cfg = GAConfig(
        data=GADataConfig(rows_per_step=1, vertical_step_ft=10.0, max_horizontal_steps=32),
        model=GAModelConfig(d_model=32, n_heads=4, lateral_layers=1, typewell_layers=1, ffn_dim=64),
        train=GATrainConfig(batch_size=2, epochs=1, device="cpu", valid_fraction=0.5, seed=7),
        run=GARunConfig(name="geoaligner_tiny", output_dir=tmp_path / "run"),
        data_dir=train_dir,
        k_wells=-1,
    )

    summary = train(cfg)

    assert summary["valid"]["normal"]["hidden_steps"] > 0
    assert (tmp_path / "run" / "geoaligner_metrics.json").exists()
    assert (tmp_path / "run" / "geoaligner_row_predictions.parquet").exists()
    assert (tmp_path / "run" / "geoaligner_alignment_steps.parquet").exists()
    assert (tmp_path / "run" / "geoaligner_report.md").exists()
    assert (tmp_path / "run" / "figures" / "alignment_example.png").exists()
    saved = json.loads((tmp_path / "run" / "geoaligner_metrics.json").read_text())
    assert saved["run"]["name"] == "geoaligner_tiny"
