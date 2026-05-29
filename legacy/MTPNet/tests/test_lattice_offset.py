from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch


def _write_tiny_lattice_data(root: Path, n_wells: int = 6, n_rows: int = 96) -> None:
    offsets = np.asarray([-0.02, 0.00, 0.02], dtype=float)
    for well_idx in range(n_wells):
        well_id = f"w{well_idx:02d}"
        offset = float(offsets[well_idx % offsets.size])
        z = -100.0 - np.linspace(0.0, 75.0, n_rows)
        tvt = np.zeros(n_rows, dtype=float)
        tvt[0] = 500.0 + well_idx
        for i in range(1, n_rows):
            tvt[i] = tvt[i - 1] - (z[i] - z[i - 1]) + offset
        gr = 80.0 + 18.0 * np.sin(tvt / 15.0) + 0.2 * tvt
        hidden_start = int(0.6 * n_rows)
        pd.DataFrame(
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
        ).to_csv(root / f"{well_id}__horizontal_well.csv", index=False)
        tw_tvt = np.linspace(float(tvt.min()) - 20.0, float(tvt.max()) + 20.0, 220)
        pd.DataFrame({"TVT": tw_tvt, "GR": 80.0 + 18.0 * np.sin(tw_tvt / 15.0) + 0.2 * tw_tvt}).to_csv(
            root / f"{well_id}__typewell.csv", index=False
        )


def test_lattice_transformer_forward_shape() -> None:
    from mtpnet.lattice_offset import LatticeTransformerScorer

    model = LatticeTransformerScorer(feature_dim=7, d_model=16, n_layers=1, n_heads=4)
    x = torch.randn(2, 5, 7)
    mask = torch.tensor([[True, True, True, False, False], [True, True, True, True, True]])
    logits = model(x, mask)

    assert logits.shape == (2, 5)
    assert torch.isfinite(logits[mask]).all()
    assert torch.all(logits[~mask] < -1e20)


def test_soft_target_prefers_low_cost() -> None:
    from mtpnet.lattice_offset import soft_target_from_cost

    cost = torch.tensor([[10.0, 1.0, 4.0]])
    mask = torch.tensor([[True, True, True]])
    target = soft_target_from_cost(cost, mask, tau=2.0)

    assert torch.allclose(target.sum(dim=1), torch.ones(1))
    assert int(torch.argmax(target, dim=1).item()) == 1


def test_lattice_samples_are_finite_and_schema_safe(tmp_path: Path) -> None:
    from mtpnet.discrete_offset import _load_wells
    from mtpnet.lattice_offset import build_lattice_samples
    from mtpnet.residual_stack import make_group_folds

    _write_tiny_lattice_data(tmp_path, n_wells=4, n_rows=96)
    well_ids = [f"w{i:02d}" for i in range(4)]
    folds = make_group_folds(well_ids, n_folds=2, seed=1)
    fold_of_well = {str(w): idx for idx, (_, valid) in enumerate(folds) for w in valid}
    wells = _load_wells(tmp_path, k_wells=-1, fold_of_well=fold_of_well)
    samples, feature_names = build_lattice_samples(
        wells,
        offsets=np.asarray([-0.02, 0.0, 0.02]),
        spans=np.asarray([16, 32]),
        state_stride=16,
        variant="normal",
        seed=1,
    )

    assert samples
    assert all(name.startswith("feat_") for name in feature_names)
    assert np.isfinite(samples[0].features).all()
    assert np.isfinite(samples[0].cost_rmse).all()
    assert samples[0].features.shape[0] == samples[0].cost_rmse.shape[0]


def test_lattice_offset_smoke_writes_artifacts(tmp_path: Path) -> None:
    from mtpnet.lattice_offset import LatticeOffsetConfig, run_lattice_offset

    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    _write_tiny_lattice_data(data_dir, n_wells=6, n_rows=96)

    metrics = run_lattice_offset(
        LatticeOffsetConfig(
            data_dir=data_dir,
            output_dir=out_dir,
            offset_grid="-0.02:0.02:0.02",
            spans="16,32",
            state_stride=16,
            n_folds=2,
            k_wells=-1,
            epochs=2,
            d_model=32,
            n_layers=1,
            batch_size=8,
            on_policy_rounds=1,
            on_policy_state_stride=16,
            on_policy_max_wells=3,
            include_shuffled=True,
        )
    )

    assert metrics["experiment"] == "lattice_offset_v0"
    assert metrics["on_policy_state_count"] > 0
    assert (out_dir / "lattice_offset_metrics.json").exists()
    assert (out_dir / "lattice_offset_report.md").exists()
    assert (out_dir / "lattice_offset_predictions.parquet").exists()
    preds = pd.read_parquet(out_dir / "lattice_offset_predictions.parquet")
    assert {"id", "well_id", "row_idx", "pred_tvt", "candidate"}.issubset(preds.columns)
    assert preds["pred_tvt"].notna().all()


def test_beam_rollout_covers_hidden_tail_with_finite_predictions(tmp_path: Path) -> None:
    from mtpnet.discrete_offset import _load_wells
    from mtpnet.lattice_offset import LatticeTransformerScorer, beam_rollout_well
    from mtpnet.residual_stack import make_group_folds

    _write_tiny_lattice_data(tmp_path, n_wells=2, n_rows=80)
    well_ids = [f"w{i:02d}" for i in range(2)]
    folds = make_group_folds(well_ids, n_folds=2, seed=4)
    fold_of_well = {str(w): idx for idx, (_, valid) in enumerate(folds) for w in valid}
    wells = _load_wells(tmp_path, k_wells=-1, fold_of_well=fold_of_well)
    well = wells[0]
    model = LatticeTransformerScorer(feature_dim=19, d_model=32, n_layers=1, n_heads=4)

    rows, pred, info = beam_rollout_well(
        model,
        well,
        offsets=np.asarray([-0.02, 0.0, 0.02]),
        spans=np.asarray([12, 24]),
        variant="normal",
        seed=4,
        beam_size=3,
        branch_top_k=2,
    )

    assert rows.size == well.hidden_idx.size
    assert np.array_equal(rows, well.hidden_idx)
    assert np.isfinite(pred).all()
    assert info["beam_size"] == 3
    assert info["branches"] >= 1


def test_collect_on_policy_samples_marks_rollout_source(tmp_path: Path) -> None:
    from mtpnet.discrete_offset import _load_wells
    from mtpnet.lattice_offset import LatticeTransformerScorer, collect_on_policy_samples
    from mtpnet.residual_stack import make_group_folds

    _write_tiny_lattice_data(tmp_path, n_wells=2, n_rows=80)
    well_ids = [f"w{i:02d}" for i in range(2)]
    folds = make_group_folds(well_ids, n_folds=2, seed=5)
    fold_of_well = {str(w): idx for idx, (_, valid) in enumerate(folds) for w in valid}
    wells = _load_wells(tmp_path, k_wells=-1, fold_of_well=fold_of_well)
    model = LatticeTransformerScorer(feature_dim=19, d_model=32, n_layers=1, n_heads=4)

    samples = collect_on_policy_samples(
        model,
        wells,
        offsets=np.asarray([-0.02, 0.0, 0.02]),
        spans=np.asarray([12, 24]),
        variant="normal",
        seed=5,
        beam_size=3,
        branch_top_k=2,
        state_stride=12,
    )

    assert samples
    assert {sample.source for sample in samples} == {"on_policy"}
    assert all(np.isfinite(sample.features).all() for sample in samples)
    assert all(np.isfinite(sample.cost_rmse).all() for sample in samples)
    assert all(sample.last_tvt is not None and np.isfinite(sample.last_tvt) for sample in samples)
