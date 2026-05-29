from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch


def _write_tiny_tto_data(root: Path, n_wells: int = 6, n_rows: int = 72) -> None:
    offsets = np.asarray([-0.02, 0.0, 0.02], dtype=float)
    for well_idx in range(n_wells):
        well_id = f"w{well_idx:02d}"
        offset = float(offsets[well_idx % offsets.size])
        z = -100.0 - np.linspace(0.0, 55.0, n_rows)
        tvt = np.zeros(n_rows, dtype=float)
        tvt[0] = 500.0 + well_idx
        for i in range(1, n_rows):
            tvt[i] = tvt[i - 1] - (z[i] - z[i - 1]) + offset
        # Monotonic typewell signal makes the correct offset identifiable in
        # a small unit test without relying on oscillatory local minima.
        gr = 0.2 * tvt + 5.0
        hidden_start = int(0.65 * n_rows)
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
        tw_tvt = np.linspace(float(tvt.min()) - 20.0, float(tvt.max()) + 20.0, 160)
        pd.DataFrame({"TVT": tw_tvt, "GR": 0.2 * tw_tvt + 5.0}).to_csv(
            root / f"{well_id}__typewell.csv", index=False
        )


def test_torch_interp_1d_is_piecewise_linear_and_differentiable() -> None:
    from mtpnet.offset_tto import torch_interp_1d

    xp = torch.tensor([0.0, 1.0, 2.0])
    fp = torch.tensor([0.0, 10.0, 20.0])
    x = torch.tensor([0.25, 1.50], requires_grad=True)

    y = torch_interp_1d(x, xp, fp)
    assert torch.allclose(y, torch.tensor([2.5, 15.0]))
    y.sum().backward()
    assert x.grad is not None
    assert torch.allclose(x.grad, torch.tensor([10.0, 10.0]))


def test_refine_offsets_moves_best_candidate_toward_true_offset(tmp_path: Path) -> None:
    from mtpnet.discrete_offset import _load_wells
    from mtpnet.offset_tto import refine_well_offsets
    from mtpnet.residual_stack import make_group_folds

    _write_tiny_tto_data(tmp_path, n_wells=3, n_rows=72)
    well_ids = [f"w{i:02d}" for i in range(3)]
    folds = make_group_folds(well_ids, n_folds=2, seed=1)
    fold_of_well = {str(w): idx for idx, (_, valid) in enumerate(folds) for w in valid}
    wells = _load_wells(tmp_path, k_wells=-1, fold_of_well=fold_of_well)
    well = wells[0]  # true offset is -0.02 by construction

    result = refine_well_offsets(
        well,
        initial_offsets=np.asarray([-0.04, 0.0, 0.04]),
        steps=80,
        learning_rate=0.08,
        max_delta=0.04,
        offset_l2=0.0,
        normalize_gr=False,
        variant="normal",
        seed=1,
    )

    assert abs(result.selected_offset - (-0.02)) < abs(0.0 - (-0.02))
    assert result.selected_loss < result.initial_loss
    assert result.pred_tvt.shape == well.hidden_idx.shape


def test_offset_tto_smoke_writes_artifacts(tmp_path: Path) -> None:
    from mtpnet.offset_tto import OffsetTTOConfig, run_offset_tto

    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    _write_tiny_tto_data(data_dir, n_wells=6, n_rows=64)

    metrics = run_offset_tto(
        OffsetTTOConfig(
            data_dir=data_dir,
            output_dir=out_dir,
            posterior_scores_path=None,
            offset_grid="-0.04:0.04:0.04",
            top_k=3,
            n_folds=2,
            k_wells=-1,
            steps=12,
            learning_rate=0.08,
            max_delta=0.04,
            include_nulls=True,
            normalize_gr=False,
        )
    )

    assert metrics["experiment"] == "offset_tto_v0"
    assert (out_dir / "offset_tto_metrics.json").exists()
    assert (out_dir / "offset_tto_report.md").exists()
    assert (out_dir / "offset_tto_predictions.parquet").exists()
    preds = pd.read_parquet(out_dir / "offset_tto_predictions.parquet")
    assert {"id", "well_id", "row_idx", "pred_tvt", "candidate", "selected_offset"}.issubset(
        preds.columns
    )
    assert preds["pred_tvt"].notna().all()
    assert any(item["candidate"] == "tto_normal" for item in metrics["candidates"])
