from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _make_horizontal(well_id: str, *, shift: float = 0.0, slope: float = 2.0) -> pd.DataFrame:
    rows = []
    for row_idx in range(10):
        tvt = 100.0 + shift + slope * row_idx
        rows.append(
            {
                "id": f"{well_id}_{row_idx}",
                "well_id": well_id,
                "row_idx": row_idx,
                "MD": float(row_idx),
                "X": float(row_idx),
                "Y": float(row_idx % 2),
                "Z": -1000.0 - row_idx,
                "GR": 50.0 + np.sin(row_idx / 2.0) * 10.0,
                "TVT": tvt,
                "TVT_input": tvt if row_idx < 4 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _make_typewell(well_id: str, *, phase: int = 0) -> pd.DataFrame:
    x = np.arange(30, dtype=np.float32)
    gr = np.sin((x + phase) / 3.0) * 20.0 + 80.0
    return pd.DataFrame({"well_id": well_id, "TVT": 90.0 + x * 2.0, "GR": gr})


def test_shifted_typewell_correlation_finds_offset_copy() -> None:
    from mtpnet.shared_typewell_neighbour import best_shifted_corr

    base = np.array([0, 1, 3, 8, 13, 8, 3, 1, 0], dtype=np.float32)
    shifted = np.roll(base, 2)

    corr, shift = best_shifted_corr(base, shifted, max_shift_bins=4)

    assert corr > 0.99
    assert abs(shift) == 2


def test_neighbour_candidate_generation_uses_only_train_wells() -> None:
    from mtpnet.shared_typewell_neighbour import (
        SharedTypewellNeighbourConfig,
        build_neighbour_candidates_from_frames,
    )

    frame = pd.concat(
        [
            _make_horizontal("valid", shift=0.0),
            _make_horizontal("train_a", shift=5.0),
            _make_horizontal("train_b", shift=-5.0),
        ],
        ignore_index=True,
    )
    typewells = {
        "valid": _make_typewell("valid", phase=0),
        "train_a": _make_typewell("train_a", phase=0),
        "train_b": _make_typewell("train_b", phase=5),
    }

    candidates, neighbours = build_neighbour_candidates_from_frames(
        frame,
        typewells,
        train_wells=["train_a", "train_b"],
        valid_wells=["valid"],
        cfg=SharedTypewellNeighbourConfig(rows_per_step=1, top_k_neighbours=1),
    )

    assert set(candidates["well_id"]) == {"valid"}
    assert set(candidates["candidate"]).issuperset({"known_tail_anchor", "neighbour_top1"})
    assert set(neighbours["query_well_id"]) == {"valid"}
    assert set(neighbours["neighbour_well_id"]) == {"train_a"}
    assert "valid" not in set(neighbours["neighbour_well_id"])


def test_neighbour_candidate_generation_does_not_require_hidden_tvt() -> None:
    from mtpnet.shared_typewell_neighbour import (
        SharedTypewellNeighbourConfig,
        build_neighbour_candidates_from_frames,
    )

    frame = pd.concat(
        [_make_horizontal("valid"), _make_horizontal("train_a", shift=3.0)],
        ignore_index=True,
    )
    no_target_for_valid = frame.copy()
    no_target_for_valid.loc[no_target_for_valid["well_id"] == "valid", "TVT"] = np.nan
    typewells = {
        "valid": _make_typewell("valid"),
        "train_a": _make_typewell("train_a"),
    }

    candidates, _ = build_neighbour_candidates_from_frames(
        no_target_for_valid,
        typewells,
        train_wells=["train_a"],
        valid_wells=["valid"],
        cfg=SharedTypewellNeighbourConfig(rows_per_step=1, top_k_neighbours=1),
    )

    assert "TVT" not in candidates.columns
    assert candidates["pred_tvt"].notna().all()


def test_shared_typewell_neighbour_smoke_writes_artifacts(tmp_path: Path) -> None:
    from mtpnet.shared_typewell_neighbour import (
        SharedTypewellNeighbourConfig,
        run_shared_typewell_neighbour_audit_from_frames,
    )

    frame = pd.concat(
        [
            _make_horizontal("a", shift=0.0),
            _make_horizontal("b", shift=4.0),
            _make_horizontal("c", shift=-4.0),
            _make_horizontal("d", shift=8.0),
        ],
        ignore_index=True,
    )
    typewells = {
        "a": _make_typewell("a", phase=0),
        "b": _make_typewell("b", phase=0),
        "c": _make_typewell("c", phase=2),
        "d": _make_typewell("d", phase=5),
    }

    metrics = run_shared_typewell_neighbour_audit_from_frames(
        frame,
        typewells,
        output_dir=tmp_path,
        cfg=SharedTypewellNeighbourConfig(
            rows_per_step=1,
            n_folds=2,
            top_k_neighbours=2,
            n_signature_bins=32,
        ),
    )

    assert metrics["wells"] == 4
    assert metrics["candidates"] >= 2
    assert "oracle" in metrics
    assert (tmp_path / "typewell_neighbours.parquet").exists()
    assert (tmp_path / "neighbour_candidate_predictions.parquet").exists()
    assert (tmp_path / "shared_typewell_neighbour_metrics.json").exists()
    assert (tmp_path / "shared_typewell_neighbour_report.md").exists()
