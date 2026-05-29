from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _tiny_horizontal() -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for idx in range(12):
        tvt = 100.0 + 2.0 * idx
        known = idx < 3 or idx >= 10
        rows.append(
            {
                "id": f"w_{idx}",
                "well_id": "w",
                "row_idx": idx,
                "MD": float(idx),
                "X": float(idx),
                "Y": 0.0,
                "Z": -float(idx),
                "GR": np.sin(tvt / 8.0),
                "TVT": tvt,
                "TVT_input": tvt if known else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _tiny_typewell() -> pd.DataFrame:
    tvt = np.linspace(90.0, 140.0, 101)
    return pd.DataFrame({"TVT": tvt, "GR": np.sin(tvt / 8.0)})


def _heel_only_horizontal(n_rows: int = 96, n_known_heel: int = 6) -> pd.DataFrame:
    """Mimic real ROGII layout: TVT_input known only at the heel, hidden toe.

    With this shape the legacy `bridge` coord proxy clips to a constant on
    the hidden region, which used to silently produce score==0 for every
    template variant. The default `bridge_slope` proxy must keep coord
    non-constant on the hidden region.
    """
    rng = np.random.default_rng(0)
    rows: list[dict[str, float | str]] = []
    for idx in range(n_rows):
        tvt = 100.0 + 0.25 * idx + 1.5 * np.sin(idx / 4.0) + 0.05 * rng.normal()
        known = idx < n_known_heel
        rows.append(
            {
                "id": f"hw_{idx}",
                "well_id": "hw",
                "row_idx": idx,
                "MD": float(idx),
                "X": float(idx),
                "Y": 0.0,
                "Z": -float(idx),
                "GR": float(np.sin(tvt / 8.0)),
                "TVT": float(tvt),
                "TVT_input": float(tvt) if known else np.nan,
            }
        )
    return pd.DataFrame(rows)


def test_schema_guard_rejects_forbidden_columns() -> None:
    from promising.template_envelope_v0.data import assert_schema_safe_columns

    assert_schema_safe_columns(["MD", "X", "Y", "Z", "GR", "TVT_input"])
    with pytest.raises(ValueError, match="forbidden"):
        assert_schema_safe_columns(["MD", "GR", "TVT", "Geology"])


def test_centered_template_path_uses_scale_without_absolute_tvt_blowup() -> None:
    from promising.template_envelope_v0.templates import build_template_path

    coord = np.array([100.0, 110.0, 120.0])
    path = build_template_path(coord, scale=1.1, offset=5.0)

    np.testing.assert_allclose(path, [104.0, 115.0, 126.0])


def test_template_score_prefers_matching_gr_over_shuffled() -> None:
    from promising.template_envelope_v0.scoring import score_template_match

    horizontal = np.sin(np.linspace(0.0, 4.0, 32))
    matching = horizontal.copy()
    shuffled = horizontal[::-1].copy()

    assert score_template_match(matching, horizontal) > score_template_match(shuffled, horizontal)


def test_envelope_midpoint_uses_selected_template_quantiles() -> None:
    from promising.template_envelope_v0.envelope import build_envelope_from_paths

    paths = np.array(
        [
            [90.0, 100.0, 110.0],
            [100.0, 110.0, 120.0],
            [110.0, 120.0, 130.0],
        ]
    )
    envelope = build_envelope_from_paths(paths)

    np.testing.assert_allclose(envelope.low, [90.0, 100.0, 110.0])
    np.testing.assert_allclose(envelope.high, [110.0, 120.0, 130.0])
    np.testing.assert_allclose(envelope.mid, [100.0, 110.0, 120.0])


def test_template_envelope_smoke_writes_schema_safe_artifacts(tmp_path: Path) -> None:
    from promising.template_envelope_v0.config import TemplateEnvelopeConfig
    from promising.template_envelope_v0.cli import run_template_envelope

    data_dir = tmp_path / "data" / "train"
    data_dir.mkdir(parents=True)
    _tiny_horizontal().to_csv(data_dir / "w__horizontal_well.csv", index=False)
    _tiny_typewell().to_csv(data_dir / "w__typewell.csv", index=False)

    metrics = run_template_envelope(
        TemplateEnvelopeConfig(
            data_dir=data_dir,
            output_dir=tmp_path / "out",
            rows_per_step=1,
            scale_grid=(1.0,),
            offset_grid=(-4.0, 0.0, 4.0),
            top_n_templates=2,
            k_wells=-1,
            seed=7,
            min_template_score=0.0,
        )
    )

    assert metrics["wells"] == 1
    assert metrics["variants"]["normal_GR"]["candidate_rows"] > 0
    assert (tmp_path / "out" / "metrics.json").exists()
    assert (tmp_path / "out" / "report.md").exists()
    assert (tmp_path / "out" / "envelope_candidates.parquet").exists()
    assert (tmp_path / "out" / "step_scores.parquet").exists()

    candidates = pd.read_parquet(tmp_path / "out" / "envelope_candidates.parquet")
    assert {"id", "well_id", "row_idx", "step", "candidate", "pred_tvt"}.issubset(
        candidates.columns
    )
    assert {"TVT", "true_TVT", "Geology", "ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"}.isdisjoint(
        candidates.columns
    )
    assert candidates["candidate"].str.startswith("template_envelope_").all()


def test_bridge_slope_keeps_coord_non_constant_on_heel_only_well() -> None:
    """Regression for the v0 smoke bug: heel-only TVT_input makes the legacy
    `bridge` coord proxy clip to a constant on the hidden region. The new
    default `bridge_slope` proxy must keep coord varying everywhere.
    """
    from promising.template_envelope_v0.data import compress_horizontal

    horizontal = _heel_only_horizontal(n_rows=96, n_known_heel=6)

    comp_bridge = compress_horizontal(horizontal, rows_per_step=4, coord_source="bridge")
    comp_slope = compress_horizontal(horizontal, rows_per_step=4, coord_source="bridge_slope")

    hidden_mask = comp_bridge["hidden_mask"].to_numpy(dtype=bool)
    assert hidden_mask.sum() >= 5

    bridge_hidden = comp_bridge["coord"].to_numpy()[hidden_mask]
    slope_hidden = comp_slope["coord"].to_numpy()[hidden_mask]

    # Legacy bridge collapses to one value on the hidden tail (this is the bug).
    assert np.nanstd(bridge_hidden) < 1e-6
    # bridge_slope extrapolates linearly and stays varied.
    assert np.nanstd(slope_hidden) > 1.0


def test_normal_GR_beats_shuffled_GR_on_synthetic_heel_only_well(tmp_path: Path) -> None:
    """End-to-end sanity: on a well whose GR is a deterministic function of TVT
    and whose typewell shares the same function, the normal variant must
    score (and predict) strictly better than the shuffled-hidden variant.
    """
    from promising.template_envelope_v0.cli import run_template_envelope
    from promising.template_envelope_v0.config import TemplateEnvelopeConfig

    data_dir = tmp_path / "data" / "train"
    data_dir.mkdir(parents=True)
    _heel_only_horizontal(n_rows=96, n_known_heel=6).to_csv(
        data_dir / "hw__horizontal_well.csv", index=False
    )
    _tiny_typewell().to_csv(data_dir / "hw__typewell.csv", index=False)

    metrics = run_template_envelope(
        TemplateEnvelopeConfig(
            data_dir=data_dir,
            output_dir=tmp_path / "out",
            rows_per_step=4,
            scale_grid=(0.85, 0.95, 1.0, 1.05, 1.15),
            offset_grid=(-4.0, -2.0, 0.0, 2.0, 4.0),
            top_n_templates=3,
            k_wells=-1,
            seed=11,
            coord_source="bridge_slope",
            min_template_score=0.0,
        )
    )

    normal = metrics["variants"]["normal_GR"]
    shuffled = metrics["variants"]["shuffled_hidden_GR"]
    assert normal["top_template_score"] > shuffled["top_template_score"], metrics
    assert normal["step_rmse"] < shuffled["step_rmse"], metrics
    assert metrics["sanity"]["normal_beats_shuffled_score"] is True
    assert metrics["sanity"]["verdict"] == "GO_to_full_run"
    assert metrics["skipped_wells"]["count"] == 0


def test_select_template_indices_respects_min_score_threshold() -> None:
    from promising.template_envelope_v0.envelope import select_template_indices

    scores = np.array([-0.5, 0.1, 0.4, np.nan, -np.inf, 0.2], dtype=np.float64)
    no_threshold = select_template_indices(scores, top_n=10)
    with_threshold = select_template_indices(scores, top_n=10, min_score=0.0)

    # Without threshold, finite scores including negative ones are eligible.
    assert set(no_threshold.tolist()) == {0, 1, 2, 5}
    # With min_score=0.0, only strictly positive scores are eligible.
    assert set(with_threshold.tolist()) == {1, 2, 5}


def test_constant_coord_well_is_skipped_and_recorded(tmp_path: Path) -> None:
    """If `coord_source='bridge'` is forced on a heel-only well, the v0
    pipeline must refuse to score it instead of silently emitting an
    all-zeros envelope. The skip should be recorded in metrics.
    """
    from promising.template_envelope_v0.cli import run_template_envelope
    from promising.template_envelope_v0.config import TemplateEnvelopeConfig

    data_dir = tmp_path / "data" / "train"
    data_dir.mkdir(parents=True)
    _heel_only_horizontal(n_rows=96, n_known_heel=6).to_csv(
        data_dir / "hw__horizontal_well.csv", index=False
    )
    _tiny_typewell().to_csv(data_dir / "hw__typewell.csv", index=False)

    metrics = run_template_envelope(
        TemplateEnvelopeConfig(
            data_dir=data_dir,
            output_dir=tmp_path / "out",
            rows_per_step=4,
            scale_grid=(1.0,),
            offset_grid=(0.0,),
            top_n_templates=1,
            k_wells=-1,
            seed=1,
            coord_source="bridge",
            min_template_score=0.0,
        )
    )

    assert metrics["skipped_wells"]["count"] == 1
    assert "constant_coord_on_hidden" in metrics["skipped_wells"]["by_reason"]
    # Nothing was scored, so no variant metrics should be present.
    assert metrics["variants"] == {}

