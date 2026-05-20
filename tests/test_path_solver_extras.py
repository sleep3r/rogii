from __future__ import annotations

import numpy as np
import pandas as pd

from rogii.direct_solver import (
    EnergyContext,
    SolverVariant,
    fit_geo_candidate,
    fit_gr_calibration,
    fit_linear_candidate,
    known_tail_mask,
    robust_line_slope,
    score_candidate_path,
)
from rogii.path_solver_extras import (
    cem_path_search,
    collect_well_signatures,
    compute_well_signature,
    matched_triples,
    signatures_to_frame,
    stage12_path,
    stage1_global_linear,
    stage2_local_refine,
)


def _build_context(seed: int = 7) -> tuple[EnergyContext, "callable[[np.ndarray], float]"]:
    rng = np.random.default_rng(int(seed))
    n = 200
    md = np.arange(n, dtype=float)
    z = 2400.0 - 0.03 * md
    tvt = 1200.0 + 0.07 * md
    tvt_input = tvt.copy()
    tvt_input[150:] = np.nan
    gr = 90.0 + 12.0 * np.sin(tvt / 18.0) + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame(
        {
            "MD": md,
            "X": md * 0.1,
            "Y": md * 0.2,
            "Z": z,
            "GR": gr,
            "TVT_input": tvt_input,
            "TVT": tvt,
            "ANCC": z - (tvt - 1190.0),
            "ASTNU": z - (tvt - 1210.0),
            "ASTNL": z - (tvt - 1230.0),
            "EGFDU": z - (tvt - 1250.0),
            "EGFDL": z - (tvt - 1270.0),
            "BUDA": z - (tvt - 1290.0),
        }
    )
    hidden_idx = np.arange(150, n, dtype=int)
    last_idx = 149
    last_tvt = float(tvt_input[last_idx])
    tail_rows = 60
    tail_mask = known_tail_mask(tvt_input, last_idx, tail_rows)
    tail_slope = robust_line_slope(md[tail_mask], tvt_input[tail_mask], default=0.0)
    linear_path = fit_linear_candidate(
        df, md, z, df["X"].to_numpy(float), df["Y"].to_numpy(float), tvt_input,
        hidden_idx, last_idx, last_tvt, tail_rows,
    )
    geo_path, _, _ = fit_geo_candidate(df, md, z, tvt_input, hidden_idx, last_idx, last_tvt, tail_rows)
    # Use geo_path as the CEM base path so the search is not anchor-pinned.
    typewell_tvt = np.linspace(1180.0, 1260.0, 80)
    typewell_gr = 90.0 + 12.0 * np.sin(typewell_tvt / 18.0)
    typewell = (typewell_tvt, typewell_gr)
    cal_a, cal_b, _ = fit_gr_calibration(typewell, tvt_input, gr, tail_mask)
    ctx = EnergyContext(
        md=md, gr=gr, z=z, typewell=typewell, hidden_indices=hidden_idx,
        last_idx=last_idx, last_tvt=last_tvt, tail_slope=tail_slope,
        cal_a=cal_a, cal_b=cal_b,
        linear_path=linear_path, geo_path=geo_path, anchor_path=geo_path,
    )
    energy_variant = SolverVariant(
        name="_eval",
        gr_weight=1.0, geo_weight=0.5,
        anchor_weight=0.0, slope_weight=0.2, endpoint_weight=0.0,
        max_gate=1.0, min_improvement=0.0,
    )
    def energy_fn(path: np.ndarray) -> float:
        return score_candidate_path(
            "_eval", path, energy_variant, typewell, gr, hidden_idx, md,
            linear_path, geo_path, None, cal_a, cal_b, tail_slope,
        )
    return ctx, energy_fn


def test_cem_path_search_returns_finite_paths() -> None:
    ctx, energy_fn = _build_context(seed=23)
    outputs, diag = cem_path_search(
        ctx, energy_fn, n_iter=3, pop_size=64, elite_frac=0.20, seed=23,
    )
    assert {"cem_raw", "cem_top_median"} <= set(outputs)
    assert np.isfinite(outputs["cem_raw"]).all()
    assert np.isfinite(outputs["cem_top_median"]).all()
    assert np.isfinite(diag["cem_best_score"])
    assert abs(diag["cem_best_offset"]) <= 24.0
    assert abs(diag["cem_best_curvature"]) <= 16.0


def test_stage1_global_linear_picks_finite_pair() -> None:
    ctx, energy_fn = _build_context(seed=31)
    path, diag = stage1_global_linear(
        ctx, energy_fn,
        a_grid=np.array([-0.10, -0.05, 0.0, 0.05, 0.07, 0.10]),
        b_grid=np.array([-0.5, 0.0, 0.5]),
    )
    assert np.isfinite(path[ctx.hidden_indices]).all()
    assert np.isfinite(diag["stage1_best_a"])
    assert np.isfinite(diag["stage1_best_b"])


def test_stage2_local_refine_respects_max_offset() -> None:
    ctx, energy_fn = _build_context(seed=37)
    base, _ = stage1_global_linear(ctx, energy_fn,
        a_grid=np.array([-0.10, 0.0, 0.07, 0.10]),
        b_grid=np.array([-0.5, 0.0, 0.5]))
    refined, diag = stage2_local_refine(
        ctx, energy_fn, base, n_knots=6, max_offset=8.0, passes=1,
    )
    assert refined.shape == base.shape
    assert np.isfinite(refined[ctx.hidden_indices]).all()
    assert diag["stage2_max_offset_used"] <= 8.0 + 1e-6


def test_stage12_path_chains_stage1_and_stage2() -> None:
    ctx, energy_fn = _build_context(seed=41)
    outputs, diag = stage12_path(ctx, energy_fn, n_knots=6, max_offset=8.0, passes=1)
    assert "stage1_path" in outputs and "stage12_path" in outputs
    assert np.isfinite(outputs["stage12_path"][ctx.hidden_indices]).all()
    assert diag["stage2_max_offset_used"] <= 8.0 + 1e-6


def test_compute_well_signature_returns_expected_keys() -> None:
    n = 80
    md = np.arange(n, dtype=float)
    tvt = 1000.0 + 0.05 * md
    tvt_input = tvt.copy()
    tvt_input[60:] = np.nan
    df = pd.DataFrame(
        {
            "MD": md,
            "Z": 2000.0 - 0.04 * md,
            "GR": 80.0 + np.sin(md / 7.0),
            "TVT_input": tvt_input,
            "TVT": tvt,
        }
    )
    hidden = np.arange(60, 80)
    sig = compute_well_signature("well0001", df, hidden)
    assert sig is not None
    assert sig.hidden_len == 20
    assert sig.tail_len > 0
    frame = signatures_to_frame([sig])
    assert "log_md_span" in frame.columns


def test_matched_triples_picks_neighbors() -> None:
    train_rows = [
        {
            "well": f"train{i:03d}",
            "hidden_len": int(50 + i * 5),
            "tail_len": int(200 + i * 10),
            "log_md_span": float(np.log(100 + i * 8)),
            "log_gr_mean": float(np.log(80 + i)),
            "log_gr_std": float(np.log(5 + (i % 4))),
            "z_drift": float(i * 1.5),
            "last_tvt": float(1000 + i * 2),
        }
        for i in range(40)
    ]
    test_rows = [
        {
            "well": "test01",
            "hidden_len": 60,
            "tail_len": 220,
            "log_md_span": float(np.log(108)),
            "log_gr_mean": float(np.log(81)),
            "log_gr_std": float(np.log(6)),
            "z_drift": 3.0,
            "last_tvt": 1004.0,
        },
        {
            "well": "test02",
            "hidden_len": 110,
            "tail_len": 320,
            "log_md_span": float(np.log(180)),
            "log_gr_mean": float(np.log(95)),
            "log_gr_std": float(np.log(8)),
            "z_drift": 15.0,
            "last_tvt": 1040.0,
        },
        {
            "well": "test03",
            "hidden_len": 200,
            "tail_len": 450,
            "log_md_span": float(np.log(280)),
            "log_gr_mean": float(np.log(105)),
            "log_gr_std": float(np.log(9)),
            "z_drift": 30.0,
            "last_tvt": 1080.0,
        },
    ]
    triples = matched_triples(
        pd.DataFrame(train_rows),
        pd.DataFrame(test_rows),
        trials=5,
        triple_size=3,
        seed=11,
        candidate_k=8,
    )
    assert len(triples) == 5
    for trip in triples:
        assert len(trip) == 3
        assert len(set(trip)) == 3


def test_collect_well_signatures_train(tmp_path) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    n = 90
    md = np.arange(n, dtype=float)
    tvt = 1000.0 + 0.05 * md
    tvt_input = tvt.copy()
    tvt_input[70:] = np.nan
    pd.DataFrame(
        {
            "MD": md, "X": md, "Y": md * 0.2,
            "Z": 2000.0 - 0.04 * md, "GR": 80.0 + np.sin(md / 5.0),
            "TVT_input": tvt_input, "TVT": tvt,
        }
    ).to_csv(train_dir / "abc__horizontal_well.csv", index=False)
    frame = collect_well_signatures(tmp_path, subset="train")
    assert not frame.empty
    assert "log_md_span" in frame.columns
    assert (frame["hidden_len"] > 0).all()
