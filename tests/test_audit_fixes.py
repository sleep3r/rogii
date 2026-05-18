from __future__ import annotations

import csv

import numpy as np
import pandas as pd
import pytest

from rogii.constants import FORMATIONS
from rogii.diagnostics import append_run_registry
from rogii.features import (
    FEATURE_CACHE_SCHEMA_VERSION,
    build_target_mask,
    build_training_table,
    build_well_features,
    feature_cache_path,
)
from rogii.modeling import (
    EnsembleRegressor,
    ResidualModel,
    apply_postprocess,
    make_xgboost,
)
from rogii.pipeline import assert_fold_context_safe
from rogii.spatial import KaggleTopContext, context_key_for_paths, context_well_overlap
from rogii.submission import predict_test
from rogii.top_signals import downsample_indices, lowres_dtw_signal


def minimal_config() -> dict:
    return {
        "data": {"target_rows": "hidden_only"},
        "features": {
            "tail_windows": [2],
            "rolling_windows": [3],
            "prediction_baseline": "last_known_tvt",
            "cache": {"enabled": False},
            "include_typewell": False,
            "include_kaggle_top_signals": False,
        },
        "postprocess": {
            "residual_weight": 1.0,
            "residual_clip": None,
            "notebook_blend": {"enabled": False},
            "smoothing": {"enabled": False},
        },
    }


def write_dwt_synthetic_well(tmp_path, name: str = "abc12345") -> tuple:
    n = 18
    idx = np.arange(n, dtype=float)
    tvt = 100.0 + 0.55 * idx
    known = idx < 12
    tvt_input = np.where(known, tvt, np.nan)
    z = 800.0 + 0.15 * idx
    x = 1000.0 + 5.0 * idx
    y = 2000.0 + 2.0 * idx
    gr = 80.0 + 8.0 * np.sin(idx / 3.0)
    frame = pd.DataFrame(
        {
            "MD": idx,
            "X": x,
            "Y": y,
            "Z": z,
            "GR": gr,
            "TVT_input": tvt_input,
            "TVT": tvt,
        }
    )
    for offset, formation in enumerate(FORMATIONS):
        frame[formation] = z + tvt + 10.0 * offset
    horizontal = tmp_path / f"{name}__horizontal_well.csv"
    frame.to_csv(horizontal, index=False)

    tw_tvt = np.linspace(95.0, 115.0, 80)
    typewell = pd.DataFrame(
        {
            "TVT": tw_tvt,
            "GR": 80.0 + 8.0 * np.sin((tw_tvt - 100.0) / 1.65 / 3.0),
            "Geology": ["ANCC"] * len(tw_tvt),
        }
    )
    typewell.to_csv(tmp_path / f"{name}__typewell.csv", index=False)
    return horizontal, frame


def dwt_config() -> dict:
    config = minimal_config()
    config["features"]["include_typewell"] = True
    config["features"]["include_kaggle_top_signals"] = True
    config["features"]["kaggle_top"] = {
        "mode": "notebook",
        "beam_configs": [[4, 6.0, 40.0, 1, "cons"], [4, 6.0, 40.0, 1, "sm5"]],
        "ncc_windows": [2, 3, 4],
        "ncc_stride": 1,
        "dtw_enabled": True,
        "dtw_max_query_points": 32,
        "dtw_max_ref_points": 32,
        "dtw_radii": [2, 4],
        "dtw_stochastic_enabled": True,
        "dtw_stochastic_radius": 2,
        "dtw_stochastic_k": 2,
        "dtw_stochastic_temperature": 1.0,
        "dwt_enabled": False,
        "particle_enabled": True,
        "particle_count": 32,
        "ancc_particle_count": 32,
        "spatial_k": 1,
        "dense_k": 1,
        "dense_fetch": 4,
        "dense_query_chunk": 2,
        "dense_samples_per_well": 8,
    }
    return config


def reference_lowres_dtw_signal(
    full_gr: np.ndarray,
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
    max_query_points: int,
    max_ref_points: int,
    radius: int,
) -> np.ndarray:
    q_idx = downsample_indices(len(full_gr), max_query_points)
    r_idx = downsample_indices(len(tw_gr), max_ref_points)
    q = full_gr[q_idx]
    r = tw_gr[r_idx]
    q = (q - np.nanmean(q)) / (np.nanstd(q) + 1e-6)
    r = (r - np.nanmean(r)) / (np.nanstd(r) + 1e-6)

    n = len(q)
    m = len(r)
    inf = 1e18
    dp = np.full((n, m), inf, dtype=float)
    parent = np.full((n, m), -1, dtype=np.int8)
    slope = (m - 1) / max(n - 1, 1)
    radius = max(int(radius), 1)

    for i in range(n):
        center = int(round(i * slope))
        lo = max(0, center - radius)
        hi = min(m - 1, center + radius)
        for j in range(lo, hi + 1):
            cost = (q[i] - r[j]) ** 2
            if i == 0 and j == 0:
                dp[i, j] = cost
                continue
            choices = []
            if i > 0 and j > 0:
                choices.append((dp[i - 1, j - 1], 0))
            if i > 0:
                choices.append((dp[i - 1, j], 1))
            if j > 0:
                choices.append((dp[i, j - 1], 2))
            prev_cost, prev_code = min(choices, key=lambda item: item[0])
            dp[i, j] = cost + prev_cost
            parent[i, j] = prev_code

    j_end = int(np.nanargmin(dp[-1]))
    i = n - 1
    j = j_end
    j_for_i = np.zeros(n, dtype=int)
    while i >= 0 and j >= 0:
        j_for_i[i] = j
        code = parent[i, j]
        if i == 0 and j == 0:
            break
        if code == 0:
            i -= 1
            j -= 1
        elif code == 1:
            i -= 1
        else:
            j -= 1
    coarse_tvt = tw_tvt[r_idx[j_for_i]]
    return np.interp(np.arange(len(full_gr)), q_idx, coarse_tvt).astype(float)


def test_build_target_mask_always_returns_bool_array() -> None:
    missing_tvt = pd.DataFrame({"TVT_input": [np.nan, 1.0, np.nan]})
    mask = build_target_mask(missing_tvt, "hidden_only")
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == bool
    assert mask.tolist() == [False, False, False]

    all_rows = pd.DataFrame({"TVT": [10.0, np.nan, 12.0], "TVT_input": [1, 2, 3]})
    assert build_target_mask(all_rows, "all").tolist() == [True, False, True]

    hidden_only = pd.DataFrame(
        {"TVT": [10.0, 11.0, np.nan, 13.0], "TVT_input": [10.0, np.nan, np.nan, 13.0]}
    )
    assert build_target_mask(hidden_only, "hidden_only").tolist() == [
        False,
        True,
        False,
        False,
    ]


def test_feature_schema_keeps_only_canonical_last_known_offsets(tmp_path) -> None:
    path = tmp_path / "abc12345__horizontal_well.csv"
    pd.DataFrame(
        {
            "MD": [0, 1, 2, 3, 4],
            "X": [0, 1, 2, 3, 4],
            "Y": [0, 0, 0, 0, 0],
            "Z": [100, 101, 102, 103, 104],
            "GR": [80, 82, 85, 86, 88],
            "TVT_input": [10.0, 11.0, 12.0, np.nan, np.nan],
            "TVT": [10.0, 11.0, 12.0, 13.0, 14.0],
        }
    ).to_csv(path, index=False)

    wf = build_well_features(path, minimal_config(), train=True)
    assert "idx_from_last_known" in wf.features.columns
    assert "md_from_last_known" in wf.features.columns
    assert "idx_since" not in wf.features.columns
    assert "md_since" not in wf.features.columns


def test_training_target_is_last_known_residual(tmp_path) -> None:
    path, frame = write_dwt_synthetic_well(tmp_path)
    config = minimal_config()

    X, residual, _groups, flat, y_true = build_training_table([path], config)
    last_known = float(frame.loc[frame["TVT_input"].notna(), "TVT_input"].iloc[-1])
    hidden_tvt = frame.loc[frame["TVT_input"].isna(), "TVT"].to_numpy(dtype=float)

    assert np.allclose(X["last_known_tvt"].to_numpy(), last_known)
    assert np.allclose(flat, last_known)
    assert np.allclose(y_true, hidden_tvt)
    assert np.allclose(residual, hidden_tvt - last_known)


def test_dwt_repro_feature_block_is_present(tmp_path) -> None:
    path, _frame = write_dwt_synthetic_well(tmp_path)
    config = dwt_config()
    context = KaggleTopContext([path], config)

    wf = build_well_features(path, config, train=True, top_context=context)
    hidden = wf.target_mask
    required = [
        "pf_ancc_delta",
        "pf_z_delta",
        "dtw_ens_d",
        "dtw_stoch_std",
        "tddtw0",
        "tdpf0",
        "tvt_dense_d",
        "beam_cons_d",
        "sc_cons_d",
    ]

    for column in required:
        assert column in wf.features.columns
        values = wf.features.loc[hidden, column].to_numpy(dtype=float)
        assert np.isfinite(values).all()


def test_context_key_is_stable_and_changes_with_context_wells(tmp_path) -> None:
    path_a, _frame_a = write_dwt_synthetic_well(tmp_path, "aaaa1111")
    path_b, _frame_b = write_dwt_synthetic_well(tmp_path, "bbbb2222")
    config = dwt_config()

    key_a1 = context_key_for_paths([path_a], config)
    key_a2 = context_key_for_paths([path_a], config)
    key_ab = context_key_for_paths([path_a, path_b], config)

    assert key_a1 == key_a2
    assert key_a1 != key_ab


def test_feature_cache_path_includes_context_key(tmp_path) -> None:
    path_a, _frame_a = write_dwt_synthetic_well(tmp_path, "aaaa1111")
    path_b, _frame_b = write_dwt_synthetic_well(tmp_path, "bbbb2222")
    config = dwt_config()
    config["features"]["cache"] = {"enabled": True, "dir": str(tmp_path / "cache")}
    key_a = context_key_for_paths([path_a], config)
    key_ab = context_key_for_paths([path_a, path_b], config)

    cache_a = feature_cache_path(path_a, config, train=True, context_key=key_a)
    cache_ab = feature_cache_path(path_a, config, train=True, context_key=key_ab)

    assert cache_a is not None
    assert cache_ab is not None
    assert cache_a != cache_ab


def test_fold_context_excludes_validation_wells(tmp_path) -> None:
    path_a, _frame_a = write_dwt_synthetic_well(tmp_path, "aaaa1111")
    path_b, _frame_b = write_dwt_synthetic_well(tmp_path, "bbbb2222")
    config = dwt_config()
    context = KaggleTopContext([path_a], config)

    assert not context_well_overlap(context, [path_b])
    assert_fold_context_safe(context, [path_b], fold_id=1)
    assert context_well_overlap(context, [path_a]) == {"aaaa1111"}
    with pytest.raises(RuntimeError, match="validation wells"):
        assert_fold_context_safe(context, [path_a], fold_id=1)


def test_ensemble_predict_uses_final_models() -> None:
    class ConstantModel(ResidualModel):
        def __init__(self, value: float) -> None:
            self.value = value

        def fit(self, X: pd.DataFrame, y: np.ndarray, **_) -> "ConstantModel":
            return self

        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return np.full(len(X), self.value, dtype=float)

    config = {
        "model": {"base_models": [{"id": "a"}, {"id": "b"}], "blend": {}},
        "validation": {"n_splits": 2},
    }
    model = EnsembleRegressor(config, seed=42)
    model.base_names = ["a", "b"]
    model.weights = np.array([0.25, 0.75])
    model.fold_models = [[ConstantModel(100.0)], [ConstantModel(200.0)]]
    model.final_models = [ConstantModel(1.0), ConstantModel(3.0)]

    pred = model.predict(pd.DataFrame({"x": [0.0, 1.0]}))
    assert np.allclose(pred, [2.5, 2.5])


def test_run_registry_appends_required_columns(tmp_path) -> None:
    path = tmp_path / "runs.csv"
    metrics = {
        "train": {"rows": 10, "wells": 2},
        "features": {"count": 5, "schema_version": FEATURE_CACHE_SCHEMA_VERSION},
        "cv": {
            "rmse": 1.23,
            "best_residual_weight": 1.0,
            "best_notebook_blend": {"alpha": 0.98, "tau": 50.0, "w_pf": 0.05},
            "best_smoothing": None,
        },
        "diagnostics": {
            "global_rmse": 1.23,
            "mean_well_rmse": 1.1,
            "p90_well_rmse": 1.4,
            "worst_well_rmse": 1.5,
            "no_typewell_rmse": None,
            "long_hidden_rmse": 1.2,
            "short_hidden_rmse": 1.0,
        },
    }

    append_run_registry(
        path,
        metrics,
        tmp_path / "config.yml",
        "test_run",
        public_lb=None,
        runtime_seconds=90.0,
        notes="unit",
    )

    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 1
    assert rows[0]["run_id"] == "test_run"
    assert rows[0]["schema_version"] == str(FEATURE_CACHE_SCHEMA_VERSION)
    assert rows[0]["notes"] == "unit"


def test_notebook_postprocess_uses_md_from_last_known() -> None:
    config = minimal_config()
    config["postprocess"]["notebook_blend"] = {
        "enabled": True,
        "pf_column": "kg_pf_ancc_tvt",
        "alpha": 1.0,
        "tau": 100.0,
        "w_pf": 0.0,
    }
    features = pd.DataFrame(
        {
            "last_known_tvt": [10.0, 10.0],
            "md_from_last_known": [0.0, 100.0],
            "kg_pf_ancc_tvt": [10.0, 20.0],
        }
    )
    pred = apply_postprocess(
        flat=np.array([10.0, 10.0]),
        residual=np.array([2.0, 2.0]),
        config=config,
        features=features,
    )
    assert np.allclose(pred, [10.0, 10.0 + 2.0 * (1.0 - np.exp(-1.0))])


def test_lowres_dtw_signal_matches_reference() -> None:
    full_gr = np.array([10.0, 12.0, 11.0, 15.0, 18.0, 17.0, 20.0])
    tw_tvt = np.linspace(100.0, 106.0, 7)
    tw_gr = np.array([9.0, 11.0, 12.0, 14.0, 17.0, 19.0, 21.0])

    actual = lowres_dtw_signal(full_gr, tw_tvt, tw_gr, 7, 7, 2)
    expected = reference_lowres_dtw_signal(full_gr, tw_tvt, tw_gr, 7, 7, 2)
    assert np.allclose(actual, expected)


def test_make_xgboost_keeps_early_stopping_rounds_in_estimator_params() -> None:
    model = make_xgboost(
        {"n_estimators": 5, "early_stopping_rounds": 17},
        seed=123,
    )
    assert model.estimator.get_params()["early_stopping_rounds"] == 17


def test_predict_test_keeps_missing_features_as_nan(tmp_path) -> None:
    class DummyModel:
        def __init__(self) -> None:
            self.seen_missing: np.ndarray | None = None

        def predict(self, frame: pd.DataFrame) -> np.ndarray:
            self.seen_missing = frame["missing_feature"].to_numpy()
            return np.zeros(len(frame), dtype=float)

    class CaptureLogger:
        def __init__(self) -> None:
            self.warnings: list[tuple[str, dict]] = []

        def warn(self, message: str, **fields) -> None:
            self.warnings.append((message, fields))

        def info(self, message: str, **fields) -> None:
            pass

    path = tmp_path / "abcdef12__horizontal_well.csv"
    pd.DataFrame(
        {
            "MD": [0, 1, 2],
            "X": [0, 1, 2],
            "Y": [0, 0, 0],
            "Z": [100, 101, 102],
            "GR": [80, 82, 85],
            "TVT_input": [10.0, np.nan, np.nan],
        }
    ).to_csv(path, index=False)
    sample = tmp_path / "sample_submission.csv"
    pd.DataFrame({"id": ["abcdef12_1", "abcdef12_2"], "tvt": [0.0, 0.0]}).to_csv(
        sample, index=False
    )

    model = DummyModel()
    logger = CaptureLogger()
    submission = predict_test(
        model=model,
        test_paths=[path],
        sample_submission_path=sample,
        config=minimal_config(),
        feature_names=["idx", "missing_feature"],
        logger=logger,
    )

    assert not submission["tvt"].isna().any()
    assert model.seen_missing is not None
    # Missing features must arrive as NaN so tree models use their trained
    # "missing" branch (matching how no-typewell rows looked at train time).
    assert np.all(np.isnan(model.seen_missing))
    assert logger.warnings
    assert logger.warnings[0][0] == "Missing inference features kept as NaN"
