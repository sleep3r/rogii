from __future__ import annotations

import numpy as np
import pandas as pd

from rogii.features import build_target_mask, build_well_features
from rogii.modeling import apply_postprocess, make_xgboost
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


def test_predict_test_warns_and_zero_fills_missing_features(tmp_path) -> None:
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
    assert np.all(model.seen_missing == 0.0)
    assert logger.warnings
    assert logger.warnings[0][0] == "Missing inference features filled with zero"
