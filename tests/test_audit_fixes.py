from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from rogii.best_public_solution import Source, claimed_score, select_best_source
from rogii.clearml_data import (
    _dataset_get_local_copy,
    apply_data_clearml_overrides,
    selected_top_level_paths,
)
from rogii.clearml_data import upload_dataset as clearml_upload_dataset
from rogii.clearml_tracking import apply_clearml_overrides
from rogii.config import DEFAULT_CONFIG, load_config
from rogii.constants import FORMATIONS
from rogii.features import (
    _build_well_features_single_layer,
    build_target_mask,
    build_training_table,
    build_well_features,
    feature_cache_path,
)
from rogii.filter_model_bundle import filter_lines
from rogii.gpu_preflight import (
    catboost_gpu_models,
    lightgbm_gpu_models,
    xgboost_gpu_models,
)
from rogii.kaggle_submit import download_clearml_artifacts_from_task
from rogii.modeling import (
    EnsembleRegressor,
    ResidualModel,
    apply_postprocess,
    fit_hill_climb_weights,
    make_xgboost,
    notebook_blend_candidates,
    postprocess_candidate_count,
    smoothing_candidates,
    tune_postprocess,
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


def test_best_public_solution_score_selection() -> None:
    assert claimed_score("9.251 ROGII-Wellbore Geology Prediction: DWT-based") == 9.251
    assert claimed_score("LB-9.830: ROGII - LGB+XGB") == 9.830
    assert claimed_score("[ROGII] SUPER SOLUTION |LB: TOP 3") is None

    selected = select_best_source(
        [
            Source("a/lb-9-830", "LB-9.830", "A", "", 30, "", 9.830),
            Source("b/9-251", "9.251 DWT", "B", "", 12, "", 9.251),
            Source("c/9-251-more-votes", "9.251 DWT fork", "C", "", 40, "", 9.251),
        ]
    )
    assert selected.source_ref == "c/9-251-more-votes"


def test_filter_model_bundle_removes_excluded_lines() -> None:
    text = "keep\n M tests/test_audit_fixes.py\nalso keep\n- tests/foo.py\n"
    assert filter_lines(text, ["tests/"]) == "keep\nalso keep\n"


def test_clearml_overrides_accept_spacebridge_style_args() -> None:
    config = {
        "project_name": "ROGII/Fallback",
        "output_uri": "s3://fallback",
        "tracking": {
            "clearml": {
                "enabled": False,
                "project": "old",
                "tags": ["old"],
                "log_model": False,
            }
        },
    }
    args = SimpleNamespace(
        clearml_enabled="true",
        clearml_project="ROGII/Test",
        clearml_task_name="remote_task",
        clearml_output_uri="s3://bucket/path",
        clearml_tags="rogii,spacebridge",
        clearml_log_artifacts="false",
        clearml_log_model="true",
        clearml_fail_on_error="true",
    )

    clearml_cfg = apply_clearml_overrides(
        config,
        args,
        run_id="run_1",
        config_path=Path("configs/stack.yml"),
    )

    assert clearml_cfg["enabled"] is True
    assert clearml_cfg["project"] == "ROGII/Test"
    assert clearml_cfg["task_name"] == "remote_task"
    assert clearml_cfg["output_uri"] == "s3://bucket/path"
    assert clearml_cfg["tags"] == ["rogii", "spacebridge"]
    assert clearml_cfg["log_artifacts"] is False
    assert clearml_cfg["log_model"] is True
    assert clearml_cfg["fail_on_error"] is True


def test_clearml_overrides_fall_back_to_project_and_output_uri() -> None:
    config = {
        "project_name": "ROGII/Fallback",
        "output_uri": "s3://fallback",
        "tracking": {
            "clearml": {"enabled": False, "project": None, "output_uri": None}
        },
    }
    args = SimpleNamespace(
        clearml_enabled=None,
        clearml_project=None,
        clearml_task_name=None,
        clearml_output_uri=None,
        clearml_tags=None,
        clearml_log_artifacts=None,
        clearml_log_model=None,
        clearml_fail_on_error=None,
    )

    clearml_cfg = apply_clearml_overrides(
        config,
        args,
        run_id="run_1",
        config_path=Path("configs/stack.yml"),
    )

    assert clearml_cfg["project"] == "ROGII/Fallback"
    assert clearml_cfg["output_uri"] == "s3://fallback"
    assert clearml_cfg["task_name"] == "rogii-stack-run_1"


def test_clearml_data_upload_selection_excludes_zip(tmp_path: Path) -> None:
    (tmp_path / "train").mkdir()
    (tmp_path / "test").mkdir()
    (tmp_path / "sample_submission.csv").write_text("id,tvt\n", encoding="utf-8")
    (tmp_path / "rogii-wellbore-geology-prediction.zip").write_text(
        "zip", encoding="utf-8"
    )

    selected = {path.name for path in selected_top_level_paths(tmp_path, ["*.zip"])}

    assert selected == {"train", "test", "sample_submission.csv"}


def test_data_clearml_overrides_accept_spacebridge_style_args() -> None:
    config = {"data": {"clearml": {"enabled": False}}}
    args = SimpleNamespace(
        data_clearml_enabled="true",
        data_clearml_project="ROGII/Test",
        data_clearml_name="rogii-data",
        data_clearml_version="v1",
        data_clearml_dataset_id="dataset-id",
        data_clearml_alias="alias",
        data_clearml_cache_dir="/tmp/rogii-cache",
    )

    clearml_cfg = apply_data_clearml_overrides(config, args)

    assert clearml_cfg["enabled"] is True
    assert clearml_cfg["project"] == "ROGII/Test"
    assert clearml_cfg["name"] == "rogii-data"
    assert clearml_cfg["version"] == "v1"
    assert clearml_cfg["dataset_id"] == "dataset-id"
    assert clearml_cfg["alias"] == "alias"
    assert clearml_cfg["cache_dir"] == "/tmp/rogii-cache"


def test_clearml_data_upload_requires_s3_before_clearml_import(tmp_path: Path) -> None:
    (tmp_path / "train").mkdir()

    with pytest.raises(ValueError, match="s3://"):
        clearml_upload_dataset(
            data_dir=tmp_path,
            project="p",
            name="n",
            version="v",
            output_uri="https://clearml.wb.ru/fileserver",
            tags=[],
            excludes=[],
            max_workers=1,
            require_output_uri=True,
            s3_only=True,
            logger=SimpleNamespace(info=lambda *args, **kwargs: None),
        )


def test_clearml_dataset_get_local_copy_supports_new_cache_arg(tmp_path: Path) -> None:
    class DatasetWithCacheArg:
        def __init__(self) -> None:
            self.local_cache_path = None

        def get_local_copy(self, local_cache_path: str | None = None) -> str:
            self.local_cache_path = local_cache_path
            return "/tmp/clearml-data"

    dataset = DatasetWithCacheArg()
    path = _dataset_get_local_copy(dataset, cache_dir=tmp_path)

    assert path == "/tmp/clearml-data"
    assert dataset.local_cache_path == str(tmp_path)


def test_clearml_dataset_get_local_copy_supports_legacy_signature(
    tmp_path: Path,
) -> None:
    class DatasetWithoutCacheArg:
        def __init__(self) -> None:
            self.called = False

        def get_local_copy(self) -> str:
            self.called = True
            return "/tmp/legacy-clearml-data"

    warnings = []
    logger = SimpleNamespace(warn=lambda message, **fields: warnings.append(message))
    dataset = DatasetWithoutCacheArg()
    path = _dataset_get_local_copy(dataset, cache_dir=tmp_path, logger=logger)

    assert path == "/tmp/legacy-clearml-data"
    assert dataset.called is True
    assert warnings == [
        "ClearML Dataset.get_local_copy does not support custom cache dir"
    ]


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


def write_spatial_context_well(
    tmp_path, name: str, x0: float, y0: float, shift: float
) -> Path:
    n = 24
    idx = np.arange(n, dtype=float)
    frame = pd.DataFrame(
        {
            "MD": idx,
            "X": x0 + 2.0 * idx,
            "Y": y0 + 0.5 * idx,
            "Z": 1000.0 + 0.1 * idx,
            "GR": 80.0 + idx,
            "TVT_input": 10.0 + idx,
            "TVT": 10.0 + idx,
        }
    )
    for offset, formation in enumerate(FORMATIONS):
        frame[formation] = 900.0 + shift + 4.0 * offset + 0.2 * idx
    horizontal = tmp_path / f"{name}__horizontal_well.csv"
    frame.to_csv(horizontal, index=False)
    return horizontal


def dwt_config() -> dict:
    config = minimal_config()
    config["features"]["include_typewell"] = True
    config["features"]["include_kaggle_top_signals"] = True
    config["features"]["kaggle_top"] = {
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
        "dwt_enabled": True,
        "dwt_wavelet": "db2",
        "dwt_level": 1,
        "dwt_radii": [2, 4],
        "dwt_max_query_points": 32,
        "dwt_max_ref_points": 32,
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


def reference_impute_formations(context, xy: np.ndarray, self_well: str | None):
    if context.formation_tree is None or len(context.formation_values) == 0:
        return (
            np.full((len(xy), len(FORMATIONS)), np.nan, dtype=float),
            np.full(len(xy), np.nan, dtype=float),
        )
    k_fetch = min(len(context.formation_values), context.spatial_k + 8)
    dist, idx = context.formation_tree.query(xy / context.formation_scale, k=k_fetch)
    if k_fetch == 1:
        dist = np.asarray(dist).reshape(len(xy), 1)
        idx = np.asarray(idx).reshape(len(xy), 1)
    else:
        dist = np.atleast_2d(dist)
        idx = np.atleast_2d(idx)
        if len(xy) == 1:
            dist = dist.reshape(1, -1)
            idx = idx.reshape(1, -1)
    if self_well is not None:
        dist = np.where(context.formation_wells[idx] == self_well, np.inf, dist)

    pred = np.empty((len(xy), len(FORMATIONS)), dtype=float)
    nearest_dist = np.empty(len(xy), dtype=float)
    global_mean = np.nanmean(context.formation_values, axis=0)
    for row in range(len(xy)):
        order = np.argsort(dist[row])[: context.spatial_k]
        valid = np.isfinite(dist[row, order])
        if not valid.any():
            pred[row] = global_mean
            nearest_dist[row] = np.nan
            continue
        chosen = order[valid]
        chosen_idx = idx[row, chosen]
        weights = 1.0 / (dist[row, chosen] + 1e-3)
        xn = context.formation_xy[chosen_idx, 0]
        yn = context.formation_xy[chosen_idx, 1]
        values = context.formation_values[chosen_idx]
        design = np.column_stack([xn, yn, np.ones_like(xn)])
        normal = design.T @ (design * weights[:, None])
        rhs = design.T @ (values * weights[:, None])
        normal += np.eye(3) * 1e-9
        try:
            coef = np.linalg.solve(normal, rhs)
        except np.linalg.LinAlgError:
            coef = np.linalg.pinv(normal) @ rhs
        pred[row] = xy[row, 0] * coef[0] + xy[row, 1] * coef[1] + coef[2]
        nearest_dist[row] = float(np.nanmin(dist[row, chosen]))
    return pred, nearest_dist


def reference_impute_dense_ancc(context, xy: np.ndarray, self_well: str | None):
    if context.dense_tree is None or len(context.dense_ancc) == 0:
        return (
            np.full(len(xy), np.nan, dtype=float),
            np.full(len(xy), np.nan, dtype=float),
            np.full(len(xy), np.nan, dtype=float),
        )
    k_fetch = len(context.dense_ancc)
    pred = np.empty(len(xy), dtype=float)
    std = np.empty(len(xy), dtype=float)
    nearest_dist = np.empty(len(xy), dtype=float)
    global_mean = float(np.nanmean(context.dense_ancc))
    chunk_size = max(1, context.dense_query_chunk)
    for start in range(0, len(xy), chunk_size):
        stop = min(start + chunk_size, len(xy))
        chunk_xy = xy[start:stop]
        dist, idx = context.dense_tree.query(chunk_xy / context.dense_scale, k=k_fetch)
        if k_fetch == 1:
            dist = np.asarray(dist).reshape(len(chunk_xy), 1)
            idx = np.asarray(idx).reshape(len(chunk_xy), 1)
        else:
            dist = np.atleast_2d(dist)
            idx = np.atleast_2d(idx)
            if len(chunk_xy) == 1:
                dist = dist.reshape(1, -1)
                idx = idx.reshape(1, -1)
        if self_well is not None:
            dist = np.where(context.dense_wells[idx] == self_well, np.inf, dist)

        for local_row in range(len(chunk_xy)):
            row = start + local_row
            order = np.argsort(dist[local_row])[: context.dense_k]
            valid = np.isfinite(dist[local_row, order])
            if not valid.any():
                pred[row] = global_mean
                std[row] = np.nan
                nearest_dist[row] = np.nan
                continue
            chosen = order[valid]
            values = context.dense_ancc[idx[local_row, chosen]]
            weights = 1.0 / (dist[local_row, chosen] + 1e-3)
            weights /= weights.sum()
            mean = float(weights @ values)
            pred[row] = mean
            std[row] = float(np.sqrt(weights @ ((values - mean) ** 2)))
            nearest_dist[row] = float(np.nanmin(dist[local_row, chosen]))
    return pred, std, nearest_dist


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


def test_feature_schema_includes_public_dwt_last_known_offsets(tmp_path) -> None:
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
    assert "x_from_last_known" in wf.features.columns
    assert "y_from_last_known" in wf.features.columns
    assert "z_from_last_known" in wf.features.columns
    assert "xy_dist_from_last_known" in wf.features.columns
    assert "md_since" in wf.features.columns
    assert "dx" in wf.features.columns
    assert "dy" in wf.features.columns
    assert "dz" in wf.features.columns
    assert "dxy" in wf.features.columns
    assert np.allclose(
        wf.features["md_since"].to_numpy(dtype=float),
        wf.features["md_from_last_known"].to_numpy(dtype=float),
    )
    assert np.allclose(
        wf.features["dx"].to_numpy(dtype=float),
        wf.features["x_from_last_known"].to_numpy(dtype=float),
    )
    assert np.allclose(
        wf.features["dy"].to_numpy(dtype=float),
        wf.features["y_from_last_known"].to_numpy(dtype=float),
    )
    assert np.allclose(
        wf.features["dz"].to_numpy(dtype=float),
        wf.features["z_from_last_known"].to_numpy(dtype=float),
    )
    assert np.allclose(
        wf.features["dxy"].to_numpy(dtype=float),
        wf.features["xy_dist_from_last_known"].to_numpy(dtype=float),
    )
    assert "idx_since" not in wf.features.columns
    assert "tvt_input_isna" not in wf.features.columns


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
        "dwt_ens_d",
        "kg_dwt_r2_tvt",
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


def test_split_context_features_match_full_builder_on_hidden_rows(tmp_path) -> None:
    path, _frame = write_dwt_synthetic_well(tmp_path)
    config = dwt_config()
    config["features"]["cache"] = {"enabled": False}
    context = KaggleTopContext([path], config)

    split = build_well_features(path, config, train=True, top_context=context)
    full = _build_well_features_single_layer(
        path,
        config,
        train=True,
        top_context=context,
        cache_layer="full",
    )
    hidden = full.target_mask

    assert set(split.features.columns) == set(full.features.columns)
    for column in full.features.columns:
        assert np.allclose(
            split.features.loc[hidden, column].to_numpy(dtype=float),
            full.features.loc[hidden, column].to_numpy(dtype=float),
            equal_nan=True,
        )


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


def test_split_feature_cache_layers_handle_context_key(tmp_path) -> None:
    path_a, _frame_a = write_dwt_synthetic_well(tmp_path, "aaaa1111")
    path_b, _frame_b = write_dwt_synthetic_well(tmp_path, "bbbb2222")
    config = dwt_config()
    config["features"]["cache"] = {"enabled": True, "dir": str(tmp_path / "cache")}
    key_a = context_key_for_paths([path_a], config)
    key_ab = context_key_for_paths([path_a, path_b], config)

    free_a = feature_cache_path(
        path_a, config, train=True, context_key=key_a, layer="context_free"
    )
    free_ab = feature_cache_path(
        path_a, config, train=True, context_key=key_ab, layer="context_free"
    )
    context_a = feature_cache_path(
        path_a, config, train=True, context_key=key_a, layer="context"
    )
    context_ab = feature_cache_path(
        path_a, config, train=True, context_key=key_ab, layer="context"
    )
    worker_config = dwt_config()
    worker_config["features"]["cache"] = config["features"]["cache"]
    worker_config["features"]["num_workers"] = 8
    free_worker = feature_cache_path(
        path_a, worker_config, train=True, context_key=key_ab, layer="context_free"
    )

    assert free_a is not None
    assert free_ab is not None
    assert free_worker is not None
    assert context_a is not None
    assert context_ab is not None
    assert free_a == free_ab
    assert free_a == free_worker
    assert context_a != context_ab


def test_parallel_feature_build_matches_serial_order_and_values(tmp_path) -> None:
    paths = [
        write_dwt_synthetic_well(tmp_path, "aaaa1111")[0],
        write_dwt_synthetic_well(tmp_path, "bbbb2222")[0],
        write_dwt_synthetic_well(tmp_path, "cccc3333")[0],
    ]
    serial_config = dwt_config()
    parallel_config = dwt_config()
    serial_config["features"]["cache"] = {"enabled": False}
    parallel_config["features"]["cache"] = {"enabled": False}
    serial_config["features"]["num_workers"] = 1
    parallel_config["features"]["num_workers"] = 2
    serial_context = KaggleTopContext(paths, serial_config)
    parallel_context = KaggleTopContext(paths, parallel_config)

    X1, residual1, groups1, flat1, y_true1 = build_training_table(
        paths,
        serial_config,
        serial_context,
    )
    X2, residual2, groups2, flat2, y_true2 = build_training_table(
        paths,
        parallel_config,
        parallel_context,
    )

    assert X1.columns.tolist() == X2.columns.tolist()
    assert groups1.tolist() == groups2.tolist()
    assert np.allclose(
        X1.to_numpy(dtype=float), X2.to_numpy(dtype=float), equal_nan=True
    )
    assert np.allclose(residual1, residual2)
    assert np.allclose(flat1, flat2)
    assert np.allclose(y_true1, y_true2)


def test_parallel_feature_worker_reports_well_name_on_error(tmp_path) -> None:
    bad_path = tmp_path / "badwell1__horizontal_well.csv"
    config = minimal_config()
    config["features"]["num_workers"] = 2

    with pytest.raises(RuntimeError, match="badwell1"):
        build_training_table([bad_path], config)


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


def test_vectorized_spatial_imputation_matches_reference(tmp_path) -> None:
    paths = [
        write_spatial_context_well(tmp_path, "aaaa1111", 1000, 2000, 0),
        write_spatial_context_well(tmp_path, "bbbb2222", 1100, 2030, 25),
        write_spatial_context_well(tmp_path, "cccc3333", 1200, 1980, 50),
        write_spatial_context_well(tmp_path, "dddd4444", 1300, 2080, 75),
        write_spatial_context_well(tmp_path, "eeee5555", 1400, 2100, 100),
    ]
    config = dwt_config()
    config["features"]["kaggle_top"].update(
        {
            "spatial_k": 3,
            "dense_k": 4,
            "dense_fetch": 8,
            "dense_query_chunk": 3,
            "dense_samples_per_well": 6,
        }
    )
    context = KaggleTopContext(paths, config)
    xy = np.array(
        [
            [1010.0, 2001.0],
            [1120.0, 2035.0],
            [1260.0, 2050.0],
            [1450.0, 2110.0],
        ],
        dtype=float,
    )

    expected_form, expected_form_dist = reference_impute_formations(
        context, xy, "bbbb2222"
    )
    actual_form, actual_form_dist = context.impute_formations(xy, "bbbb2222")
    assert np.allclose(actual_form, expected_form, equal_nan=True)
    assert np.allclose(actual_form_dist, expected_form_dist, equal_nan=True)

    expected_dense = reference_impute_dense_ancc(context, xy, "bbbb2222")
    actual_dense = context.impute_dense_ancc(xy, "bbbb2222")
    for actual, expected in zip(actual_dense, expected_dense, strict=True):
        assert np.allclose(actual, expected, equal_nan=True)


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


def test_ensemble_predict_uses_fold_average_without_final_models() -> None:
    class ConstantModel(ResidualModel):
        def __init__(self, value: float) -> None:
            self.value = value

        def fit(self, X: pd.DataFrame, y: np.ndarray, **_) -> "ConstantModel":
            return self

        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return np.full(len(X), self.value, dtype=float)

    config = {
        "model": {"base_models": [{"id": "a"}, {"id": "b"}], "blend": {}},
        "validation": {"n_splits": 2, "final_model_strategy": "fold_average"},
    }
    model = EnsembleRegressor(config, seed=42)
    model.base_names = ["a", "b"]
    model.weights = np.array([0.25, 0.75])
    model.fold_models = [
        [ConstantModel(80.0), ConstantModel(120.0)],
        [ConstantModel(180.0), ConstantModel(220.0)],
    ]

    pred = model.predict(pd.DataFrame({"x": [0.0, 1.0]}))
    assert np.allclose(pred, [175.0, 175.0])


def test_notebook_blend_candidates_support_ranges() -> None:
    config = minimal_config()
    config["postprocess"]["notebook_blend"] = {
        "enabled": True,
        "alpha_range": [0.5, 0.7, 0.1],
        "tau_range": [0, 10, 5],
        "w_pf_range": [0, 0.2, 0.1],
    }
    candidates = notebook_blend_candidates(config)
    assert len(candidates) == 27
    assert candidates[0] == {"alpha": 0.5, "tau": 0.0, "w_pf": 0.0}
    assert candidates[-1] == {"alpha": 0.7, "tau": 10.0, "w_pf": 0.2}


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


def test_fast_postprocess_tuning_matches_bruteforce() -> None:
    config = minimal_config()
    config["postprocess"] = {
        "residual_weight": 1.0,
        "residual_weight_grid": [0.7, 1.0, 1.1],
        "residual_clip": 10.0,
        "notebook_blend": {
            "enabled": True,
            "pf_column": "kg_pf_ancc_tvt",
            "alpha_grid": [0.95, 1.0],
            "tau_grid": [0, 20],
            "w_pf_grid": [0, 0.2],
        },
        "smoothing": {
            "enabled": True,
            "candidates": [{"enabled": False}, {"window": 5, "polyorder": 2}],
        },
    }
    flat = np.array([10, 10, 10, 10, 20, 20, 20, 20], dtype=float)
    residual = np.array([0.5, 1.5, 2.0, 2.5, -1.0, -1.5, -2.0, -2.5])
    y_true = np.array([10.8, 11.4, 12.2, 12.9, 18.8, 18.3, 17.8, 17.3])
    features = pd.DataFrame(
        {
            "last_known_tvt": [10.0] * 4 + [20.0] * 4,
            "md_from_last_known": [0.0, 10.0, 20.0, 30.0] * 2,
            "kg_pf_ancc_tvt": [10.7, 11.6, 12.1, 12.8, 19.0, 18.6, 17.9, 17.1],
        }
    )
    groups = np.array(["a"] * 4 + ["b"] * 4)

    best_weight, scores, best_blend, best_smoothing = tune_postprocess(
        flat,
        residual,
        y_true,
        config,
        features,
        groups,
    )

    brute_scores = []
    for weight in config["postprocess"]["residual_weight_grid"]:
        for alpha in config["postprocess"]["notebook_blend"]["alpha_grid"]:
            for tau in config["postprocess"]["notebook_blend"]["tau_grid"]:
                for w_pf in config["postprocess"]["notebook_blend"]["w_pf_grid"]:
                    blend = {"alpha": alpha, "tau": tau, "w_pf": w_pf}
                    for smoothing in [None, {"window": 5, "polyorder": 2}]:
                        pred = apply_postprocess(
                            flat,
                            residual,
                            config,
                            residual_weight=weight,
                            features=features,
                            groups=groups,
                            notebook_blend=blend,
                            smoothing=smoothing,
                        )
                        score = {
                            "weight": float(weight),
                            "rmse": float(np.sqrt(np.mean((pred - y_true) ** 2))),
                            "blend_alpha": float(alpha),
                            "blend_tau": float(tau),
                            "blend_w_pf": float(w_pf),
                        }
                        if smoothing is not None:
                            score["smooth_window"] = 5.0
                            score["smooth_polyorder"] = 2.0
                        brute_scores.append(score)

    assert len(scores) == len(brute_scores)
    assert np.allclose(
        [score["rmse"] for score in scores],
        [score["rmse"] for score in brute_scores],
    )
    best_brute = min(brute_scores, key=lambda item: item["rmse"])
    assert best_weight == pytest.approx(best_brute["weight"])
    assert best_blend == {
        "alpha": best_brute["blend_alpha"],
        "tau": best_brute["blend_tau"],
        "w_pf": best_brute["blend_w_pf"],
    }
    if "smooth_window" in best_brute:
        assert best_smoothing == {"window": 5.0, "polyorder": 2.0}
    else:
        assert best_smoothing is None


def test_hill_climb_can_use_negative_weights() -> None:
    x = np.linspace(-2.0, 2.0, 50)
    stack = np.column_stack([x, 2.0 * x])
    y_true = -x

    non_negative = fit_hill_climb_weights(
        stack,
        y_true,
        iterations=200,
        alpha_grid=[0.5, 0.25, 0.1, 0.05, 0.02, 0.01],
        allow_negative_weights=False,
    )
    negative = fit_hill_climb_weights(
        stack,
        y_true,
        iterations=200,
        alpha_grid=[0.5, 0.25, 0.1, 0.05, 0.02, 0.01],
        allow_negative_weights=True,
    )

    assert negative.sum() == pytest.approx(1.0)
    assert np.any(negative < 0.0)
    assert np.sqrt(np.mean((stack @ negative - y_true) ** 2)) < np.sqrt(
        np.mean((stack @ non_negative - y_true) ** 2)
    )


def test_fixed_smoothing_config_returns_only_savgol() -> None:
    config = minimal_config()
    config["postprocess"]["smoothing"] = {
        "enabled": True,
        "window": 17,
        "polyorder": 3,
        "candidates": [],
    }

    assert smoothing_candidates(config) == [{"window": 17, "polyorder": 3}]


def test_optuna_postprocess_tuning_uses_trial_budget() -> None:
    config = minimal_config()
    config["postprocess"] = {
        "search_method": "optuna",
        "optuna_trials": 12,
        "optuna_seed": 123,
        "progress_interval": 100,
        "residual_weight": 1.0,
        "residual_weight_grid": [],
        "residual_clip": 10.0,
        "notebook_blend": {
            "enabled": True,
            "pf_column": "kg_pf_ancc_tvt",
            "alpha_range": [0.95, 1.05, 0.01],
            "tau_range": [0, 50, 10],
            "w_pf_range": [0, 0.2, 0.05],
        },
        "smoothing": {
            "enabled": True,
            "window": 5,
            "polyorder": 2,
            "candidates": [],
        },
    }
    flat = np.array([10, 10, 10, 10, 20, 20, 20, 20], dtype=float)
    residual = np.array([0.5, 1.5, 2.0, 2.5, -1.0, -1.5, -2.0, -2.5])
    y_true = np.array([10.8, 11.4, 12.2, 12.9, 18.8, 18.3, 17.8, 17.3])
    features = pd.DataFrame(
        {
            "last_known_tvt": [10.0] * 4 + [20.0] * 4,
            "md_from_last_known": [0.0, 10.0, 20.0, 30.0] * 2,
            "kg_pf_ancc_tvt": [10.7, 11.6, 12.1, 12.8, 19.0, 18.6, 17.9, 17.1],
        }
    )
    groups = np.array(["a"] * 4 + ["b"] * 4)

    best_weight, scores, best_blend, best_smoothing = tune_postprocess(
        flat,
        residual,
        y_true,
        config,
        features,
        groups,
    )

    assert postprocess_candidate_count(config) == 12
    assert len(scores) == 12
    assert best_weight == pytest.approx(1.0)
    assert set(best_blend or {}) == {"alpha", "tau", "w_pf"}
    assert best_smoothing == {"window": 5.0, "polyorder": 2.0}


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


def test_configs_use_canonical_pf_postprocess_column() -> None:
    assert DEFAULT_CONFIG["postprocess"]["notebook_blend"]["pf_column"] == "pf_ancc"
    for path in [Path("configs/quick.yml"), Path("configs/stack.yml")]:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert config["postprocess"]["notebook_blend"]["pf_column"] == "pf_ancc"


def test_stack_gpu_uses_catboost_and_xgboost_gpu_without_lightgbm_gpu() -> None:
    config = load_config(Path("configs/stack_gpu.yml"))

    assert [model["id"] for model in catboost_gpu_models(config)] == [
        "cat_lr025",
        "cat_lr020",
        "cat_lr030",
    ]
    assert [model["id"] for model in xgboost_gpu_models(config)] == [
        "xgb_lr025",
        "xgb_lr020",
        "xgb_lr030",
    ]
    assert lightgbm_gpu_models(config) == []


def test_download_clearml_artifacts_from_task(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    artifact_names = (
        "model.pkl",
        "features.json",
        "metrics.json",
        "config.yml",
        "source_config.yml",
    )
    for name in artifact_names:
        (source_dir / name).write_text(f"{name}\n", encoding="utf-8")

    class FakeArtifact:
        def __init__(self, path: Path) -> None:
            self.path = path

        def get_local_copy(self) -> str:
            return str(self.path)

    task = SimpleNamespace(
        id="task123",
        name="clearml-run",
        artifacts={
            f"output/{path.name}": FakeArtifact(path)
            for path in sorted(source_dir.iterdir())
        },
    )
    model_dir = tmp_path / "model"

    download_clearml_artifacts_from_task(task=task, model_dir=model_dir)

    assert sorted(path.name for path in model_dir.iterdir()) == [
        "config.yml",
        "features.json",
        "metrics.json",
        "model.pkl",
        "source_config.yml",
    ]
    assert (model_dir / "model.pkl").read_text(encoding="utf-8") == "model.pkl\n"


def test_notebook_postprocess_requires_explicit_pf_column() -> None:
    config = minimal_config()
    config["postprocess"]["notebook_blend"] = {
        "enabled": True,
        "alpha": 1.0,
        "tau": 0.0,
        "w_pf": 0.0,
    }
    features = pd.DataFrame(
        {
            "last_known_tvt": [10.0],
            "md_from_last_known": [0.0],
            "pf_ancc": [11.0],
        }
    )

    with pytest.raises(ValueError, match="pf_column"):
        apply_postprocess(
            flat=np.array([10.0]),
            residual=np.array([1.0]),
            config=config,
            features=features,
        )


def test_notebook_postprocess_noops_when_pf_feature_missing() -> None:
    config = minimal_config()
    config["postprocess"]["notebook_blend"] = {
        "enabled": True,
        "pf_column": "pf_ancc",
        "alpha": 1.0,
        "tau": 0.0,
        "w_pf": 0.5,
    }
    pred = apply_postprocess(
        flat=np.array([10.0]),
        residual=np.array([2.0]),
        config=config,
        features=pd.DataFrame(
            {
                "last_known_tvt": [10.0],
                "md_from_last_known": [0.0],
            }
        ),
    )

    assert np.allclose(pred, [12.0])


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


def test_predict_test_rejects_out_of_range_sample_row_id(tmp_path) -> None:
    class DummyModel:
        def predict(self, frame: pd.DataFrame) -> np.ndarray:
            return np.zeros(len(frame), dtype=float)

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
    pd.DataFrame({"id": ["abcdef12_99"], "tvt": [0.0]}).to_csv(
        sample,
        index=False,
    )

    with pytest.raises(IndexError, match="out of range"):
        predict_test(
            model=DummyModel(),
            test_paths=[path],
            sample_submission_path=sample,
            config=minimal_config(),
            feature_names=["idx"],
        )
