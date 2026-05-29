from __future__ import annotations

import importlib
import importlib.util
import random
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch


def _synthetic_well(tmp_path: Path) -> dict[str, Path | str]:
    n = 64
    m = 220
    row = np.arange(n, dtype=np.float32)
    md = 10_000.0 + row
    tvt = 11_000.0 + 0.04 * row + 2.0 * np.sin(row / 9.0)
    tw_tvt = np.linspace(tvt.min() - 30.0, tvt.max() + 30.0, m).astype(np.float32)
    tw_gr = (85.0 + 18.0 * np.sin((tw_tvt - tw_tvt.min()) / 8.0)).astype(np.float32)
    gr = np.interp(tvt, tw_tvt, tw_gr).astype(np.float32)
    gr[10:14] = np.nan

    tvt_input = np.full(n, np.nan, dtype=np.float32)
    tvt_input[:8] = tvt[:8]
    tvt_input[-5:] = tvt[-5:]

    hw = pd.DataFrame(
        {
            "MD": md,
            "X": 2_900_000.0 + row * 8.0,
            "Y": 1_100_000.0 + row * 2.0,
            "Z": -9_200.0 - row * 0.1,
            "ANCC": -9_350.0,
            "ASTNU": -9_520.0,
            "ASTNL": -9_560.0,
            "EGFDU": -9_640.0,
            "EGFDL": -9_690.0,
            "BUDA": -9_820.0,
            "TVT": tvt,
            "GR": gr,
            "TVT_input": tvt_input,
        }
    )
    tw = pd.DataFrame({"TVT": tw_tvt, "GR": tw_gr, "Geology": ["A"] * m})

    hw_path = tmp_path / "abc12345__horizontal_well.csv"
    tw_path = tmp_path / "abc12345__typewell.csv"
    hw.to_csv(hw_path, index=False)
    tw.to_csv(tw_path, index=False)
    return {"well_id": "abc12345", "hw_path": hw_path, "tw_path": tw_path}


def test_module_entrypoints_are_importable() -> None:
    assert importlib.import_module("bphwt.train.__main__")
    assert importlib.import_module("bphwt.infer.__main__")


def test_gr_features_accept_dataframe_and_array() -> None:
    from bphwt.features.gr_processing import compute_gr_features

    gr = np.array([80.0, np.nan, 90.0, 95.0], dtype=np.float32)
    from_array = compute_gr_features(gr, smooth_sigmas=[1.0])
    from_df = compute_gr_features(pd.DataFrame({"GR": gr}), smooth_sigmas=[1.0])

    np.testing.assert_allclose(from_array["gr_filled"], from_df["gr_filled"])
    assert np.isfinite(from_array["gr_filled"]).all()
    assert from_array["gr_valid"].tolist() == [1.0, 0.0, 1.0, 1.0]


def test_process_one_well_emits_plan_priors(tmp_path: Path) -> None:
    from bphwt.config import load_config
    from bphwt.data.build_cache import process_one_well

    cfg = load_config("configs/bphwt_lite.yml")
    cfg.data.hmm_tvt_step = 2.0
    cfg.surface.enabled = False
    cfg.neighbor.enabled = False

    out = process_one_well(_synthetic_well(tmp_path), cfg, norm_stats={})

    assert out is not None
    for key in [
        "X",
        "tvt_base",
        "tvt_linear",
        "tvt_last",
        "tvt_hmm",
        "tvt_dtw",
        "tvt_neighbor",
        "candidate_confidence",
        "hmm_entropy",
        "hmm_gr_mismatch",
        "dtw_score",
        "dtw_orientation",
    ]:
        assert key in out
    assert out["X"].shape[0] == 64
    assert np.isfinite(out["tvt_base"]).all()


def test_process_one_well_can_disable_hmm_prior(tmp_path: Path) -> None:
    from bphwt.config import load_config
    from bphwt.data.build_cache import process_one_well

    cfg = load_config("configs/bphwt_lite.yml")
    cfg.data.use_hmm_prior = False
    cfg.surface.enabled = False
    cfg.neighbor.enabled = False

    out = process_one_well(_synthetic_well(tmp_path), cfg, norm_stats={})

    assert out is not None
    np.testing.assert_allclose(out["tvt_hmm"], out["tvt_linear"])
    np.testing.assert_allclose(out["hmm_std"], np.full_like(out["tvt_linear"], 999.0))
    np.testing.assert_allclose(out["hmm_entropy"], np.full_like(out["tvt_linear"], 99.0))
    assert out["_meta"]["hmm_enabled"] is False


def test_cache_status_detects_k_wells_mismatch(tmp_path: Path) -> None:
    from bphwt.config import load_config
    from bphwt.data.build_cache import cache_status

    data_train = tmp_path / "data" / "train"
    data_train.mkdir(parents=True)
    for well_id in ["aaa00000", "bbb00000"]:
        (data_train / f"{well_id}__horizontal_well.csv").write_text("MD,TVT_input\n")
        (data_train / f"{well_id}__typewell.csv").write_text("TVT,GR\n")

    cache_root = tmp_path / "cache"
    (cache_root / "train").mkdir(parents=True)
    pd.DataFrame([{"well_id": "aaa00000"}]).to_csv(cache_root / "meta_train.csv", index=False)
    np.savez_compressed(cache_root / "train" / "aaa00000.npz", dummy=np.array([1], dtype=np.float32))

    cfg = load_config("configs/bphwt_lite.yml")
    cfg.data_dir = str(tmp_path / "data")
    cfg.cache_dir = str(cache_root)
    cfg.k_wells = 2

    status = cache_status(cfg, split="train")

    assert not status.valid
    assert status.expected_count == 2
    assert status.meta_count == 1
    assert "do not match" in status.reason


def test_cache_status_detects_hmm_flag_mismatch(tmp_path: Path) -> None:
    from bphwt.config import load_config
    from bphwt.data.build_cache import cache_status

    data_train = tmp_path / "data" / "train"
    data_train.mkdir(parents=True)
    (data_train / "aaa00000__horizontal_well.csv").write_text("MD,TVT_input\n")
    (data_train / "aaa00000__typewell.csv").write_text("TVT,GR\n")

    cache_root = tmp_path / "cache"
    (cache_root / "train").mkdir(parents=True)
    pd.DataFrame([{"well_id": "aaa00000", "hmm_enabled": True}]).to_csv(
        cache_root / "meta_train.csv",
        index=False,
    )
    np.savez_compressed(cache_root / "train" / "aaa00000.npz", dummy=np.array([1], dtype=np.float32))

    cfg = load_config("configs/bphwt_lite.yml")
    cfg.data_dir = str(tmp_path / "data")
    cfg.cache_dir = str(cache_root)
    cfg.k_wells = 1
    cfg.data.use_hmm_prior = False

    status = cache_status(cfg, split="train")

    assert not status.valid
    assert "HMM prior flag" in status.reason


def test_model_forward_loss_with_collated_well(tmp_path: Path) -> None:
    from bphwt.config import load_config
    from bphwt.data.build_cache import process_one_well
    from bphwt.data.well_dataset import WellDataset, collate_wells
    from bphwt.models.bphwt import BPHWT
    from bphwt.models.losses import BPHWTLoss

    cfg = load_config("configs/bphwt_lite.yml")
    cfg.data.max_seq_len = 0
    cfg.surface.enabled = False
    cfg.neighbor.enabled = False
    cfg.loss.w_forward_gr = 0.1
    cfg.loss.w_distill = 0.1
    cfg.loss.w_seg_boundary = 0.1

    out = process_one_well(_synthetic_well(tmp_path), cfg, norm_stats={})
    assert out is not None
    meta = out.pop("_meta")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    np.savez_compressed(cache_dir / f"{meta['well_id']}.npz", **out, _meta=np.array(meta, dtype=object))

    batch = collate_wells([WellDataset(cache_dir, [meta["well_id"]], cfg)[0]])
    model = BPHWT(
        in_channels=batch["X"].shape[1],
        stage_channels=[16, 24],
        stage_strides=[2, 2],
        decoder_channels=[24, 16],
        attn_heads=4,
        dropout=0.0,
        stoch_depth=0.0,
    )
    preds = model(batch["X"], batch["tvt_base"], batch["tvt_input_filled"], batch["known_mask"], batch["md"])
    loss, logs = BPHWTLoss(cfg.loss)(preds, batch)

    assert torch.isfinite(loss)
    assert "loss_fwd_gr" in logs
    assert "loss_distill" in logs
    assert "loss_seg" in logs


def test_segment_boundary_head_exposes_logits_for_amp_safe_loss() -> None:
    from bphwt.models.bphwt import BPHWT

    model = BPHWT(
        in_channels=8,
        stage_channels=[16, 24],
        stage_strides=[2, 2],
        decoder_channels=[24, 16],
        n_blocks=1,
        use_bottleneck_attn=False,
        predict_seg_boundary=True,
    )
    x = torch.randn(1, 8, 32)
    tvt_base = torch.randn(1, 32) * 5.0 + 1000.0
    tvt_input = tvt_base.clone()
    known_mask = torch.zeros_like(tvt_base)
    md = torch.arange(32, dtype=torch.float32).unsqueeze(0)

    out = model(x, tvt_base, tvt_input, known_mask, md)

    assert "seg_boundary_logits" in out
    torch.testing.assert_close(out["seg_boundary"], torch.sigmoid(out["seg_boundary_logits"]))


def test_lite_config_is_recipe_sized() -> None:
    from bphwt.config import load_config
    from bphwt.models.bphwt import BPHWT

    cfg = load_config("configs/bphwt_lite.yml")
    model = BPHWT(
        in_channels=80,
        stage_channels=cfg.model.stage_channels,
        stage_strides=cfg.model.stage_strides,
        decoder_channels=cfg.model.decoder_channels,
        n_blocks=cfg.model.n_blocks,
        use_bottleneck_attn=cfg.model.use_bottleneck_attn,
        attn_heads=cfg.model.attn_heads,
        dropout=cfg.model.dropout,
        stoch_depth=cfg.model.stoch_depth,
        predict_velocity=cfg.model.predict_velocity,
        predict_dip_sign=cfg.model.predict_dip_sign,
        predict_seg_boundary=cfg.model.predict_seg_boundary,
    )

    assert model.param_count() < cfg.model.max_params
    assert not cfg.augmentation.enabled
    assert cfg.train.device == "cpu"
    assert cfg.loss.w_forward_gr == 0.0


def test_server_config_enables_clearml_and_gpu_auto() -> None:
    from bphwt.config import load_config

    cfg = load_config("configs/bphwt_server.yml")

    assert cfg.tracking.enabled is True
    assert cfg.data_clearml.enabled is True
    assert cfg.train.device == "auto"
    assert cfg.data.use_hmm_prior is False
    assert cfg.data.random_train_crop is True
    assert cfg.data.val_max_seq_len == 0
    assert cfg.run.log_every >= 25
    assert cfg.train.validate_every_steps >= 200
    assert cfg.train.early_stop_patience >= 30
    assert cfg.k_wells == -1


def test_validation_config_uses_full_sequence_without_mutating_train_config() -> None:
    from bphwt.config import load_config
    from bphwt.train.train_fold import make_validation_config

    cfg = load_config("configs/bphwt_server.yml")
    cfg.data.max_seq_len = 512
    cfg.data.val_max_seq_len = 0

    val_cfg = make_validation_config(cfg)

    assert cfg.data.max_seq_len == 512
    assert val_cfg is not cfg
    assert val_cfg.data.max_seq_len == 0
    assert val_cfg.data.random_train_crop is False


def test_v1_config_enables_recipe_physics_and_aux_heads() -> None:
    from bphwt.config import load_config
    from bphwt.models.bphwt import BPHWT

    cfg = load_config("configs/bphwt_v1.yml")
    model = BPHWT(
        in_channels=80,
        stage_channels=cfg.model.stage_channels,
        stage_strides=cfg.model.stage_strides,
        decoder_channels=cfg.model.decoder_channels,
        n_blocks=cfg.model.n_blocks,
        use_bottleneck_attn=cfg.model.use_bottleneck_attn,
        attn_heads=cfg.model.attn_heads,
        dropout=cfg.model.dropout,
        stoch_depth=cfg.model.stoch_depth,
        predict_velocity=cfg.model.predict_velocity,
        predict_dip_sign=cfg.model.predict_dip_sign,
        predict_seg_boundary=cfg.model.predict_seg_boundary,
    )

    assert cfg.run.name == "bphwt_v1"
    assert cfg.data.max_seq_len == 1024
    assert cfg.data.val_max_seq_len == 0
    assert cfg.run.log_first_steps == 5
    assert cfg.train.batch_size == 1
    assert cfg.train.grad_accum == 4
    assert cfg.augmentation.enabled is True
    assert cfg.loss.w_forward_gr == 0.03
    assert cfg.loss.w_gr_corr == 0.0
    assert cfg.model.use_bottleneck_attn is True
    assert cfg.model.predict_velocity is True
    assert cfg.model.predict_dip_sign is True
    assert cfg.model.predict_seg_boundary is True
    assert model.param_count() < cfg.model.max_params


def test_bpi_conv_tiny_config_is_small_physics_cnn() -> None:
    from bphwt.config import load_config
    from bphwt.models.bphwt import BPHWT

    cfg = load_config("configs/bpi_conv_tiny.yml")
    model = BPHWT(
        in_channels=59,
        stage_channels=cfg.model.stage_channels,
        stage_strides=cfg.model.stage_strides,
        decoder_channels=cfg.model.decoder_channels,
        n_blocks=cfg.model.n_blocks,
        use_bottleneck_attn=cfg.model.use_bottleneck_attn,
        attn_heads=cfg.model.attn_heads,
        dropout=cfg.model.dropout,
        stoch_depth=cfg.model.stoch_depth,
        predict_velocity=cfg.model.predict_velocity,
        predict_dip_sign=cfg.model.predict_dip_sign,
        predict_seg_boundary=cfg.model.predict_seg_boundary,
    )

    assert cfg.run.name == "bpi_conv_tiny"
    assert cfg.data.val_max_seq_len == 0
    assert cfg.data.use_hmm_prior is False
    assert cfg.model.use_bottleneck_attn is False
    assert cfg.model.predict_velocity is False
    assert cfg.loss.w_forward_gr == 0.03
    assert cfg.loss.w_smooth == 0.005
    assert model.param_count() < cfg.model.max_params


def test_bpi_conv_tiny_local_config_is_cpu_smoke() -> None:
    from bphwt.config import load_config

    cfg = load_config("configs/bpi_conv_tiny_local.yml")

    assert cfg.run.name == "bpi_conv_tiny_local"
    assert cfg.k_wells == 40
    assert cfg.tracking.enabled is False
    assert cfg.data_clearml.enabled is False
    assert cfg.train.device == "cpu"
    assert cfg.train.num_workers == 0
    assert cfg.train.n_folds == 2
    assert cfg.train.epochs == 2
    assert cfg.data.val_max_seq_len == 0


def test_model_starts_as_tvt_base_residual_baseline() -> None:
    from bphwt.models.bphwt import BPHWT

    model = BPHWT(
        in_channels=8,
        stage_channels=[16, 24],
        stage_strides=[2, 2],
        decoder_channels=[24, 16],
        n_blocks=1,
        use_bottleneck_attn=False,
        predict_velocity=False,
        predict_dip_sign=False,
        predict_seg_boundary=False,
    )
    x = torch.randn(2, 8, 32)
    tvt_base = torch.randn(2, 32) * 5.0 + 1000.0
    tvt_input = tvt_base.clone()
    known_mask = torch.zeros_like(tvt_base)
    md = torch.arange(32, dtype=torch.float32).unsqueeze(0).repeat(2, 1)

    out = model(x, tvt_base, tvt_input, known_mask, md)

    torch.testing.assert_close(out["mu_delta"], torch.zeros_like(out["mu_delta"]))
    torch.testing.assert_close(out["tvt_pred"], tvt_base)


def test_train_crop_randomizes_even_when_augmentation_disabled(tmp_path: Path) -> None:
    from bphwt.config import load_config
    from bphwt.data.build_cache import process_one_well
    from bphwt.data.well_dataset import WellDataset

    cfg = load_config("configs/bphwt_lite.yml")
    cfg.data.max_seq_len = 16
    cfg.augmentation.enabled = False
    cfg.surface.enabled = False
    cfg.neighbor.enabled = False

    out = process_one_well(_synthetic_well(tmp_path), cfg, norm_stats={})
    assert out is not None
    meta = out.pop("_meta")
    full_md = out["md"].copy()
    cache_dir = tmp_path / "cache_crop"
    cache_dir.mkdir()
    np.savez_compressed(cache_dir / f"{meta['well_id']}.npz", **out, _meta=np.array(meta, dtype=object))

    ds = WellDataset(cache_dir, [meta["well_id"]], cfg, augment=True)
    random.seed(123)
    starts = [float(ds[0]["md"][0]) for _ in range(12)]
    assert len(set(starts)) > 1

    cfg.data.random_train_crop = False
    fixed_ds = WellDataset(cache_dir, [meta["well_id"]], cfg, augment=True)
    first = fixed_ds[0]["md"]
    second = fixed_ds[0]["md"]

    torch.testing.assert_close(first, second)
    assert float(first[0]) == float(full_md[-cfg.data.max_seq_len])


def test_clearml_task_init_disables_framework_auto_upload(monkeypatch) -> None:
    from bphwt.clearml_utils import init_clearml_task
    from bphwt.config import load_config

    class FakeTask:
        init_kwargs = None
        connected = None

        @classmethod
        def init(cls, **kwargs):
            cls.init_kwargs = kwargs
            return cls()

        def connect(self, payload, name=None):
            FakeTask.connected = (payload, name)

    fake_clearml = types.SimpleNamespace(Task=FakeTask)
    monkeypatch.setitem(sys.modules, "clearml", fake_clearml)

    cfg = load_config("configs/bphwt_server.yml")
    cfg.tracking.fail_on_error = True
    task = init_clearml_task(cfg)

    assert task is not None
    assert FakeTask.init_kwargs["auto_connect_frameworks"] is False
    assert FakeTask.init_kwargs["auto_connect_arg_parser"] is False
    assert FakeTask.init_kwargs["auto_connect_streams"] is True


def test_clearml_close_marks_failed_before_close() -> None:
    from bphwt.clearml_utils import close_clearml_task

    class FakeTask:
        def __init__(self) -> None:
            self.calls: list[tuple[str, object]] = []

        def mark_failed(self, *, force=False, status_message=None):
            self.calls.append(("mark_failed", force, status_message))

        def close(self) -> None:
            self.calls.append(("close",))

    task = FakeTask()

    close_clearml_task(task, failed=True, status_message="training crashed")

    assert task.calls == [
        ("mark_failed", True, "training crashed"),
        ("close",),
    ]


def test_train_main_marks_clearml_failed_on_exception(monkeypatch, tmp_path: Path) -> None:
    from bphwt.train import __main__ as train_main

    task = object()
    close_calls: list[dict[str, object]] = []
    cfg = types.SimpleNamespace(
        run=types.SimpleNamespace(output_dir=str(tmp_path / "out")),
        resolved_cache_dir=lambda: tmp_path / "cache",
    )

    monkeypatch.setattr(train_main, "load_config", lambda _: cfg)
    monkeypatch.setattr(train_main, "apply_runtime_overrides", lambda _: None)
    monkeypatch.setattr(train_main, "save_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_main, "init_clearml_task", lambda *_args, **_kwargs: task)
    monkeypatch.setattr(train_main, "resolve_clearml_data_dir", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        train_main,
        "cache_status",
        lambda *_args, **_kwargs: types.SimpleNamespace(valid=True, meta_count=1, extra_npz_count=0),
    )
    monkeypatch.setattr(train_main, "run_cv", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(
        train_main,
        "close_clearml_task",
        lambda task_arg, **kwargs: close_calls.append(kwargs) if task_arg is not None else None,
    )

    with pytest.raises(RuntimeError, match="boom"):
        train_main.main(["config.yml"])

    assert close_calls
    assert close_calls[0]["failed"] is True
    assert "RuntimeError: boom" in str(close_calls[0]["status_message"])


def test_segment_dtw_fast_kernel_matches_python() -> None:
    from bphwt.priors.segment_dtw import _banded_dtw, _banded_dtw_python

    x = np.sin(np.linspace(0.0, 3.0, 32)).astype(np.float64)
    y = np.sin(np.linspace(0.1, 3.1, 35)).astype(np.float64)

    cost_py, path_i_py, path_j_py = _banded_dtw_python(x, y, band=8)
    cost_fast, path_i_fast, path_j_fast = _banded_dtw(x, y, band=8)

    assert np.isfinite(cost_fast)
    np.testing.assert_allclose(cost_fast, cost_py, rtol=1e-8, atol=1e-8)
    assert path_i_fast == path_i_py
    assert path_j_fast == path_j_py


def test_train_logging_helpers_format_breakdowns_and_csv(tmp_path: Path) -> None:
    from bphwt.train.train_fold import (
        MetricTracker,
        append_metric_row,
        format_cv_summary_table,
        format_loss_metrics,
        format_progress_header,
        format_progress_row,
        should_log_train_step,
    )

    tracker = MetricTracker()
    tracker.update({"loss_total": 2.0, "loss_tvt": 1.0, "loss_fwd_gr": 0.25})
    tracker.update({"loss_total": 4.0, "loss_tvt": 3.0})

    averages = tracker.average()
    assert averages["loss_total"] == 3.0
    assert averages["loss_tvt"] == 2.0
    assert averages["loss_fwd_gr"] == 0.25

    line = format_loss_metrics("train", averages)
    assert "train total=3.0000" in line
    assert "tvt=2.0000" in line
    assert "fwd_gr=0.2500" in line

    header = format_progress_header(1)
    row = format_progress_row(
        fold=1,
        event="VAL*",
        epoch=20,
        epochs=20,
        step=80,
        lr=3e-4,
        train_losses={"loss_total": 389.2834, "loss_tvt": 389.2834},
        val_losses={"loss_total": 30.6791, "loss_tvt": 30.6791},
        rmse=9.6085,
        base_rmse=10.1000,
        rmse_gain=0.4915,
        best=9.6085,
        hidden_rows=2048,
        elapsed_s=0.4,
        note="saved",
    )
    assert "event" in header
    assert "train" in header
    assert "base" in header
    assert "gain" in header
    assert "tr_tvt" not in header
    assert "val_tvt" not in header
    assert "[fold 1] VAL*" in row
    assert "020/020" in row
    assert "000080" in row
    assert "9.6085" in row
    assert "10.1000" in row
    assert "+0.4915" in row
    assert "2048" in row

    summary = format_cv_summary_table(
        [
            {"fold": 0, "val_rmse": 136.5961, "val_base_rmse": 140.0, "val_rmse_gain": 3.4039, "val_loss": 123.0},
            {"fold": 1, "val_rmse": 9.6085, "val_base_rmse": 10.1, "val_rmse_gain": 0.4915, "val_loss": 30.6791},
        ],
        mean_rmse=73.1023,
        std_rmse=63.4938,
    )
    assert "CV SUMMARY" in summary
    assert "val_max_seq_len" in summary
    assert "WINDOW" not in summary
    assert "136.5961" in summary

    metrics_path = tmp_path / "metrics.csv"
    append_metric_row(metrics_path, {"fold": 0, "epoch": 1, "step": 2, "val_rmse": 9.5})
    append_metric_row(metrics_path, {"fold": 0, "epoch": 1, "step": 3, "val_rmse": 9.0})

    df = pd.read_csv(metrics_path)
    assert df["val_rmse"].tolist() == [9.5, 9.0]
    assert should_log_train_step(1, log_every=50, log_first_steps=3)
    assert should_log_train_step(3, log_every=50, log_first_steps=3)
    assert not should_log_train_step(4, log_every=50, log_first_steps=3)
    assert should_log_train_step(50, log_every=50, log_first_steps=3)


def test_clearml_runtime_env_overrides(monkeypatch) -> None:
    from bphwt.clearml_utils import apply_runtime_overrides
    from bphwt.config import load_config

    cfg = load_config("configs/bphwt_lite.yml")
    monkeypatch.setenv("BPHWT_TRACKING_ENABLED", "true")
    monkeypatch.setenv("BPHWT_TRACKING_PROJECT", "Proj/X")
    monkeypatch.setenv("BPHWT_TRACKING_TAGS", "gpu,server,test")
    monkeypatch.setenv("BPHWT_DATA_CLEARML_ENABLED", "true")
    monkeypatch.setenv("BPHWT_DATA_CLEARML_VERSION", "v123")
    monkeypatch.setenv("BPHWT_DATA_CLEARML_MAX_WORKERS", "16")
    monkeypatch.setenv("BPHWT_DATA_RANDOM_TRAIN_CROP", "false")
    monkeypatch.setenv("BPHWT_DATA_VAL_MAX_SEQ_LEN", "2048")
    monkeypatch.setenv("BPHWT_TRAIN_DEVICE", "auto")
    monkeypatch.setenv("BPHWT_K_WELLS", "42")

    apply_runtime_overrides(cfg)

    assert cfg.tracking.enabled is True
    assert cfg.tracking.project == "Proj/X"
    assert cfg.tracking.tags == ["gpu", "server", "test"]
    assert cfg.data_clearml.enabled is True
    assert cfg.data_clearml.version == "v123"
    assert cfg.data_clearml.max_workers == 16
    assert cfg.data.random_train_crop is False
    assert cfg.data.val_max_seq_len == 2048
    assert cfg.train.device == "auto"
    assert cfg.k_wells == 42


def test_clearml_data_module_imports_and_parser_builds() -> None:
    from bphwt.clearml_data import build_parser

    parser = build_parser()
    args = parser.parse_args(["download", "--project", "P", "--name", "D", "--version", "V"])

    assert args.command == "download"
    assert args.project == "P"
    assert args.name == "D"
    assert args.version == "V"


def test_docker_entrypoint_parses_spacebridge_args() -> None:
    from bphwt.docker_entrypoint import parse_entrypoint_args

    args = parse_entrypoint_args(
        [
            "--config=configs/bphwt_server.yml",
            "--rebuild-cache=true",
            "--keep-alive-on-fail=120",
            "--preflight-sleep=3",
            "--sleep-only=true",
        ]
    )

    assert args.config == "configs/bphwt_server.yml"
    assert args.rebuild_cache is True
    assert args.keep_alive_on_fail == 120
    assert args.preflight_sleep == 3
    assert args.sleep_only is True


def test_docker_entrypoint_reads_rebuild_cache_env(monkeypatch) -> None:
    from bphwt.docker_entrypoint import parse_entrypoint_args

    monkeypatch.setenv("BPHWT_REBUILD_CACHE", "1")

    args = parse_entrypoint_args(["--config=configs/bphwt_v1.yml"])

    assert args.config == "configs/bphwt_v1.yml"
    assert args.rebuild_cache is True


def test_dockerfile_runs_bphwt_entrypoint() -> None:
    dockerfile = Path("Dockerfile").read_text()

    assert "ARG BPHWT_BASE_IMAGE" in dockerfile
    assert "FROM ${BPHWT_BASE_IMAGE}" in dockerfile
    assert "uv sync" not in dockerfile
    assert "bphwt.docker_entrypoint" in dockerfile
    assert "rogii" not in dockerfile


def test_dockerfile_full_builds_dependency_base() -> None:
    dockerfile = Path("Dockerfile.full").read_text()

    assert "uv sync --locked" in dockerfile
    assert "COPY pyproject.toml uv.lock .python-version ./" in dockerfile
    assert "COPY . ." not in dockerfile


def test_bpi_spline_script_predicts_one_synthetic_cache(tmp_path: Path) -> None:
    from bphwt.config import load_config
    from bphwt.data.build_cache import process_one_well

    script_path = Path("scripts/run_bpi_spline_oof.py")
    spec = importlib.util.spec_from_file_location("run_bpi_spline_oof", script_path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    cfg = load_config("configs/bphwt_lite.yml")
    cfg.data.use_hmm_prior = False
    cfg.surface.enabled = False
    cfg.neighbor.enabled = False
    out = process_one_well(_synthetic_well(tmp_path), cfg, norm_stats={})
    assert out is not None
    meta = out.pop("_meta")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    np.savez_compressed(cache_dir / f"{meta['well_id']}.npz", **out, _meta=np.array(meta, dtype=object))

    args = types.SimpleNamespace(
        n_knots=8,
        n_steps=2,
        lr=0.1,
        prior_sigma=10.0,
        lambda_prior=0.6,
        lambda_smooth=0.03,
        lambda_anchor=100.0,
        clip_correction=5.0,
        gr_huber_delta=15.0,
        min_gr_valid=0.2,
    )

    row = mod.predict_one(cache_dir / f"{meta['well_id']}.npz", args)

    assert row["well_id"] == meta["well_id"]
    assert row["pred"].shape == row["base"].shape
    assert np.isfinite(row["rmse_bpi"])
    assert np.isfinite(row["rmse_base"])


def test_train_server_ensures_dependency_base_image() -> None:
    makefile = Path("Makefile").read_text()

    assert "ensure-docker-base:" in makefile
    assert "docker manifest inspect" in makefile
    assert "train-server train-spacebridge: check-server-env check-server-config ensure-docker-base" in makefile


def test_codebase_snapshot_excludes_portainer_secrets() -> None:
    makefile = Path("Makefile").read_text()
    root_files_lines = [line for line in makefile.splitlines() if line.startswith("CODEBASE_ROOT_FILES")]

    assert root_files_lines
    assert "portainer.yml" not in root_files_lines[0]


def test_error_analysis_helpers_are_masked_and_row_weighted() -> None:
    from bphwt.infer.oof_diagnostics import row_weighted_rmse, summarize_well_errors

    true = np.array([0.0, 10.0, 100.0], dtype=np.float32)
    hidden = np.array([1.0, 0.0, 1.0], dtype=np.float32)
    data = {
        "tvt_true": true,
        "hidden_mask": hidden,
        "gr_valid": np.array([1.0, 0.0, 1.0], dtype=np.float32),
        "tvt_base": np.array([3.0, 10.0, 104.0], dtype=np.float32),
        "tvt_linear": np.array([0.0, 10.0, 100.0], dtype=np.float32),
        "tvt_hmm": np.array([30.0, 10.0, 140.0], dtype=np.float32),
    }
    nn_pred = np.array([0.0, 10.0, 110.0], dtype=np.float32)

    row = summarize_well_errors(fold=0, well_id="w0", data=data, nn_pred=nn_pred)

    assert row["hidden_rows"] == 2
    np.testing.assert_allclose(row["nn_rmse"], np.sqrt(50.0))
    np.testing.assert_allclose(row["base_rmse"], np.sqrt(12.5))
    assert row["best_prior"] == "linear"
    assert row["linear_rmse"] == 0.0

    df = pd.DataFrame(
        [
            {"source": "nn", "sse": 100.0, "n": 1},
            {"source": "nn", "sse": 0.0, "n": 3},
        ]
    )
    np.testing.assert_allclose(row_weighted_rmse(df, "nn"), 5.0)


def test_blend_tvt_base_downweights_hmm_that_disagrees_with_safe_priors() -> None:
    from bphwt.data.build_cache import _blend_tvt_base

    linear = np.array([100.0, 101.0, 102.0], dtype=np.float32)
    last = linear.copy()
    neighbor = linear.copy()
    dtw = linear.copy()
    bad_hmm = np.array([700.0, 701.0, 702.0], dtype=np.float32)

    blended = _blend_tvt_base(
        linear_tvt=linear,
        last_tvt=last,
        hmm_tvt=bad_hmm,
        dtw_tvt=dtw,
        neighbor_tvt=neighbor,
        hmm_entropy=np.zeros_like(linear),
        dtw_score=np.zeros_like(linear),
        gr_valid_ratio=0.9,
    )

    assert float(np.max(np.abs(blended - linear))) < 10.0
