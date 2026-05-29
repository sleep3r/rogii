from __future__ import annotations

from types import SimpleNamespace

from racformer.config import RACFormerConfig
from racformer.train import _optimizer_steps_per_epoch, _should_validate_step_mode, _train_val_splits


def _sample(well_id: str):
    return SimpleNamespace(well_id=well_id)


def test_debug_overfit_splits_use_same_first_samples_for_train_and_val() -> None:
    cfg = RACFormerConfig()
    cfg.train.debug_overfit_samples = 3
    samples = [_sample(f"well_{idx}") for idx in range(5)]

    splits = _train_val_splits(samples, cfg)

    assert splits == [([0, 1, 2], [0, 1, 2])]


def test_regular_splits_still_use_group_kfold() -> None:
    cfg = RACFormerConfig()
    cfg.train.debug_overfit_samples = 0
    cfg.train.n_folds = 2
    samples = [_sample(f"well_{idx}") for idx in range(4)]

    splits = _train_val_splits(samples, cfg)

    assert len(splits) == 2
    assert all(set(train).isdisjoint(val) for train, val in splits)


def test_optimizer_steps_per_epoch_rounds_up_accumulated_batches() -> None:
    assert _optimizer_steps_per_epoch(train_batches=39, grad_accum=2) == 20
    assert _optimizer_steps_per_epoch(train_batches=1, grad_accum=8) == 1


def test_step_mode_validation_runs_on_interval_and_final_step() -> None:
    assert _should_validate_step_mode(
        optimizer_steps_done=99,
        max_optimizer_steps=1000,
        validate_every_steps=100,
    ) is False
    assert _should_validate_step_mode(
        optimizer_steps_done=100,
        max_optimizer_steps=1000,
        validate_every_steps=100,
    ) is True
    assert _should_validate_step_mode(
        optimizer_steps_done=1000,
        max_optimizer_steps=1000,
        validate_every_steps=100,
    ) is True
