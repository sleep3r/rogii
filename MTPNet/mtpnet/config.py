from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    data_dir: Path = Path("data")
    train_dir: Path | None = None
    test_dir: Path | None = None
    k_wells: int = -1
    copy_from: Path | None = None

    @property
    def use_all_wells(self) -> bool:
        return self.k_wells == -1


@dataclass(frozen=True)
class WindowConfig:
    rows_per_step: int = 32
    history_steps: int = 8
    future_steps: int = 16
    vertical_bins: int = 64
    vertical_radius_ft: float = 160.0
    stride_steps: int = 4
    max_windows_per_well: int = 64
    channels: tuple[str, ...] = (
        "gr_diff",
        "abs_gr_diff",
        "history_mask",
        "history_sdf",
        "finite_mask",
    )


@dataclass(frozen=True)
class ModelConfig:
    k_modes: int = 8
    conv_channels: tuple[int, ...] = (16, 32, 64, 128)
    hidden_dims: tuple[int, ...] = (512, 1024)
    dropout: float = 0.05


@dataclass(frozen=True)
class LossConfig:
    path_loss: str = "mae"
    alpha_cls: float = 0.2
    smooth_lambda: float = 0.01


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 64
    epochs: int = 8
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    device: str = "auto"
    num_workers: int = 0
    seed: int = 42


@dataclass(frozen=True)
class ValidationConfig:
    valid_fraction: float = 0.2
    seed: int = 42


@dataclass(frozen=True)
class RunConfig:
    name: str = "mtp_v0"
    output_dir: Path = Path("artifacts/mtp_v0")


@dataclass(frozen=True)
class MTPConfig:
    data: DataConfig = field(default_factory=DataConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    run: RunConfig = field(default_factory=RunConfig)


def _as_path(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    return Path(str(value))


def _tuple(value: Any, default: tuple[Any, ...]) -> tuple[Any, ...]:
    if value is None:
        return default
    return tuple(value)


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} config section must be a mapping")
    return value


def load_config(path: str | Path) -> MTPConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    data_raw = _section(raw, "data")
    k_wells = int(data_raw.get("k_wells", -1))
    if k_wells == 0 or k_wells < -1:
        raise ValueError("data.k_wells must be -1 or a positive integer")
    data = DataConfig(
        data_dir=Path(data_raw.get("data_dir", "data")),
        train_dir=_as_path(data_raw.get("train_dir")),
        test_dir=_as_path(data_raw.get("test_dir")),
        k_wells=k_wells,
        copy_from=_as_path(data_raw.get("copy_from")),
    )

    window_raw = _section(raw, "window")
    default_window = WindowConfig()
    window = WindowConfig(
        rows_per_step=int(
            window_raw.get("rows_per_step", default_window.rows_per_step)
        ),
        history_steps=int(
            window_raw.get("history_steps", default_window.history_steps)
        ),
        future_steps=int(window_raw.get("future_steps", default_window.future_steps)),
        vertical_bins=int(
            window_raw.get("vertical_bins", default_window.vertical_bins)
        ),
        vertical_radius_ft=float(
            window_raw.get("vertical_radius_ft", default_window.vertical_radius_ft)
        ),
        stride_steps=int(window_raw.get("stride_steps", default_window.stride_steps)),
        max_windows_per_well=int(
            window_raw.get(
                "max_windows_per_well", default_window.max_windows_per_well
            )
        ),
        channels=tuple(window_raw.get("channels", default_window.channels)),
    )

    model_raw = _section(raw, "model")
    default_model = ModelConfig()
    model = ModelConfig(
        k_modes=int(model_raw.get("k_modes", default_model.k_modes)),
        conv_channels=tuple(
            int(v)
            for v in _tuple(model_raw.get("conv_channels"), default_model.conv_channels)
        ),
        hidden_dims=tuple(
            int(v)
            for v in _tuple(model_raw.get("hidden_dims"), default_model.hidden_dims)
        ),
        dropout=float(model_raw.get("dropout", default_model.dropout)),
    )

    loss_raw = _section(raw, "loss")
    default_loss = LossConfig()
    loss = LossConfig(
        path_loss=str(loss_raw.get("path_loss", default_loss.path_loss)),
        alpha_cls=float(loss_raw.get("alpha_cls", default_loss.alpha_cls)),
        smooth_lambda=float(loss_raw.get("smooth_lambda", default_loss.smooth_lambda)),
    )

    train_raw = _section(raw, "train")
    default_train = TrainConfig()
    train = TrainConfig(
        batch_size=int(train_raw.get("batch_size", default_train.batch_size)),
        epochs=int(train_raw.get("epochs", default_train.epochs)),
        learning_rate=float(
            train_raw.get("learning_rate", default_train.learning_rate)
        ),
        weight_decay=float(train_raw.get("weight_decay", default_train.weight_decay)),
        device=str(train_raw.get("device", default_train.device)),
        num_workers=int(train_raw.get("num_workers", default_train.num_workers)),
        seed=int(train_raw.get("seed", default_train.seed)),
    )

    validation_raw = _section(raw, "validation")
    default_validation = ValidationConfig()
    validation = ValidationConfig(
        valid_fraction=float(
            validation_raw.get("valid_fraction", default_validation.valid_fraction)
        ),
        seed=int(validation_raw.get("seed", default_validation.seed)),
    )

    run_raw = _section(raw, "run")
    default_run = RunConfig()
    run = RunConfig(
        name=str(run_raw.get("name", default_run.name)),
        output_dir=Path(run_raw.get("output_dir", default_run.output_dir)),
    )

    return MTPConfig(
        data=data,
        window=window,
        model=model,
        loss=loss,
        train=train,
        validation=validation,
        run=run,
    )
