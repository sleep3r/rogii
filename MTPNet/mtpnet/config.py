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
class PriorConfig:
    enabled: bool = False
    id_column: str = "id"
    base_path: Path | None = None
    b2_path: Path | None = None
    a_path: Path | None = None
    base_column: str = "schema10_oof_pp"
    b2_column: str = "b2_guarded_submit"
    b2_danger_column: str = "b2_submit_danger_score"
    a_p50_column: str = "formation_sample_median"
    a_p10_column: str = "formation_sample_p10"
    a_p90_column: str = "formation_sample_p90"
    strict: bool = True


@dataclass(frozen=True)
class AugmentationConfig:
    enabled: bool = False
    drop_anchor_sdf_prob: float = 0.0
    drop_b2_sdf_prob: float = 0.0
    drop_a_density_prob: float = 0.0
    drop_all_priors_prob: float = 0.0
    anchor_jitter_ft: tuple[float, ...] = ()
    anchor_swap_prob: float = 0.0
    anchor_swap: tuple[str, ...] = ()


@dataclass(frozen=True)
class WindowConfig:
    rows_per_step: int = 32
    history_steps: int = 8
    future_steps: int = 16
    vertical_bins: int = 64
    vertical_radius_ft: float = 160.0
    stride_steps: int = 4
    max_windows_per_well: int = 64
    train_history_mode: str = "teacher_forcing"
    valid_history_mode: str = "known_tail_only"
    train_center_source: str = "true_tvt"
    valid_center_source: str = "tvt_input_tail"
    train_sample_mix: dict[str, float] = field(default_factory=dict)
    valid_sample_types: tuple[str, ...] = ()
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
    bounded_output: bool = False
    mode_bias_init: bool = False
    mode_bias_span_bins: float = 20.0


@dataclass(frozen=True)
class LossConfig:
    path_loss: str = "mae"
    alpha_cls: float = 0.2
    cls_warmup_epochs: int = 0
    alpha_cls_warmup_value: float = 0.0
    smooth_lambda: float = 0.01
    entropy_lambda: float = 0.0
    entropy_warmup_epochs: int = 0
    entropy_final_lambda: float = 0.0
    diversity_lambda: float = 0.0
    diversity_margin_bins: float = 0.0
    soft_prob_alpha: float = 0.0
    soft_prob_tau_bins: float = 2.0
    top3_margin_alpha: float = 0.0
    top3_margin: float = 0.0
    continuation_alpha: float = 0.0
    continuation_tau_bins: float = 3.0


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
    train_wells: tuple[str, ...] = ()
    valid_wells: tuple[str, ...] = ()


@dataclass(frozen=True)
class RunConfig:
    name: str = "mtp_v0"
    output_dir: Path = Path("artifacts/mtp_v0")


@dataclass(frozen=True)
class MTPConfig:
    data: DataConfig = field(default_factory=DataConfig)
    priors: PriorConfig = field(default_factory=PriorConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
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

    priors_raw = _section(raw, "priors")
    default_priors = PriorConfig()
    priors = PriorConfig(
        enabled=bool(priors_raw.get("enabled", default_priors.enabled)),
        id_column=str(priors_raw.get("id_column", default_priors.id_column)),
        base_path=_as_path(priors_raw.get("base_path")),
        b2_path=_as_path(priors_raw.get("b2_path")),
        a_path=_as_path(priors_raw.get("a_path")),
        base_column=str(priors_raw.get("base_column", default_priors.base_column)),
        b2_column=str(priors_raw.get("b2_column", default_priors.b2_column)),
        b2_danger_column=str(
            priors_raw.get("b2_danger_column", default_priors.b2_danger_column)
        ),
        a_p50_column=str(
            priors_raw.get("a_p50_column", default_priors.a_p50_column)
        ),
        a_p10_column=str(
            priors_raw.get("a_p10_column", default_priors.a_p10_column)
        ),
        a_p90_column=str(
            priors_raw.get("a_p90_column", default_priors.a_p90_column)
        ),
        strict=bool(priors_raw.get("strict", default_priors.strict)),
    )

    augmentation_raw = _section(raw, "augmentation")
    default_augmentation = AugmentationConfig()
    augmentation = AugmentationConfig(
        enabled=bool(augmentation_raw.get("enabled", default_augmentation.enabled)),
        drop_anchor_sdf_prob=float(
            augmentation_raw.get(
                "drop_anchor_sdf_prob", default_augmentation.drop_anchor_sdf_prob
            )
        ),
        drop_b2_sdf_prob=float(
            augmentation_raw.get(
                "drop_b2_sdf_prob", default_augmentation.drop_b2_sdf_prob
            )
        ),
        drop_a_density_prob=float(
            augmentation_raw.get(
                "drop_a_density_prob", default_augmentation.drop_a_density_prob
            )
        ),
        drop_all_priors_prob=float(
            augmentation_raw.get(
                "drop_all_priors_prob", default_augmentation.drop_all_priors_prob
            )
        ),
        anchor_jitter_ft=tuple(
            float(value)
            for value in _tuple(
                augmentation_raw.get("anchor_jitter_ft"),
                default_augmentation.anchor_jitter_ft,
            )
        ),
        anchor_swap_prob=float(
            augmentation_raw.get(
                "anchor_swap_prob", default_augmentation.anchor_swap_prob
            )
        ),
        anchor_swap=tuple(
            str(value)
            for value in _tuple(
                augmentation_raw.get("anchor_swap"),
                default_augmentation.anchor_swap,
            )
        ),
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
        train_history_mode=str(
            window_raw.get(
                "train_history_mode", default_window.train_history_mode
            )
        ),
        valid_history_mode=str(
            window_raw.get(
                "valid_history_mode", default_window.valid_history_mode
            )
        ),
        train_center_source=str(
            window_raw.get(
                "train_center_source", default_window.train_center_source
            )
        ),
        valid_center_source=str(
            window_raw.get(
                "valid_center_source", default_window.valid_center_source
            )
        ),
        train_sample_mix={
            str(key): float(value)
            for key, value in (
                window_raw.get(
                    "train_sample_mix", default_window.train_sample_mix
                )
                or {}
            ).items()
        },
        valid_sample_types=tuple(
            window_raw.get("valid_sample_types", default_window.valid_sample_types)
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
        bounded_output=bool(
            model_raw.get("bounded_output", default_model.bounded_output)
        ),
        mode_bias_init=bool(
            model_raw.get("mode_bias_init", default_model.mode_bias_init)
        ),
        mode_bias_span_bins=float(
            model_raw.get("mode_bias_span_bins", default_model.mode_bias_span_bins)
        ),
    )

    loss_raw = _section(raw, "loss")
    default_loss = LossConfig()
    loss = LossConfig(
        path_loss=str(loss_raw.get("path_loss", default_loss.path_loss)),
        alpha_cls=float(loss_raw.get("alpha_cls", default_loss.alpha_cls)),
        cls_warmup_epochs=int(
            loss_raw.get("cls_warmup_epochs", default_loss.cls_warmup_epochs)
        ),
        alpha_cls_warmup_value=float(
            loss_raw.get(
                "alpha_cls_warmup_value", default_loss.alpha_cls_warmup_value
            )
        ),
        smooth_lambda=float(loss_raw.get("smooth_lambda", default_loss.smooth_lambda)),
        entropy_lambda=float(
            loss_raw.get("entropy_lambda", default_loss.entropy_lambda)
        ),
        entropy_warmup_epochs=int(
            loss_raw.get(
                "entropy_warmup_epochs", default_loss.entropy_warmup_epochs
            )
        ),
        entropy_final_lambda=float(
            loss_raw.get("entropy_final_lambda", default_loss.entropy_final_lambda)
        ),
        diversity_lambda=float(
            loss_raw.get("diversity_lambda", default_loss.diversity_lambda)
        ),
        diversity_margin_bins=float(
            loss_raw.get("diversity_margin_bins", default_loss.diversity_margin_bins)
        ),
        soft_prob_alpha=float(
            loss_raw.get("soft_prob_alpha", default_loss.soft_prob_alpha)
        ),
        soft_prob_tau_bins=float(
            loss_raw.get("soft_prob_tau_bins", default_loss.soft_prob_tau_bins)
        ),
        top3_margin_alpha=float(
            loss_raw.get("top3_margin_alpha", default_loss.top3_margin_alpha)
        ),
        top3_margin=float(
            loss_raw.get("top3_margin", default_loss.top3_margin)
        ),
        continuation_alpha=float(
            loss_raw.get("continuation_alpha", default_loss.continuation_alpha)
        ),
        continuation_tau_bins=float(
            loss_raw.get("continuation_tau_bins", default_loss.continuation_tau_bins)
        ),
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
        train_wells=tuple(
            str(value)
            for value in _tuple(
                validation_raw.get("train_wells"),
                default_validation.train_wells,
            )
        ),
        valid_wells=tuple(
            str(value)
            for value in _tuple(
                validation_raw.get("valid_wells"),
                default_validation.valid_wells,
            )
        ),
    )

    run_raw = _section(raw, "run")
    default_run = RunConfig()
    run = RunConfig(
        name=str(run_raw.get("name", default_run.name)),
        output_dir=Path(run_raw.get("output_dir", default_run.output_dir)),
    )

    return MTPConfig(
        data=data,
        priors=priors,
        augmentation=augmentation,
        window=window,
        model=model,
        loss=loss,
        train=train,
        validation=validation,
        run=run,
    )
