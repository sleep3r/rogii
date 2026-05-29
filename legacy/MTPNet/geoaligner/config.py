from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class GADataConfig:
    rows_per_step: int = 32
    max_horizontal_steps: int = 512
    vertical_step_ft: float = 5.0
    max_typewell_bins: int = 512


@dataclass
class GAModelConfig:
    d_model: int = 128
    n_heads: int = 4
    lateral_layers: int = 2
    typewell_layers: int = 2
    ffn_dim: int = 256
    dropout: float = 0.10


@dataclass
class GADecodeConfig:
    max_jump_bins: int = 6
    jump_penalty: float = 0.03
    anchor_band_radius_ft: float = 160.0
    anchor_band_penalty: float = 0.02
    topk: tuple[int, ...] = (1, 3, 10)


@dataclass
class GATrainConfig:
    batch_size: int = 2
    epochs: int = 8
    lr: float = 3e-4
    weight_decay: float = 1e-4
    device: str = "auto"
    seed: int = 42
    valid_fraction: float = 0.15
    clip_grad_norm: float = 1.0


@dataclass
class GARunConfig:
    name: str = "geoaligner_v0"
    output_dir: Path = field(default_factory=lambda: Path("artifacts/geoaligner_v0"))


@dataclass
class GAConfig:
    data: GADataConfig = field(default_factory=GADataConfig)
    model: GAModelConfig = field(default_factory=GAModelConfig)
    decode: GADecodeConfig = field(default_factory=GADecodeConfig)
    train: GATrainConfig = field(default_factory=GATrainConfig)
    run: GARunConfig = field(default_factory=GARunConfig)
    data_dir: Path = field(default_factory=lambda: Path("data"))
    k_wells: int = -1


def _coerce_tuple(value: Any, item_type: type = int) -> tuple:
    if value is None:
        return tuple()
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(item_type(v) for v in value)
    return (item_type(value),)


def _coerce(cls: type, raw: Any) -> Any:
    if not isinstance(raw, dict):
        return raw
    kwargs: dict[str, Any] = {}
    fields = cls.__dataclass_fields__
    for key, value in raw.items():
        if key not in fields:
            continue
        ftype = fields[key].type
        if isinstance(ftype, str):
            import sys

            ftype = eval(ftype, sys.modules[cls.__module__].__dict__)  # noqa: S307
        if ftype is Path:
            value = Path(value)
        elif hasattr(ftype, "__args__") and type(None) in ftype.__args__:
            inner = [arg for arg in ftype.__args__ if arg is not type(None)][0]
            if inner is Path and value is not None:
                value = Path(value)
        elif ftype in (tuple[int, ...], tuple):
            value = _coerce_tuple(value, int)
        elif isinstance(value, dict) and hasattr(ftype, "__dataclass_fields__"):
            value = _coerce(ftype, value)
        kwargs[key] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> GAConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return _coerce(GAConfig, raw)


def config_to_dict(cfg: GAConfig) -> dict[str, Any]:
    def convert(value: Any) -> Any:
        if is_dataclass(value):
            return {k: convert(v) for k, v in asdict(value).items()}
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, tuple):
            return [convert(v) for v in value]
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        return value

    return convert(cfg)
