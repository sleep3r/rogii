"""PathFormer configuration dataclasses + YAML loading."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Feature dimensions (fixed, derived from dataset.py feature list)
# ---------------------------------------------------------------------------
N_FEATURES = 28  # must match build_features() in dataset.py


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class PFDataConfig:
    rows_per_step: int = 16      # compress this many raw rows into one step
    max_seq_len: int = 1024      # truncate wells longer than this


@dataclass
class PFModelConfig:
    d_model: int = 192
    n_heads: int = 8
    n_encoder_layers: int = 4
    ffn_dim: int = 768           # feed-forward hidden dim (= 4 * d_model)
    dropout: float = 0.10
    anchor_skip: bool = True
    anchor_feature_index: int = 17
    anchor_valid_index: int = 22
    anchor_scale: float = 100.0


@dataclass
class PFAugConfig:
    """Prior dropout augmentation during training."""
    drop_b2_prob: float = 0.30
    drop_base_prob: float = 0.20
    drop_a_prob: float = 0.25
    drop_all_priors_prob: float = 0.15  # overrides individual drops


@dataclass
class PFPriorConfig:
    enabled: bool = False
    id_column: str = "id"
    base_path: Path | None = None
    b2_path: Path | None = None
    a_path: Path | None = None
    base_column: str = "schema10_oof_pp"
    b2_column: str = "b2_guarded_submit"
    a_p50_column: str = "formation_sample_median"
    a_p10_column: str = "formation_sample_p10"
    a_p90_column: str = "formation_sample_p90"


@dataclass
class PFTrainConfig:
    batch_size: int = 4          # wells per gradient update (with padding)
    epochs: int = 16
    lr: float = 3e-4
    weight_decay: float = 1e-4
    device: str = "auto"
    seed: int = 42
    valid_fraction: float = 0.15
    # tail-balanced sampling
    tail_oversample_factor: float = 2.0   # non-OK-or-mixed wells sampled this many times more
    # loss
    huber_delta: float = 5.0     # ft
    smooth_lambda: float = 0.005
    clip_grad_norm: float = 1.0
    # checkpointing
    save_every_n_epochs: int = 2


@dataclass
class PFRunConfig:
    name: str = "pathformer_v0"
    output_dir: Path = field(default_factory=lambda: Path("artifacts/pathformer_v0"))


@dataclass
class PathFormerConfig:
    data: PFDataConfig = field(default_factory=PFDataConfig)
    model: PFModelConfig = field(default_factory=PFModelConfig)
    augmentation: PFAugConfig = field(default_factory=PFAugConfig)
    train: PFTrainConfig = field(default_factory=PFTrainConfig)
    priors: PFPriorConfig = field(default_factory=PFPriorConfig)
    run: PFRunConfig = field(default_factory=PFRunConfig)
    # top-level shortcuts
    data_dir: Path = field(default_factory=lambda: Path("data"))
    k_wells: int = -1
    tail_audit_path: Path | None = None   # well_tail_audit.csv for tail balancing


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------

def _coerce(cls: type, raw: Any) -> Any:
    """Recursively coerce a dict into a dataclass instance."""
    if not isinstance(raw, dict):
        return raw
    hints: dict[str, Any] = {f.name: f for f in cls.__dataclass_fields__.values()}
    kwargs: dict[str, Any] = {}
    for key, val in raw.items():
        if key not in hints:
            continue
        f = hints[key]
        ftype = f.type
        # resolve string annotations
        if isinstance(ftype, str):
            import sys
            ftype = eval(ftype, sys.modules[cls.__module__].__dict__)  # noqa: S307
        origin = getattr(ftype, "__origin__", None)
        # handle Path | None
        if hasattr(ftype, "__args__") and type(None) in ftype.__args__:
            inner = [a for a in ftype.__args__ if a is not type(None)][0]
            if inner is Path and val is not None:
                val = Path(val)
        elif ftype is Path:
            val = Path(val)
        elif isinstance(val, dict) and hasattr(ftype, "__dataclass_fields__"):
            val = _coerce(ftype, val)
        kwargs[key] = val
    return cls(**kwargs)


def load_config(path: str | Path) -> PathFormerConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return _coerce(PathFormerConfig, raw or {})
