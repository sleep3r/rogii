"""RAC-Former v1 configuration (No-Prior anchored C-Former).

Architecture constants:
  ROWS_PER_STEP = 32    raw rows → 1 step token
  K_SEG = 16            learned segment queries for C-field correction
  D_MODEL = 128         embedding dimension
  N_HEADS = 4           attention heads
  ENC_LAYERS = 3        transformer encoder layers
  DEC_LAYERS = 2        segment decoder layers
  FF_DIM = 512          FFN hidden dim
  DROPOUT = 0.06
  STOCH_DEPTH = 0.05

Physical decomposition (no external priors):
  base_tvt[i] = anchor_tvt - (Z[i] - Z_anchor) + c0 * (i - anchor_row)
  pred_tvt[i] = base_tvt[i] + cumsum(segment_slope_residual) + direct_residual

  c0 = median(forward diff(TVT_input + Z)) over the last 512 known rows;
  set to 0 if the well has < 2 known rows in the window.

Feature layout per step (N_RAW_FEATURES = 42):
  see dataset.py FEATURE_NAMES for the full list.
  N_FOURIER = 32 (4 inputs × 4 freqs × 2 sin/cos)
  N_FEATURES = N_RAW_FEATURES + N_FOURIER = 74
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Architecture constants (shared with dataset + model)
# ---------------------------------------------------------------------------

ROWS_PER_STEP: int = 32
K_SEG: int = 16
D_MODEL: int = 128
N_HEADS: int = 4
ENC_LAYERS: int = 3
DEC_LAYERS: int = 2
FF_DIM: int = 512
DROPOUT: float = 0.06
STOCH_DEPTH: float = 0.05

# Feature dims (no-prior version: 6 external-prior channels removed)
N_RAW_FEATURES: int = 42
N_FOURIER: int = 32          # 4 inputs × 4 freqs × 2
N_FEATURES: int = N_RAW_FEATURES + N_FOURIER   # 74

# Soft-bucket centers for the auxiliary distributional head
N_BUCKETS: int = 21
BUCKET_LO: float = -0.08   # ft/row
BUCKET_HI: float = 0.08    # ft/row

# Event detection threshold on |dC| ft/step
EVENT_THRESHOLD: float = 0.10


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class RACDataConfig:
    rows_per_step: int = ROWS_PER_STEP
    max_seq_len: int = 384          # 384*32 = 12288 raw rows — covers all 773 wells
    last_known_window: int = 512    # raw rows used for anchor statistics
    use_c0_drift: bool = True       # if False, base_tvt = anchor - dZ (c0=0)
    top_teacher_eps: float = 0.10   # ANCC d-step threshold for top-event distillation


@dataclass
class RACModelConfig:
    d_model: int = D_MODEL
    n_heads: int = N_HEADS
    enc_layers: int = ENC_LAYERS
    dec_layers: int = DEC_LAYERS
    ff_dim: int = FF_DIM
    k_seg: int = K_SEG
    dropout: float = DROPOUT
    stoch_depth: float = STOCH_DEPTH
    n_buckets: int = N_BUCKETS
    bucket_lo: float = BUCKET_LO
    bucket_hi: float = BUCKET_HI
    # head output bounds
    seg_slope_bound: float = 0.08    # tanh bound for residual slope ft/row
    direct_resid_bound: float = 3.0  # tanh bound for direct residual ft (tighter than v0 6.0)


@dataclass
class RACAugConfig:
    pseudo_anchor_prob: float = 0.0           # OFF by default (clean rebuild)
    pseudo_anchor_lo: float = 0.45
    pseudo_anchor_hi: float = 0.85
    pseudo_anchor_loss_weight: float = 0.35
    bin_shift_options: tuple[int, ...] = (0, 8, 16, 24)   # step-bin shift TTA
    gr_dropout_prob: float = 0.05             # very mild GR robustness aug
    gr_dropout_frac: float = 0.20             # fraction of hidden steps to drop GR
    feature_noise_std: float = 0.01           # Gaussian noise on normalized features


@dataclass
class RACTrainConfig:
    batch_size: int = 16        # wells per gradient update (CPU: use 4)
    grad_accum: int = 2         # effective batch = batch_size * grad_accum
    epochs: int = 160
    max_optimizer_steps: int = 0  # >0: stop by optimizer steps instead of epochs
    validate_every_steps: int = 0  # step mode only; 0 validates every pass through loader
    warmup_epochs: int = 10
    lr: float = 8e-4
    lr_min: float = 2e-5
    weight_decay: float = 2e-4
    clip_grad_norm: float = 1.0
    ema_start_epoch: int = 20
    ema_decay: float = 0.999
    early_stop_patience: int = 25
    n_folds: int = 5
    debug_overfit_samples: int = 0  # >0: train and validate on first N wells in one fold
    seed: int = 42
    device: str = "auto"
    num_workers: int = 0
    # loss weights
    w_tvt_mse: float = 1.00
    w_tvt_huber: float = 0.25
    w_endpoint: float = 0.15
    w_seg: float = 0.40
    w_local: float = 0.10
    w_smooth: float = 0.03
    w_direct_reg: float = 0.02
    w_event: float = 0.03
    w_bucket: float = 0.00         # disabled until segment model works
    w_top_event: float = 0.03
    w_top_dir: float = 0.03
    # soft bucket KL target width
    bucket_soft_sigma: float = 0.006   # ft/row
    # event focal loss params
    event_focal_alpha: float = 0.75
    event_focal_gamma: float = 2.0


@dataclass
class RACRunConfig:
    name: str = "racformer_v1"
    output_dir: Path = field(default_factory=lambda: Path("artifacts/racformer_v1"))


# ---------------------------------------------------------------------------
# ClearML integration
# ---------------------------------------------------------------------------

@dataclass
class RACClearMLConfig:
    """Experiment tracking config (tracking.clearml)."""
    enabled: bool = False
    project: str = "ROGII/Wellbore/RACFormer"
    task_name: str = ""                # auto: f"{run.name}-{run_id}" if empty
    output_uri: str | None = None      # e.g. "s3://s3-basket-cold.wb.ru/ds-experiments"
    tags: list[str] = field(default_factory=lambda: ["racformer"])
    log_artifacts: bool = True
    log_model: bool = True
    fail_on_error: bool = False


@dataclass
class RACDataClearMLConfig:
    """Dataset resolver config (data_clearml).  When enabled, cfg.data_dir
    is overridden with the local cache path of the resolved ClearML Dataset."""
    enabled: bool = False
    project: str = "ROGII/Wellbore"
    name: str = "rogii-wellbore-geology-prediction"
    version: str | None = None
    dataset_id: str | None = None
    alias: str | None = None
    cache_dir: Path | None = field(default_factory=lambda: Path("~/.cache/clearml/rogii"))


@dataclass
class RACFormerConfig:
    data: RACDataConfig = field(default_factory=RACDataConfig)
    model: RACModelConfig = field(default_factory=RACModelConfig)
    augmentation: RACAugConfig = field(default_factory=RACAugConfig)
    train: RACTrainConfig = field(default_factory=RACTrainConfig)
    run: RACRunConfig = field(default_factory=RACRunConfig)
    tracking: RACClearMLConfig = field(default_factory=RACClearMLConfig)
    data_clearml: RACDataClearMLConfig = field(default_factory=RACDataClearMLConfig)
    # top-level shortcuts
    data_dir: Path = field(default_factory=lambda: Path("data"))
    k_wells: int = -1    # -1 = all wells


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------

def _coerce(cls: type, raw: Any) -> Any:
    if not isinstance(raw, dict):
        return raw
    hints = {f.name: f for f in cls.__dataclass_fields__.values()}
    kwargs: dict[str, Any] = {}
    for key, val in raw.items():
        if key not in hints:
            continue
        f = hints[key]
        ftype = f.type
        if isinstance(ftype, str):
            import sys
            ftype = eval(ftype, sys.modules[cls.__module__].__dict__)  # noqa: S307
        if hasattr(ftype, "__args__") and type(None) in ftype.__args__:
            inner = [a for a in ftype.__args__ if a is not type(None)][0]
            if inner is Path and val is not None:
                val = Path(val)
        elif ftype is Path:
            val = Path(val)
        elif isinstance(val, dict) and hasattr(ftype, "__dataclass_fields__"):
            val = _coerce(ftype, val)
        elif hasattr(ftype, "__origin__") and ftype.__origin__ is tuple and val is not None:
            val = tuple(val)
        kwargs[key] = val
    return cls(**kwargs)


def load_config(path: str | Path) -> RACFormerConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    cfg = _coerce(RACFormerConfig, raw)
    _apply_env_overrides(cfg)
    return cfg


# ---------------------------------------------------------------------------
# Env-var overrides
# ---------------------------------------------------------------------------

_TRUE = {"1", "true", "yes", "y", "on"}
_FALSE = {"0", "false", "no", "n", "off"}


def _env_bool(name: str) -> bool | None:
    import os
    val = os.environ.get(name)
    if val is None or val == "":
        return None
    v = val.strip().lower()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    return None


def _env_str(name: str) -> str | None:
    import os
    val = os.environ.get(name)
    return val if val not in (None, "") else None


def _apply_env_overrides(cfg: RACFormerConfig) -> None:
    """Apply RACFORMER_* env vars on top of YAML config.  No-op if unset."""
    # tracking.clearml
    if (v := _env_bool("RACFORMER_CLEARML_ENABLED")) is not None:
        cfg.tracking.enabled = v
    if (v := _env_str("RACFORMER_CLEARML_PROJECT")) is not None:
        cfg.tracking.project = v
    if (v := _env_str("RACFORMER_CLEARML_TASK_NAME")) is not None:
        cfg.tracking.task_name = v
    if (v := _env_str("RACFORMER_CLEARML_OUTPUT_URI")) is not None:
        cfg.tracking.output_uri = v
    if (v := _env_str("RACFORMER_CLEARML_TAGS")) is not None:
        cfg.tracking.tags = [t.strip() for t in v.split(",") if t.strip()]
    if (v := _env_bool("RACFORMER_CLEARML_LOG_MODEL")) is not None:
        cfg.tracking.log_model = v
    if (v := _env_bool("RACFORMER_CLEARML_FAIL_ON_ERROR")) is not None:
        cfg.tracking.fail_on_error = v

    # data_clearml
    if (v := _env_bool("RACFORMER_DATA_CLEARML_ENABLED")) is not None:
        cfg.data_clearml.enabled = v
    if (v := _env_str("RACFORMER_DATA_CLEARML_PROJECT")) is not None:
        cfg.data_clearml.project = v
    if (v := _env_str("RACFORMER_DATA_CLEARML_NAME")) is not None:
        cfg.data_clearml.name = v
    if (v := _env_str("RACFORMER_DATA_CLEARML_VERSION")) is not None:
        cfg.data_clearml.version = v
    if (v := _env_str("RACFORMER_DATA_CLEARML_ID")) is not None:
        cfg.data_clearml.dataset_id = v
    if (v := _env_str("RACFORMER_DATA_CLEARML_CACHE_DIR")) is not None:
        from pathlib import Path as _P
        cfg.data_clearml.cache_dir = _P(v)


def config_to_dict(cfg: RACFormerConfig) -> dict:
    import dataclasses
    def _convert(obj):
        if dataclasses.is_dataclass(obj):
            return {k: _convert(v) for k, v in dataclasses.asdict(obj).items()}
        if isinstance(obj, Path):
            return str(obj)
        return obj
    return _convert(cfg)
