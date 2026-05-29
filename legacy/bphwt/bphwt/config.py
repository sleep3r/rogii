"""BPHWT configuration — Pydantic models + YAML loader."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------


class DataConfig(BaseModel):
    # Sequence/batching
    max_seq_len: int = 512  # max number of rows per sample (pad/truncate)
    val_max_seq_len: int | None = 0  # validation length; 0 = full sequence, None = use max_seq_len
    stride: int = 1  # row stride when building the sequence
    random_train_crop: bool = True  # sample random windows during training when sequences are truncated
    # GR processing
    gr_smooth_sigmas: list[float] = [5.0, 15.0, 50.0, 200.0]
    gr_fill_method: str = "linear"  # "linear" | "forward"
    # Typewell offset channels
    tw_offsets: list[float] = [-160.0, -80.0, -40.0, -20.0, 0.0, 20.0, 40.0, 80.0, 160.0]
    # HMM prior
    use_hmm_prior: bool = True
    hmm_tvt_step: float = 2.0  # ft per TVT bin
    hmm_tvt_margin: float = 100.0  # ft beyond anchor range to extend TVT grid
    hmm_velocity_sigma: float = 0.025  # ft/ft per MD step
    hmm_gr_sigma: float = 15.0  # GR observation noise (API)
    hmm_max_velocity: float = 0.12  # ft/ft maximum TVT change per MD ft
    # Feature normalisation stats (computed from train, stored in cache)
    normalise: bool = True


class ModelConfig(BaseModel):
    # Input channels (computed dynamically, kept here for reference)
    in_channels: int = 64  # will be overridden by build_cache
    # Tiny U-Net defaults for the Karpathy-style recipe stage.
    stage_channels: list[int] = [32, 64, 96]
    stage_strides: list[int] = [2, 2, 2]
    n_blocks: int = 1
    max_params: int = 500_000
    # Bottleneck attention
    use_bottleneck_attn: bool = False
    attn_heads: int = 4
    attn_window: int = 256  # local attention window size
    # Typewell cross-attention
    use_typewell_cross_attn: bool = False  # enable in step 5
    tw_channels: int = 32
    # Decoder
    decoder_channels: list[int] = [64, 48, 32]
    # Dropout
    dropout: float = 0.0
    stoch_depth: float = 0.0
    # Output heads
    predict_velocity: bool = False
    predict_dip_sign: bool = False
    predict_seg_boundary: bool = False


class AugmentationConfig(BaseModel):
    enabled: bool = False
    # GR degradation
    gr_dropout_prob: float = 0.15  # probability of applying GR dropout
    gr_dropout_frac: float = 0.30  # fraction of GR to blank
    gr_scale_range: list[float] = [0.85, 1.15]
    gr_shift_range: list[float] = [-10.0, 10.0]
    gr_noise_std: float = 3.0
    # Typewell perturbation
    tw_smooth_prob: float = 0.10
    tw_noise_std: float = 2.0
    # Hidden mask randomisation (simulate different anchor structures)
    pseudo_anchor_prob: float = 0.25  # add random known TVT anchors in hidden zone
    pseudo_anchor_lo: float = 0.40
    pseudo_anchor_hi: float = 0.80
    # Feature noise
    feature_noise_std: float = 0.01


class LossConfig(BaseModel):
    # Primary TVT loss on hidden rows
    w_tvt: float = 1.00
    tvt_huber_delta: float = 5.0
    # Forward-GR physics loss
    w_forward_gr: float = 0.0
    gr_huber_delta: float = 10.0
    w_gr_grad: float = 0.5
    w_gr_corr: float = 0.2
    gr_corr_window: int = 64
    # Anchor loss on known rows
    w_anchor: float = 0.05
    # Velocity auxiliary loss
    w_velocity: float = 0.0
    # Dip-sign classification loss
    w_dip_sign: float = 0.0
    dip_sign_threshold: float = 0.005  # |dTVT/dMD| < threshold => flat
    # Segment boundary loss
    w_seg_boundary: float = 0.0
    # Smoothness / curvature regularisation
    w_smooth: float = 0.0
    # NLL (uncertainty) loss
    w_nll: float = 0.0
    nll_nu: float = 4.0  # Student-t degrees of freedom
    # Candidate distillation (best prior)
    w_distill: float = 0.0
    distill_confidence_thresh: float = 0.3
    # Weighting: hidden rows get higher weight
    hidden_weight: float = 1.00
    known_weight: float = 0.15


class TrainConfig(BaseModel):
    # CV
    n_folds: int = 5
    cv_group: str = "well_id"  # "well_id" main CV, "typewell_id" stress CV
    seed: int = 42
    # Optimiser
    optimizer: str = "adamw"
    lr: float = 3e-4
    lr_min: float = 3e-4
    weight_decay: float = 0.0
    clip_grad_norm: float = 5.0
    # Schedule
    epochs: int = 20
    warmup_epochs: int = 0
    # Batch
    batch_size: int = 1  # wells per batch
    grad_accum: int = 1
    # EMA
    use_ema: bool = False
    ema_decay: float = 0.999
    ema_start_epoch: int = 5
    # Validation
    validate_every_steps: int = 25
    early_stop_patience: int = 20
    # Hardware
    device: str = "cpu"
    num_workers: int = 0
    mixed_precision: bool = False
    # Curriculum (progressive hidden length exposure)
    curriculum_enabled: bool = False
    curriculum_stages: list[int] = [256, 512, 1024, 2048, -1]  # -1 = full
    curriculum_epochs: list[int] = [5, 10, 15, 20, 999]


class InferConfig(BaseModel):
    batch_size: int = 1
    device: str = "cpu"
    use_ema: bool = True
    # Per-well post-optimiser
    run_postopt: bool = False
    postopt_n_knots: int = 48
    postopt_n_steps: int = 150
    postopt_lambda_prior: float = 1.0
    postopt_lambda_hmm: float = 0.5
    postopt_lambda_smooth: float = 0.02
    postopt_lambda_anchor: float = 50.0
    postopt_clip_correction: float = 20.0
    # Blend weights (NN ensemble + priors)
    blend_nn_opt: float = 0.55
    blend_nn_raw: float = 0.20
    blend_hmm: float = 0.10
    blend_linear: float = 0.05
    blend_neighbor: float = 0.10


class SurfacePriorConfig(BaseModel):
    enabled: bool = False
    k_neighbors: int = 15
    use_rbf: bool = True
    rbf_epsilon: float = 1e-3
    formations: list[str] = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]


class NeighborConfig(BaseModel):
    enabled: bool = False
    k_neighbors: int = 8
    spatial_weight: float = 1.0
    typewell_weight: float = 2.0
    trajectory_weight: float = 0.5


class RunConfig(BaseModel):
    name: str = "bphwt_lite"
    output_dir: str = "artifacts/bphwt_lite"
    log_every: int = 50
    log_first_steps: int = 3


class TrackingConfig(BaseModel):
    enabled: bool = False
    project: str = "ROGII/Wellbore/BPHWT"
    task_name: str = ""
    output_uri: str = ""
    tags: list[str] = ["bphwt"]
    log_artifacts: bool = True
    log_model: bool = True
    fail_on_error: bool = False


class ClearMLDataConfig(BaseModel):
    enabled: bool = False
    project: str = "ROGII/Wellbore"
    name: str = "rogii-wellbore-geology-prediction"
    version: str = "20260519_s3"
    cache_dir: str = "~/.cache/clearml/rogii"
    max_workers: int = 8


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------


class BPHWTConfig(BaseModel):
    data_dir: str = "data"
    k_wells: int = -1  # -1 = all wells; N = first N wells (debug)
    cache_dir: str | None = None  # defaults to {output_dir}/cache

    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    infer: InferConfig = Field(default_factory=InferConfig)
    surface: SurfacePriorConfig = Field(default_factory=SurfacePriorConfig)
    neighbor: NeighborConfig = Field(default_factory=NeighborConfig)

    run: RunConfig = Field(default_factory=RunConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    data_clearml: ClearMLDataConfig = Field(default_factory=ClearMLDataConfig)

    def resolved_cache_dir(self) -> Path:
        if self.cache_dir:
            return Path(self.cache_dir)
        return Path(self.run.output_dir) / "cache"

    def resolved_data_dir(self) -> Path:
        return Path(self.data_dir)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> BPHWTConfig:
    """Load a YAML config file and return a BPHWTConfig."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    return BPHWTConfig(**raw)


def save_config(cfg: BPHWTConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(cfg.model_dump(), f, default_flow_style=False, sort_keys=False)
