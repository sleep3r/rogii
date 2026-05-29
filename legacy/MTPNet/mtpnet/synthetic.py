from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .config import MTPConfig
from .heatmap import build_channels
from .windows import WindowSample, _target_bins


@dataclass(frozen=True)
class SyntheticTemplate:
    crop_tvt: np.ndarray
    typewell_gr: np.ndarray
    target_bins: np.ndarray
    history_bins: np.ndarray
    well_id: str
    start_step: int
    center_tvt: float


def sample_typewell_gr(typewell_gr: np.ndarray, path_bins: np.ndarray) -> np.ndarray:
    typewell = np.asarray(typewell_gr, dtype=np.float32)
    path = np.asarray(path_bins, dtype=np.float32)
    grid = np.arange(typewell.shape[0], dtype=np.float32)
    return np.interp(path, grid, typewell).astype(np.float32)


def stretch_gr_sequence(values: np.ndarray, factor: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size <= 1 or abs(float(factor) - 1.0) < 1e-8:
        return arr.copy()
    source = np.arange(arr.size, dtype=np.float32)
    center = (arr.size - 1) / 2.0
    query = center + (source - center) / max(float(factor), 1e-6)
    return np.interp(query, source, arr).astype(np.float32)


def interpolate_dropped_gr(values: np.ndarray, finite: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).copy()
    mask = np.asarray(finite, dtype=bool)
    if mask.all() or not mask.any():
        return arr
    steps = np.arange(arr.size, dtype=np.float32)
    arr[~mask] = np.interp(steps[~mask], steps[mask], arr[mask]).astype(np.float32)
    return arr


def _rng(cfg: MTPConfig, index: int) -> np.random.Generator:
    return np.random.default_rng(int(cfg.synthetic.seed) + int(index))


def _choose_range(rng: np.random.Generator, bounds: tuple[float, ...]) -> float:
    if len(bounds) == 0:
        return 0.0
    if len(bounds) == 1:
        return float(bounds[0])
    return float(rng.uniform(float(bounds[0]), float(bounds[1])))


def _clip_path(path: np.ndarray, height: int) -> np.ndarray:
    return np.clip(np.asarray(path, dtype=np.float32), 0.0, float(height - 1)).astype(
        np.float32
    )


def _target_tvt(crop_tvt: np.ndarray, bins: np.ndarray) -> np.ndarray:
    grid = np.arange(len(crop_tvt), dtype=np.float32)
    return np.interp(np.asarray(bins, dtype=np.float32), grid, crop_tvt).astype(np.float32)


def _bin_size_ft(cfg: MTPConfig) -> float:
    return (2.0 * float(cfg.window.vertical_radius_ft)) / max(
        int(cfg.window.vertical_bins) - 1, 1
    )


def _generate_path_family(
    template_path: np.ndarray,
    cfg: MTPConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    height = int(cfg.window.vertical_bins)
    width = len(template_path)
    family = str(rng.choice(np.asarray(cfg.synthetic.path_families, dtype=object)))
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)
    base = np.asarray(template_path, dtype=np.float32)
    if family == "real_residual":
        path = base.copy()
    elif family == "linear":
        slope = float((base[-1] - base[0]) / max(width - 1, 1))
        slope += float(rng.normal(0.0, max(0.15, abs(slope) * 0.35)))
        start = float(base[0] + rng.normal(0.0, 1.0))
        path = start + slope * np.arange(width, dtype=np.float32)
    elif family == "curved":
        start = float(base[0] + rng.normal(0.0, 1.0))
        end = float(base[-1] + rng.normal(0.0, 1.0))
        curvature = float(rng.normal(0.0, 4.0))
        path = start + (end - start) * x + curvature * (x - 0.5) ** 2
    elif family == "piecewise":
        knot_count = int(rng.integers(3, 6))
        knot_x = np.linspace(0.0, 1.0, knot_count, dtype=np.float32)
        knot_base = np.interp(knot_x, x, base)
        knot_y = knot_base + rng.normal(0.0, 2.0, size=knot_count).astype(np.float32)
        path = np.interp(x, knot_x, knot_y).astype(np.float32)
    else:
        raise ValueError(f"Unsupported synthetic path family: {family}")
    return _clip_path(path, height)


def _prior_paths(
    full_path: np.ndarray, cfg: MTPConfig, rng: np.random.Generator, prior_kind: str
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    height = cfg.window.vertical_bins
    if prior_kind == "no_prior":
        nan_path = np.full_like(full_path, np.nan, dtype=np.float32)
        return (
            {
                "anchor": nan_path,
                "base": nan_path,
                "b2": nan_path,
                "a_p50": nan_path,
                "a_p10": nan_path,
                "a_p90": nan_path,
            },
            {
                "anchor_offset": np.zeros_like(full_path, dtype=np.float32),
                "base_offset": np.zeros_like(full_path, dtype=np.float32),
                "b2_delta": np.zeros_like(full_path, dtype=np.float32),
            },
        )
    if prior_kind == "bad":
        bad_shift_bins = rng.uniform(40.0, 100.0) / max(_bin_size_ft(cfg), 1e-6)
        bias = float(rng.choice(np.asarray([-bad_shift_bins, bad_shift_bins])))
        noise_scale = 1.25
    elif prior_kind == "medium":
        bias = float(rng.normal(0.0, 1.0))
        noise_scale = 0.75
    else:
        bias = float(rng.normal(0.0, 0.25))
        noise_scale = 0.25
    smooth_noise = np.cumsum(rng.normal(0.0, noise_scale, size=len(full_path))).astype(
        np.float32
    )
    smooth_noise -= float(smooth_noise.mean())
    anchor = _clip_path(full_path + bias + smooth_noise, height)
    b2 = _clip_path(anchor + rng.normal(0.0, 0.35, size=len(full_path)), height)
    a_p50 = _clip_path(anchor + rng.normal(0.0, 0.75, size=len(full_path)), height)
    a_p10 = _clip_path(a_p50 - 1.0, height)
    a_p90 = _clip_path(a_p50 + 1.0, height)
    return (
        {
            "anchor": anchor,
            "base": anchor,
            "b2": b2,
            "a_p50": a_p50,
            "a_p10": a_p10,
            "a_p90": a_p90,
        },
        {
            "anchor_offset": (anchor - full_path) / max(1.0, float(height)),
            "base_offset": (anchor - full_path) / max(1.0, float(height)),
            "b2_delta": (b2 - anchor) / max(1.0, float(height)),
        },
    )


def _choose_prior_kind(cfg: MTPConfig, rng: np.random.Generator) -> str:
    draw = float(rng.random())
    if draw < cfg.synthetic.no_prior_prob:
        return "no_prior"
    if draw < cfg.synthetic.no_prior_prob + cfg.synthetic.bad_prior_prob:
        return "bad"
    if draw < cfg.synthetic.no_prior_prob + cfg.synthetic.bad_prior_prob + 0.30:
        return "medium"
    return "good"


def generate_synthetic_sample(
    template: SyntheticTemplate,
    cfg: MTPConfig,
    *,
    index: int,
    prior_kind: str | None = None,
    horizontal_flip: bool = False,
) -> WindowSample:
    rng = _rng(cfg, index)
    history_steps = cfg.window.history_steps
    future_steps = cfg.window.future_steps
    width = history_steps + future_steps
    height = cfg.window.vertical_bins
    history = _clip_path(template.history_bins, height)
    target = _clip_path(template.target_bins, height)
    full_path = _generate_path_family(
        np.concatenate([history, target]).astype(np.float32),
        cfg,
        rng,
    )
    history = full_path[:history_steps]
    target = full_path[history_steps:]
    path_gr = sample_typewell_gr(template.typewell_gr, full_path)
    stretch = _choose_range(rng, cfg.synthetic.stretch_range)
    horizontal_gr = stretch_gr_sequence(path_gr, stretch)
    horizontal_gr *= _choose_range(rng, cfg.synthetic.amplitude_scale_range)
    horizontal_gr += _choose_range(rng, cfg.synthetic.baseline_shift_range)
    noise_std = _choose_range(rng, cfg.synthetic.noise_std_range)
    if noise_std > 0.0:
        horizontal_gr += rng.normal(0.0, noise_std, size=width).astype(np.float32)
    finite = np.ones(width, dtype=np.float32)
    if cfg.synthetic.dropout_max_frac > 0.0:
        dropout_count = int(round(width * rng.uniform(0.0, cfg.synthetic.dropout_max_frac)))
        if dropout_count > 0:
            dropped = rng.choice(np.arange(width), size=dropout_count, replace=False)
            finite[dropped] = 0.0
            horizontal_gr = interpolate_dropped_gr(horizontal_gr, finite)
    prior_rng = np.random.default_rng(int(cfg.synthetic.seed) + 10_000_000 + int(index))
    prior_kind = prior_kind or _choose_prior_kind(cfg, prior_rng)
    prior_bins, prior_values = _prior_paths(full_path, cfg, prior_rng, prior_kind)
    if horizontal_flip:
        full_path = full_path[::-1].copy()
        history = full_path[:history_steps]
        target = full_path[history_steps:]
        horizontal_gr = horizontal_gr[::-1].copy()
        finite = finite[::-1].copy()
        prior_bins = {key: value[::-1].copy() for key, value in prior_bins.items()}
        prior_values = {key: value[::-1].copy() for key, value in prior_values.items()}
    history_bins = np.full(width, np.nan, dtype=np.float32)
    history_bins[:history_steps] = history
    x = build_channels(
        horizontal_gr=horizontal_gr,
        typewell_gr=template.typewell_gr,
        history_bins=history_bins,
        finite_steps=finite,
        channels=cfg.window.channels,
        prior_bins=prior_bins,
        prior_values=prior_values,
    )
    if prior_kind == "no_prior":
        prior_names = {
            "anchor_sdf",
            "base_sdf",
            "b2_sdf",
            "a_p50_sdf",
            "a_density",
            "a_p10_p90_band",
            "anchor_offset_value",
            "base_offset_value",
            "b2_delta_value",
        }
        for channel_index, channel_name in enumerate(cfg.window.channels):
            if channel_name in prior_names:
                x[channel_index] = 0.0
    crop_tvt = np.asarray(template.crop_tvt, dtype=np.float32)
    return WindowSample(
        x=x,
        target_bins=target.astype(np.float32),
        target_tvt=_target_tvt(crop_tvt, target),
        history_tvt=_target_tvt(crop_tvt, history),
        crop_tvt=crop_tvt,
        well_id=f"synthetic_{template.well_id}_{index}",
        start_step=int(template.start_step),
        center_tvt=float(template.center_tvt),
        sample_type=f"synthetic:{prior_kind}",
        horizontal_gr=horizontal_gr.astype(np.float32),
        typewell_gr=np.asarray(template.typewell_gr, dtype=np.float32),
    )


def flip_synthetic_sample(sample: WindowSample) -> WindowSample:
    horizontal_gr = None if sample.horizontal_gr is None else sample.horizontal_gr[::-1].copy()
    x = sample.x[:, :, ::-1].copy()
    return replace(
        sample,
        x=x,
        target_bins=sample.target_bins[::-1].copy(),
        target_tvt=sample.target_tvt[::-1].copy(),
        history_tvt=sample.history_tvt[::-1].copy(),
        horizontal_gr=horizontal_gr,
    )


def templates_from_samples(samples: list[WindowSample]) -> list[SyntheticTemplate]:
    templates: list[SyntheticTemplate] = []
    for sample in samples:
        if sample.typewell_gr is None:
            continue
        history_bins = _target_bins(sample.history_tvt, sample.crop_tvt)
        templates.append(
            SyntheticTemplate(
                crop_tvt=sample.crop_tvt.astype(np.float32),
                typewell_gr=sample.typewell_gr.astype(np.float32),
                target_bins=sample.target_bins.astype(np.float32),
                history_bins=history_bins.astype(np.float32),
                well_id=sample.well_id,
                start_step=sample.start_step,
                center_tvt=sample.center_tvt,
            )
        )
    return templates


def generate_synthetic_samples(
    templates: list[SyntheticTemplate],
    cfg: MTPConfig,
    *,
    count: int,
    seed_offset: int = 0,
) -> list[WindowSample]:
    if not templates or count <= 0:
        return []
    rng = np.random.default_rng(int(cfg.synthetic.seed) + int(seed_offset))
    out: list[WindowSample] = []
    for index in range(count):
        template = templates[int(rng.integers(0, len(templates)))]
        out.append(
            generate_synthetic_sample(
                template,
                cfg,
                index=index + seed_offset,
                horizontal_flip=bool(rng.random() < 0.5),
            )
        )
    return out
