from __future__ import annotations

import numpy as np


KNOWN_CHANNELS = {
    "gr_diff",
    "gr_z_diff",
    "dgr_diff",
    "abs_gr_diff",
    "history_mask",
    "history_sdf",
    "finite_mask",
    "base_sdf",
    "b2_sdf",
    "a_p50_sdf",
    "a_density",
    "a_p10_p90_band",
    "base_offset_value",
    "b2_delta_value",
}

PRIOR_CHANNELS = {
    "base_sdf",
    "b2_sdf",
    "a_p50_sdf",
    "a_density",
    "a_p10_p90_band",
    "base_offset_value",
    "b2_delta_value",
}


def fill_nan(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(arr)
    if finite.all():
        return arr.astype(np.float32), finite.astype(np.float32)
    if not finite.any():
        return np.zeros_like(arr, dtype=np.float32), finite.astype(np.float32)
    idx = np.arange(len(arr))
    filled = np.interp(idx, idx[finite], arr[finite]).astype(np.float32)
    return filled, finite.astype(np.float32)


def rasterize_path(path_bins: np.ndarray, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.float32)
    path = np.asarray(path_bins, dtype=np.float32)
    for x, y in enumerate(path[:width]):
        if not np.isfinite(y):
            continue
        y0 = int(np.clip(round(float(y)), 0, height - 1))
        mask[y0, x] = 1.0
        if y0 > 0:
            mask[y0 - 1, x] = 0.5
        if y0 + 1 < height:
            mask[y0 + 1, x] = 0.5
    return mask


def path_sdf(path_bins: np.ndarray, height: int, width: int) -> np.ndarray:
    yy = np.arange(height, dtype=np.float32)[:, None]
    out = np.zeros((height, width), dtype=np.float32)
    path = np.asarray(path_bins, dtype=np.float32)
    for x in range(width):
        if x < len(path) and np.isfinite(path[x]):
            out[:, x] = (yy[:, 0] - path[x]) / max(1.0, float(height))
    return out


def band_mask(
    low_bins: np.ndarray, high_bins: np.ndarray, height: int, width: int
) -> np.ndarray:
    yy = np.arange(height, dtype=np.float32)[:, None]
    out = np.zeros((height, width), dtype=np.float32)
    low = np.asarray(low_bins, dtype=np.float32)
    high = np.asarray(high_bins, dtype=np.float32)
    for x in range(width):
        if x >= len(low) or x >= len(high) or not np.isfinite(low[x] + high[x]):
            continue
        lo = min(float(low[x]), float(high[x]))
        hi = max(float(low[x]), float(high[x]))
        out[:, x] = ((yy[:, 0] >= lo) & (yy[:, 0] <= hi)).astype(np.float32)
    return out


def density_from_quantiles(
    p50_bins: np.ndarray,
    p10_bins: np.ndarray,
    p90_bins: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    yy = np.arange(height, dtype=np.float32)[:, None]
    out = np.zeros((height, width), dtype=np.float32)
    p50 = np.asarray(p50_bins, dtype=np.float32)
    p10 = np.asarray(p10_bins, dtype=np.float32)
    p90 = np.asarray(p90_bins, dtype=np.float32)
    for x in range(width):
        if x >= len(p50) or not np.isfinite(p50[x]):
            continue
        if x < len(p10) and x < len(p90) and np.isfinite(p10[x] + p90[x]):
            sigma = max(abs(float(p90[x] - p10[x])) / 2.563, 1.0)
        else:
            sigma = 2.0
        z = (yy[:, 0] - float(p50[x])) / sigma
        out[:, x] = np.exp(-0.5 * z * z).astype(np.float32)
    return out


def broadcast_values(values: np.ndarray, height: int, width: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if len(arr) != width:
        raise ValueError(f"Expected {width} values, got {len(arr)}")
    return np.broadcast_to(arr[None, :], (height, width)).astype(np.float32)


def _robust_z(values: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=np.float32)
    median = float(np.nanmedian(arr[finite]))
    q25, q75 = np.nanpercentile(arr[finite], [25.0, 75.0])
    scale = max(float(q75 - q25), eps)
    return ((arr - median) / scale).astype(np.float32)


def _gradient(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size < 2:
        return np.zeros_like(arr, dtype=np.float32)
    return np.gradient(arr).astype(np.float32)


def build_channels(
    horizontal_gr: np.ndarray,
    typewell_gr: np.ndarray,
    history_bins: np.ndarray,
    finite_steps: np.ndarray,
    channels: tuple[str, ...],
    prior_bins: dict[str, np.ndarray] | None = None,
    prior_values: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    unknown = sorted(set(channels).difference(KNOWN_CHANNELS))
    if unknown:
        raise ValueError(f"Unknown input channels: {unknown}")
    h = np.asarray(horizontal_gr, dtype=np.float32)
    t = np.asarray(typewell_gr, dtype=np.float32)
    heatmap = h[None, :] - t[:, None]
    h_z = _robust_z(h)
    t_z = _robust_z(t)
    z_heatmap = h_z[None, :] - t_z[:, None]
    dgr = _gradient(h_z)[None, :] - _gradient(t_z)[:, None]
    height, width = heatmap.shape
    history_mask = rasterize_path(history_bins, height=height, width=width)
    sdf = path_sdf(history_bins, height=height, width=width)
    finite = np.broadcast_to(
        np.asarray(finite_steps, dtype=np.float32)[None, :], (height, width)
    )
    values = {
        "gr_diff": heatmap / 100.0,
        "gr_z_diff": z_heatmap,
        "dgr_diff": dgr,
        "abs_gr_diff": np.abs(heatmap) / 100.0,
        "history_mask": history_mask,
        "history_sdf": sdf,
        "finite_mask": finite,
    }
    prior_bins = prior_bins or {}
    prior_values = prior_values or {}
    if PRIOR_CHANNELS.intersection(channels):
        missing: list[str] = []
        for key, channel in (
            ("base", "base_sdf"),
            ("b2", "b2_sdf"),
            ("a_p50", "a_p50_sdf"),
            ("a_p50", "a_density"),
            ("a_p10", "a_p10_p90_band"),
            ("a_p90", "a_p10_p90_band"),
        ):
            if channel in channels and key not in prior_bins:
                missing.append(key)
        for key, channel in (
            ("base_offset", "base_offset_value"),
            ("b2_delta", "b2_delta_value"),
        ):
            if channel in channels and key not in prior_values:
                missing.append(key)
        if missing:
            raise ValueError(
                f"Prior channels requested but missing priors: {sorted(set(missing))}"
            )
    if "base_sdf" in channels:
        values["base_sdf"] = path_sdf(prior_bins["base"], height=height, width=width)
    if "b2_sdf" in channels:
        values["b2_sdf"] = path_sdf(prior_bins["b2"], height=height, width=width)
    if "a_p50_sdf" in channels:
        values["a_p50_sdf"] = path_sdf(prior_bins["a_p50"], height=height, width=width)
    if "a_density" in channels:
        values["a_density"] = density_from_quantiles(
            prior_bins["a_p50"],
            prior_bins.get("a_p10", prior_bins["a_p50"]),
            prior_bins.get("a_p90", prior_bins["a_p50"]),
            height=height,
            width=width,
        )
    if "a_p10_p90_band" in channels:
        values["a_p10_p90_band"] = band_mask(
            prior_bins["a_p10"], prior_bins["a_p90"], height=height, width=width
        )
    if "base_offset_value" in channels:
        values["base_offset_value"] = broadcast_values(
            prior_values["base_offset"], height=height, width=width
        )
    if "b2_delta_value" in channels:
        values["b2_delta_value"] = broadcast_values(
            prior_values["b2_delta"], height=height, width=width
        )
    return np.stack([values[name] for name in channels]).astype(np.float32)
