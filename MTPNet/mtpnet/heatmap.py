from __future__ import annotations

import numpy as np


KNOWN_CHANNELS = {"gr_diff", "abs_gr_diff", "history_mask", "history_sdf", "finite_mask"}


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


def build_channels(
    horizontal_gr: np.ndarray,
    typewell_gr: np.ndarray,
    history_bins: np.ndarray,
    finite_steps: np.ndarray,
    channels: tuple[str, ...],
) -> np.ndarray:
    unknown = sorted(set(channels).difference(KNOWN_CHANNELS))
    if unknown:
        raise ValueError(f"Unknown input channels: {unknown}")
    h = np.asarray(horizontal_gr, dtype=np.float32)
    t = np.asarray(typewell_gr, dtype=np.float32)
    heatmap = h[None, :] - t[:, None]
    height, width = heatmap.shape
    history_mask = rasterize_path(history_bins, height=height, width=width)
    sdf = path_sdf(history_bins, height=height, width=width)
    finite = np.broadcast_to(
        np.asarray(finite_steps, dtype=np.float32)[None, :], (height, width)
    )
    values = {
        "gr_diff": heatmap / 100.0,
        "abs_gr_diff": np.abs(heatmap) / 100.0,
        "history_mask": history_mask,
        "history_sdf": sdf,
        "finite_mask": finite,
    }
    return np.stack([values[name] for name in channels]).astype(np.float32)
