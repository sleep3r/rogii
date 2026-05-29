from __future__ import annotations

import numpy as np
import pandas as pd


def build_template_path(coord_tvt: np.ndarray, *, scale: float, offset: float) -> np.ndarray:
    """Build a centered scaled TVT path.

    Centering avoids absolute TVT blow-up when scale is near but not equal to 1.0.
    """
    coord = np.asarray(coord_tvt, dtype=np.float64)
    finite = np.isfinite(coord)
    if not finite.any():
        return np.full_like(coord, np.nan, dtype=np.float64)
    center = float(np.nanmean(coord[finite]))
    return center + float(scale) * (coord - center) + float(offset)


def sample_typewell_gr(typewell: pd.DataFrame, path_tvt: np.ndarray) -> np.ndarray:
    tvt = pd.to_numeric(typewell["TVT"], errors="coerce").to_numpy(dtype=np.float64)
    gr = pd.to_numeric(typewell["GR"], errors="coerce").to_numpy(dtype=np.float64)
    finite = np.isfinite(tvt) & np.isfinite(gr)
    if finite.sum() < 2:
        return np.full_like(np.asarray(path_tvt, dtype=np.float64), np.nan, dtype=np.float64)
    order = np.argsort(tvt[finite])
    x = tvt[finite][order]
    y = gr[finite][order]
    path = np.asarray(path_tvt, dtype=np.float64)
    sampled = np.interp(path, x, y, left=np.nan, right=np.nan)
    return sampled.astype(np.float64)

