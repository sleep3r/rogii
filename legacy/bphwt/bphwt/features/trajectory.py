"""Trajectory feature extraction: inclination, azimuth, dogleg, curvature."""

from __future__ import annotations

import numpy as np
import pandas as pd

_EPS = 1e-8


def compute_trajectory_features(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """
    Compute trajectory-based features from MD, X, Y, Z columns.

    Returns a dict of 1-D numpy arrays, all length == len(df).
    """
    n = len(df)
    md = df["MD"].to_numpy(dtype=np.float32)
    x = df["X"].to_numpy(dtype=np.float32)
    y = df["Y"].to_numpy(dtype=np.float32)
    z = df["Z"].to_numpy(dtype=np.float32)

    # Differences (central where possible, forward/backward at edges)
    def _diff(v: np.ndarray) -> np.ndarray:
        d = np.empty(n, dtype=np.float32)
        if n == 1:
            d[0] = 0.0
            return d
        d[1:-1] = v[2:] - v[:-2]
        d[0] = v[1] - v[0]
        d[-1] = v[-1] - v[-2]
        return d

    dmd = np.diff(md, prepend=md[0])  # step size (usually ~1 ft)
    dmd = np.where(np.abs(dmd) < _EPS, 1.0, dmd)

    dx = _diff(x)
    dy = _diff(y)
    dz = _diff(z)

    # Unit tangent vector
    norm = np.sqrt(dx**2 + dy**2 + dz**2) + _EPS
    tx = dx / norm
    ty = dy / norm
    tz = dz / norm

    # Inclination proxy: angle from vertical (0 = vertical, 90 = horizontal)
    inclination = np.degrees(np.arccos(np.clip(-tz, -1.0, 1.0)))  # [-tz] because Z is negative down

    # Azimuth proxy: angle from North (Y axis)
    azimuth = np.degrees(np.arctan2(tx, ty)) % 360.0

    # Dogleg (change in tangent direction, degrees per 100 ft)
    dot = np.clip(tx[:-1] * tx[1:] + ty[:-1] * ty[1:] + tz[:-1] * tz[1:], -1.0, 1.0)
    dogleg_angle = np.degrees(np.arccos(dot))  # [n-1]
    dmd_step = np.diff(md)
    dogleg_per_100ft = np.where(dmd_step > _EPS, dogleg_angle / dmd_step * 100.0, 0.0)
    dogleg = np.concatenate([[dogleg_per_100ft[0]], dogleg_per_100ft])

    # Curvature (second derivative of position w.r.t. MD)
    ddx = _diff(dx / dmd)
    ddy = _diff(dy / dmd)
    ddz = _diff(dz / dmd)
    curvature = np.sqrt(ddx**2 + ddy**2 + ddz**2)

    # dX/dMD, dY/dMD, dZ/dMD (unit-normalised tangent components)
    # Already computed as tx, ty, tz

    # Relative position (offset from well head)
    x0, y0, z0 = x[0], y[0], z[0]
    rel_x = x - x0
    rel_y = y - y0
    rel_z = z - z0

    # Cumulative MD (normalised)
    cum_md = md - md[0]

    return {
        "traj_tx": tx,
        "traj_ty": ty,
        "traj_tz": tz,
        "traj_inclination": inclination,
        "traj_azimuth_sin": np.sin(np.radians(azimuth)).astype(np.float32),
        "traj_azimuth_cos": np.cos(np.radians(azimuth)).astype(np.float32),
        "traj_dogleg": dogleg.astype(np.float32),
        "traj_curvature": curvature.astype(np.float32),
        "traj_rel_x": rel_x,
        "traj_rel_y": rel_y,
        "traj_rel_z": rel_z,
        "traj_cum_md": cum_md,
    }
