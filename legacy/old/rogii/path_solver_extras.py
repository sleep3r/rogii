"""Direct test-time TVT path solvers used by ``direct_solver``.

After the May-20 simplification this module hosts two solver families plus
the matched-triple harness:

* CEM iterative path search: cross-entropy refit over a low-dimensional
  ``(offset, slope_offset, curvature)`` correction family applied to a
  base path (the geological tailfit, not the submission anchor). Five
  iterations with elite-fraction proposal updates, returns the top-k
  median path plus the best single path.
* Stage1 / Stage2: Stage1 grid-searches a global
  ``tvt = last_tvt + a*(md-md0) + b*(z-z0)`` family, Stage2 applies
  knot-by-knot bounded refinement (max +/-12 ft) with a smoothness
  penalty.

The tie-point / landmark solver that used to live here was removed during
the May-20 simplification: it was a cosmetic family that rarely improved
matched-triple ranks and added 220 lines of code.

A small ``EnergyContext`` carries the per-well inputs an energy function
needs. The scorer in ``direct_solver.score_candidate_path`` is shared, so
both families are ranked on the same loss.

Matched-triple sampling for the 3-well pseudo-public harness lives here
too. Triples are drawn near the public test wells by hidden_len, GR
statistics and MD/Z drift; this is a standardized Euclidean nearest
neighbor search, not uniform-random sampling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EnergyContext:
    """Per-well inputs that any energy function needs.

    ``anchor_path`` here is the *base* path the CEM correction is applied
    on top of, not the submission anchor. After the simplification we
    always pass the geological tailfit so the search is not pinned to a
    known-suboptimal submission anchor.
    """

    md: np.ndarray
    gr: np.ndarray
    z: np.ndarray
    typewell: tuple[np.ndarray, np.ndarray] | None
    hidden_indices: np.ndarray
    last_idx: int
    last_tvt: float
    tail_slope: float
    cal_a: float
    cal_b: float
    linear_path: np.ndarray
    geo_path: np.ndarray
    anchor_path: np.ndarray


EnergyFn = Callable[[np.ndarray], float]


def _finite(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]



def _affine_correction_path(
    *,
    anchor: np.ndarray,
    hidden_idx: np.ndarray,
    offset: float,
    slope_offset: float,
    curvature: float,
) -> np.ndarray:
    path = anchor.copy()
    if len(hidden_idx) == 0:
        return path
    fracs = (
        np.linspace(0.0, 1.0, len(hidden_idx)) if len(hidden_idx) > 1 else np.zeros(1)
    )
    centered = 2.0 * fracs - 1.0
    curve_shape = centered**2 - 0.5
    path[hidden_idx] = anchor[hidden_idx] + offset + slope_offset * centered + curvature * curve_shape
    return path


def cem_path_search(
    ctx: EnergyContext,
    energy_fn: EnergyFn,
    *,
    n_iter: int = 5,
    pop_size: int = 320,
    elite_frac: float = 0.18,
    init_std: tuple[float, float, float] = (10.0, 10.0, 6.0),
    min_std: tuple[float, float, float] = (0.6, 0.6, 0.4),
    max_abs: tuple[float, float, float] = (24.0, 24.0, 16.0),
    seed: int = 17,
    top_k: int = 5,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Cross-entropy search over (offset, slope_offset, curvature), plan item 3.

    The proposal distribution is a diagonal Gaussian over the three parameters.
    Each iteration scores ``pop_size`` candidates, keeps the elite fraction by
    energy, and refits the mean/std. ``min_std`` keeps the proposal from
    collapsing too early; ``max_abs`` clips to plausible per-well shifts.
    """
    rng = np.random.default_rng(int(seed))
    hidden_idx = ctx.hidden_indices
    mean = np.zeros(3, dtype=float)
    std = np.asarray(init_std, dtype=float)
    min_std_arr = np.asarray(min_std, dtype=float)
    max_abs_arr = np.asarray(max_abs, dtype=float)
    best_score = np.inf
    best_params = mean.copy()
    history: list[dict[str, float]] = []
    pool_params: list[np.ndarray] = []
    pool_scores: list[float] = []

    for iteration in range(int(n_iter)):
        samples = rng.normal(loc=mean, scale=std, size=(int(pop_size), 3))
        samples = np.clip(samples, -max_abs_arr, max_abs_arr)
        scores = np.full(int(pop_size), np.inf, dtype=float)
        for k in range(int(pop_size)):
            params = samples[k]
            path = _affine_correction_path(
                anchor=ctx.anchor_path,
                hidden_idx=hidden_idx,
                offset=float(params[0]),
                slope_offset=float(params[1]),
                curvature=float(params[2]),
            )
            scores[k] = float(energy_fn(path))
            pool_params.append(params.copy())
            pool_scores.append(float(scores[k]))
        finite_mask = np.isfinite(scores)
        if not finite_mask.any():
            break
        n_elite = max(4, int(round(float(elite_frac) * pop_size)))
        order = np.argsort(scores)
        elite = samples[order[:n_elite]]
        mean = elite.mean(axis=0)
        std = np.maximum(elite.std(axis=0), min_std_arr)
        iter_best = float(scores[order[0]])
        if iter_best < best_score:
            best_score = iter_best
            best_params = samples[order[0]].copy()
        history.append({
            "iteration": float(iteration),
            "best_score": iter_best,
            "mean_offset": float(mean[0]),
            "mean_slope_offset": float(mean[1]),
            "mean_curvature": float(mean[2]),
            "std_offset": float(std[0]),
            "std_slope_offset": float(std[1]),
            "std_curvature": float(std[2]),
        })

    if not np.isfinite(best_score):
        diag = {
            "cem_best_score": np.nan,
            "cem_iterations": float(len(history)),
            "cem_best_offset": 0.0,
            "cem_best_slope_offset": 0.0,
            "cem_best_curvature": 0.0,
            "cem_history": history,
        }
        return {
            "cem_raw": ctx.anchor_path.copy(),
            "cem_top_median": ctx.anchor_path.copy(),
        }, diag

    best_path = _affine_correction_path(
        anchor=ctx.anchor_path,
        hidden_idx=hidden_idx,
        offset=float(best_params[0]),
        slope_offset=float(best_params[1]),
        curvature=float(best_params[2]),
    )
    pool_params_arr = np.asarray(pool_params, dtype=float)
    pool_scores_arr = np.asarray(pool_scores, dtype=float)
    finite = np.isfinite(pool_scores_arr)
    top_paths: list[np.ndarray] = [best_path]
    if finite.any():
        order_pool = np.argsort(pool_scores_arr[finite])
        valid_params = pool_params_arr[finite][order_pool]
        k = min(int(top_k), len(valid_params))
        for params in valid_params[:k]:
            top_paths.append(
                _affine_correction_path(
                    anchor=ctx.anchor_path,
                    hidden_idx=hidden_idx,
                    offset=float(params[0]),
                    slope_offset=float(params[1]),
                    curvature=float(params[2]),
                )
            )
    stack = np.vstack(top_paths)
    top_median = np.nanmedian(stack, axis=0)

    diag: dict[str, object] = {
        "cem_best_score": float(best_score),
        "cem_iterations": float(len(history)),
        "cem_best_offset": float(best_params[0]),
        "cem_best_slope_offset": float(best_params[1]),
        "cem_best_curvature": float(best_params[2]),
        "cem_history": history,
    }
    return {
        "cem_raw": best_path,
        "cem_top_median": top_median,
    }, diag


def stage1_global_linear(
    ctx: EnergyContext,
    energy_fn: EnergyFn,
    *,
    a_grid: np.ndarray | None = None,
    b_grid: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Stage1: choose ``tvt = last_tvt + a*(md-md0) + b*(z-z0)``.

    Grid search over ``(a, b)`` pairs. Returns the best path on all rows; only
    hidden rows are used for scoring.
    """
    if a_grid is None:
        a_grid = np.concatenate(
            [
                np.linspace(-0.30, -0.04, 14),
                np.linspace(-0.03, 0.03, 13),
                np.linspace(0.04, 0.30, 14),
            ]
        )
    if b_grid is None:
        b_grid = np.linspace(-3.0, 3.0, 13)

    md = ctx.md
    z = ctx.z
    last_idx = int(ctx.last_idx)
    last_tvt = float(ctx.last_tvt)
    md0 = float(md[last_idx])
    z0 = float(z[last_idx])
    dmd = md - md0
    dz = z - z0
    best_score = np.inf
    best_path = ctx.anchor_path.copy()
    best_a = 0.0
    best_b = 0.0
    for a in a_grid:
        for b in b_grid:
            path = last_tvt + float(a) * dmd + float(b) * dz
            path[: last_idx + 1] = last_tvt
            score = float(energy_fn(path))
            if np.isfinite(score) and score < best_score:
                best_score = score
                best_a = float(a)
                best_b = float(b)
                best_path = path
    return best_path, {
        "stage1_best_a": best_a,
        "stage1_best_b": best_b,
        "stage1_best_score": float(best_score) if np.isfinite(best_score) else np.nan,
    }


def _knot_indices(hidden_idx: np.ndarray, n_knots: int) -> np.ndarray:
    if len(hidden_idx) == 0:
        return np.zeros(0, dtype=int)
    n_knots = max(2, min(int(n_knots), max(2, len(hidden_idx) // 4)))
    positions = np.linspace(0.0, 1.0, n_knots)
    selected = (positions * (len(hidden_idx) - 1)).round().astype(int)
    return hidden_idx[np.unique(selected)]


def _apply_knot_offsets(
    path: np.ndarray, knot_positions: np.ndarray, knot_offsets: np.ndarray, md: np.ndarray
) -> np.ndarray:
    if len(knot_positions) == 0:
        return path.copy()
    pad_md = np.concatenate([[float(md[0])], md[knot_positions], [float(md[-1])]])
    pad_offsets = np.concatenate([[float(knot_offsets[0])], knot_offsets, [float(knot_offsets[-1])]])
    interpolated = np.interp(md, pad_md, pad_offsets)
    return path + interpolated


def stage2_local_refine(
    ctx: EnergyContext,
    energy_fn: EnergyFn,
    base_path: np.ndarray,
    *,
    n_knots: int = 10,
    max_offset: float = 12.0,
    offsets: np.ndarray | None = None,
    passes: int = 2,
    smoothness_weight: float = 0.05,
) -> tuple[np.ndarray, dict[str, float]]:
    """Stage2: iterative knot-by-knot bounded local refinement.

    For each knot we search a small offset grid (default ``±max_offset`` in
    seven steps), apply the change as a smooth linear-tapered bump and accept
    if the energy plus a smoothness penalty on knot offsets improves.
    """
    if offsets is None:
        offsets = np.array(
            [-max_offset, -0.66 * max_offset, -0.33 * max_offset, 0.0,
             0.33 * max_offset, 0.66 * max_offset, max_offset],
            dtype=float,
        )
    knots = _knot_indices(ctx.hidden_indices, n_knots)
    if len(knots) == 0:
        return base_path.copy(), {
            "stage2_knots": 0.0,
            "stage2_passes": 0.0,
            "stage2_score": np.nan,
        }
    current_offsets = np.zeros(len(knots), dtype=float)
    current_path = base_path.copy()
    current_score = float(energy_fn(current_path))
    history_scores: list[float] = [current_score]
    accepted_changes = 0
    for _ in range(int(max(1, passes))):
        for k_idx in range(len(knots)):
            best_local_score = current_score
            best_offset = current_offsets[k_idx]
            for candidate_offset in offsets:
                trial_offsets = current_offsets.copy()
                trial_offsets[k_idx] = float(candidate_offset)
                trial_path = _apply_knot_offsets(base_path, knots, trial_offsets, ctx.md)
                penalty = smoothness_weight * float(np.sum(np.diff(trial_offsets) ** 2)) / max(len(trial_offsets), 1)
                score = float(energy_fn(trial_path)) + penalty
                if np.isfinite(score) and score < best_local_score:
                    best_local_score = score
                    best_offset = float(candidate_offset)
            if best_offset != current_offsets[k_idx]:
                current_offsets[k_idx] = best_offset
                current_path = _apply_knot_offsets(base_path, knots, current_offsets, ctx.md)
                current_score = best_local_score
                accepted_changes += 1
        history_scores.append(current_score)
    return current_path, {
        "stage2_knots": float(len(knots)),
        "stage2_passes": float(int(passes)),
        "stage2_accepted": float(accepted_changes),
        "stage2_score": float(current_score) if np.isfinite(current_score) else np.nan,
        "stage2_max_offset_used": float(np.nanmax(np.abs(current_offsets))) if len(current_offsets) else 0.0,
    }


def stage12_path(
    ctx: EnergyContext,
    energy_fn: EnergyFn,
    *,
    n_knots: int = 10,
    max_offset: float = 12.0,
    passes: int = 2,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Run Stage1 followed by Stage2 and return both paths."""
    stage1, diag1 = stage1_global_linear(ctx, energy_fn)
    stage2, diag2 = stage2_local_refine(
        ctx, energy_fn, stage1, n_knots=n_knots, max_offset=max_offset, passes=passes
    )
    diag: dict[str, float] = {**diag1, **diag2}
    return {"stage1_path": stage1, "stage12_path": stage2}, diag


# ---------------------------------------------------------------------------
# Matched triples for pseudo-public harness
# ---------------------------------------------------------------------------


WELL_SIGNATURE_COLUMNS: tuple[str, ...] = (
    "hidden_len",
    "tail_len",
    "log_md_span",
    "log_gr_mean",
    "log_gr_std",
    "z_drift",
    "last_tvt",
)


@dataclass
class WellSignature:
    well: str
    hidden_len: int
    tail_len: int
    log_md_span: float
    log_gr_mean: float
    log_gr_std: float
    z_drift: float
    last_tvt: float
    rows: int = 0
    extras: dict[str, float] = field(default_factory=dict)


def _safe_log(value: float, floor: float = 1e-3) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(np.log(max(float(value), floor)))


def compute_well_signature(
    well: str, frame: pd.DataFrame, hidden_indices: np.ndarray
) -> WellSignature | None:
    """Extract per-well signature features the matched-triple sampler uses."""
    if len(hidden_indices) == 0 or len(frame) == 0:
        return None
    md = pd.to_numeric(frame.get("MD", pd.Series(np.zeros(len(frame)))), errors="coerce").to_numpy(float)
    z = pd.to_numeric(frame.get("Z", pd.Series(np.zeros(len(frame)))), errors="coerce").to_numpy(float)
    gr = pd.to_numeric(frame.get("GR", pd.Series(np.zeros(len(frame)))), errors="coerce").to_numpy(float)
    tvt_input = pd.to_numeric(frame.get("TVT_input", pd.Series(np.full(len(frame), np.nan))), errors="coerce").to_numpy(float)
    hidden_idx = np.asarray(hidden_indices, dtype=int)
    hidden_idx = hidden_idx[(hidden_idx >= 0) & (hidden_idx < len(frame))]
    if len(hidden_idx) == 0:
        return None
    finite_tvt = np.flatnonzero(np.isfinite(tvt_input))
    tail_len = int(finite_tvt[-1] - finite_tvt[0] + 1) if len(finite_tvt) else 0
    last_tvt = float(tvt_input[finite_tvt[-1]]) if len(finite_tvt) else np.nan
    md_span = float(np.nanmax(md[hidden_idx]) - np.nanmin(md[hidden_idx])) if np.isfinite(md[hidden_idx]).any() else 0.0
    gr_hidden = gr[hidden_idx]
    z_hidden = z[hidden_idx]
    gr_mean = float(np.nanmean(gr_hidden)) if np.isfinite(gr_hidden).any() else 0.0
    gr_std = float(np.nanstd(gr_hidden)) if np.isfinite(gr_hidden).any() else 0.0
    z_drift = (
        float(np.nanmax(z_hidden) - np.nanmin(z_hidden))
        if np.isfinite(z_hidden).any()
        else 0.0
    )
    return WellSignature(
        well=well,
        hidden_len=int(len(hidden_idx)),
        tail_len=tail_len,
        log_md_span=_safe_log(md_span),
        log_gr_mean=_safe_log(abs(gr_mean) + 1.0),
        log_gr_std=_safe_log(gr_std + 1.0),
        z_drift=float(z_drift),
        last_tvt=float(last_tvt) if np.isfinite(last_tvt) else 0.0,
        rows=int(len(frame)),
    )


def signatures_to_frame(signatures: list[WellSignature]) -> pd.DataFrame:
    rows = [
        {
            "well": sig.well,
            "hidden_len": sig.hidden_len,
            "tail_len": sig.tail_len,
            "log_md_span": sig.log_md_span,
            "log_gr_mean": sig.log_gr_mean,
            "log_gr_std": sig.log_gr_std,
            "z_drift": sig.z_drift,
            "last_tvt": sig.last_tvt,
        }
        for sig in signatures
    ]
    return pd.DataFrame(rows)


def _standardize_block(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    safe_std = np.where(std > 1e-6, std, 1.0)
    return (values - mean) / safe_std


def _signature_matrix(frame: pd.DataFrame) -> np.ndarray:
    return frame[list(WELL_SIGNATURE_COLUMNS)].to_numpy(float)


def matched_triples(
    train_signatures: pd.DataFrame,
    test_signatures: pd.DataFrame,
    *,
    trials: int,
    triple_size: int = 3,
    seed: int = 17,
    candidate_k: int = 80,
) -> list[list[str]]:
    """Sample train triples matched to each test well by signature distance.

    For each trial we sample one matched train well per test well from a
    nearest-``candidate_k`` candidate pool. This biases the harness toward
    triples that look like the actual public test wells (hidden length, GR
    statistics, MD/Z drift, last known TVT) instead of uniform-random triples.
    """
    if train_signatures.empty or test_signatures.empty or trials <= 0:
        return []
    matrix_train = _signature_matrix(train_signatures)
    matrix_test = _signature_matrix(test_signatures)
    mean = matrix_train.mean(axis=0)
    std = matrix_train.std(axis=0)
    train_std = _standardize_block(matrix_train, mean, std)
    test_std = _standardize_block(matrix_test, mean, std)
    distances = np.linalg.norm(
        train_std[:, None, :] - test_std[None, :, :], axis=-1
    )
    train_names = train_signatures["well"].astype(str).to_numpy()
    rng = np.random.default_rng(int(seed))
    n_test = matrix_test.shape[0]
    triple_size = min(int(triple_size), int(n_test))
    candidate_k = max(min(int(candidate_k), len(train_names)), triple_size)
    triples: list[list[str]] = []
    for _ in range(int(trials)):
        if triple_size <= n_test:
            picked_test_idx = rng.choice(n_test, size=triple_size, replace=False)
        else:
            picked_test_idx = rng.choice(n_test, size=triple_size, replace=True)
        chosen: list[str] = []
        used: set[str] = set()
        for t_idx in picked_test_idx:
            ordering = np.argsort(distances[:, int(t_idx)])
            pool = [
                train_names[int(i)] for i in ordering[: candidate_k]
                if train_names[int(i)] not in used
            ]
            if not pool:
                pool = [
                    train_names[int(i)] for i in ordering
                    if train_names[int(i)] not in used
                ]
            if not pool:
                continue
            pick = str(rng.choice(np.asarray(pool)))
            chosen.append(pick)
            used.add(pick)
        if chosen:
            triples.append(chosen)
    return triples


def collect_well_signatures(
    data_dir: Path,
    *,
    subset: str,
    well_rows: dict[str, list[int]] | None = None,
) -> pd.DataFrame:
    """Compute well signatures for ``train`` or ``test`` wells."""
    if subset not in {"train", "test"}:
        raise ValueError("subset must be 'train' or 'test'")
    base = data_dir / subset
    if not base.is_dir():
        return pd.DataFrame()
    rows: list[WellSignature] = []
    for path in sorted(base.glob("*__horizontal_well.csv")):
        well = path.name.replace("__horizontal_well.csv", "")
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if well_rows is not None:
            hidden = np.asarray(well_rows.get(well, []), dtype=int)
        elif subset == "train":
            tvt_input = pd.to_numeric(
                frame.get("TVT_input", pd.Series(np.nan, index=frame.index)),
                errors="coerce",
            )
            tvt = pd.to_numeric(
                frame.get("TVT", pd.Series(np.nan, index=frame.index)), errors="coerce"
            )
            hidden = np.flatnonzero(tvt_input.isna().to_numpy() & tvt.notna().to_numpy())
        else:
            tvt_input = pd.to_numeric(
                frame.get("TVT_input", pd.Series(np.nan, index=frame.index)),
                errors="coerce",
            )
            hidden = np.flatnonzero(tvt_input.isna().to_numpy())
        if len(hidden) == 0:
            continue
        sig = compute_well_signature(well, frame, hidden)
        if sig is not None:
            rows.append(sig)
    return signatures_to_frame(rows)
