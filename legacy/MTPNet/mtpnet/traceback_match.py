from __future__ import annotations

import numpy as np
import pandas as pd


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    finite = np.isfinite(aa) & np.isfinite(bb)
    if finite.sum() < 3:
        return 0.0
    aa = aa[finite] - np.mean(aa[finite])
    bb = bb[finite] - np.mean(bb[finite])
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom <= 1e-9:
        return 0.0
    return float(np.dot(aa, bb) / denom)


def _mad_score(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    finite = np.isfinite(aa) & np.isfinite(bb)
    if not finite.any():
        return -10.0
    return -float(np.nanmedian(np.abs(aa[finite] - bb[finite])))


def score_event_against_dictionary(
    event: pd.Series,
    dictionary: pd.DataFrame,
    *,
    location_weight: float,
    bridge_tvt: float | None = None,
    location_sigma_ft: float = 80.0,
    top_k: int = 10,
    max_candidates: int | None = None,
) -> pd.DataFrame:
    if dictionary.empty:
        return pd.DataFrame()
    radius = int(event["patch_radius"])
    candidates = dictionary[dictionary["patch_radius"].astype(int) == radius].copy()
    if candidates.empty:
        return pd.DataFrame()
    if max_candidates is not None and len(candidates) > max_candidates:
        type_mismatch = (
            candidates["event_type"].astype(str) != str(event["event_type"])
        ).astype(int)
        prom_delta = (
            pd.to_numeric(candidates["prominence"], errors="coerce")
            .sub(float(event["prominence"]))
            .abs()
        )
        candidates = (
            candidates.assign(_type_mismatch=type_mismatch, _prom_delta=prom_delta)
            .sort_values(["_type_mismatch", "_prom_delta"], kind="mergesort")
            .head(int(max_candidates))
        )
    rows = []
    event_gr = np.asarray(event["patch_gr"], dtype=np.float32)
    event_dgr = np.asarray(event["patch_dgr"], dtype=np.float32)
    for _, cand in candidates.iterrows():
        shape_corr = _corr(event_gr, cand["patch_gr"])
        dgr_corr = _corr(event_dgr, cand["patch_dgr"])
        mad = _mad_score(event_gr, cand["patch_gr"])
        event_type_hit = 1.0 if str(event["event_type"]) == str(cand["event_type"]) else 0.0
        prom_score = -abs(float(event["prominence"]) - float(cand["prominence"]))
        finite_weight = min(float(event["finite_frac"]), float(cand["finite_frac"]))
        location_score = 0.0
        if bridge_tvt is not None and np.isfinite(bridge_tvt):
            location_score = -abs(float(cand["source_TVT"]) - float(bridge_tvt)) / max(
                location_sigma_ft, 1.0
            )
        score = finite_weight * (
            shape_corr
            + 0.5 * dgr_corr
            + 0.3 * mad
            + 0.2 * event_type_hit
            + 0.15 * prom_score
            + location_weight * location_score
        )
        out = cand.to_dict()
        out.update(
            {
                "event_well_id": str(event.get("well_id", "")),
                "event_step": int(event.get("step", -1)),
                "score": float(score),
                "shape_corr": float(shape_corr),
                "dgr_corr": float(dgr_corr),
                "mad_score": float(mad),
                "event_type_hit": float(event_type_hit),
                "location_score": float(location_score),
            }
        )
        rows.append(out)
    return (
        pd.DataFrame(rows)
        .sort_values("score", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )


def make_sanity_events(events: pd.DataFrame, *, seed: int) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    normal = events.copy()
    shuffled = events.copy()
    if "patch_radius" in shuffled.columns:
        groups = shuffled.groupby("patch_radius").groups.values()
    else:
        groups = [shuffled.index]
    for idx in groups:
        idx_list = list(idx)
        patches = list(shuffled.loc[idx_list, "patch_gr"])
        dpatches = list(shuffled.loc[idx_list, "patch_dgr"])
        order = rng.permutation(len(idx_list))
        for dst, src in zip(idx_list, order, strict=False):
            shuffled.at[dst, "patch_gr"] = patches[int(src)]
            shuffled.at[dst, "patch_dgr"] = dpatches[int(src)]
    zero = events.copy()
    zero["patch_gr"] = [
        np.zeros_like(np.asarray(p, dtype=np.float32)) for p in zero["patch_gr"]
    ]
    zero["patch_dgr"] = [
        np.zeros_like(np.asarray(p, dtype=np.float32)) for p in zero["patch_dgr"]
    ]
    return {
        "normal_GR": normal,
        "shuffled_hidden_GR": shuffled,
        "zero_hidden_GR": zero,
    }


def evaluate_event_matches(
    matches: pd.DataFrame,
    *,
    tolerance_ft: float = 10.0,
) -> dict[str, float | int]:
    if matches.empty or "true_TVT" not in matches:
        return {
            "events": 0,
            "event_top1_rmse_ft": float("nan"),
            "event_top10_oracle_rmse_ft": float("nan"),
            "event_true_top10_rate_at_10ft": 0.0,
        }
    rows = []
    for _, group in matches.groupby(["event_well_id", "event_step"], sort=False):
        true_tvt = float(group["true_TVT"].iloc[0])
        if not np.isfinite(true_tvt):
            continue
        tvt = pd.to_numeric(group["source_TVT"], errors="coerce").to_numpy(dtype=np.float64)
        sqerr = (tvt - true_tvt) ** 2
        rows.append(
            {
                "top1_sqerr": float(sqerr[0]),
                "top10_sqerr": float(np.nanmin(sqerr)),
                "top10_hit": bool(np.nanmin(np.sqrt(sqerr)) <= tolerance_ft),
            }
        )
    if not rows:
        return {
            "events": 0,
            "event_top1_rmse_ft": float("nan"),
            "event_top10_oracle_rmse_ft": float("nan"),
            "event_true_top10_rate_at_10ft": 0.0,
        }
    frame = pd.DataFrame(rows)
    return {
        "events": int(len(frame)),
        "event_top1_rmse_ft": float(np.sqrt(frame["top1_sqerr"].mean())),
        "event_top10_oracle_rmse_ft": float(np.sqrt(frame["top10_sqerr"].mean())),
        "event_true_top10_rate_at_10ft": float(frame["top10_hit"].mean()),
    }
