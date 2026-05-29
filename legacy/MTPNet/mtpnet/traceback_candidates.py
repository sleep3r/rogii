from __future__ import annotations

import numpy as np
import pandas as pd


def matches_to_anchors(
    matches: pd.DataFrame,
    *,
    min_score_quantile: float = 0.75,
    min_top1_gap: float = 0.0,
    max_bridge_delta_ft: float | None = None,
) -> pd.DataFrame:
    if matches.empty:
        return pd.DataFrame(
            columns=["well_id", "step", "anchor_tvt", "confidence", "score_gap"]
        )
    ordered = matches.sort_values(
        ["event_well_id", "event_step", "score"],
        ascending=[True, True, False],
        kind="mergesort",
    ).copy()
    ordered["_rank"] = ordered.groupby(["event_well_id", "event_step"]).cumcount()
    top = ordered[ordered["_rank"] == 0].copy()
    second = ordered[ordered["_rank"] == 1][
        ["event_well_id", "event_step", "score"]
    ].rename(columns={"score": "_second_score"})
    top = top.merge(second, on=["event_well_id", "event_step"], how="left")
    top["_second_score"] = top["_second_score"].fillna(-np.inf)
    top["score_gap"] = (
        pd.to_numeric(top["score"], errors="coerce")
        - pd.to_numeric(top["_second_score"], errors="coerce")
    )
    top = top[top["score_gap"] >= float(min_top1_gap)].copy()
    if (
        max_bridge_delta_ft is not None
        and "event_bridge_tvt" in top.columns
        and max_bridge_delta_ft >= 0
    ):
        bridge = pd.to_numeric(top["event_bridge_tvt"], errors="coerce")
        source = pd.to_numeric(top["source_TVT"], errors="coerce")
        close = (source - bridge).abs() <= float(max_bridge_delta_ft)
        unknown_bridge = ~np.isfinite(bridge)
        top = top[close | unknown_bridge].copy()
    threshold = float(top["score"].quantile(min_score_quantile)) if len(top) else float("inf")
    top = top[top["score"] >= threshold].copy()
    if top.empty:
        return pd.DataFrame(
            columns=["well_id", "step", "anchor_tvt", "confidence", "score_gap"]
        )
    score = top["score"].to_numpy(dtype=np.float64)
    denom = max(float(np.nanmax(score) - np.nanmin(score)), 1e-6)
    conf = (score - float(np.nanmin(score))) / denom
    return pd.DataFrame(
        {
            "well_id": top["event_well_id"].astype(str).to_numpy(),
            "step": top["event_step"].astype(int).to_numpy(),
            "anchor_tvt": pd.to_numeric(top["source_TVT"], errors="coerce").to_numpy(
                dtype=np.float64
            ),
            "confidence": np.clip(conf, 0.05, 1.0),
            "score_gap": pd.to_numeric(top["score_gap"], errors="coerce").to_numpy(
                dtype=np.float64
            ),
        }
    )


def build_traceback_bands(
    comp: pd.DataFrame,
    anchors: pd.DataFrame,
    *,
    widths_ft: tuple[float, ...] = (40.0, 80.0, 120.0),
) -> pd.DataFrame:
    if anchors.empty or "well_id" not in anchors.columns:
        return pd.DataFrame(
            columns=["well_id", "step", "band_center_tvt", "band_width_ft", "candidate"]
        )
    rows = []
    for well_id, well in comp.groupby("well_id", sort=False):
        well = well.sort_values("step")
        well_anchors = anchors[anchors["well_id"].astype(str) == str(well_id)].sort_values(
            "step"
        )
        steps = well["step"].to_numpy(dtype=np.float64)
        if well_anchors.empty:
            continue
        a_steps = well_anchors["step"].to_numpy(dtype=np.float64)
        a_tvt = well_anchors["anchor_tvt"].to_numpy(dtype=np.float64)
        center = np.interp(steps, a_steps, a_tvt, left=a_tvt[0], right=a_tvt[-1])
        for width in widths_ft:
            for step, c in zip(well["step"].to_numpy(dtype=int), center, strict=False):
                rows.append(
                    {
                        "well_id": str(well_id),
                        "step": int(step),
                        "band_center_tvt": float(c) if np.isfinite(c) else np.nan,
                        "band_width_ft": float(width),
                        "candidate": f"traceback_band_w{int(width)}",
                    }
                )
    return pd.DataFrame(rows)


def build_traceback_candidates(hidden_rows: pd.DataFrame, bands: pd.DataFrame) -> pd.DataFrame:
    return build_traceback_candidates_with_offsets(hidden_rows, bands)


def build_traceback_candidates_with_offsets(
    hidden_rows: pd.DataFrame,
    bands: pd.DataFrame,
    *,
    offset_fracs: tuple[float, ...] = (0.0,),
) -> pd.DataFrame:
    if hidden_rows.empty or bands.empty:
        return pd.DataFrame(
            columns=["id", "well_id", "row_idx", "step", "candidate", "pred_tvt"]
        )
    hidden = hidden_rows.copy()
    if "step" not in hidden:
        hidden["step"] = hidden["row_idx"] // 32
    keep = ["id", "well_id", "row_idx", "step"]
    out_parts = []
    for candidate, band in bands.groupby("candidate", sort=False):
        merged_base = hidden[keep].merge(
            band[["well_id", "step", "band_center_tvt", "band_width_ft"]],
            on=["well_id", "step"],
            how="left",
        )
        merged_base = merged_base[
            pd.to_numeric(merged_base["band_center_tvt"], errors="coerce").notna()
        ].copy()
        if merged_base.empty:
            continue
        for frac in offset_fracs:
            merged = merged_base.copy()
            offset = pd.to_numeric(merged["band_width_ft"], errors="coerce") * float(frac)
            suffix = "" if frac == 0.0 else f"_off{float(frac):+.2f}".replace(".", "p")
            merged["candidate"] = f"{candidate}{suffix}"
            merged["pred_tvt"] = (
                pd.to_numeric(merged["band_center_tvt"], errors="coerce") + offset
            )
            out_parts.append(
                merged[["id", "well_id", "row_idx", "step", "candidate", "pred_tvt"]]
            )
    if not out_parts:
        return pd.DataFrame(
            columns=["id", "well_id", "row_idx", "step", "candidate", "pred_tvt"]
        )
    return pd.concat(out_parts, ignore_index=True)


def _rmse(pred: np.ndarray, target: np.ndarray) -> float:
    finite = np.isfinite(pred) & np.isfinite(target)
    if not finite.any():
        return float("nan")
    return float(np.sqrt(np.mean((pred[finite] - target[finite]) ** 2)))


def evaluate_traceback_candidate_oracle(
    hidden_rows: pd.DataFrame,
    traceback_candidates: pd.DataFrame,
) -> dict[str, float | int]:
    if hidden_rows.empty:
        return {
            "rows": 0,
            "b2_row_rmse": float("nan"),
            "traceback_oracle_row_rmse": float("nan"),
            "oracle_gain_ft": 0.0,
        }
    truth = hidden_rows[["id", "TVT", "b2_tvt"]].copy()
    b2_rmse = _rmse(
        truth["b2_tvt"].to_numpy(dtype=np.float64),
        truth["TVT"].to_numpy(dtype=np.float64),
    )
    bridge_rmse = (
        _rmse(
            pd.to_numeric(hidden_rows["bridge_TVT"], errors="coerce").to_numpy(
                dtype=np.float64
            ),
            truth["TVT"].to_numpy(dtype=np.float64),
        )
        if "bridge_TVT" in hidden_rows.columns
        else float("nan")
    )
    if traceback_candidates.empty:
        return {
            "rows": int(len(hidden_rows)),
            "covered_rows": 0,
            "coverage_frac": 0.0,
            "b2_row_rmse": b2_rmse,
            "bridge_row_rmse": bridge_rmse,
            "traceback_oracle_row_rmse": float("nan"),
            "oracle_gain_ft": 0.0,
        }
    merged = traceback_candidates.merge(truth[["id", "TVT"]], on="id", how="inner")
    covered_rows = int(merged["id"].nunique())
    coverage_frac = float(covered_rows / max(len(hidden_rows), 1))
    merged["sqerr"] = (
        pd.to_numeric(merged["pred_tvt"], errors="coerce")
        - pd.to_numeric(merged["TVT"], errors="coerce")
    ) ** 2
    best = merged.groupby("id", as_index=False)["sqerr"].min()
    oracle_rmse = float(np.sqrt(best["sqerr"].mean())) if not best.empty else float("nan")
    truth_sqerr = (
        pd.to_numeric(truth["b2_tvt"], errors="coerce")
        - pd.to_numeric(truth["TVT"], errors="coerce")
    ) ** 2
    b2_by_id = truth.assign(b2_sqerr=truth_sqerr)[["id", "b2_sqerr"]]
    bank = best.merge(b2_by_id, on="id", how="right")
    bank["sqerr"] = np.minimum(
        pd.to_numeric(bank["sqerr"], errors="coerce").fillna(np.inf),
        pd.to_numeric(bank["b2_sqerr"], errors="coerce").fillna(np.inf),
    )
    bank = bank[np.isfinite(bank["sqerr"])]
    bank_oracle_rmse = (
        float(np.sqrt(bank["sqerr"].mean())) if not bank.empty else float("nan")
    )
    return {
        "rows": int(len(hidden_rows)),
        "covered_rows": covered_rows,
        "coverage_frac": coverage_frac,
        "candidate_rows": int(len(traceback_candidates)),
        "b2_row_rmse": b2_rmse,
        "bridge_row_rmse": bridge_rmse,
        "traceback_oracle_row_rmse": oracle_rmse,
        "oracle_gain_ft": float(b2_rmse - oracle_rmse) if np.isfinite(oracle_rmse) else 0.0,
        "b2_plus_traceback_oracle_row_rmse": bank_oracle_rmse,
        "b2_plus_traceback_oracle_gain_ft": (
            float(b2_rmse - bank_oracle_rmse) if np.isfinite(bank_oracle_rmse) else 0.0
        ),
    }
