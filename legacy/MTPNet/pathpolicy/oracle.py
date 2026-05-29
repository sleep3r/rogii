"""
pathpolicy/oracle.py — ACTION_ORACLE_REPORT

Greedy per-chunk oracle: at every chunk of hidden rows, pick the action
(from a fixed candidate set) that minimises RMSE vs true TVT.
Then aggregate globally and by tail class.

This is a GATE: if oracle mean-well RMSE ≤ 8.5 ft → action space viable;
proceed to PolicyFormer. Otherwise, expand action set.

Usage:
    uv run --extra dev python -m pathpolicy.oracle \
        --data_dir data/train \
        --output_dir artifacts/oracle_v0

Output (artifacts/oracle_v0/):
    oracle_report.json   — aggregated metrics
    oracle_chunks.parquet — per-chunk results (well_id, chunk, winner, rmses)
    oracle_wells.parquet  — per-well summary
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Import canonical vocab so oracle RMSE columns are always index-aligned with model
try:
    from pathpolicy.actions import ACTION_VOCAB, ACTION_TO_IDX
    _VOCAB_AVAILABLE = True
except ImportError:
    ACTION_VOCAB = []
    ACTION_TO_IDX = {}
    _VOCAB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LEVEL_SHIFTS = [-80.0, -60.0, -40.0, -20.0, 20.0, 40.0, 60.0, 80.0]  # ft
SLOPE_DELTAS = [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0]  # ft per 32-row step
CHUNK_LEN = 256  # rows — 8 steps × 32 rows/step (aligns with MTP stride=4, rps=32)
ROWS_PER_STEP = 32  # for slope estimation

# Coverage threshold: actions with fewer than this fraction of finite-vs-true rows
# are rejected as oracle candidates. Prevents low-coverage actions (e.g. 20/256 rows)
# from winning by RMSE on an unrepresentative subset.
MIN_ORACLE_COVERAGE: float = 0.95


# ---------------------------------------------------------------------------
# Prior loading
# ---------------------------------------------------------------------------

def load_priors(
    base_path: Path,
    base_col: str,
    b2_path: Path,
    b2_col: str,
    a_path: Path,
    a_p50_col: str,
    a_p10_col: str,
    a_p90_col: str,
) -> dict[str, pd.Series]:
    """Return dict name→Series indexed by (id,) string like '000d7d20_1234'."""
    print("[oracle] loading priors...", end=" ", flush=True)
    t0 = time.time()

    base_df = pd.read_parquet(base_path, columns=["id", base_col]).set_index("id")[base_col]
    b2_df = pd.read_parquet(b2_path, columns=["id", b2_col]).set_index("id")[b2_col]
    a_df = pd.read_parquet(
        a_path, columns=["id", a_p50_col, a_p10_col, a_p90_col]
    ).set_index("id")

    priors = {
        "base": base_df,
        "b2": b2_df,
        "a_p50": a_df[a_p50_col],
        "a_p10": a_df[a_p10_col],
        "a_p90": a_df[a_p90_col],
    }
    print(f"done ({time.time()-t0:.1f}s)")
    return priors


def load_mtp_track(track_path: Path) -> pd.DataFrame:
    """Return pivot: index=(well_id, row_idx), cols=candidate names."""
    print("[oracle] loading MTP track...", end=" ", flush=True)
    t0 = time.time()
    tr = pd.read_parquet(track_path, columns=["well_id", "row_idx", "pred_tvt", "candidate"])
    piv = tr.pivot_table(index=["well_id", "row_idx"], columns="candidate", values="pred_tvt")
    piv.columns = [f"mtp_{c}" for c in piv.columns]
    print(f"done ({time.time()-t0:.1f}s) — {piv.shape[1]} MTP candidates, {tr.well_id.nunique()} wells")
    return piv


# ---------------------------------------------------------------------------
# Action generator
# ---------------------------------------------------------------------------

def generate_actions(
    true_tvt: np.ndarray,         # [L] true TVT for chunk (for oracle only; NaN in test)
    prior_segs: dict[str, np.ndarray],  # name → [L] or None
    last_tvt: float,              # last known TVT just before chunk
    last_slope_per_row: float,    # estimated dTVT/row at chunk start
) -> dict[str, np.ndarray]:
    """Return all candidate action paths for this chunk."""
    L = len(true_tvt)
    actions: dict[str, np.ndarray] = {}

    # --- anchor actions ---
    for name, seg in prior_segs.items():
        if seg is not None and len(seg) == L and np.isfinite(seg).any():
            actions[name] = seg.astype(np.float32)

    # --- level shift on B2 / base / A ---
    for anchor in ["b2", "base", "a_p50"]:
        seg = prior_segs.get(anchor)
        if seg is not None and len(seg) == L and np.isfinite(seg).any():
            for sh in LEVEL_SHIFTS:
                actions[f"{anchor}{sh:+.0f}"] = (seg + sh).astype(np.float32)

    # --- slope continuation ---
    steps = np.arange(1, L + 1, dtype=np.float32)  # rows 1..L ahead
    for sd in SLOPE_DELTAS:
        slope_row = last_slope_per_row + sd / ROWS_PER_STEP  # ft/row
        seg = last_tvt + steps * slope_row
        actions[f"slope{sd:+.0f}"] = seg.astype(np.float32)

    return actions


def _rmse(pred: np.ndarray, true: np.ndarray, min_coverage: float = MIN_ORACLE_COVERAGE) -> float:
    """RMSE over rows where both pred and true are finite.

    Returns NaN if coverage (fraction of jointly-finite rows) is below min_coverage.
    This prevents low-coverage actions (e.g. 20/256 rows with finite predictions)
    from winning the oracle due to their RMSE being computed on an unrepresentative subset.
    """
    mask = np.isfinite(pred) & np.isfinite(true)
    coverage = mask.sum() / max(len(pred), 1)
    if coverage < min_coverage:
        return np.nan
    if mask.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((pred[mask] - true[mask]) ** 2)))


# ---------------------------------------------------------------------------
# Per-well oracle
# ---------------------------------------------------------------------------

def oracle_well(
    well_id: str,
    well_df: pd.DataFrame,
    priors: dict[str, pd.Series],
    mtp_piv: pd.DataFrame | None,  # indexed by (well_id, row_idx)
    min_coverage: float = MIN_ORACLE_COVERAGE,
) -> list[dict]:
    """Run greedy per-chunk oracle for one well. Returns list of chunk records."""

    # Hidden rows: TVT_input is NaN
    hidden_mask = well_df["TVT_input"].isna().values
    if hidden_mask.sum() == 0:
        return []

    true_tvt = well_df["TVT"].values          # full well, true TVT
    row_indices = np.arange(len(well_df))     # absolute row indices in well CSV

    known_indices = row_indices[~hidden_mask]
    hidden_indices = row_indices[hidden_mask]

    last_known_idx = int(known_indices[-1])
    last_known_tvt = float(well_df["TVT_input"].iloc[last_known_idx])

    # Estimate slope from last 3 known rows
    if len(known_indices) >= 3:
        k3 = known_indices[-3:]
        slope_per_row = float(
            np.polyfit(np.arange(len(k3)), well_df["TVT_input"].iloc[k3].values, 1)[0]
        )
    else:
        slope_per_row = 0.0

    # Build id strings for prior lookup: "{well_id}_{row_idx}"
    def get_prior_seg(prior_series: pd.Series, ridxs: np.ndarray) -> np.ndarray | None:
        ids = [f"{well_id}_{r}" for r in ridxs]
        vals = prior_series.reindex(ids).values.astype(np.float32)
        if not np.isfinite(vals).any():
            return None
        return vals

    # MTP segments for this well (if available)
    well_mtp: pd.DataFrame | None = None
    if mtp_piv is not None and well_id in mtp_piv.index.get_level_values("well_id"):
        well_mtp = mtp_piv.xs(well_id, level="well_id")  # index=row_idx

    # Pre-load full B2 prior for this well (for slope anchoring between chunks)
    all_hidden_ids = [f"{well_id}_{r}" for r in hidden_indices]
    b2_full = priors["b2"].reindex(all_hidden_ids).values.astype(np.float32)

    chunk_records = []
    n_hidden = len(hidden_indices)

    for c_start in range(0, n_hidden, CHUNK_LEN):
        c_end = min(c_start + CHUNK_LEN, n_hidden)
        chunk_rows = hidden_indices[c_start:c_end]
        L = len(chunk_rows)
        true_chunk = true_tvt[chunk_rows].astype(np.float32)

        if not np.isfinite(true_chunk).any():
            continue

        # Slope anchor: use B2's endpoint at the row just before this chunk.
        # This anchors slope actions to the B2 prior's trajectory (not the true path).
        # For the first chunk, fall back to last_known_tvt / slope from known section.
        if c_start > 0:
            # Look at B2 values in the previous chunk
            prev_b2 = b2_full[max(0, c_start - CHUNK_LEN): c_start]
            valid_prev_b2 = prev_b2[np.isfinite(prev_b2)]
            if len(valid_prev_b2) >= 2:
                prev_tvt = float(valid_prev_b2[-1])
                slope_per_row = float(
                    np.polyfit(np.arange(len(valid_prev_b2)), valid_prev_b2, 1)[0]
                )
            elif len(valid_prev_b2) == 1:
                prev_tvt = float(valid_prev_b2[-1])
                # keep slope_per_row from previous iteration
            # else: keep from previous iteration
        else:
            prev_tvt = last_known_tvt
            # slope_per_row already estimated from known section

        # prior segments for this chunk
        prior_segs: dict[str, np.ndarray | None] = {}
        for name, s in priors.items():
            prior_segs[name] = get_prior_seg(s, chunk_rows)

        # MTP segments for this chunk
        if mtp_piv is not None and well_id in mtp_piv.index.get_level_values("well_id"):
            for col in well_mtp.columns:
                seg = well_mtp.reindex(chunk_rows.astype(int))[col].values.astype(np.float32)
                if np.isfinite(seg).any():
                    prior_segs[col] = seg

        actions = generate_actions(
            true_tvt=true_chunk,
            prior_segs=prior_segs,
            last_tvt=prev_tvt,
            last_slope_per_row=slope_per_row,
        )

        if not actions:
            continue

        # oracle: argmin RMSE (with coverage filter)
        rmses = {name: _rmse(seg, true_chunk, min_coverage) for name, seg in actions.items()}
        valid = {k: v for k, v in rmses.items() if np.isfinite(v)}
        if not valid:
            continue

        oracle_name = min(valid, key=valid.get)
        oracle_r = valid[oracle_name]
        b2_r = rmses.get("b2", np.nan)

        # Record the oracle winner's raw coverage (for diagnostics)
        oracle_seg = actions[oracle_name]
        coverage_mask = np.isfinite(oracle_seg) & np.isfinite(true_chunk)
        oracle_winner_coverage = float(coverage_mask.sum()) / max(len(true_chunk), 1)

        rec = {
            "well_id": well_id,
            "chunk_start_hidden": c_start,
            "chunk_len": L,
            "oracle_action": oracle_name,
            "oracle_rmse": oracle_r,
            "oracle_winner_coverage": oracle_winner_coverage,
            "b2_rmse": b2_r,
            "base_rmse": rmses.get("base", np.nan),
            "a_p50_rmse": rmses.get("a_p50", np.nan),
            "mtp_top1_rmse": rmses.get("mtp_mtp_track_top1", np.nan),
            "mtp_weighted_rmse": rmses.get("mtp_mtp_track_weighted", np.nan),
            "n_actions": len(valid),
        }

        # Per-action RMSE for all canonical vocab actions (index-aligned with model).
        # Stored as rmse_0, rmse_1, ..., rmse_{N-1} using ACTION_TO_IDX ordering.
        # NaN means the action was not available for this chunk.
        if _VOCAB_AVAILABLE:
            for aname, aidx in ACTION_TO_IDX.items():
                rec[f"rmse_{aidx}"] = rmses.get(aname, np.nan)

        chunk_records.append(rec)

    return chunk_records


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _well_rmse(chunks: list[dict], well_id: str) -> float:
    """Compute well-level RMSE from chunk records (weighted by chunk_len)."""
    rows = [c for c in chunks if c["well_id"] == well_id]
    if not rows:
        return np.nan
    total_sq_err = sum(c["oracle_rmse"] ** 2 * c["chunk_len"] for c in rows if np.isfinite(c["oracle_rmse"]))
    total_len = sum(c["chunk_len"] for c in rows if np.isfinite(c["oracle_rmse"]))
    if total_len == 0:
        return np.nan
    return float(np.sqrt(total_sq_err / total_len))


def _well_rmse_for(chunks: list[dict], well_id: str, field: str = "b2_rmse") -> float:
    rows = [c for c in chunks if c["well_id"] == well_id and np.isfinite(c.get(field, np.nan))]
    if not rows:
        return np.nan
    total_sq = sum(c[field] ** 2 * c["chunk_len"] for c in rows)
    total_len = sum(c["chunk_len"] for c in rows)
    if total_len == 0:
        return np.nan
    return float(np.sqrt(total_sq / total_len))


def _action_group(name: str) -> str:
    """Map action name to a human-readable group."""
    if name.startswith("mtp_"):
        return "mtp"
    if "+" in name or name in ("b2", "base", "a_p50", "a_p10", "a_p90"):
        for prefix in ("b2", "base", "a_p50", "a_p10", "a_p90"):
            if name == prefix:
                return prefix
            if name.startswith(prefix + "+") or name.startswith(prefix + "-"):
                return f"{prefix}_shift"
    if name.startswith("slope"):
        return "slope"
    return "other"


# ---------------------------------------------------------------------------
# Main report builder
# ---------------------------------------------------------------------------

def build_report(
    chunks: list[dict],
    tail_audit: pd.DataFrame | None,
    well_ids: list[str],
) -> dict:
    chunk_df = pd.DataFrame(chunks)
    if chunk_df.empty:
        return {"error": "no chunks processed"}

    # per-well summaries
    well_records = []
    for wid in well_ids:
        oracle_r = _well_rmse(chunks, wid)
        b2_r = _well_rmse_for(chunks, wid, "b2_rmse")
        base_r = _well_rmse_for(chunks, wid, "base_rmse")
        well_records.append(
            {
                "well_id": wid,
                "oracle_mwr": oracle_r,
                "b2_mwr": b2_r,
                "base_mwr": base_r,
                "gain_vs_b2": (b2_r - oracle_r) if np.isfinite(oracle_r) and np.isfinite(b2_r) else np.nan,
            }
        )
    well_df = pd.DataFrame(well_records)

    if tail_audit is not None:
        well_df = well_df.merge(tail_audit[["well_id", "tail_class"]], on="well_id", how="left")
        well_df["tail_class"] = well_df["tail_class"].fillna("unknown")
    else:
        well_df["tail_class"] = "unknown"

    # global metrics
    valid_oracle = well_df["oracle_mwr"].dropna()
    valid_b2 = well_df["b2_mwr"].dropna()

    def tail_class_stats(df: pd.DataFrame) -> dict:
        out = {}
        for cls in df["tail_class"].unique():
            sub = df[df["tail_class"] == cls]
            out[cls] = {
                "n_wells": int(len(sub)),
                "oracle_mwr": float(sub["oracle_mwr"].mean()),
                "b2_mwr": float(sub["b2_mwr"].mean()),
                "gain_vs_b2": float(sub["gain_vs_b2"].mean()),
            }
        return out

    # winner distribution by action group
    chunk_df["action_group"] = chunk_df["oracle_action"].apply(_action_group)
    group_counts = chunk_df["action_group"].value_counts(normalize=True).to_dict()

    # row-level global oracle RMSE (weighted by chunk_len)
    total_sq = sum(
        c["oracle_rmse"] ** 2 * c["chunk_len"]
        for c in chunks
        if np.isfinite(c["oracle_rmse"])
    )
    total_b2_sq = sum(
        c["b2_rmse"] ** 2 * c["chunk_len"]
        for c in chunks
        if np.isfinite(c["b2_rmse"])
    )
    total_len = sum(c["chunk_len"] for c in chunks if np.isfinite(c["oracle_rmse"]))
    total_b2_len = sum(c["chunk_len"] for c in chunks if np.isfinite(c["b2_rmse"]))

    row_oracle_rmse = float(np.sqrt(total_sq / total_len)) if total_len > 0 else np.nan
    row_b2_rmse = float(np.sqrt(total_b2_sq / total_b2_len)) if total_b2_len > 0 else np.nan

    # % chunks where oracle < B2
    frac_oracle_beats_b2 = float(
        (chunk_df["oracle_rmse"] < chunk_df["b2_rmse"]).mean()
    )
    frac_large_shift = float(
        chunk_df["action_group"].isin(["b2_shift", "base_shift", "a_p50_shift"]).mean()
    )
    frac_slope = float((chunk_df["action_group"] == "slope").mean())
    frac_mtp = float((chunk_df["action_group"] == "mtp").mean())

    # Oracle winner coverage distribution
    win_cov = chunk_df["oracle_winner_coverage"]
    frac_full_coverage = float((win_cov >= 0.95).mean()) if "oracle_winner_coverage" in chunk_df.columns else float("nan")
    mean_winner_coverage = float(win_cov.mean()) if "oracle_winner_coverage" in chunk_df.columns else float("nan")

    report = {
        "global": {
            "n_wells": int(len(well_ids)),
            "n_chunks": int(len(chunks)),
            "row_oracle_rmse_ft": row_oracle_rmse,
            "row_b2_rmse_ft": row_b2_rmse,
            "mean_well_oracle_rmse_ft": float(valid_oracle.mean()),
            "mean_well_b2_rmse_ft": float(valid_b2.mean()),
            "p50_oracle_mwr": float(valid_oracle.quantile(0.5)),
            "p90_oracle_mwr": float(valid_oracle.quantile(0.9)),
            "p95_oracle_mwr": float(valid_oracle.quantile(0.95)),
            "worst_oracle_mwr": float(valid_oracle.max()),
            "mean_gain_vs_b2_ft": float(well_df["gain_vs_b2"].mean()),
        },
        "tail_class": tail_class_stats(well_df),
        "winner_distribution": {
            "by_group": {k: round(v, 4) for k, v in group_counts.items()},
            "top_actions": chunk_df["oracle_action"].value_counts(normalize=True).head(20).to_dict(),
        },
        "coverage": {
            "frac_chunks_oracle_beats_b2": frac_oracle_beats_b2,
            "frac_chunks_large_shift_wins": frac_large_shift,
            "frac_chunks_slope_wins": frac_slope,
            "frac_chunks_mtp_wins": frac_mtp,
            "frac_oracle_winner_full_coverage": frac_full_coverage,
            "mean_oracle_winner_coverage": mean_winner_coverage,
        },
        "gate": {
            "oracle_mwr_ft": float(valid_oracle.mean()),
            "b2_mwr_ft": float(valid_b2.mean()),
            "gain_vs_b2_ft": float(well_df["gain_vs_b2"].mean()),
            "decision": (
                "GO — proceed to PolicyFormer"
                if valid_oracle.mean() <= 8.5
                else "EXPAND — action space too narrow, add more actions"
            ),
        },
    }
    return report, well_df, chunk_df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Compute ACTION_ORACLE_REPORT")
    parser.add_argument("--data_dir", default="data/train")
    parser.add_argument("--output_dir", default="artifacts/oracle_v0")
    parser.add_argument(
        "--base_path",
        default="../old/artifacts/oof_baseline/schema10_oof.parquet",
    )
    parser.add_argument("--base_col", default="schema10_oof_pp")
    parser.add_argument(
        "--b2_path",
        default="../old/artifacts/formation_b2_danger_guard_a2_full_schema10/guarded_predictions.parquet",
    )
    parser.add_argument("--b2_col", default="b2_guarded_submit")
    parser.add_argument(
        "--a_path",
        default="../old/artifacts/formation_plane_knn/oof_candidates.parquet",
    )
    parser.add_argument("--a_p50_col", default="formation_sample_median")
    parser.add_argument("--a_p10_col", default="formation_sample_p10")
    parser.add_argument("--a_p90_col", default="formation_sample_p90")
    parser.add_argument(
        "--mtp_track",
        default="artifacts/mtp_v3_gr_forced/track_row_predictions.parquet",
    )
    parser.add_argument(
        "--tail_audit",
        default="artifacts/tail_audit_v1/well_tail_audit.csv",
    )
    parser.add_argument("--k_wells", type=int, default=-1, help="-1 = all")
    parser.add_argument(
        "--min_coverage",
        type=float,
        default=MIN_ORACLE_COVERAGE,
        help="Minimum fraction of jointly-finite rows for an action to be oracle-eligible "
             f"(default: {MIN_ORACLE_COVERAGE}). Set to 0.0 to disable (permissive oracle).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[oracle] min_coverage={args.min_coverage:.2f} "
          f"({'strict — partial-coverage actions rejected' if args.min_coverage > 0 else 'permissive — all actions eligible'})")

    # --- Load priors ---
    priors = load_priors(
        base_path=Path(args.base_path),
        base_col=args.base_col,
        b2_path=Path(args.b2_path),
        b2_col=args.b2_col,
        a_path=Path(args.a_path),
        a_p50_col=args.a_p50_col,
        a_p10_col=args.a_p10_col,
        a_p90_col=args.a_p90_col,
    )

    # --- Load MTP track ---
    mtp_piv: pd.DataFrame | None = None
    mtp_track_path = Path(args.mtp_track)
    if mtp_track_path.exists():
        mtp_piv = load_mtp_track(mtp_track_path)
    else:
        print(f"[oracle] WARNING: MTP track not found at {mtp_track_path}, skipping")

    # --- Load tail audit ---
    tail_audit: pd.DataFrame | None = None
    ta_path = Path(args.tail_audit)
    if ta_path.exists():
        tail_audit = pd.read_csv(ta_path)
        print(f"[oracle] tail audit: {len(tail_audit)} wells, classes: {sorted(tail_audit.tail_class.unique())}")

    # --- Discover wells ---
    data_dir = Path(args.data_dir)
    well_paths = sorted(data_dir.glob("*__horizontal_well.csv"))
    if args.k_wells > 0:
        well_paths = well_paths[: args.k_wells]
    print(f"[oracle] {len(well_paths)} wells to process")

    # --- Run oracle ---
    all_chunks: list[dict] = []
    all_well_ids: list[str] = []
    t0 = time.time()

    for i, wp in enumerate(well_paths):
        well_id = wp.stem.replace("__horizontal_well", "")
        well_df = pd.read_csv(wp)
        # add row_idx = original CSV row number
        well_df["_row_idx"] = np.arange(len(well_df))

        chunks = oracle_well(well_id, well_df, priors, mtp_piv, min_coverage=args.min_coverage)
        all_chunks.extend(chunks)
        all_well_ids.append(well_id)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(
                f"  [{i+1}/{len(well_paths)}] {elapsed:.0f}s elapsed, "
                f"{len(all_chunks)} chunks so far"
            )

    print(f"[oracle] done: {len(all_well_ids)} wells, {len(all_chunks)} chunks in {time.time()-t0:.1f}s")

    # --- Build report ---
    report, well_df, chunk_df = build_report(all_chunks, tail_audit, all_well_ids)
    report["config"] = {"min_coverage": args.min_coverage}

    # --- Save ---
    report_path = output_dir / "oracle_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    well_df.to_parquet(output_dir / "oracle_wells.parquet", index=False)
    chunk_df.to_parquet(output_dir / "oracle_chunks.parquet", index=False)

    # --- Print summary ---
    print("\n" + "=" * 60)
    print("ACTION_ORACLE_REPORT")
    print("=" * 60)
    g = report["global"]
    print(f"\nB2 baseline:   row_rmse={g['row_b2_rmse_ft']:.3f} ft   mwr={g['mean_well_b2_rmse_ft']:.3f} ft")
    print(f"Oracle:        row_rmse={g['row_oracle_rmse_ft']:.3f} ft   mwr={g['mean_well_oracle_rmse_ft']:.3f} ft")
    print(f"Gain vs B2:    {g['mean_gain_vs_b2_ft']:+.3f} ft")
    print(f"p50/p90/p95/worst: {g['p50_oracle_mwr']:.2f} / {g['p90_oracle_mwr']:.2f} / {g['p95_oracle_mwr']:.2f} / {g['worst_oracle_mwr']:.2f} ft")

    print("\nTail-class oracle mwr vs B2:")
    for cls, stats in sorted(report["tail_class"].items()):
        print(
            f"  {cls:<35} n={stats['n_wells']:3d}  "
            f"oracle={stats['oracle_mwr']:.2f}  b2={stats['b2_mwr']:.2f}  "
            f"gain={stats['gain_vs_b2']:+.2f}"
        )

    print("\nWinner distribution (by action group):")
    for grp, frac in sorted(report["winner_distribution"]["by_group"].items(), key=lambda x: -x[1]):
        print(f"  {grp:<25} {frac*100:.1f}%")

    print("\nCoverage:")
    cov = report["coverage"]
    print(f"  oracle beats B2:    {cov['frac_chunks_oracle_beats_b2']*100:.1f}% of chunks")
    print(f"  large shift wins:   {cov['frac_chunks_large_shift_wins']*100:.1f}%")
    print(f"  slope wins:         {cov['frac_chunks_slope_wins']*100:.1f}%")
    print(f"  MTP wins:           {cov['frac_chunks_mtp_wins']*100:.1f}%")

    gate = report["gate"]
    print(f"\nGATE: {gate['decision']}")
    print(f"  oracle mwr={gate['oracle_mwr_ft']:.3f} ft  (threshold ≤ 8.5 ft)")
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
