"""
pathpolicy/costgbm.py — PP-0.5: CostGBM diagnostic.

Builds a tabular (chunk × action) dataset from oracle_chunks.parquet and
trains LightGBM to predict per-action regret.  Greedy MWR is then computed
by argmin(predicted_regret) per chunk.

Two modes (controlled by --privileged flag):

  DEFAULT (deployable):
    Chunk-level features are restricted to what is knowable at test time:
      b2_available, base_available, a_p50_available  — bool: prior exists for chunk
      n_valid_actions                                 — count of available actions
      chunk_len                                       — 256 or partial (last chunk)
      chunk_idx                                       — ordinal position in well (0-based)

  --privileged (diagnostic upper bound only — NOT deployable):
    Adds oracle-derived chunk features that require true TVT (not available at test):
      b2_rmse, base_rmse, a_p50_rmse
      b2_vs_base, b2_vs_a50, log_b2_rmse
    Useful as an upper-bound reference; do NOT conclude deployable signal exists from
    privileged results.

Action-level features (same in both modes):
  action_idx, is_slope, is_anchor, is_shift, is_mtp,
  shift_amount, anchor_b2, anchor_base, anchor_a, slope_delta

Target:
  regret = clip(rmse_i - min_chunk_rmse, 0, cap=30)  [ft]

Trivial baselines evaluated:
  always-b2:       pick action whose name is "b2"
  always-slope+0:  pick "slope+0"
  always-lowest-mean: pick the action with lowest mean train regret
  gbm:             argmin(pred_regret) per chunk (greedy)

Usage:
    # Deployable mode (default):
    uv run --extra dev python -m pathpolicy.costgbm \\
        --oracle_chunks artifacts/oracle_v1/oracle_chunks.parquet \\
        --output_dir    artifacts/costgbm_v1

    # Privileged diagnostic (uses true-TVT-derived features — upper bound only):
    uv run --extra dev python -m pathpolicy.costgbm \\
        --oracle_chunks artifacts/oracle_v1/oracle_chunks.parquet \\
        --output_dir    artifacts/costgbm_v1_privileged \\
        --privileged
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from pathpolicy.actions import ACTION_VOCAB, ACTION_TO_IDX, N_ACTION_TYPES


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def _parse_action_features(vocab: list[str]) -> pd.DataFrame:
    """Build a DataFrame of action-level features for all 44 canonical actions."""
    rows = []
    for idx, name in enumerate(vocab):
        feat: dict = {
            "action_idx":   idx,
            "action_name":  name,
            "is_slope":     0,
            "is_anchor":    0,
            "is_shift":     0,
            "is_mtp":       0,
            "shift_amount": 0.0,
            "anchor_b2":    0,
            "anchor_base":  0,
            "anchor_a":     0,
            "slope_delta":  0.0,
        }

        if name.startswith("slope"):
            feat["is_slope"] = 1
            m = re.search(r"slope([+-]\d+(?:\.\d+)?)", name)
            feat["slope_delta"] = float(m.group(1)) if m else 0.0

        elif name.startswith("mtp_"):
            feat["is_mtp"] = 1

        else:
            # Check for level shift: "b2+20", "base-40", "a_p50+60", etc.
            m_shift = re.match(
                r"^(b2|base|a_p\d+)([+-]\d+(?:\.\d+)?)$", name
            )
            if m_shift:
                feat["is_shift"] = 1
                anchor = m_shift.group(1)
                feat["shift_amount"] = float(m_shift.group(2))
            else:
                # raw anchor
                feat["is_anchor"] = 1
                anchor = name  # e.g. "b2", "base", "a_p50"

            if anchor == "b2" or anchor.startswith("b2"):
                feat["anchor_b2"] = 1
            elif anchor == "base" or anchor.startswith("base"):
                feat["anchor_base"] = 1
            else:
                feat["anchor_a"] = 1

        rows.append(feat)
    return pd.DataFrame(rows)


def _well_rmse_from_chunks(
    chunk_df: pd.DataFrame,
    rmse_col: str,
) -> pd.Series:
    """Compute per-well mean-well-RMSE weighted by chunk_len from a chunk-level column."""
    valid = chunk_df[["well_id", "chunk_len", rmse_col]].dropna(subset=[rmse_col])
    grp = valid.groupby("well_id")
    sq_sum = grp.apply(lambda g: (g[rmse_col] ** 2 * g["chunk_len"]).sum())
    len_sum = grp["chunk_len"].sum()
    return np.sqrt(sq_sum / len_sum.replace(0, np.nan))


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_costgbm_dataset(
    oracle_chunks: Path,
    privileged: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], int]:
    """Return (long_df, chunk_df, feature_cols, n_chunk_feats).

    long_df has one row per (chunk, action) with columns [feature_cols] + ["regret", "well_id"].
    Only rows where the action was available (non-NaN rmse) are kept.

    n_chunk_feats is the number of chunk-level features (first n_chunk_feats entries of
    feature_cols).  Use this to correctly split chunk-level vs action-level features when
    building prediction rows in run().

    privileged=True adds oracle-derived features that require true TVT (not deployable at
    test time).  Use as a diagnostic upper bound only.
    """
    print(f"[costgbm] loading {oracle_chunks}", flush=True)
    chunk_df = pd.read_parquet(oracle_chunks)
    print(f"[costgbm] {len(chunk_df):,} chunks, {chunk_df['well_id'].nunique()} wells")
    if privileged:
        print("[costgbm] WARNING: --privileged mode — features include true-TVT-derived "
              "RMSEs (b2_rmse, base_rmse, a_p50_rmse). Results are an upper bound; "
              "do NOT conclude deployable signal exists from privileged GBM.", flush=True)

    # ── Action-level feature matrix ─────────────────────────────────────────
    act_feat = _parse_action_features(ACTION_VOCAB)

    # ── Chunk-level features ─────────────────────────────────────────────────
    rmse_cols = [f"rmse_{i}" for i in range(N_ACTION_TYPES)]
    rmse_matrix = chunk_df[rmse_cols].values.astype(np.float32)  # [C, K]

    # Minimum RMSE per chunk (oracle best)
    min_rmse = np.nanmin(rmse_matrix, axis=1)   # [C]

    # Count valid actions per chunk
    n_valid = (~np.isnan(rmse_matrix)).sum(axis=1).astype(np.float32)

    # Deployable availability flags: True if the named prior exists for this chunk
    # (computed from rmse matrix — if rmse_i is finite, the prior was available).
    b2_idx    = ACTION_TO_IDX.get("b2",    0)
    base_idx  = ACTION_TO_IDX.get("base",  1)
    a_p50_idx = ACTION_TO_IDX.get("a_p50", 2)

    chunk_df = chunk_df.copy()
    chunk_df["min_rmse"]        = min_rmse
    chunk_df["n_valid_actions"] = n_valid
    chunk_df["b2_available"]    = (~np.isnan(rmse_matrix[:, b2_idx])).astype(np.float32)
    chunk_df["base_available"]  = (~np.isnan(rmse_matrix[:, base_idx])).astype(np.float32)
    chunk_df["a_p50_available"] = (~np.isnan(rmse_matrix[:, a_p50_idx])).astype(np.float32)

    # chunk_idx = ordinal within well (0 = first chunk)
    chunk_df["chunk_idx"] = (
        chunk_df.groupby("well_id")["chunk_start_hidden"]
        .rank(method="first")
        .astype(int) - 1
    )

    # Deployable chunk features
    chunk_feat_cols = [
        "b2_available", "base_available", "a_p50_available",
        "n_valid_actions", "chunk_len", "chunk_idx",
    ]

    if privileged:
        # Add oracle-derived features (require true TVT — not deployable)
        chunk_df["log_b2_rmse"] = np.log1p(chunk_df["b2_rmse"].fillna(0))
        chunk_df["b2_vs_base"]  = chunk_df["b2_rmse"] - chunk_df["base_rmse"]
        chunk_df["b2_vs_a50"]   = chunk_df["b2_rmse"] - chunk_df["a_p50_rmse"]
        chunk_feat_cols = [
            "b2_rmse", "base_rmse", "a_p50_rmse",
            "log_b2_rmse", "b2_vs_base", "b2_vs_a50",
        ] + chunk_feat_cols

    n_chunk_feats = len(chunk_feat_cols)

    # ── Melt to long format ──────────────────────────────────────────────────
    print("[costgbm] melting to long format...", flush=True)

    long_rows: list[pd.DataFrame] = []
    chunk_meta = chunk_df[["well_id", "chunk_start_hidden", "min_rmse"] + chunk_feat_cols].copy()

    for aidx in range(N_ACTION_TYPES):
        col = f"rmse_{aidx}"
        sub = chunk_meta.copy()
        sub["action_idx"] = aidx
        sub["action_rmse"] = chunk_df[col].values
        sub["regret"] = (sub["action_rmse"] - sub["min_rmse"]).clip(lower=0, upper=30)
        # only keep rows where action was available
        sub = sub.dropna(subset=["action_rmse"])
        long_rows.append(sub)

    long_df = pd.concat(long_rows, ignore_index=True)
    # Merge action-level features
    long_df = long_df.merge(act_feat.drop(columns=["action_name"]), on="action_idx", how="left")

    act_feat_cols = [
        "action_idx", "is_slope", "is_anchor", "is_shift", "is_mtp",
        "shift_amount", "anchor_b2", "anchor_base", "anchor_a", "slope_delta",
    ]
    feature_cols = chunk_feat_cols + act_feat_cols

    print(f"[costgbm] long_df: {len(long_df):,} rows, {len(feature_cols)} features "
          f"({'privileged' if privileged else 'deployable'})")
    return long_df, chunk_df, feature_cols, n_chunk_feats


# ---------------------------------------------------------------------------
# Greedy MWR from predicted regret
# ---------------------------------------------------------------------------

def greedy_mwr(
    chunk_df: pd.DataFrame,
    pred_regret_by_chunk: np.ndarray,   # [C, K] predicted regret; nan for unavailable
) -> float:
    """Compute mean-well RMSE for greedy argmin(pred_regret) policy."""
    rmse_cols = [f"rmse_{i}" for i in range(N_ACTION_TYPES)]
    rmse_mat = chunk_df[rmse_cols].values.astype(np.float32)   # [C, K]

    # argmin(pred), fallback to b2 index if all nan
    b2_idx = ACTION_TO_IDX.get("b2", 0)
    chosen_idxs = np.full(len(chunk_df), b2_idx, dtype=int)
    for c in range(len(chunk_df)):
        row = pred_regret_by_chunk[c]
        valid = ~np.isnan(row)
        if valid.any():
            chosen_idxs[c] = int(np.nanargmin(row))

    chosen_rmse = rmse_mat[np.arange(len(chunk_df)), chosen_idxs]   # [C]

    # Per-well aggregation
    cdf = chunk_df[["well_id", "chunk_len"]].copy()
    cdf["chosen_rmse"] = chosen_rmse
    cdf = cdf.dropna(subset=["chosen_rmse"])

    grp = cdf.groupby("well_id")
    sq_sum  = grp.apply(lambda g: (g["chosen_rmse"] ** 2 * g["chunk_len"]).sum())
    len_sum = grp["chunk_len"].sum()
    well_rmse = np.sqrt(sq_sum / len_sum.replace(0, np.nan))
    return float(well_rmse.dropna().mean())


# ---------------------------------------------------------------------------
# Trivial baselines
# ---------------------------------------------------------------------------

def _trivial_baseline_mwr(chunk_df: pd.DataFrame, action_name: str) -> float:
    """MWR when always choosing a fixed action (by name)."""
    if action_name not in ACTION_TO_IDX:
        return float("nan")
    aidx = ACTION_TO_IDX[action_name]
    rmse_cols = [f"rmse_{i}" for i in range(N_ACTION_TYPES)]
    rmse_mat = chunk_df[rmse_cols].values.astype(np.float32)

    K = rmse_mat.shape[1]
    pred = np.full((len(chunk_df), K), np.nan)
    pred[:, aidx] = 0.0  # lowest possible regret at chosen action
    return greedy_mwr(chunk_df, pred)


def _lowest_mean_regret_action(
    train_df: pd.DataFrame, chunk_df_val: pd.DataFrame
) -> float:
    """MWR when picking the action with lowest mean regret in training split."""
    mean_reg = train_df.groupby("action_idx")["regret"].mean()
    best_idx = int(mean_reg.idxmin())

    rmse_cols = [f"rmse_{i}" for i in range(N_ACTION_TYPES)]
    rmse_mat = chunk_df_val[rmse_cols].values.astype(np.float32)
    K = rmse_mat.shape[1]
    pred = np.full((len(chunk_df_val), K), np.nan)
    pred[:, best_idx] = 0.0
    return greedy_mwr(chunk_df_val, pred)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(oracle_chunks: Path, output_dir: Path, privileged: bool = False) -> None:
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("lightgbm not installed — run: uv add lightgbm --extra dev")

    output_dir.mkdir(parents=True, exist_ok=True)

    long_df, chunk_df, feature_cols, n_chunk_feats = build_costgbm_dataset(
        oracle_chunks, privileged=privileged
    )

    # ── Train / val split by well_id (same 80/20 split as train.py default) ─
    rng = np.random.default_rng(42)
    all_wells = chunk_df["well_id"].unique()
    rng.shuffle(all_wells)
    n_val = max(1, int(0.2 * len(all_wells)))
    val_wells  = set(all_wells[:n_val])
    train_wells = set(all_wells[n_val:])

    train_df = long_df[long_df["well_id"].isin(train_wells)].copy()
    val_df   = long_df[long_df["well_id"].isin(val_wells)].copy()
    chunk_val = chunk_df[chunk_df["well_id"].isin(val_wells)].copy().reset_index(drop=True)

    print(f"[costgbm] train chunks={len(train_df):,}  val chunks={len(val_df):,}")

    # ── LightGBM training ────────────────────────────────────────────────────
    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df["regret"].values.astype(np.float32)
    X_val   = val_df[feature_cols].values.astype(np.float32)
    y_val   = val_df["regret"].values.astype(np.float32)

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    dval   = lgb.Dataset(X_val,   label=y_val,   feature_name=feature_cols, reference=dtrain)

    params = {
        "objective":    "regression_l1",    # MAE more robust to heavy tail
        "metric":       "mae",
        "num_leaves":   63,
        "learning_rate": 0.05,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq":     5,
        "verbose":      -1,
    }

    print("[costgbm] training LightGBM...", flush=True)
    booster = lgb.train(
        params, dtrain,
        num_boost_round=500,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )
    booster.save_model(str(output_dir / "costgbm.lgb"))

    # ── Build predicted-regret matrix for val chunks ─────────────────────────
    rmse_cols_val = [f"rmse_{i}" for i in range(N_ACTION_TYPES)]
    rmse_mat_val  = chunk_val[rmse_cols_val].values.astype(np.float32)   # [C, K]
    pred_regret   = np.full_like(rmse_mat_val, np.nan)

    # Build action feature lookup once (avoid repeated DataFrame creation inside loop)
    act_feat_all = _parse_action_features(ACTION_VOCAB)
    act_feat_cols_only = feature_cols[n_chunk_feats:]   # action-level column names

    for aidx in range(N_ACTION_TYPES):
        # Build feature rows for this action on all val chunks
        af_row = act_feat_all.iloc[aidx].to_dict()
        af = pd.DataFrame([af_row] * len(chunk_val))[act_feat_cols_only].reset_index(drop=True)
        chunk_feats = chunk_val[feature_cols[:n_chunk_feats]].reset_index(drop=True)
        row = pd.concat([chunk_feats, af], axis=1)
        X_pred = row[feature_cols].values.astype(np.float32)
        preds_aidx = booster.predict(X_pred)

        # only fill where action was available
        available = ~np.isnan(rmse_mat_val[:, aidx])
        pred_regret[available, aidx] = preds_aidx[available]

    mwr_gbm = greedy_mwr(chunk_val, pred_regret)

    # ── Trivial baselines on val split ───────────────────────────────────────
    mwr_b2      = _trivial_baseline_mwr(chunk_val, "b2")
    mwr_slope0  = _trivial_baseline_mwr(chunk_val, "slope+0")
    mwr_oracle  = float(
        _well_rmse_from_chunks(chunk_val, "oracle_rmse" if "oracle_rmse" in chunk_val.columns
                               else "min_rmse").dropna().mean()
    )
    mwr_low_mean = _lowest_mean_regret_action(train_df, chunk_val)

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PP-0.5 COSTGBM DIAGNOSTIC REPORT")
    print("=" * 60)
    print(f"Oracle upper bound:         {mwr_oracle:.3f} ft")
    print(f"Trivial always-B2:          {mwr_b2:.3f} ft")
    print(f"Trivial always-slope+0:     {mwr_slope0:.3f} ft")
    print(f"Trivial lowest-mean-regret: {mwr_low_mean:.3f} ft")
    print(f"CostGBM greedy:             {mwr_gbm:.3f} ft")

    best_trivial = min(v for v in [mwr_b2, mwr_slope0, mwr_low_mean] if np.isfinite(v))
    gbm_gain = best_trivial - mwr_gbm
    print(f"\nGBM gain vs best trivial:   {gbm_gain:+.3f} ft")

    if gbm_gain >= 0.05:
        verdict = "SIGNAL EXISTS — contextual features help; proceed to neural PP-1"
    elif gbm_gain >= 0.01:
        verdict = "WEAK SIGNAL — GBM marginally better; neural may still help"
    else:
        verdict = "NO SIGNAL — GBM cannot beat trivial baselines; features insufficient"
    print(f"Verdict: {verdict}")

    # Feature importance
    fi = pd.DataFrame({
        "feature":    feature_cols,
        "importance": booster.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)
    print("\nTop-10 features by gain:")
    print(fi.head(10).to_string(index=False))

    results = {
        "mode":          "privileged" if privileged else "deployable",
        "mwr_oracle":    mwr_oracle,
        "mwr_b2":        mwr_b2,
        "mwr_slope0":    mwr_slope0,
        "mwr_low_mean":  mwr_low_mean,
        "mwr_gbm":       mwr_gbm,
        "gbm_gain_vs_best_trivial": gbm_gain,
        "verdict":       verdict,
        "n_val_wells":   int(len(val_wells)),
        "n_train_wells": int(len(train_wells)),
        "n_features":    len(feature_cols),
    }
    with open(output_dir / "costgbm_report.json", "w") as f:
        json.dump(results, f, indent=2)
    fi.to_csv(output_dir / "feature_importance.csv", index=False)
    print(f"\nReport saved to {output_dir}/costgbm_report.json")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle_chunks", default="artifacts/oracle_v1/oracle_chunks.parquet")
    parser.add_argument("--output_dir",    default="artifacts/costgbm_v1")
    parser.add_argument(
        "--privileged", action="store_true",
        help="Include oracle-derived features (b2_rmse, base_rmse, a_p50_rmse, etc.) that "
             "require true TVT. Results are a privileged upper bound — not deployable. "
             "Default: deployable mode (availability flags + action type only).",
    )
    args = parser.parse_args()
    run(Path(args.oracle_chunks), Path(args.output_dir), privileged=args.privileged)


if __name__ == "__main__":
    main()
