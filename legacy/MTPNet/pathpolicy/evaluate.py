"""
pathpolicy/evaluate.py — Greedy PolicyFormer evaluation.

Runs greedy action selection per chunk, assembles full TVT prediction,
and reports mean-well RMSE vs B2 baseline.

Usage:
    uv run --extra dev python -m pathpolicy.evaluate \
        --checkpoint artifacts/pathpolicy_v0/best.pt \
        --output_dir artifacts/pathpolicy_v0
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pathpolicy.actions import (
    CHUNK_LEN,
    N_ACTION_TYPES,
    ACTION_VOCAB,
    generate_action_segments,
)
from pathpolicy.dataset import (
    load_well_csvs,
    load_priors,
    load_mtp_pivot,
    extract_context_features_fast,
    _extract_prior_arrays,
    _extract_mtp_arrays,
    PAST_LEN,
    D_FEAT,
    _DATA_DIR,
    _BASE_PATH,
    _B2_PATH,
    _A_PATH,
    _MTP_PATH,
)
from pathpolicy.model import PolicyFormer


# ---------------------------------------------------------------------------
# Per-well greedy evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_well(
    well_id: str,
    well_df: pd.DataFrame,
    prior_arrs: dict[str, np.ndarray],   # pre-extracted: {pname: [n_rows]}
    mtp_arrs: dict[str, np.ndarray],     # pre-extracted: {col: [n_rows]}
    model: PolicyFormer,
    device: torch.device,
    head_type: str = "classifier",
) -> dict:
    """Return dict with well_id, pred_tvt, b2_tvt, true_tvt, chosen_actions."""
    hidden_mask = well_df["TVT_input"].isna().values
    if hidden_mask.sum() == 0:
        return {}

    row_indices = np.arange(len(well_df))
    known_indices  = row_indices[~hidden_mask]
    hidden_indices = row_indices[hidden_mask]

    tvt_input_arr = well_df["TVT_input"].values.astype(np.float32)
    last_known_idx = int(known_indices[-1])
    last_known_tvt = float(tvt_input_arr[last_known_idx])

    if len(known_indices) >= 3:
        k3 = known_indices[-3:]
        slope_per_row = float(np.polyfit(np.arange(3), tvt_input_arr[k3], 1)[0])
    else:
        slope_per_row = 0.0

    # Collect predictions
    pred_tvt = np.full(len(hidden_indices), np.nan, dtype=np.float32)
    chosen_actions: list[str] = []

    for c_start in range(0, len(hidden_indices), CHUNK_LEN):
        c_end = min(c_start + CHUNK_LEN, len(hidden_indices))
        chunk_rows = hidden_indices[c_start:c_end]
        L = len(chunk_rows)

        # B2-anchored slope for subsequent chunks
        _last_tvt = last_known_tvt
        _slope = slope_per_row
        if c_start > 0:
            b2_arr = prior_arrs.get("b2")
            if b2_arr is not None:
                prev_idxs = hidden_indices[max(0, c_start - CHUNK_LEN): c_start]
                prev_b2 = b2_arr[prev_idxs]
                valid_b2 = prev_b2[np.isfinite(prev_b2)]
                if len(valid_b2) >= 2:
                    _last_tvt = float(valid_b2[-1])
                    _slope = float(np.polyfit(np.arange(len(valid_b2)), valid_b2, 1)[0])
                elif len(valid_b2) == 1:
                    _last_tvt = float(valid_b2[-1])

        # Build prior_segs from pre-extracted numpy arrays
        prior_segs: dict[str, np.ndarray | None] = {}
        for pname, parr in prior_arrs.items():
            seg = parr[chunk_rows]
            prior_segs[pname] = seg if np.isfinite(seg).any() else None
        for col, marr in mtp_arrs.items():
            seg = marr[chunk_rows]
            if np.isfinite(seg).any():
                prior_segs[col] = seg

        segs, mask = generate_action_segments(prior_segs, _last_tvt, _slope, L=L)

        if not mask.any():
            b2_seg = prior_segs.get("b2")
            if b2_seg is not None:
                pred_tvt[c_start:c_end] = b2_seg[:L]
            continue

        # Pad to CHUNK_LEN — edge-value fill, not zeros (matches dataset.py)
        if L < CHUNK_LEN:
            edge = segs[:, L - 1 : L]
            pad  = np.repeat(edge, CHUNK_LEN - L, axis=1)
            segs_full = np.concatenate([segs, pad], axis=1)
            b2_seg = prior_segs.get("b2")
            if b2_seg is not None:
                b2_edge = b2_seg[-1] if np.isfinite(b2_seg[-1]) else np.nan
                b2_full = np.concatenate([b2_seg, np.full(CHUNK_LEN - L, b2_edge, dtype=np.float32)])
            else:
                b2_full = None
        else:
            segs_full = segs
            b2_full = prior_segs.get("b2")

        from pathpolicy.actions import normalize_action_segs
        delta_segs = normalize_action_segs(segs_full, mask, b2_full, _last_tvt)

        # Context features
        context = extract_context_features_fast(
            well_df=well_df,
            prior_arrs=prior_arrs,
            chunk_rows=chunk_rows,
            known_indices=known_indices,
            last_known_tvt=last_known_tvt,
        )

        ctx_t   = torch.from_numpy(context).unsqueeze(0).to(device)         # [1, T, D_FEAT]
        segs_t  = torch.from_numpy(delta_segs).unsqueeze(0).to(device)      # [1, K, L]
        mask_t  = torch.from_numpy(mask).unsqueeze(0).to(device)            # [1, K]
        tids_t  = torch.arange(N_ACTION_TYPES, dtype=torch.long).unsqueeze(0).to(device)  # [1, K]

        logits, _ = model(ctx_t, segs_t, tids_t, mask_t)
        if head_type == "rmse_regressor":
            scores_safe = logits.masked_fill(~mask_t, 1e9)
            chosen_idx = int(scores_safe.argmin(dim=-1).item())
        else:
            chosen_idx = int(logits.argmax(dim=-1).item())
        chosen_seg = segs[chosen_idx, :L]
        pred_tvt[c_start:c_end] = chosen_seg
        chosen_actions.append(ACTION_VOCAB[chosen_idx])

    # B2 baseline for this well (fast numpy indexing)
    b2_arr = prior_arrs.get("b2")
    b2_pred = b2_arr[hidden_indices] if b2_arr is not None else np.full(len(hidden_indices), np.nan, dtype=np.float32)
    true_tvt = well_df["TVT"].values.astype(np.float32)[hidden_indices]

    return {
        "well_id": well_id,
        "hidden_row_idxs": hidden_indices,
        "pred_tvt": pred_tvt,
        "b2_tvt": b2_pred,
        "true_tvt": true_tvt,
        "chosen_actions": chosen_actions,
    }


def _well_rmse(pred: np.ndarray, true: np.ndarray) -> float:
    mask = np.isfinite(pred) & np.isfinite(true)
    if mask.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((pred[mask] - true[mask]) ** 2)))


# ---------------------------------------------------------------------------
# Main evaluation runner
# ---------------------------------------------------------------------------

def run_evaluation(
    checkpoint: Path,
    output_dir: Path,
    data_dir: Path = _DATA_DIR,
    base_path: Path = _BASE_PATH,
    b2_path: Path = _B2_PATH,
    a_path: Path = _A_PATH,
    mtp_path: Path = _MTP_PATH,
    oracle_chunks: Path = Path("artifacts/oracle_v0/oracle_chunks.parquet"),
    k_wells: int = -1,
    d_model: int = 128,
    n_layers: int = 2,
    nhead: int = 4,
    dim_ff: int = 256,
    dropout: float = 0.1,
    head_type: str = "classifier",
    split: str = "val",   # "val" | "train" | "all"
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )

    # Auto-load model dims from co-located config.json (written by train.py)
    cfg_path = checkpoint.parent / "config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            saved_cfg = json.load(f)
        d_model   = saved_cfg.get("d_model",    d_model)
        n_layers  = saved_cfg.get("n_layers",   n_layers)
        nhead     = saved_cfg.get("nhead",      nhead)
        dim_ff    = saved_cfg.get("dim_ff",     dim_ff)
        dropout   = saved_cfg.get("dropout",    dropout)
        head_type = saved_cfg.get("head_type",  head_type)
        print(f"[eval] loaded config from {cfg_path}: d_model={d_model} n_layers={n_layers} head_type={head_type}")

    # Load model
    model = PolicyFormer(
        d_model=d_model, n_layers=n_layers, nhead=nhead, dim_ff=dim_ff,
        dropout=dropout, head_type=head_type,
    )
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"[eval] loaded checkpoint: {checkpoint}  params={model.n_params:,}")

    # Load data
    chunks = pd.read_parquet(oracle_chunks)
    all_well_ids = chunks["well_id"].unique().tolist()

    # Restrict to the requested split (val-only by default to avoid train contamination)
    split_file_map = {
        "val":   checkpoint.parent / "val_wells.json",
        "train": checkpoint.parent / "train_wells.json",
    }
    if split in split_file_map:
        split_path = split_file_map[split]
        if split_path.exists():
            with open(split_path) as f:
                split_wells = set(json.load(f))
            well_ids = [w for w in all_well_ids if w in split_wells]
            print(f"[eval] split='{split}': {len(well_ids)}/{len(all_well_ids)} wells "
                  f"(loaded from {split_path.name})", flush=True)
        else:
            print(f"[eval] WARNING: split='{split}' requested but {split_path} not found — "
                  f"falling back to all {len(all_well_ids)} wells (CONTAMINATED)", flush=True)
            well_ids = all_well_ids
    else:
        # split="all": intentional; no filtering
        well_ids = all_well_ids
        print(f"[eval] split='all': evaluating all {len(well_ids)} wells (train+val)", flush=True)

    if k_wells > 0:
        well_ids = well_ids[:k_wells]

    print("[eval] loading priors...", flush=True)
    priors = load_priors(base_path, b2_path, a_path)
    print("[eval] loading MTP pivot...", flush=True)
    mtp_piv = load_mtp_pivot(mtp_path)
    print(f"[eval] loading {len(well_ids)} well CSVs...", flush=True)
    wells = load_well_csvs(data_dir, well_ids)

    print("[eval] pre-extracting per-well arrays...", flush=True)
    prior_arrs_all = _extract_prior_arrays(priors, wells)
    mtp_arrs_all   = _extract_mtp_arrays(mtp_piv, wells)

    # Load tail audit for class breakdown
    tail_path = Path("artifacts/tail_audit_v1/well_tail_audit.csv")
    tail_map: dict[str, str] = {}
    if tail_path.exists():
        ta = pd.read_csv(tail_path, usecols=["well_id", "tail_class"])
        tail_map = dict(zip(ta["well_id"], ta["tail_class"]))

    results: list[dict] = []
    all_chosen_actions: list[str] = []
    t0 = time.time()

    for i, wid in enumerate(well_ids):
        if wid not in wells:
            continue
        res = evaluate_well(
            wid, wells[wid],
            prior_arrs_all[wid],
            mtp_arrs_all.get(wid, {}),
            model, device, head_type=head_type,
        )
        if not res:
            continue

        pp_rmse = _well_rmse(res["pred_tvt"], res["true_tvt"])
        b2_rmse = _well_rmse(res["b2_tvt"],   res["true_tvt"])
        results.append({
            "well_id": wid,
            "tail_class": tail_map.get(wid, "unknown"),
            "pp_rmse": pp_rmse,
            "b2_rmse": b2_rmse,
            "gain": b2_rmse - pp_rmse,
        })
        all_chosen_actions.extend(res["chosen_actions"])

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(well_ids)}] {time.time()-t0:.0f}s elapsed", flush=True)

    df = pd.DataFrame(results)
    df.to_parquet(output_dir / "eval_wells.parquet", index=False)

    # Metrics
    mwr_pp = float(df["pp_rmse"].dropna().mean())
    mwr_b2 = float(df["b2_rmse"].dropna().mean())
    gain   = mwr_b2 - mwr_pp

    print("\n" + "=" * 60)
    print("PATHPOLICY EVAL REPORT")
    print("=" * 60)
    print(f"Wells evaluated:  {len(df)}")
    print(f"B2 baseline MWR:  {mwr_b2:.3f} ft")
    print(f"PathPolicy MWR:   {mwr_pp:.3f} ft")
    print(f"Gain vs B2:       {gain:+.3f} ft")
    print()

    if tail_map:
        print("Tail-class breakdown:")
        for cls, grp in df.groupby("tail_class"):
            print(f"  {cls:<35} n={len(grp):3d}  pp={grp['pp_rmse'].mean():.2f}  "
                  f"b2={grp['b2_rmse'].mean():.2f}  gain={grp['gain'].mean():+.2f}")

    if all_chosen_actions:
        from collections import Counter
        action_counts = Counter(all_chosen_actions)
        total_chunks = len(all_chosen_actions)
        print("\nChosen action distribution (top 15):")
        for action, cnt in action_counts.most_common(15):
            print(f"  {action:<40} {cnt:5d}  ({100*cnt/total_chunks:.1f}%)")
    print("=" * 60)

    gate = gain >= 0.15
    print(f"\nGATE: {'GO' if gate else 'NO-GO'} — gain={gain:+.3f} ft (threshold +0.15 ft)")

    metrics = {
        "n_wells": len(df),
        "mwr_pp":  mwr_pp,
        "mwr_b2":  mwr_b2,
        "gain":    gain,
        "gate":    "GO" if gate else "NO-GO",
    }
    with open(output_dir / "eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nReport saved to {output_dir}/eval_metrics.json")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",    default="artifacts/pathpolicy_v0/best.pt")
    parser.add_argument("--output_dir",    default="artifacts/pathpolicy_v0")
    parser.add_argument("--oracle_chunks", default="artifacts/oracle_v0/oracle_chunks.parquet")
    parser.add_argument("--k_wells",       type=int, default=-1)
    parser.add_argument("--d_model",       type=int, default=128)
    parser.add_argument("--n_layers",      type=int, default=2)
    parser.add_argument(
        "--split", default="val",
        choices=["val", "train", "all"],
        help="Which split to evaluate. 'val' (default) uses val_wells.json saved by train.py. "
             "'all' evaluates every well (in-sample contaminated — diagnostic only).",
    )
    args = parser.parse_args()

    run_evaluation(
        checkpoint=Path(args.checkpoint),
        output_dir=Path(args.output_dir),
        oracle_chunks=Path(args.oracle_chunks),
        k_wells=args.k_wells,
        d_model=args.d_model,
        n_layers=args.n_layers,
        split=args.split,
    )


if __name__ == "__main__":
    main()
