"""Build final submission CSV from model predictions.

Handles:
  - Loading test well cache
  - Running NN inference (all fold checkpoints)
  - Per-well post-optimization (optional)
  - Static/dynamic blending
  - Writing submission.csv in required format: id,tvt

The submission ID format is: {well_id}_{row_index}
where row_index is the integer index from sample_submission.csv.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def make_submission(
    cfg,
    output_dir: Path,
    submission_path: Path,
    sample_submission_path: Path | None = None,
) -> None:
    """
    Full inference pipeline → submission.csv.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load sample submission to get required IDs -----------------
    data_dir = cfg.resolved_data_dir()
    if sample_submission_path is None:
        sample_submission_path = data_dir / "sample_submission.csv"

    sample_df = pd.read_csv(sample_submission_path)
    # Parse well_id and row_index from id column
    sample_df["well_id"] = sample_df["id"].str.split("_").str[0]
    sample_df["row_index"] = sample_df["id"].str.split("_").str[1].astype(int)

    required_wells = sample_df["well_id"].unique().tolist()
    logger.info(f"Required {len(required_wells)} test wells, {len(sample_df)} rows total")

    # ---- Find fold checkpoints ------------------------------------
    fold_checkpoints = _find_fold_checkpoints(output_dir, cfg.train.n_folds)
    if not fold_checkpoints:
        raise FileNotFoundError(f"No fold checkpoints found in {output_dir}")
    logger.info(f"Found {len(fold_checkpoints)} fold checkpoints")

    # ---- Run NN inference ----------------------------------------
    from bphwt.infer.predict_nn import predict_all_wells

    cache_dir = cfg.resolved_cache_dir() / "test"

    # Filter to wells that have cache
    available = [w for w in required_wells if (cache_dir / f"{w}.npz").exists()]
    missing_cache = [w for w in required_wells if w not in available]
    if missing_cache:
        logger.warning(f"{len(missing_cache)} test wells have no cache: {missing_cache[:5]}")

    nn_preds = predict_all_wells(cfg, fold_checkpoints, available, cache_dir)

    # ---- Per-well post-optimization (optional) -------------------
    ic = cfg.infer
    if ic.run_postopt:
        logger.info("Running per-well post-optimization...")
        from bphwt.infer.optimize_curve import optimize_well_curve

        for well_id, data in nn_preds.items():
            try:
                tvt_hmm = data.get("tvt_hmm")
                tvt_opt = optimize_well_curve(
                    tvt_nn=data["tvt_pred"],
                    log_sigma=data["log_sigma_ens"],
                    md=data["md"],
                    known_mask=data["known_mask"],
                    tvt_input=data["tvt_input"],
                    tvt_input_filled=data["tvt_input_filled"],
                    gr_obs=data["gr_obs"],
                    gr_valid=data["gr_valid"],
                    tw_tvt=data["tw_tvt"],
                    tw_gr=data["tw_gr"],
                    tvt_hmm=tvt_hmm,
                    n_knots=ic.postopt_n_knots,
                    n_steps=ic.postopt_n_steps,
                    lambda_prior=ic.postopt_lambda_prior,
                    lambda_hmm=ic.postopt_lambda_hmm,
                    lambda_smooth=ic.postopt_lambda_smooth,
                    lambda_anchor=ic.postopt_lambda_anchor,
                    clip_correction=ic.postopt_clip_correction,
                )
                data["tvt_opt"] = tvt_opt
            except Exception as e:
                logger.warning(f"Post-opt failed for {well_id}: {e}")
                data["tvt_opt"] = data["tvt_pred"]
    else:
        for data in nn_preds.values():
            data["tvt_opt"] = data["tvt_pred"]

    # ---- Static blend --------------------------------------------
    from bphwt.infer.blend import enforce_anchors, static_blend

    for well_id, data in nn_preds.items():
        tvt_final = static_blend(
            tvt_nn_opt=data["tvt_opt"],
            tvt_nn_raw=data["tvt_pred"],
            tvt_hmm=data.get("tvt_hmm"),
            tvt_dtw=data.get("tvt_dtw"),
            tvt_neighbor=data.get("tvt_neighbor"),
            tvt_linear=data.get("tvt_linear"),
            w_nn_opt=ic.blend_nn_opt,
            w_nn_raw=ic.blend_nn_raw,
            w_hmm=ic.blend_hmm,
            w_dtw=0.07,
            w_neighbor=ic.blend_neighbor,
            w_linear=ic.blend_linear,
        )
        # Hard-enforce anchors
        tvt_final = enforce_anchors(tvt_final, data["tvt_input"], data["known_mask"])
        data["tvt_final"] = tvt_final

    # ---- Build submission dataframe -------------------------------
    rows = []
    for _, row in sample_df.iterrows():
        well_id = row["well_id"]
        row_idx = int(row["row_index"])

        if well_id in nn_preds and row_idx < len(nn_preds[well_id]["tvt_final"]):
            tvt_val = float(nn_preds[well_id]["tvt_final"][row_idx])
        else:
            # Fallback: use 0.0 (will be penalized but avoids missing rows)
            logger.warning(f"No prediction for {well_id} row {row_idx}, using 0.0")
            tvt_val = 0.0

        rows.append({"id": row["id"], "tvt": tvt_val})

    sub_df = pd.DataFrame(rows)
    sub_df.to_csv(submission_path, index=False)
    logger.info(f"Saved submission: {submission_path} ({len(sub_df)} rows)")


def make_oof(
    cfg,
    output_dir: Path,
    oof_path: Path,
) -> None:
    """
    Generate OOF predictions on train wells for CV scoring.
    Uses fold validation sets (no data leakage).
    """
    output_dir = Path(output_dir)
    import json

    from bphwt.train.cv_score import compute_oof_rmse, score_by_buckets

    cache_dir = cfg.resolved_cache_dir() / "train"
    all_rows = []

    for fold in range(cfg.train.n_folds):
        fold_dir = output_dir / f"fold_{fold}"
        ckpt = fold_dir / "best_ema.pt"
        if not ckpt.exists():
            logger.warning(f"No checkpoint for fold {fold}, skipping")
            continue

        from bphwt.infer.predict_nn import predict_all_wells

        # We need val_ids — stored in cv_summary.json
        summary_path = output_dir / "cv_summary.json"
        if not summary_path.exists():
            logger.warning("cv_summary.json not found, cannot determine val splits")
            continue

        with open(summary_path) as f:
            summary = json.load(f)

        val_ids = summary["fold_results"][fold]["val_ids"]

        fold_preds = predict_all_wells(cfg, [ckpt], val_ids, cache_dir)

        for well_id, data in fold_preds.items():
            L = len(data["tvt_pred"])
            hm = data["hidden_mask"] > 0.5

            # Load true TVT from cache
            npz = np.load(cache_dir / f"{well_id}.npz", allow_pickle=True)
            if "tvt_true" not in npz:
                continue
            tvt_true = npz["tvt_true"].astype(np.float32)

            for i in range(L):
                if hm[i]:
                    all_rows.append(
                        {
                            "well_id": well_id,
                            "fold": fold,
                            "row_idx": i,
                            "is_hidden": 1,
                            "tvt_pred": float(data["tvt_pred"][i]),
                            "tvt_true": float(tvt_true[i]),
                            "tvt_base": float(data["tvt_base"][i]),
                            "gr_valid_ratio": float(data["gr_valid"].mean()),
                            "hidden_ratio": float(hm.sum() / max(L, 1)),
                        }
                    )

    oof_df = pd.DataFrame(all_rows)
    oof_df.to_csv(oof_path, index=False)

    if len(oof_df) > 0:
        rmse = compute_oof_rmse(oof_df)
        buckets = score_by_buckets(oof_df)
        logger.info(f"OOF RMSE (hidden): {rmse:.4f}")
        logger.info(f"Bucket scores: {buckets}")

        bucket_path = oof_path.with_suffix(".buckets.json")
        with open(bucket_path, "w") as f:
            json.dump(buckets, f, indent=2)

    logger.info(f"Saved OOF: {oof_path} ({len(oof_df)} rows)")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_fold_checkpoints(output_dir: Path, n_folds: int) -> list[Path]:
    """Find best_ema.pt for each fold."""
    paths = []
    for fold in range(n_folds):
        p = output_dir / f"fold_{fold}" / "best_ema.pt"
        if p.exists():
            paths.append(p)
    return paths
