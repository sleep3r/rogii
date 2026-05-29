"""NN inference: load fold checkpoints and predict for one or all wells."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


def load_model(ckpt_path: Path, device: torch.device):
    """Load BPHWT model from checkpoint."""
    from bphwt.models.bphwt import BPHWT

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg_dict = ckpt.get("cfg", {})
    mc = cfg_dict.get("model", {})

    model = BPHWT(
        in_channels=ckpt["in_channels"],
        stage_channels=mc.get("stage_channels", [96, 160, 256, 384]),
        stage_strides=mc.get("stage_strides", [2, 2, 2, 2]),
        decoder_channels=mc.get("decoder_channels", [256, 160, 96, 64]),
        n_blocks=mc.get("n_blocks", 2),
        use_bottleneck_attn=mc.get("use_bottleneck_attn", True),
        attn_heads=mc.get("attn_heads", 8),
        dropout=0.0,
        stoch_depth=0.0,
        predict_velocity=mc.get("predict_velocity", True),
        predict_dip_sign=mc.get("predict_dip_sign", True),
        predict_seg_boundary=mc.get("predict_seg_boundary", True),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_well(
    model,
    npz_path: Path,
    device: torch.device,
    cfg,
) -> dict[str, np.ndarray]:
    """
    Run NN prediction for a single well.

    Returns dict:
        tvt_pred      [L] — anchor-projected prediction
        tvt_raw       [L] — raw prediction before anchor projection
        log_sigma     [L] — log std
        velocity      [L] — predicted dTVT/dMD (if head active)
        seg_boundary  [L] — segment boundary probability (if head active)
        tvt_base      [L] — prior TVT base used as input
        known_mask    [L]
        hidden_mask   [L]
        md            [L]
        gr_obs        [L]  (NaN where invalid)
        gr_valid      [L]
        tw_tvt        [M]
        tw_gr         [M]
    """
    from bphwt.features.candidate_features import _backward_fill, _forward_fill

    data = np.load(npz_path, allow_pickle=True)

    X = torch.from_numpy(data["X"].astype(np.float32)).unsqueeze(0).permute(0, 2, 1).to(device)  # [1, C, L]
    md = torch.from_numpy(data["md"].astype(np.float32)).unsqueeze(0).to(device)
    known_mask = torch.from_numpy(data["known_mask"].astype(np.float32)).unsqueeze(0).to(device)
    tvt_base = torch.from_numpy(data["tvt_base"].astype(np.float32)).unsqueeze(0).to(device)
    tvt_input = data["tvt_input"].astype(np.float32)

    tvt_input_filled = _forward_fill(tvt_input.copy())
    tvt_input_filled = _backward_fill(tvt_input_filled)
    tvt_input_filled_t = torch.from_numpy(tvt_input_filled).unsqueeze(0).to(device)

    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
        preds = model(
            x=X,
            tvt_base=tvt_base,
            tvt_input=tvt_input_filled_t,
            known_mask=known_mask,
            md=md,
        )

    out = {
        "tvt_pred": preds["tvt_pred"][0].cpu().float().numpy(),
        "tvt_raw": preds["tvt_raw"][0].cpu().float().numpy(),
        "log_sigma": preds["log_sigma"][0].cpu().float().numpy(),
        "tvt_base": data["tvt_base"].astype(np.float32),
        "tvt_linear": data["tvt_linear"].astype(np.float32)
        if "tvt_linear" in data
        else data["tvt_base"].astype(np.float32),
        "tvt_hmm": data["tvt_hmm"].astype(np.float32)
        if "tvt_hmm" in data
        else data["tvt_base"].astype(np.float32),
        "tvt_dtw": data["tvt_dtw"].astype(np.float32)
        if "tvt_dtw" in data
        else data["tvt_base"].astype(np.float32),
        "tvt_neighbor": data["tvt_neighbor"].astype(np.float32)
        if "tvt_neighbor" in data
        else data["tvt_base"].astype(np.float32),
        "hmm_std": data["hmm_std"].astype(np.float32)
        if "hmm_std" in data
        else np.full_like(data["tvt_base"], 20.0),
        "hmm_entropy": data["hmm_entropy"].astype(np.float32)
        if "hmm_entropy" in data
        else np.zeros_like(data["tvt_base"]),
        "hmm_gr_mismatch": data["hmm_gr_mismatch"].astype(np.float32)
        if "hmm_gr_mismatch" in data
        else np.full_like(data["tvt_base"], 999.0),
        "known_mask": data["known_mask"].astype(np.float32),
        "hidden_mask": data["hidden_mask"].astype(np.float32),
        "md": data["md"].astype(np.float32),
        "gr_obs": data["gr_obs"].astype(np.float32),
        "gr_valid": data["gr_valid"].astype(np.float32),
        "tw_tvt": data["tw_tvt"].astype(np.float32),
        "tw_gr": data["tw_gr"].astype(np.float32),
        "tvt_input": tvt_input,
        "tvt_input_filled": tvt_input_filled,
    }
    if "velocity" in preds:
        out["velocity"] = preds["velocity"][0].cpu().float().numpy()
    if "seg_boundary" in preds:
        out["seg_boundary"] = preds["seg_boundary"][0].cpu().float().numpy()

    return out


def predict_all_wells(
    cfg,
    fold_checkpoint_paths: list[Path],
    well_ids: list[str],
    cache_dir: Path,
) -> dict[str, np.ndarray]:
    """
    Ensemble prediction over all folds for each well.
    Returns dict well_id -> averaged tvt_pred [L].
    """
    from bphwt.train.train_fold import resolve_device

    device = resolve_device(cfg.infer.device)

    models = [load_model(p, device) for p in fold_checkpoint_paths]
    logger.info(f"Loaded {len(models)} fold models")

    results = {}
    for well_id in well_ids:
        npz_path = cache_dir / f"{well_id}.npz"
        if not npz_path.exists():
            logger.warning(f"Cache not found for {well_id}, skipping")
            continue

        fold_preds = []
        fold_sigmas = []
        for model in models:
            p = predict_well(model, npz_path, device, cfg)
            fold_preds.append(p["tvt_pred"])
            fold_sigmas.append(np.exp(np.clip(p["log_sigma"], -4, 4)))

        # Precision-weighted ensemble
        sigmas = np.stack(fold_sigmas, axis=0)  # [F, L]
        preds = np.stack(fold_preds, axis=0)  # [F, L]
        precision = 1.0 / (sigmas**2 + 1e-6)
        weights = precision / precision.sum(axis=0, keepdims=True)
        tvt_ens = (preds * weights).sum(axis=0)

        # Keep per-fold data for optional post-opt
        results[well_id] = {
            "tvt_pred": tvt_ens,
            "tvt_pred_folds": preds,
            "log_sigma_ens": np.log(np.mean(sigmas, axis=0) + 1e-8),
            **{
                k: p[k]
                for k in [
                    "known_mask",
                    "hidden_mask",
                    "md",
                    "gr_obs",
                    "gr_valid",
                    "tw_tvt",
                    "tw_gr",
                    "tvt_input",
                    "tvt_input_filled",
                    "tvt_base",
                    "tvt_linear",
                    "tvt_hmm",
                    "tvt_dtw",
                    "tvt_neighbor",
                    "hmm_std",
                    "hmm_entropy",
                    "hmm_gr_mismatch",
                ]
            },
        }

    logger.info(f"Predicted {len(results)} wells")
    return results
