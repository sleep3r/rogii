from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import GAConfig, config_to_dict, load_config
from .dataset import AlignmentDataset, collate_alignment_samples, load_alignment_samples
from .evaluate import evaluate_model
from .loss import alignment_ce_loss
from .model import GeoAligner, count_parameters


def _device(name: str) -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _split_samples(samples: list, valid_fraction: float, seed: int) -> tuple[list, list]:
    rng = np.random.default_rng(seed)
    order = np.arange(len(samples))
    rng.shuffle(order)
    valid_n = max(1, int(round(len(samples) * valid_fraction)))
    valid_idx = set(order[:valid_n].tolist())
    train = [s for i, s in enumerate(samples) if i not in valid_idx]
    valid = [s for i, s in enumerate(samples) if i in valid_idx]
    if not train:
        train, valid = samples, samples
    return train, valid


def _train_one_epoch(model, loader, optimizer, device, clip_grad_norm: float) -> float:
    model.train()
    losses: list[float] = []
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        logits = model(
            batch["lateral_features"].to(device),
            batch["typewell_features"].to(device),
            batch["lateral_pad_mask"].to(device),
            batch["typewell_pad_mask"].to(device),
        )
        loss = alignment_ce_loss(
            logits,
            batch["target_bins"].to(device),
            batch["hidden_mask"].to(device),
            batch["lateral_pad_mask"].to(device),
            batch["typewell_pad_mask"].to(device),
        )
        loss.backward()
        if clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_report(output_dir: Path, summary: dict) -> None:
    valid = summary["valid"]
    normal = valid.get("normal", {})
    shuffled = valid.get("shuffled_gr", {})
    zero = valid.get("zero_gr", {})
    lines = [
        "# GEOALIGNER_V0_REPORT",
        "",
        "## Summary",
        "",
        f"- run: `{summary['run']['name']}`",
        f"- train wells: `{summary['data']['train_wells']}`",
        f"- valid wells: `{summary['data']['valid_wells']}`",
        f"- parameters: `{summary['model']['parameters']}`",
        "",
        "## Normal Metrics",
        "",
        f"- emission top1 RMSE ft: `{normal.get('emission_top1_rmse_ft')}`",
        f"- emission top10 oracle RMSE ft: `{normal.get('emission_top10_oracle_rmse_ft')}`",
        f"- target top10 rate: `{normal.get('target_top10_rate')}`",
        f"- DP row RMSE ft: `{normal.get('row_rmse_ft')}`",
        f"- known-tail anchor RMSE ft: `{normal.get('known_tail_anchor_rmse_ft')}`",
        "",
        "## Sanity",
        "",
        f"- shuffled GR DP row RMSE ft: `{shuffled.get('row_rmse_ft')}`",
        f"- zero GR DP row RMSE ft: `{zero.get('row_rmse_ft')}`",
        f"- normal minus shuffled row RMSE ft: `{None if not normal or not shuffled else normal.get('row_rmse_ft') - shuffled.get('row_rmse_ft')}`",
        "",
        "## Artifacts",
        "",
        "- `geoaligner_metrics.json`",
        "- `geoaligner_row_predictions.parquet`",
        "- `geoaligner_alignment_steps.parquet`",
        "- `figures/alignment_example.png`",
    ]
    (output_dir / "geoaligner_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_example_figure(output_dir: Path, steps) -> None:
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if steps.empty:
        (fig_dir / "alignment_example.png").touch()
        return
    data = steps[steps["variant"] == "normal"]
    if data.empty:
        data = steps
    well = str(data.iloc[0]["well_id"])
    data = data[data["well_id"] == well].sort_values("step")
    plt.figure(figsize=(8, 4))
    plt.plot(data["step"], data["true_tvt"], label="true TVT", linewidth=2)
    plt.plot(data["step"], data["dp_tvt"], label="GeoAligner DP", linewidth=2)
    plt.plot(data["step"], data["anchor_tvt"], label="known-tail anchor", linestyle="--")
    plt.gca().invert_yaxis()
    plt.xlabel("compressed step")
    plt.ylabel("TVT ft")
    plt.title(f"GeoAligner alignment example: {well}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "alignment_example.png", dpi=150)
    plt.close()


def train(cfg: GAConfig, override_epochs: int | None = None) -> dict:
    _seed(cfg.train.seed)
    device = _device(cfg.train.device)
    output_dir = cfg.run.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output_dir / "config_resolved.json").write_text(
        json.dumps(config_to_dict(cfg), indent=2), encoding="utf-8"
    )

    samples = load_alignment_samples(cfg.data_dir, cfg.data, k_wells=cfg.k_wells)
    train_samples, valid_samples = _split_samples(samples, cfg.train.valid_fraction, cfg.train.seed)
    loader = DataLoader(
        AlignmentDataset(train_samples),
        batch_size=cfg.train.batch_size,
        shuffle=True,
        collate_fn=collate_alignment_samples,
    )
    model = GeoAligner(cfg.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    best_rmse = float("inf")
    best_epoch = 0
    epochs = override_epochs or cfg.train.epochs
    print(f"[geoaligner] device={device}", flush=True)
    print(
        f"[geoaligner] train={len(train_samples)} valid={len(valid_samples)} params={count_parameters(model)}",
        flush=True,
    )
    for epoch in range(1, epochs + 1):
        train_loss = _train_one_epoch(model, loader, optimizer, device, cfg.train.clip_grad_norm)
        valid_metrics, _, _ = evaluate_model(
            model,
            valid_samples,
            device,
            variants=("normal",),
            max_jump_bins=cfg.decode.max_jump_bins,
            jump_penalty=cfg.decode.jump_penalty,
            anchor_band_radius_ft=cfg.decode.anchor_band_radius_ft,
            anchor_band_penalty=cfg.decode.anchor_band_penalty,
        )
        score = float(valid_metrics["normal"]["row_rmse_ft"])
        is_best = score < best_rmse
        if is_best:
            best_rmse = score
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "epoch": epoch, "score": score}, output_dir / "checkpoints" / "best.pt")
        print(
            json.dumps(
                {
                    "event": "geoaligner_epoch",
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "valid_row_rmse_ft": score,
                    "valid_emission_top10_oracle_rmse_ft": valid_metrics["normal"].get("emission_top10_oracle_rmse_ft"),
                    "is_best": is_best,
                }
            ),
            flush=True,
        )

    state = torch.load(output_dir / "checkpoints" / "best.pt", map_location=device)
    model.load_state_dict(state["model"])
    valid_metrics, rows, steps = evaluate_model(
        model,
        valid_samples,
        device,
        variants=("normal", "shuffled_gr", "zero_gr", "typewell_shuffled"),
        max_jump_bins=cfg.decode.max_jump_bins,
        jump_penalty=cfg.decode.jump_penalty,
        anchor_band_radius_ft=cfg.decode.anchor_band_radius_ft,
        anchor_band_penalty=cfg.decode.anchor_band_penalty,
    )
    rows.to_parquet(output_dir / "geoaligner_row_predictions.parquet", index=False)
    steps.to_parquet(output_dir / "geoaligner_alignment_steps.parquet", index=False)
    _write_example_figure(output_dir, steps)
    summary = {
        "run": {"name": cfg.run.name, "output_dir": str(output_dir)},
        "data": {"train_wells": len(train_samples), "valid_wells": len(valid_samples)},
        "model": {"parameters": count_parameters(model)},
        "checkpoint": {"epoch": best_epoch, "score": best_rmse},
        "valid": valid_metrics,
    }
    (output_dir / "geoaligner_metrics.json").write_text(
        json.dumps(_json_safe(summary), indent=2), encoding="utf-8"
    )
    _write_report(output_dir, _json_safe(summary))
    return _json_safe(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GeoAligner")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    summary = train(load_config(args.config), override_epochs=args.epochs)
    print(json.dumps(summary["valid"]["normal"], indent=2), flush=True)


if __name__ == "__main__":
    main()
