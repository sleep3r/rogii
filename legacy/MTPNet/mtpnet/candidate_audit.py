"""Per-chunk audit of new candidates vs ``b2`` and the existing bank oracle.

Reads a baseline ``chunk_policy_dataset.parquet`` (which already contains per-chunk
``target_mse`` for every existing candidate plus row counts) and one or more new
candidate prediction parquets in the ``residual_stack`` schema
``(id, well_id, row_idx, pred_tvt)``. For each new candidate it re-aggregates
per-chunk MSE against the held-out hidden ``TVT``, then reports:

* pooled row-RMSE of the new candidate vs ``b2`` vs bank-oracle on all/safe/disaster chunks
* how often the new candidate beats ``b2``
* how often it is within ``0.5 ft`` of the bank oracle
* the augmented oracle pooled RMSE if the new candidate is added to the bank

This is the "does it help" verdict that should run in seconds once each candidate
artifact lands — without needing to launch a full ``chunk-policy`` cycle.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class _CandidateInput:
    label: str
    parquet: Path


def _load_chunk_baseline(path: Path) -> pd.DataFrame:
    ds = pd.read_parquet(path)
    required = {"well_id", "chunk_id", "candidate", "row_count", "target_mse"}
    missing = required - set(ds.columns)
    if missing:
        raise ValueError(f"baseline missing columns: {sorted(missing)}")
    ds["well_id"] = ds["well_id"].astype(str)
    ds["chunk_id"] = pd.to_numeric(ds["chunk_id"], errors="coerce").astype(int)
    return ds


def _per_chunk_b2_and_oracle(ds: pd.DataFrame) -> pd.DataFrame:
    b2 = ds[ds["candidate"].astype(str) == "b2"][
        ["well_id", "chunk_id", "row_count", "target_mse"]
    ].rename(columns={"target_mse": "b2_mse"})
    oracle = ds.groupby(["well_id", "chunk_id"]).agg(best_mse=("target_mse", "min")).reset_index()
    merged = b2.merge(oracle, on=["well_id", "chunk_id"], how="left")
    merged["b2_rmse"] = np.sqrt(merged["b2_mse"].clip(lower=0))
    merged["best_rmse"] = np.sqrt(merged["best_mse"].clip(lower=0))
    merged["gap_rmse"] = merged["b2_rmse"] - merged["best_rmse"]
    return merged


def _reconstruct_chunk_id(
    data_dir: Path,
    well_ids: Iterable[str],
    *,
    chunk_size: int,
) -> pd.DataFrame:
    """Build ``(well_id, row_idx, chunk_id, TVT)`` for hidden rows.

    Hidden chunking matches the rule in ``chunk_policy._chunk_frame``:
    chunks index *hidden* rows in their original ``row_idx`` order with
    ``chunk_id = local_index // chunk_size``.
    """
    rows: list[pd.DataFrame] = []
    for well_id in well_ids:
        path = data_dir / f"{well_id}__horizontal_well.csv"
        if not path.exists():
            continue
        w = pd.read_csv(path)
        tvt = pd.to_numeric(w.get("TVT"), errors="coerce")
        tvt_in = pd.to_numeric(w.get("TVT_input"), errors="coerce")
        hidden_mask = (~tvt_in.notna()) & tvt.notna()
        hidden = w.loc[hidden_mask].copy()
        if hidden.empty:
            continue
        if "row_idx" not in hidden.columns:
            hidden["row_idx"] = np.arange(len(w))[hidden_mask]
        hidden = hidden.sort_values("row_idx").reset_index(drop=True)
        local = np.arange(len(hidden))
        hidden["chunk_id"] = (local // max(int(chunk_size), 1)).astype(int)
        hidden["well_id"] = well_id
        hidden["TVT"] = tvt.values[hidden_mask][hidden.index.to_numpy()]
        rows.append(hidden[["well_id", "row_idx", "chunk_id", "TVT"]])
    if not rows:
        raise ValueError(f"No hidden rows found under {data_dir}")
    return pd.concat(rows, ignore_index=True)


def _per_chunk_new_candidate(
    truth: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    label: str,
) -> pd.DataFrame:
    pred = candidate[["well_id", "row_idx", "pred_tvt"]].copy()
    pred["well_id"] = pred["well_id"].astype(str)
    pred["row_idx"] = pd.to_numeric(pred["row_idx"], errors="coerce").astype("Int64")
    pred = pred.dropna(subset=["row_idx"]).copy()
    pred["row_idx"] = pred["row_idx"].astype(int)
    pred["pred_tvt"] = pd.to_numeric(pred["pred_tvt"], errors="coerce")
    joined = truth.merge(pred, on=["well_id", "row_idx"], how="inner")
    if joined.empty:
        raise ValueError(f"candidate {label!r} has no matching hidden rows")
    joined["err"] = joined["pred_tvt"] - joined["TVT"]
    joined = joined.dropna(subset=["err"])
    grouped = (
        joined.groupby(["well_id", "chunk_id"])
        .agg(
            new_rows=("err", "size"),
            new_mse=("err", lambda s: float(np.mean(np.square(s)))),
        )
        .reset_index()
    )
    grouped["new_rmse"] = np.sqrt(grouped["new_mse"].clip(lower=0))
    grouped = grouped.rename(
        columns={
            "new_rows": f"{label}_rows",
            "new_mse": f"{label}_mse",
            "new_rmse": f"{label}_rmse",
        }
    )
    return grouped


def _format_section(label: str, summary: pd.DataFrame, candidates: list[str]) -> list[str]:
    lines = [
        "",
        f"=== {label} (n={len(summary)}) ===",
    ]
    if summary.empty:
        return lines + ["  (empty)"]
    b2_pool = float(
        np.sqrt(
            (summary["b2_rmse"] ** 2 * summary["row_count"]).sum()
            / summary["row_count"].sum()
        )
    )
    oracle_pool = float(
        np.sqrt(
            (summary["best_rmse"] ** 2 * summary["row_count"]).sum()
            / summary["row_count"].sum()
        )
    )
    lines.append(f"  b2 pooled RMSE:                    {b2_pool:.3f} ft")
    lines.append(f"  bank-oracle pooled RMSE:           {oracle_pool:.3f} ft")
    for cand in candidates:
        col_mse = f"{cand}_mse"
        col_rmse = f"{cand}_rmse"
        col_rows = f"{cand}_rows"
        if col_mse not in summary.columns:
            continue
        sub = summary.dropna(subset=[col_mse])
        if sub.empty:
            lines.append(f"  {cand}: no chunks matched")
            continue
        cand_pool = float(
            np.sqrt(
                (sub[col_rmse] ** 2 * sub[col_rows]).sum() / sub[col_rows].sum()
            )
        )
        beats_b2 = float((sub[col_rmse] < sub["b2_rmse"]).mean())
        near_oracle = float(((sub[col_rmse] - sub["best_rmse"]).abs() < 0.5).mean())
        in_top1 = float((sub[col_rmse] <= sub["best_rmse"] + 0.5).mean())
        new_best = float((sub[col_mse] < sub["best_mse"] - 1e-6).mean())
        lines.append(
            f"  {cand}: pooled={cand_pool:.3f} ft  "
            f"beats_b2={100*beats_b2:.1f}%  "
            f"≤0.5ft of oracle={100*near_oracle:.1f}%  "
            f"new-best={100*new_best:.1f}%"
        )
    return lines


def audit_candidates(
    *,
    baseline: Path,
    data_dir: Path,
    chunk_size: int,
    candidates: list[_CandidateInput],
    output: Path | None = None,
) -> dict:
    ds = _load_chunk_baseline(baseline)
    chunk_truth = _per_chunk_b2_and_oracle(ds)
    well_ids = sorted(chunk_truth["well_id"].unique())
    print(
        f"[audit] baseline rows={len(ds):,} chunks={len(chunk_truth):,} wells={len(well_ids)}",
        file=sys.stderr,
        flush=True,
    )
    print(f"[audit] reconstructing per-row chunk ids (chunk_size={chunk_size})...",
          file=sys.stderr, flush=True)
    row_truth = _reconstruct_chunk_id(data_dir, well_ids, chunk_size=chunk_size)
    print(f"[audit] reconstructed hidden rows={len(row_truth):,}", file=sys.stderr, flush=True)

    summary = chunk_truth.copy()
    cand_labels: list[str] = []
    for cand in candidates:
        if not cand.parquet.exists():
            print(f"[audit] WARN candidate {cand.label!r} not found at {cand.parquet}",
                  file=sys.stderr, flush=True)
            continue
        candidate_df = pd.read_parquet(cand.parquet)
        per_chunk = _per_chunk_new_candidate(row_truth, candidate_df, label=cand.label)
        summary = summary.merge(per_chunk, on=["well_id", "chunk_id"], how="left")
        cand_labels.append(cand.label)
        n_matched = summary[f"{cand.label}_rmse"].notna().sum()
        print(
            f"[audit] candidate {cand.label!r}: rows={len(candidate_df):,} "
            f"chunks_matched={n_matched}/{len(summary)}",
            file=sys.stderr,
            flush=True,
        )

    if not cand_labels:
        raise ValueError("no candidate parquets matched")

    # Quantile-based "disaster" threshold per the existing convention.
    if not summary.empty:
        disaster_thr = float(np.quantile(summary["gap_rmse"], 0.9))
    else:
        disaster_thr = float("nan")

    sections = [
        f"Per-chunk audit (chunk_size={chunk_size}, disaster threshold gap>{disaster_thr:.2f} ft)",
    ]
    sections.extend(_format_section("ALL CHUNKS", summary, cand_labels))
    sections.extend(
        _format_section(
            "SAFE (gap ≤ p90)",
            summary[summary["gap_rmse"] <= disaster_thr],
            cand_labels,
        )
    )
    sections.extend(
        _format_section(
            "DISASTER (gap > p90)",
            summary[summary["gap_rmse"] > disaster_thr],
            cand_labels,
        )
    )

    # Augmented oracle if each candidate is added to the bank
    sections.append("")
    sections.append("=== If candidates added to bank ===")
    base_oracle = float(
        np.sqrt(
            (summary["best_rmse"] ** 2 * summary["row_count"]).sum()
            / summary["row_count"].sum()
        )
    )
    sections.append(f"  current bank-oracle:           {base_oracle:.3f} ft")
    for cand in cand_labels:
        col_mse = f"{cand}_mse"
        if col_mse not in summary.columns:
            continue
        merged = summary.copy()
        merged[col_mse] = merged[col_mse].fillna(merged["best_mse"])
        merged["aug_mse"] = np.minimum(merged["best_mse"], merged[col_mse])
        aug_oracle = float(
            np.sqrt(
                (merged["aug_mse"] * merged["row_count"]).sum() / merged["row_count"].sum()
            )
        )
        new_best = int((merged[col_mse] < merged["best_mse"] - 1e-6).sum())
        sections.append(
            f"  + {cand}:                     {aug_oracle:.3f} ft  "
            f"(Δ {base_oracle - aug_oracle:+.3f} ft, new-best on {new_best} chunks)"
        )

    text = "\n".join(sections)
    print(text)
    metrics = {
        "baseline_path": str(baseline),
        "chunk_size": int(chunk_size),
        "disaster_threshold_ft": disaster_thr,
        "candidates_labels": cand_labels,
        "base_oracle_pooled_rmse": base_oracle,
        "b2_pooled_rmse": float(
            np.sqrt(
                (summary["b2_rmse"] ** 2 * summary["row_count"]).sum()
                / summary["row_count"].sum()
            )
        ),
        "n_chunks": int(len(summary)),
        "n_wells": int(summary["well_id"].nunique()),
        "report": text,
    }
    if output is not None:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Per-chunk audit of new candidate predictions vs b2 and bank-oracle."
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("artifacts/chunk_ranker_dp_v2_selfcal_c512/chunk_policy_dataset.parquet"),
        help="chunk_policy_dataset.parquet from a previous chunk-policy run.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument(
        "--k-offset",
        type=Path,
        default=Path("artifacts/k_segment_offset_v0/k_offset_oof_predictions.parquet"),
        help="k_segment_offset OOF predictions. Ignored if file absent.",
    )
    parser.add_argument(
        "--dtvt-state",
        type=Path,
        default=Path("artifacts/dtvt_state_model_v0/dtvt_state_oof_predictions.parquet"),
        help="dtvt_state_model OOF predictions. Ignored if file absent.",
    )
    parser.add_argument(
        "--extra-candidate",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Additional candidate; repeat for several.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    candidates = [
        _CandidateInput("k_segment_offset_v0", Path(args.k_offset)),
        _CandidateInput("dtvt_state_model_v0", Path(args.dtvt_state)),
    ]
    for spec in args.extra_candidate:
        if "=" not in spec:
            raise ValueError(f"--extra-candidate expects LABEL=PATH (got {spec!r})")
        label, path = spec.split("=", 1)
        candidates.append(_CandidateInput(label.strip(), Path(path.strip())))
    audit_candidates(
        baseline=Path(args.baseline),
        data_dir=Path(args.data_dir),
        chunk_size=int(args.chunk_size),
        candidates=candidates,
        output=Path(args.output) if args.output is not None else None,
    )


if __name__ == "__main__":
    main()
