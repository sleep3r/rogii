"""
analyze_oof.py  —  Load a run's oof_predictions.pkl and print detailed analysis.

Usage:
    python analyze_oof.py artifacts/dlmtp/run1/oof_predictions.pkl [--tag Run1]
    python analyze_oof.py artifacts/dlmtp/run2/oof_predictions.pkl [--tag Run2]
    python analyze_oof.py artifacts/dlmtp/run1/oof_predictions.pkl artifacts/dlmtp/run2/oof_predictions.pkl
"""
import sys, argparse, pickle
import numpy as np
sys.path.insert(0, '.')
from mtpnet.data import load_offset_samples
from mtpnet.metrics import row_rmse

CACHE_PATH = "artifacts/mtpnet_cache/samples.pkl"

def analyze(path, tag=""):
    with open(path, "rb") as f:
        d = pickle.load(f)
    preds = d["oof_preds"]
    trues = d["oof_trues"]

    samples = load_offset_samples('', k_wells=0, cache_path=CACHE_PATH, verbose=False)
    idx_map = {i: samples[i] for i in sorted(preds)}

    all_p = np.concatenate([preds[i] for i in sorted(preds)])
    all_t = np.concatenate([trues[i] for i in sorted(trues)])
    oof   = row_rmse(all_p, all_t)

    buckets = {"short": [], "medium": [], "long": [], "xlong": []}
    per_well = []
    for i in sorted(preds):
        s  = samples[i]
        nh = len(s.hidden_rows)
        p  = preds[i]; t = trues[i]
        pw = float(np.sqrt(np.mean((p - t) ** 2)))
        per_well.append((pw, i, s.well_id))
        bk = ("xlong" if nh >= 8000 else "long" if nh >= 5000
              else "medium" if nh >= 2000 else "short")
        buckets[bk].append((p, t, pw, s.well_id))

    lbl = f"[{tag}] " if tag else ""
    print(f"\n{'─'*55}")
    print(f"{lbl}OOF pooled RMSE : {oof:.4f} ft  ({len(per_well)} wells)")
    for bk, items in buckets.items():
        if not items:
            continue
        bp = np.concatenate([x[0] for x in items])
        bt = np.concatenate([x[1] for x in items])
        print(f"  {bk:7s} ({len(items):3d} wells): {row_rmse(bp, bt):.3f} ft")

    print("\nTop-10 hardest wells:")
    worst = sorted(per_well, reverse=True)[:10]
    for pw, i, wid in worst:
        nh = len(samples[i].hidden_rows)
        print(f"  {wid[:30]:30s}  nh={nh:5d}  RMSE={pw:.2f} ft")

    return oof

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--tags", nargs="*")
    args = parser.parse_args()
    tags = args.tags or [f"Run{i+1}" for i in range(len(args.paths))]
    for p, t in zip(args.paths, tags):
        analyze(p, t)

if __name__ == "__main__":
    main()
