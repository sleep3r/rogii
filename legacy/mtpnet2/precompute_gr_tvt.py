"""
precompute_gr_tvt.py  —  Precompute greedy-GR TVT predictions for all 773 wells.
Uses the best config: multi-la [300,500,800], λ_o=40, λ_s=1.

Output: artifacts/results/oof_gr_tvt.pkl
  {well_id: pred_tvt_array (nh,), ...}
  + 'oof_rmse': float (sanity check on train wells)
"""
import sys, time, pickle
import numpy as np
sys.path.insert(0, '.')
from mtpnet.data          import load_offset_samples
from mtpnet.metrics       import row_rmse
from mtpnet.local_gr_search import run_greedy, LocalSearchConfig
from mtpnet.local_search  import load_typewell

DATA_DIR   = "/Users/alexander/Desktop/rogii/MTPNet/data/train"
CACHE_PATH = "artifacts/mtpnet_cache/samples.pkl"
OOF_K3_PATH= "artifacts/results/oof_k3.pkl"
OUT_PATH   = "artifacts/results/oof_gr_tvt.pkl"

cfg = LocalSearchConfig(
    lookaheads     = (300, 500, 800),
    lambda_offset  = 40.0,
    lambda_smooth  = 1.0,
)

print("Loading samples…")
samples = load_offset_samples('', k_wells=0, cache_path=CACHE_PATH, verbose=False)
with open(OOF_K3_PATH, 'rb') as f:
    oof_k3 = pickle.load(f)['oof_k3']

print(f"Computing greedy-GR TVT for {len(samples)} wells…")
t0 = time.time()
results = {}
preds, trues = [], []
errors = []

for i, s in enumerate(samples):
    try:
        tw_tvt, tw_gr = load_typewell(s.well_id, DATA_DIR)
        k3 = oof_k3[i]
        pred = run_greedy(s, k3, tw_tvt, tw_gr, cfg)
        results[s.well_id] = pred
        if s.has_true:
            true = s.tvt_true[s.hidden_rows]
            n2 = min(len(pred), len(true))
            preds.append(pred[:n2])
            trues.append(true[:n2])
    except Exception as e:
        errors.append((s.well_id, str(e)))
        results[s.well_id] = None

    if (i + 1) % 50 == 0 or i == len(samples) - 1:
        elapsed = time.time() - t0
        print(f"  {i+1}/{len(samples)}  elapsed={elapsed:.0f}s", flush=True)

oof_rmse = row_rmse(np.concatenate(preds), np.concatenate(trues))
print(f"\nOOF RMSE (train wells, greedy-GR): {oof_rmse:.4f} ft")
print(f"Errors: {len(errors)}")
if errors:
    for wid, e in errors[:5]:
        print(f"  {wid}: {e}")

with open(OUT_PATH, 'wb') as f:
    pickle.dump({'gr_tvt': results, 'oof_rmse': oof_rmse}, f)
print(f"Saved → {OUT_PATH}")
