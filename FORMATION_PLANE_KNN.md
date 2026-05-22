# FormationPlaneKNN Experiment

Deterministic local surface solver, not a global Surface Student:

```text
train-fold wells around target well
  -> row-sampled FormationPlaneKNN / IDW / dense ANCC
  -> pseudo-surfaces ANCC ... BUDA
  -> b_well calibration from known TVT_input anchor
  -> hidden-zone TVT candidates and diagnostics
```

Run quick public-sample smoke:

```bash
make formation-plane-knn-quick
```

Run full fold-safe OOF:

```bash
make formation-plane-knn
```

Main outputs under `artifacts/formation_plane_knn*/`:

- `oof_candidates.parquet` or `.csv` fallback
- `candidate_scores.csv`
- `surface_scores.csv`
- `anchor_hidden_correlation.csv`
- `oracle_scores.csv`
- `oracle_winners.csv`
- `metrics.json`
- `report.md`

Candidate families:

- `tvtF_<FORMATION>_{full,late,wls}`
- `row_ancc_tvt`, `dense_ancc_tvt`
- `formation_median_tvt`, `formation_wls_median_tvt`
- nearby-well path samples: top1/top3/top5/weighted/p10/p90
- bootstrap formation samples: best-by-anchor/mean/median/p10/p90

Validation is fold-safe: each validation well's true formation columns are used
only for diagnostics after prediction, never for building the fold KNN context.
