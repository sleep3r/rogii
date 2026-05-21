# Drift Targeting + NCC Notebook Replay

Source notebook:
`mitchgansemer/drift-targeting-ncc-tree-based-rogii-wellbore`

The Kaggle notebook is a writeup plus inference kernel. Its exact public run used
the private attached dataset `mitchgansemer/rogii-wellbore-models`, which contains
`utils.py`, KNN artifacts, fold models, `feature_cols.json`, OOF arrays, and tuned
parameters. The notebook source and public output were pulled into
`.kaggle_mining/code/mitchgansemer_drift_targeting/`, but the private model
dataset returns 403 outside Kaggle.

The active replay is therefore a local rebuild of the same approach:

- target is drift: `TVT - last_known_tvt`
- fold validation is grouped by well
- features keep the notebook family: formation KNN, row/dense ANCC, multi-scale
  NCC, Viterbi beam, particle filter, GR signal, anchor/geometry context
- GBDT stack uses LightGBM, XGBoost, and CatBoost with the published Optuna
  parameters
- blend is non-negative least squares
- postprocess is the notebook ramp-up (`tau = 100`) plus Savitzky-Golay smoothing
  (`window = 7`, `polyorder = 3`)

Commands:

```bash
make drift-ncc-quick
make drift-ncc-train
```

Config files:

- `configs/drift_ncc_quick.yml` for the three public sample wells
- `configs/drift_ncc.yml` for the full local train/test run
