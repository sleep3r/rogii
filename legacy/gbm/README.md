# ROGII GBM / Public Artifact Inference Stack

This is the clean submit project for the public Kojimar notebook:

`kojimar/rogii-inference-stack-with-pf-beam-and-tabicl`

It is inference-only. The kernel loads public artifact datasets, builds PF/beam/formation features for the hidden test wells, runs the saved LGB/CatBoost/TabICL stack, and writes `submission.csv`.

## Inputs

The Kaggle kernel metadata pins:

- Competition: `rogii-wellbore-geology-prediction`
- Artifacts: `thbdh5765/rogii-v10-fresh-artifacts`
- TabICL wheel/checkpoint: `needless090/rogii-tabicl-mirror`
- GPU shape: `NvidiaTeslaT4`

The T4 pin matters. A P100 run failed with:

`CUDA error: no kernel image is available for execution on the device`

## Commands

From this directory:

```bash
make push
make status
make wait
make output
make submit-code VERSION=2
```

`make output` downloads Kaggle output and validates `submission.csv`.

For code competition submit, `VERSION` must be the Kaggle kernel version that completed successfully.

## Current Kernel

Default kernel ref:

`sleep3r/rogii-kojimar-pf-beam-tabicl-submit`

Version 1 failed on P100 because the fork metadata did not preserve `machine_shape`.

Version 2 was pushed with explicit T4 metadata and entered the expected long TabICL burn-in path. The notebook has 30 TabICL contexts:

- `tabicl_A`: 5 folds x 5 seeds
- `tabicl_B`: 5 folds x 1 seed

So a healthy run can take a long time. Log lines like this are normal:

```text
burn tabicl_A fold0 seed0: 763074 rows
burn tabicl_A fold0 seed1: 763074 rows
```

## Development Rule

Keep this project focused on the GBM/public artifact stack. Put experiments around our formation B2 correction, direct solver, or old ClearML pipeline under `old/` or another separate project.
