# ROGII Old

Standalone training workspace for the ROGII wellbore TVT residual stack.

The supported path is intentionally small: local data, three configs, one
training entrypoint, optional Kaggle inference packaging, and tests.

## Layout

- `rogii/` - training, feature building, inference, and artifact code.
- `configs/quick.yml` - small local smoke config.
- `configs/stack.yml` - CPU production config.
- `configs/stack_gpu.yml` - GPU production config.
- `tests/` - regression and unit tests.
- `Makefile` - the supported command surface.

Generated data and outputs live under `data/`, `artifacts/`, `catboost_info/`,
and `submission.csv`. They are ignored by git.

## Setup

```bash
make install
make data
```

`make data` downloads the Kaggle competition archive into `data/` and unpacks it.
If you already have `data/train` and `data/test`, training commands will use
them as-is.

## Train

```bash
make quick
make train
make train-gpu
make train-server INSTANCE=gpu
```

Defaults:

- `make quick` uses `configs/quick.yml`.
- `make train` uses `configs/stack.yml`.
- `make train-gpu` runs `rogii.gpu_preflight` first, then uses
  `configs/stack_gpu.yml`.
- `make train-server` validates Docker/Buildx/Portainer config, then launches
  `SERVER_CONFIG` through `spacebridge`.

Override configs with Make variables:

```bash
make train CONFIG=configs/stack.yml
make train-gpu GPU_CONFIG=configs/stack_gpu.yml
make train-server INSTANCE=gpu SERVER_CONFIG=configs/stack_gpu.yml
```

## Server Runs

Server training is still supported, but it is kept in a small separate block
instead of mixed into every local command.

1. Copy `portainer.yml.example` to `portainer.yml`.
2. Fill in your registry username, Portainer URL/access key, device ids, and
   `CLEAR_ML_CONF_PATH`.
3. Run:

```bash
make server-doctor INSTANCE=gpu
make train-server INSTANCE=gpu SERVER_NOTES=test_safe_stack
```

The Docker image runs `rogii.gpu_preflight` before training, so a broken GPU
runtime fails immediately instead of after feature generation.

## Leakage Policy

Train horizontal files contain formation-top annotation columns such as `ANCC`,
`ASTNU`, and `BUDA`; public test horizontal files do not. Those columns are
teacher labels, not inference features.

Current defaults are test-safe:

- direct-path features ignore current-well formation columns unless
  `features.direct_path.allow_current_well_formations: true` is set explicitly;
- fold-safe spatial context excludes validation wells;
- `top_state_predictions` is allowed because it is a distilled OOF/student
  artifact trained from test-safe inputs (`MD/X/Y/Z/GR/TVT_input`).

The observed `sign(dANCC)` direction accuracy around `0.927` belongs in that
teacher-student lane: use formation annotations to train state models, then feed
only their OOF/test-safe predictions into stack or DP selection.

## Quality

```bash
make test
make check
make format
```

`make check` runs ruff and pytest. `make test TESTS=tests/test_path_features.py`
runs a focused subset.

## Kaggle Inference

After training writes model artifacts under `artifacts/stack`, build or run the
inference kernel:

```bash
make submit-dry
make submit MESSAGE="stack test-safe direct path"
```

Override model or output paths with `MODEL_DIR`, `CONFIG`, and `SUBMISSION`.

## Profiling

```bash
make profile PROFILE_NAME=stack
```

This writes a cProfile file, raw log, and Markdown report under
`artifacts/profiles/`.
