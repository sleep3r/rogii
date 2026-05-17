# ROGII - Wellbore Geology Prediction

Working repository for the Kaggle competition
`rogii-wellbore-geology-prediction`.

The goal is to predict `TVT` for hidden evaluation intervals in horizontal
wellbores. The repo currently contains a configurable gradient boosting
training pipeline and local Kaggle mining tools for tracking public
notebooks/discussions.

## Quick Start

```bash
make install-deps
make unzip-data
make quick-train
make train
```

The project uses `uv`, `.python-version`, `pyproject.toml`, and `uv.lock`.
Python and Kaggle commands are run through `uv run` by default, which avoids
stale global binaries.

Experiment history, tried ideas, and measured results are tracked in
`CHANGELOG.md`.

## Setup

Install `uv` if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create or update the environment:

```bash
make install-deps
```

Kaggle auth uses the new access-token flow:

```bash
mkdir -p ~/.kaggle
echo KGAT_xxx > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
```

Check that the project Kaggle CLI is working:

```bash
uv run kaggle --version
uv run kaggle competitions files -c rogii-wellbore-geology-prediction
```

## Data

Download the competition archive:

```bash
make download-data
```

Download and unpack into `data/`:

```bash
make unzip-data
```

`make train` depends on `ensure-data`, so it will unzip the archive if
`data/train` or `data/test` is missing.

## Training

Run the main training config:

```bash
make train
```

Equivalent command:

```bash
uv run python -m rogii --config configs/stack.yml
```

For a fast smoke-test on the small public sample:

```bash
make quick-train
```

For a Kaggle submission rerun, skip CV and final train-set prediction:

```bash
make train-submit
```

Equivalent command:

```bash
uv run python -m rogii --config configs/submit.yml
```

To run inference from an already trained local artifact:

```bash
make infer
```

Default model artifact:

```text
artifacts/stack/model.pkl
```

To run training remotely on Kaggle without submitting to the competition:

```bash
make train-kaggle MESSAGE="remote gbm stack train"
```

This pushes a separate private Kaggle script, `sleep3r/rogii-gbm-stack-train`,
waits for it to finish, downloads all outputs into
`artifacts/kaggle_train_output`, and stops before the competition submit step.
By default it uses `configs/stack.yml`, so it is meant for experiment-quality
remote validation.

Monitor the remote training kernel:

```bash
make status-kaggle-train
make logs-kaggle-train
```

Training pipeline:

- builds row-level features from trajectory, GR, `TVT_input`, and typewell logs;
- keeps `flat_tvt` as a physical anchor feature and can train residuals from
  either `flat_tvt` or `last_known_tvt`;
- adds public Kaggle top-solution alignment signals;
- trains a LightGBM/XGBoost/CatBoost residual stack;
- validates with grouped CV by well;
- tunes residual blend weight from OOF predictions;
- writes `submission.csv`;
- saves model, feature list, metrics, and config under `artifacts/`.

## Kaggle Public Top Signals

On 2026-05-17, public notebook mining found the strongest current signals in:

- https://www.kaggle.com/code/nihilisticneuralnet/9-251-rogii-wellbore-geology-prediction-dwt-based
- https://www.kaggle.com/code/romantamrazov/rogii-super-solution-lb-top-3
- https://www.kaggle.com/code/ravaghi/wellbore-geology-prediction-hill-climbing
- https://www.kaggle.com/code/ravaghi/wellbore-geology-prediction-lightgbm
- https://www.kaggle.com/code/cdeotte/xgb-starter-cv-15

These notebooks point to the same core idea: this is less a generic tabular
problem and more a GR/typewell alignment problem. The current `rogii` package
implements a portable version of that idea:

- beam-style matching of hidden horizontal-well GR against the typewell GR;
- multi-scale normalized cross-correlation anchors;
- low-resolution constrained DTW alignment;
- DWT/wavelet-smoothed alignment against the typewell;
- PF-style sequential consensus tracks over ANCC/Z candidate signals;
- spatial KNN priors from train formation surfaces (`ANCC`, `ASTNU`, `ASTNL`,
  `EGFDU`, `EGFDL`, `BUDA`);
- dense ANCC spatial prior sampled from train rows;
- consensus features measuring disagreement between alignment signals.

The current default model is a residual stack over LightGBM, XGBoost, and
CatBoost with Ridge blending; the stack also supports convex hill-climb
blending via `model.blend.method: hill_climb`. `configs/hgb.yml` is kept as a
legacy baseline for comparison.

`configs/stack.yml` also has a small 9.251-style postprocess search: CV can
choose whether to blend the model delta with the `kg_pf_ancc_tvt` delta. The
candidate list always includes a no-op candidate, so this is a controlled knob
rather than a forced correction.

The feature block is controlled by:

```yaml
features:
  prediction_baseline: last_known_tvt
  include_kaggle_top_signals: true
  kaggle_top:
    beam_configs:
      - [20.0, 144.0, 2, cons]
      - [8.0, 64.0, 2, loose]
      - [35.0, 220.0, 1, vcons]
      - [14.0, 90.0, 5, sm5]
      - [4.0, 36.0, 3, vloose]
      - [12.0, 100.0, 3, mid]
      - [25.0, 180.0, 2, stiff]
    ncc_windows: [8, 15, 25]
    dtw_enabled: true
    dwt_enabled: true
    particle_enabled: true
```

Latest smoke-test on the public sample:

```text
Baseline: last_known_tvt
Baseline RMSE: 11.53934
Legacy HGB CV RMSE: 10.11490
GBM stack CV RMSE:  9.63298
Best stack residual_weight: 1.0
Best notebook blend: {alpha: 1.0, tau: 0.0, w_pf: 0.0}
Features: 150
```

Latest full local validations on visible train wells:

```text
Legacy HGB, 150-well CV:
  Features: 122
  CV RMSE:  16.63554
  Flat CV:  19.06126

GBM stack, all-well CV:
  Rows:     3,783,989
  Wells:    773
  Features: 131
  CV RMSE:  13.50285
  Flat CV:  17.50671
  Best residual_weight: 0.75
  Fold RMSE range: 11.50514-15.36182
```

The full GBM stack numbers above are from the previous `flat_tvt` residual
baseline. After the 9.251-notebook alignment update, `configs/stack.yml` now
uses `last_known_tvt`; rerun `make train` before treating full-CV numbers as
current.

Public LB anchors:

```text
Legacy HGB submit: 12.803
GBM stack submit:  13.033
```

The stack improved local CV but worsened public LB, so current validation is
not reliable enough for model ranking.

## Runtime Notes

The 9 hour number is the Kaggle code-competition notebook limit, not the
expected inference time. Locally, the legacy full visible HGB run with CV took
about 6.5 minutes, while the all-well GBM stack CV took about 38 minutes.

```text
Spatial context:       0:04
Training feature build 3:44
CV:                    1:25
Final model:           1:17
Test sample predict:   0:01
```

For a real Kaggle submission, the visible train set is still the same, but the
local 3-well public `test/` sample is replaced by the hidden test set. The
hidden inference step will therefore be longer than local `test_wells=3`, but
the training feature build is still the main cost in the current pipeline.

Use `configs/submit.yml` or `make train-submit` for submission notebooks. It
inherits `configs/stack.yml`, disables CV, and skips final train RMSE
prediction, so the rerun spends time on fitting and hidden-test prediction
instead of local diagnostics. The stack is expected to be heavier than HGB
because it trains LightGBM, XGBoost, and CatBoost.

For faster reruns after a local full train, use inference-only submit:

```bash
make submit-infer MESSAGE="gbm stack inference"
```

This publishes `artifacts/stack/model.pkl`, `features.json`, and `metrics.json`
to a private Kaggle Dataset, attaches that dataset to the inference kernel, and
runs `python -m rogii.inference` on Kaggle. It still builds the train-derived
spatial context and hidden-test features, but skips the full train feature
table and model fitting. On the local public sample, inference-only reproduced
the full train `submission.csv` byte-for-byte.

Default model dataset:

```text
sleep3r/rogii-stack-artifacts
```

By default inference-only submit pushes a new version of the existing
`sleep3r/rogii-gbm-stack-submit` Kaggle kernel. This avoids Kaggle API failures
when creating a brand-new kernel slug with the current access token.

Latest local submission-mode run:

```text
make train-submit
Total duration: not remeasured after stack switch
CV: disabled
Final train RMSE: skipped
Output artifacts: artifacts/submit_stack
```

Configs:

- `configs/stack.yml` - main GBM stack training run;
- `configs/hgb.yml` - legacy sklearn HGB baseline;
- `configs/quick.yml` - small public-sample smoke-test;
- `configs/submit.yml` - no-CV stack submission rerun config;
- `configs/best.yml` - the current best experiment pointer used by
  `make submit`.

Override the config:

```bash
make train CONFIG=configs/quick.yml
```

## Kaggle Research DB

Two local Codex skills were added for mining competition knowledge into SQLite:

- `kaggle-code-miner` - scans Kaggle notebooks/kernels;
- `kaggle-discussion-miner` - scans Kaggle topics/comments via the official MCP endpoint.

Default database:

```text
.kaggle_mining/ideas.sqlite
```

One-command refresh:

```bash
make research-db
```

That updates public notebooks, Kaggle discussions, and the Markdown brief.

Mine public notebooks only:

```bash
make mine-code
```

Mine discussions only:

```bash
make mine-discussions
```

Discussion mining uses Kaggle's official remote MCP endpoint
`https://www.kaggle.com/mcp`, specifically `list_forum_topics` and
`get_forum_topic`.

Build the Markdown brief from the current DB:

```bash
make research-brief
```

Useful query:

```bash
sqlite3 .kaggle_mining/ideas.sqlite \
  "select source_type, source_ref, idea_type, summary, score from ideas order by score desc limit 30;"
```

## Submit

This is a Kaggle code competition, so the real submission must reference a
Kaggle kernel version. The end-to-end path is:

```bash
make submit MESSAGE="gbm stack top-signal model"
```

That command:

- builds a self-contained Kaggle `run.py` under `artifacts/kaggle_kernel`;
- embeds the current `rogii/` package and `configs/` into that script;
- pushes `sleep3r/rogii-gbm-stack-submit` as a private Kaggle script;
- waits for the Kaggle run to finish;
- downloads and validates `submission.csv`;
- submits that kernel version to the competition.

The submit runner resolves Kaggle data robustly: it searches the configured
input path, nested `/kaggle/input` folders, and zipped competition archives
before starting the training pipeline.

By default it uses `configs/best.yml`, which currently inherits
`configs/submit.yml`. To try another experiment without editing files:

```bash
make submit BEST_CONFIG=configs/my_experiment.yml KERNEL=rogii-my-experiment MESSAGE="my experiment"
```

To mark an experiment as the default best, update `configs/best.yml`:

```yaml
inherits: my_experiment.yml
```

Dry-run the packaging step without pushing or submitting:

```bash
make submit-kaggle-dry
```

Monitor the submit kernel:

```bash
make status-submit
make logs-submit
```

The faster inference-only path is:

```bash
make submit-infer MESSAGE="gbm stack inference"
```

Dry-run inference packaging:

```bash
make submit-infer-dry
```

Monitor the inference kernel:

```bash
make status-infer
make logs-infer
```

Prepare only the Kaggle kernel workspace:

```bash
make prepare-kaggle-kernel
```

Manual package path for attaching the current source tree to a hand-made
Kaggle notebook/dataset:

```bash
make package-kaggle
```

This writes:

```text
artifacts/rogii_source.zip
```

Inside a Kaggle notebook, attach/unzip that source package and run:

```bash
python -m rogii \
  --config configs/submit.yml \
  --data-dir /kaggle/input/rogii-wellbore-geology-prediction
```

The notebook must write `submission.csv` in its working directory.

```bash
make submit-version NOTEBOOK=<NOTEBOOK> VERSION=<VERSION> MESSAGE="Message"
```

Example:

```bash
make submit-version NOTEBOOK=rogii-gbm-stack-submit VERSION=3 MESSAGE="gbm stack top-signal model"
```

Expands to:

```bash
uv run kaggle competitions submit \
  -c rogii-wellbore-geology-prediction \
  -f submission.csv \
  -k sleep3r/rogii-gbm-stack-submit \
  -v 3 \
  -m "gbm stack top-signal model"
```

## Ignored Outputs

The following are intentionally ignored:

- `.venv/`
- `data/`
- `artifacts/`
- `.kaggle_mining/`
- `submission.csv`

## Project Layout

```text
.
├── .python-version
├── CHANGELOG.md
├── Makefile
├── README.md
├── pyproject.toml
├── uv.lock
├── configs/
│   ├── hgb.yml
│   ├── quick.yml
│   ├── stack.yml
│   ├── best.yml
│   └── submit.yml
├── rogii/
│   ├── __main__.py     # package CLI entrypoint
│   ├── kaggle_package.py # source zip packager
│   ├── kaggle_submit.py # end-to-end Kaggle kernel submitter
│   ├── config.py       # config defaults and YAML loading
│   ├── constants.py    # formation names and Kaggle paths
│   ├── features.py     # well-level tabular feature assembly
│   ├── inference.py    # inference-only runner for saved model artifacts
│   ├── io.py           # data path discovery and well file helpers
│   ├── modeling.py     # model factory, CV, residual postprocess
│   ├── numeric.py      # numeric helpers and flat TVT baseline
│   ├── pipeline.py     # train orchestration
│   ├── runlog.py       # status/timing logger
│   ├── spatial.py      # spatial KNN/ANCC priors
│   ├── submission.py   # test prediction and artifact writing
│   └── top_signals.py  # beam/NCC/DTW/DWT/PF public-solution signals
└── data/                 # ignored
```
