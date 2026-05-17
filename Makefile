COMPETITION := rogii-wellbore-geology-prediction
KAGGLE_USER ?= sleep3r
UV ?= uv
PYTHON ?= $(UV) run python
KAGGLE ?= $(UV) run kaggle

DATA_DIR ?= data
CONFIG ?= configs/stack.yml
SUBMISSION ?= submission.csv

KAGGLE_PACKAGE ?= artifacts/rogii_source.zip
KAGGLE_DATA_DIR ?= /kaggle/input/$(COMPETITION)
KERNEL_TIMEOUT ?= 32400
KERNEL_WAIT_TIMEOUT ?= 36000
KERNEL_POLL_INTERVAL ?= 60
MESSAGE ?= Submission

KERNEL ?= rogii-baseline-train
KERNEL_TITLE ?= ROGII Baseline Train
KERNEL_DIR ?= artifacts/kaggle_kernel
KERNEL_OUTPUT_DIR ?= artifacts/kaggle_output
KAGGLE_ARTIFACT_DIR ?= artifacts/kaggle_train

INFER_MODEL_DIR ?= artifacts/stack
INFER_ARTIFACT_DIR ?= artifacts/infer
INFER_KERNEL ?= rogii-baseline-infer
INFER_KERNEL_TITLE ?= ROGII Baseline Infer
INFER_KERNEL_DIR ?= artifacts/kaggle_infer_kernel
INFER_OUTPUT_DIR ?= artifacts/kaggle_infer_output
MODEL_DATASET ?= $(KAGGLE_USER)/rogii-baseline-artifacts
MODEL_DATASET_DIR ?= artifacts/kaggle_model_dataset

CODEX_HOME ?= $(HOME)/.codex
MINE_DB ?= .kaggle_mining/ideas.sqlite
MINE_WORK_DIR ?= .kaggle_mining/code
RESEARCH_BRIEF ?= .kaggle_mining/research_brief.md
CODE_MINER ?= $(CODEX_HOME)/skills/kaggle-code-miner/scripts/mine_kaggle_code.py
DISCUSSION_MINER ?= $(CODEX_HOME)/skills/kaggle-discussion-miner/scripts/mine_kaggle_discussions.py
BRIEF_BUILDER ?= $(CODEX_HOME)/skills/kaggle-research-brief/scripts/build_research_brief.py
CODE_PULL_LIMIT ?= 100
DISCUSSION_PAGES ?= 6
DISCUSSION_SORT ?= hot top new recent
DISCUSSION_MESSAGE_PAGE_SIZE ?= 500
BRIEF_MAX_IDEAS ?= 160

.PHONY: install-deps download-data unzip-data ensure-data train quick-train infer train-kaggle train-kaggle-dry status-kaggle logs-kaggle mine-code mine-discussions research-brief research-db package-kaggle prepare-kaggle-kernel prepare-kaggle-infer submit submit-dry submit-infer submit-infer-dry status-infer logs-infer format check

install-deps:
	$(UV) sync

download-data:
	mkdir -p $(DATA_DIR)
	$(KAGGLE) competitions download -c $(COMPETITION) -p $(DATA_DIR)

unzip-data: download-data
	unzip -q -o $(DATA_DIR)/$(COMPETITION).zip -d $(DATA_DIR)

ensure-data:
	@if [ ! -d "$(DATA_DIR)/train" ] || [ ! -d "$(DATA_DIR)/test" ]; then \
		$(MAKE) unzip-data; \
	fi

train: ensure-data
	$(PYTHON) -m rogii --config $(CONFIG)

quick-train:
	$(PYTHON) -m rogii --config configs/quick.yml

infer: ensure-data
	$(PYTHON) -m rogii.inference --model-dir $(INFER_MODEL_DIR) --output-dir $(INFER_ARTIFACT_DIR) --submission $(SUBMISSION)

train-kaggle:
	$(PYTHON) -m rogii.kaggle_submit run --competition $(COMPETITION) --user $(KAGGLE_USER) --kernel $(KERNEL) --title "$(KERNEL_TITLE)" --config $(CONFIG) --kernel-dir $(KERNEL_DIR) --data-dir $(KAGGLE_DATA_DIR) --artifact-dir $(KAGGLE_ARTIFACT_DIR) --output-dir $(KERNEL_OUTPUT_DIR) --submission-file $(SUBMISSION) --kernel-timeout $(KERNEL_TIMEOUT) --wait-timeout $(KERNEL_WAIT_TIMEOUT) --poll-interval $(KERNEL_POLL_INTERVAL) --message "$(MESSAGE)" --skip-competition-submit --download-all-output

train-kaggle-dry:
	$(PYTHON) -m rogii.kaggle_submit run --competition $(COMPETITION) --user $(KAGGLE_USER) --kernel $(KERNEL) --title "$(KERNEL_TITLE)" --config $(CONFIG) --kernel-dir $(KERNEL_DIR) --data-dir $(KAGGLE_DATA_DIR) --artifact-dir $(KAGGLE_ARTIFACT_DIR) --output-dir $(KERNEL_OUTPUT_DIR) --submission-file $(SUBMISSION) --kernel-timeout $(KERNEL_TIMEOUT) --wait-timeout $(KERNEL_WAIT_TIMEOUT) --poll-interval $(KERNEL_POLL_INTERVAL) --message "$(MESSAGE)" --skip-competition-submit --download-all-output --dry-run

status-kaggle:
	$(KAGGLE) kernels status $(KAGGLE_USER)/$(KERNEL)

logs-kaggle:
	$(KAGGLE) kernels logs $(KAGGLE_USER)/$(KERNEL)

mine-code:
	$(PYTHON) $(CODE_MINER) --competition $(COMPETITION) --db $(MINE_DB) --work-dir $(MINE_WORK_DIR) --pull-limit $(CODE_PULL_LIMIT) --kaggle-cmd "$(KAGGLE)"

mine-discussions:
	$(PYTHON) $(DISCUSSION_MINER) --competition $(COMPETITION) --db $(MINE_DB) --source mcp --pages $(DISCUSSION_PAGES) --sort-by $(DISCUSSION_SORT) --message-page-size $(DISCUSSION_MESSAGE_PAGE_SIZE)

research-brief:
	$(PYTHON) $(BRIEF_BUILDER) --db $(MINE_DB) --repo . --output $(RESEARCH_BRIEF) --max-all-ideas $(BRIEF_MAX_IDEAS)

research-db: mine-code mine-discussions research-brief

package-kaggle:
	$(PYTHON) -m rogii.kaggle_package --output $(KAGGLE_PACKAGE)

prepare-kaggle-kernel:
	$(PYTHON) -m rogii.kaggle_submit prepare --competition $(COMPETITION) --user $(KAGGLE_USER) --kernel $(KERNEL) --title "$(KERNEL_TITLE)" --config $(CONFIG) --kernel-dir $(KERNEL_DIR) --data-dir $(KAGGLE_DATA_DIR) --artifact-dir $(KAGGLE_ARTIFACT_DIR) --submission-file $(SUBMISSION)

prepare-kaggle-infer:
	$(PYTHON) -m rogii.kaggle_submit prepare --mode infer --competition $(COMPETITION) --user $(KAGGLE_USER) --kernel $(INFER_KERNEL) --title "$(INFER_KERNEL_TITLE)" --config $(CONFIG) --model-dir $(INFER_MODEL_DIR) --model-dataset $(MODEL_DATASET) --model-dataset-dir $(MODEL_DATASET_DIR) --kernel-dir $(INFER_KERNEL_DIR) --data-dir $(KAGGLE_DATA_DIR) --artifact-dir $(INFER_ARTIFACT_DIR) --submission-file $(SUBMISSION)

submit: submit-infer

submit-dry: submit-infer-dry

submit-infer:
	$(PYTHON) -m rogii.kaggle_submit run --mode infer --competition $(COMPETITION) --user $(KAGGLE_USER) --kernel $(INFER_KERNEL) --title "$(INFER_KERNEL_TITLE)" --config $(CONFIG) --model-dir $(INFER_MODEL_DIR) --model-dataset $(MODEL_DATASET) --model-dataset-dir $(MODEL_DATASET_DIR) --publish-model-dataset --kernel-dir $(INFER_KERNEL_DIR) --data-dir $(KAGGLE_DATA_DIR) --artifact-dir $(INFER_ARTIFACT_DIR) --output-dir $(INFER_OUTPUT_DIR) --submission-file $(SUBMISSION) --kernel-timeout $(KERNEL_TIMEOUT) --wait-timeout $(KERNEL_WAIT_TIMEOUT) --poll-interval $(KERNEL_POLL_INTERVAL) --message "$(MESSAGE)"

submit-infer-dry:
	$(PYTHON) -m rogii.kaggle_submit run --mode infer --competition $(COMPETITION) --user $(KAGGLE_USER) --kernel $(INFER_KERNEL) --title "$(INFER_KERNEL_TITLE)" --config $(CONFIG) --model-dir $(INFER_MODEL_DIR) --model-dataset $(MODEL_DATASET) --model-dataset-dir $(MODEL_DATASET_DIR) --kernel-dir $(INFER_KERNEL_DIR) --data-dir $(KAGGLE_DATA_DIR) --artifact-dir $(INFER_ARTIFACT_DIR) --output-dir $(INFER_OUTPUT_DIR) --submission-file $(SUBMISSION) --kernel-timeout $(KERNEL_TIMEOUT) --wait-timeout $(KERNEL_WAIT_TIMEOUT) --poll-interval $(KERNEL_POLL_INTERVAL) --message "$(MESSAGE)" --dry-run

status-infer:
	$(KAGGLE) kernels status $(KAGGLE_USER)/$(INFER_KERNEL)

logs-infer:
	$(KAGGLE) kernels logs $(KAGGLE_USER)/$(INFER_KERNEL)

format:
	$(PYTHON) -m ruff check --fix --select I .
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .

check:
	$(PYTHON) -m ruff check .
