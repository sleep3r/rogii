COMPETITION := rogii-wellbore-geology-prediction
KAGGLE_USER ?= sleep3r
UV ?= uv
PYTHON ?= $(UV) run python
KAGGLE ?= $(UV) run kaggle

DATA_DIR ?= data
CONFIG ?= configs/stack.yml
SUBMISSION ?= submission.csv
MESSAGE ?= Submission

KAGGLE_DATA_DIR ?= /kaggle/input/$(COMPETITION)
KERNEL_TIMEOUT ?= 32400
KERNEL_WAIT_TIMEOUT ?= 36000
KERNEL_POLL_INTERVAL ?= 60

TRAIN_KERNEL ?= rogii-baseline-train
TRAIN_TITLE ?= ROGII Baseline Train
TRAIN_KERNEL_DIR ?= artifacts/kaggle_train_kernel
TRAIN_OUTPUT_DIR ?= artifacts/kaggle_train_output
TRAIN_ARTIFACT_DIR ?= artifacts/kaggle_train

SUBMIT_KERNEL ?= rogii-baseline-infer
SUBMIT_TITLE ?= ROGII Baseline Infer
SUBMIT_KERNEL_DIR ?= artifacts/kaggle_submit_kernel
SUBMIT_OUTPUT_DIR ?= artifacts/kaggle_submit_output
SUBMIT_ARTIFACT_DIR ?= artifacts/kaggle_submit

INFER_MODEL_DIR ?= artifacts/stack
MODEL_DATASET ?= $(KAGGLE_USER)/rogii-baseline-artifacts
MODEL_DATASET_DIR ?= artifacts/kaggle_model_dataset

CODEX_HOME ?= $(HOME)/.codex
MINE_DB ?= .kaggle_mining/ideas.sqlite
MINE_WORK_DIR ?= .kaggle_mining/code
RESEARCH_BRIEF ?= .kaggle_mining/research_brief.md
MODEL_BUNDLE ?= .kaggle_mining/model_bundle.md
OVERVIEW_DOC ?= .kaggle_mining/overview.tex
MANUAL_IDEAS_DOC ?= .kaggle_mining/manual_ideas.md
COMPETITION_DESC ?= COMPETITION.md
CODE_MINER ?= $(CODEX_HOME)/skills/kaggle-code-miner/scripts/mine_kaggle_code.py
DISCUSSION_MINER ?= $(CODEX_HOME)/skills/kaggle-discussion-miner/scripts/mine_kaggle_discussions.py
BRIEF_BUILDER ?= $(CODEX_HOME)/skills/kaggle-research-brief/scripts/build_research_brief.py
CODE_PULL_LIMIT ?= 100
DISCUSSION_PAGES ?= 6
DISCUSSION_SORT ?= hot top new recent
DISCUSSION_MESSAGE_PAGE_SIZE ?= 500
BRIEF_MAX_IDEAS ?= 160

.PHONY: install-deps download-data unzip-data ensure-data train train-local quick-train train-kaggle train-kaggle-dry submit submit-dry status-train logs-train status-submit logs-submit mine-code mine-discussions research-brief model-bundle research-db format check

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

train-local: ensure-data
	$(PYTHON) -m rogii --config $(CONFIG)

train: train-local

quick-train:
	$(PYTHON) -m rogii --config configs/quick.yml

train-kaggle:
	@echo "Kaggle CPU full training is disabled: stack.yml exceeds the 9h notebook limit."
	@echo "Use: make train-local"
	@echo "Then: make submit MESSAGE=\"...\""
	@exit 1

train-kaggle-dry:
	@echo "Kaggle CPU full training is disabled. Dry run skipped."

submit:
	$(PYTHON) -m rogii.kaggle_submit run --mode infer --competition $(COMPETITION) --user $(KAGGLE_USER) --kernel $(SUBMIT_KERNEL) --title "$(SUBMIT_TITLE)" --config $(CONFIG) --model-dir $(INFER_MODEL_DIR) --model-dataset $(MODEL_DATASET) --model-dataset-dir $(MODEL_DATASET_DIR) --publish-model-dataset --kernel-dir $(SUBMIT_KERNEL_DIR) --data-dir $(KAGGLE_DATA_DIR) --artifact-dir $(SUBMIT_ARTIFACT_DIR) --output-dir $(SUBMIT_OUTPUT_DIR) --submission-file $(SUBMISSION) --kernel-timeout $(KERNEL_TIMEOUT) --wait-timeout $(KERNEL_WAIT_TIMEOUT) --poll-interval $(KERNEL_POLL_INTERVAL) --message "$(MESSAGE)" --skip-competition-submit --download-all-output

submit-dry:
	$(PYTHON) -m rogii.kaggle_submit run --mode infer --competition $(COMPETITION) --user $(KAGGLE_USER) --kernel $(SUBMIT_KERNEL) --title "$(SUBMIT_TITLE)" --config $(CONFIG) --model-dir $(INFER_MODEL_DIR) --model-dataset $(MODEL_DATASET) --model-dataset-dir $(MODEL_DATASET_DIR) --kernel-dir $(SUBMIT_KERNEL_DIR) --data-dir $(KAGGLE_DATA_DIR) --artifact-dir $(SUBMIT_ARTIFACT_DIR) --output-dir $(SUBMIT_OUTPUT_DIR) --submission-file $(SUBMISSION) --kernel-timeout $(KERNEL_TIMEOUT) --wait-timeout $(KERNEL_WAIT_TIMEOUT) --poll-interval $(KERNEL_POLL_INTERVAL) --message "$(MESSAGE)" --skip-competition-submit --download-all-output --dry-run

status-train:
	$(KAGGLE) kernels status $(KAGGLE_USER)/$(TRAIN_KERNEL)

logs-train:
	$(KAGGLE) kernels logs $(KAGGLE_USER)/$(TRAIN_KERNEL)

status-submit:
	$(KAGGLE) kernels status $(KAGGLE_USER)/$(SUBMIT_KERNEL)

logs-submit:
	$(KAGGLE) kernels logs $(KAGGLE_USER)/$(SUBMIT_KERNEL)

mine-code:
	$(PYTHON) $(CODE_MINER) --competition $(COMPETITION) --db $(MINE_DB) --work-dir $(MINE_WORK_DIR) --pull-limit $(CODE_PULL_LIMIT) --kaggle-cmd "$(KAGGLE)"

mine-discussions:
	$(PYTHON) $(DISCUSSION_MINER) --competition $(COMPETITION) --db $(MINE_DB) --source mcp --pages $(DISCUSSION_PAGES) --sort-by $(DISCUSSION_SORT) --message-page-size $(DISCUSSION_MESSAGE_PAGE_SIZE)

research-brief:
	$(PYTHON) $(BRIEF_BUILDER) --db $(MINE_DB) --repo . --output $(RESEARCH_BRIEF) --competition-description-file $(COMPETITION_DESC) --max-all-ideas $(BRIEF_MAX_IDEAS)

model-bundle:
	$(PYTHON) $(BRIEF_BUILDER) --db $(MINE_DB) --repo . --output $(MODEL_BUNDLE) --competition-description-file $(COMPETITION_DESC) --context-file $(OVERVIEW_DOC) --context-file $(MANUAL_IDEAS_DOC) --max-all-ideas 0 --max-discussion-ideas 0

research-db: mine-code mine-discussions model-bundle

format:
	$(PYTHON) -m ruff check --fix --select I .
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .

check:
	$(PYTHON) -m ruff check .
