COMPETITION := rogii-wellbore-geology-prediction
KAGGLE_USER ?= sleep3r
SHELL := /bin/bash
UV ?= uv
PYTHON ?= $(UV) run python
KAGGLE ?= $(UV) run kaggle

DATA_DIR ?= data
CONFIG ?= configs/stack.yml
SUBMISSION ?= submission.csv
MESSAGE ?= Submission
PROJECT_NAME ?= ROGII/Wellbore
OUTPUT_URI ?= s3://s3-basket-cold.wb.ru/ds-experiments
PROFILE_DIR ?= artifacts/profiles
PROFILE_NAME ?= profile
PROFILE_FILE ?= $(PROFILE_DIR)/$(PROFILE_NAME).prof
PROFILE_LOG ?= $(PROFILE_DIR)/$(PROFILE_NAME).log
PROFILE_REPORT ?= $(PROFILE_DIR)/$(PROFILE_NAME).md
PROFILE_REPORT_LIMIT ?= 40
FEATURE_PROFILE_WELLS ?= 50
FEATURE_PROFILE_FOLD ?= 1
FEATURE_PROFILE_CONTEXT ?= fold-train
FEATURE_PROFILE_DISABLE_CACHE ?= true
FEATURE_PROFILE_CACHE_ARG := $(if $(filter true 1 yes,$(FEATURE_PROFILE_DISABLE_CACHE)),--disable-cache,)

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

INSTANCE ?=
SPACEBRIDGE_TASK ?= rogii
SERVER_NOTES ?= spacebridge_train
CLEARML_ENABLED ?= true
CLEARML_PROJECT ?= $(PROJECT_NAME)
CLEARML_TASK_NAME ?=
CLEARML_OUTPUT_URI ?= $(OUTPUT_URI)
CLEARML_TAGS ?= rogii,spacebridge,stack
CLEARML_LOG_MODEL ?= true
CLEARML_DATA_PROJECT ?= $(PROJECT_NAME)
CLEARML_DATA_NAME ?= rogii-wellbore-geology-prediction
CLEARML_DATA_VERSION ?= 20260519_s3
CLEARML_DATA_OUTPUT_URI ?= $(OUTPUT_URI)
CLEARML_DATA_CACHE_DIR ?= ~/.cache/clearml/rogii
CLEARML_DATA_MAX_WORKERS ?= 8
SPACEBRIDGE_CMD_ARGS ?= config=$(CONFIG) clearml_enabled=$(CLEARML_ENABLED) clearml_project=$(CLEARML_PROJECT) clearml_task_name=$(CLEARML_TASK_NAME) clearml_output_uri=$(CLEARML_OUTPUT_URI) clearml_tags=$(CLEARML_TAGS) clearml_log_model=$(CLEARML_LOG_MODEL) data_clearml_enabled=true data_clearml_project=$(CLEARML_DATA_PROJECT) data_clearml_name=$(CLEARML_DATA_NAME) data_clearml_version=$(CLEARML_DATA_VERSION) data_clearml_cache_dir=$(CLEARML_DATA_CACHE_DIR) notes=$(SERVER_NOTES)

CODEX_HOME ?= $(HOME)/.codex
MINE_DB ?= .kaggle_mining/ideas.sqlite
MINE_WORK_DIR ?= .kaggle_mining/code
RESEARCH_BRIEF ?= .kaggle_mining/research_brief.md
MODEL_BUNDLE ?= .kaggle_mining/model_bundle.md
MANUAL_IDEAS_DOC ?= .kaggle_mining/manual_ideas.md
PROFILE_DOC ?= PROFILING.md
BEST_PUBLIC_SOLUTION_DOC ?= .kaggle_mining/best_public_solution.md
COMPETITION_DESC ?= COMPETITION.md
CODE_MINER ?= $(CODEX_HOME)/skills/kaggle-code-miner/scripts/mine_kaggle_code.py
DISCUSSION_MINER ?= $(CODEX_HOME)/skills/kaggle-discussion-miner/scripts/mine_kaggle_discussions.py
BRIEF_BUILDER ?= $(CODEX_HOME)/skills/kaggle-research-brief/scripts/build_research_brief.py
CODE_PULL_LIMIT ?= 100
DISCUSSION_PAGES ?= 6
DISCUSSION_SORT ?= hot top new recent
DISCUSSION_MESSAGE_PAGE_SIZE ?= 500
BRIEF_MAX_IDEAS ?= 160
BUNDLE_SOLUTION_FILES ?= pyproject.toml configs/stack.yml configs/quick.yml rogii/config.py rogii/clearml_data.py rogii/clearml_tracking.py rogii/features.py rogii/top_signals.py rogii/spatial.py rogii/modeling.py rogii/pipeline.py rogii/submission.py rogii/inference.py rogii/kaggle_submit.py
BUNDLE_SOLUTION_ARGS := $(foreach file,$(BUNDLE_SOLUTION_FILES),--solution-file $(file))
BUNDLE_EXCLUDE ?= tests/
BUNDLE_EXCLUDE_ARGS := $(foreach item,$(BUNDLE_EXCLUDE),--exclude $(item))

.PHONY: install-deps download-data unzip-data ensure-data upload-clearml-data clearml-data-local-path check-server-env train train-local train-server train-spacebridge quick-train profile profile-quick profile-train profile-features profile-report train-kaggle train-kaggle-dry submit submit-dry status-train logs-train status-submit logs-submit mine-code mine-discussions research-brief best-public-solution model-bundle research-db format check

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

upload-clearml-data: ensure-data
	$(PYTHON) -m rogii.clearml_data upload --data-dir $(DATA_DIR) --project "$(CLEARML_DATA_PROJECT)" --name "$(CLEARML_DATA_NAME)" --version "$(CLEARML_DATA_VERSION)" --output-uri "$(CLEARML_DATA_OUTPUT_URI)" --require-output-uri --s3-only --max-workers $(CLEARML_DATA_MAX_WORKERS)

clearml-data-local-path:
	$(PYTHON) -m rogii.clearml_data download --project "$(CLEARML_DATA_PROJECT)" --name "$(CLEARML_DATA_NAME)" --version "$(CLEARML_DATA_VERSION)" --cache-dir "$(CLEARML_DATA_CACHE_DIR)"

check-server-env:
	docker version
	docker context show
	docker buildx version
	$(PYTHON) -m rogii.spacebridge_preflight $(if $(INSTANCE),--instance "$(INSTANCE)",)

train-local: ensure-data
	$(PYTHON) -m rogii --config $(CONFIG)

train: train-local

train-server train-spacebridge: check-server-env
	@test -n "$(INSTANCE)" || (echo "Set INSTANCE=<portainer instance from portainer.yml>" && exit 1)
	$(UV) run spacebridge train $(SPACEBRIDGE_TASK) --instance $(INSTANCE) --cmd-args "$(SPACEBRIDGE_CMD_ARGS)"

quick-train:
	$(PYTHON) -m rogii --config configs/quick.yml

profile: ensure-data
	mkdir -p $(PROFILE_DIR)
	set -o pipefail; $(PYTHON) -m cProfile -o $(PROFILE_FILE) -m rogii --config $(CONFIG) 2>&1 | tee $(PROFILE_LOG)
	$(PYTHON) -m rogii.profile_report --profile $(PROFILE_FILE) --log $(PROFILE_LOG) --output $(PROFILE_REPORT) --limit $(PROFILE_REPORT_LIMIT) --title "ROGII profile: $(PROFILE_NAME)"

profile-quick:
	$(MAKE) profile CONFIG=configs/quick.yml PROFILE_NAME=quick

profile-train:
	$(MAKE) profile CONFIG=$(CONFIG) PROFILE_NAME=stack

profile-features: ensure-data
	mkdir -p $(PROFILE_DIR)
	set -o pipefail; $(PYTHON) -m cProfile -o $(PROFILE_FILE) -m rogii.feature_profile --config $(CONFIG) --max-wells $(FEATURE_PROFILE_WELLS) --fold-id $(FEATURE_PROFILE_FOLD) --context $(FEATURE_PROFILE_CONTEXT) $(FEATURE_PROFILE_CACHE_ARG) 2>&1 | tee $(PROFILE_LOG)
	$(PYTHON) -m rogii.profile_report --profile $(PROFILE_FILE) --log $(PROFILE_LOG) --output $(PROFILE_REPORT) --limit $(PROFILE_REPORT_LIMIT) --title "ROGII feature profile: $(PROFILE_NAME)"

profile-report:
	$(PYTHON) -m rogii.profile_report --profile $(PROFILE_FILE) --log $(PROFILE_LOG) --output $(PROFILE_REPORT) --limit $(PROFILE_REPORT_LIMIT) --title "ROGII profile: $(PROFILE_NAME)"

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

best-public-solution:
	$(PYTHON) -m rogii.best_public_solution --db $(MINE_DB) --work-dir .kaggle_mining --output $(BEST_PUBLIC_SOLUTION_DOC)

model-bundle: best-public-solution
	$(PYTHON) $(BRIEF_BUILDER) --db $(MINE_DB) --repo . --output $(MODEL_BUNDLE) --competition-description-file $(COMPETITION_DESC) --context-file $(MANUAL_IDEAS_DOC) --context-file $(PROFILE_DOC) --context-file $(BEST_PUBLIC_SOLUTION_DOC) $(BUNDLE_SOLUTION_ARGS) --max-all-ideas 0 --max-discussion-ideas 0
	$(PYTHON) -m rogii.filter_model_bundle --path $(MODEL_BUNDLE) $(BUNDLE_EXCLUDE_ARGS)

research-db: mine-code mine-discussions model-bundle

format:
	$(PYTHON) -m ruff check --fix --select I .
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .

check:
	$(PYTHON) -m ruff check .
