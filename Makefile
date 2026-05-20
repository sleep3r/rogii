COMPETITION := rogii-wellbore-geology-prediction
KAGGLE_USER ?= sleep3r
SHELL := /bin/bash
UV ?= uv
PYTHON ?= $(UV) run python
KAGGLE ?= $(UV) run kaggle

DATA_DIR ?= data
CONFIG ?= configs/stack.yml
QUICK_CONFIG ?= configs/quick.yml
QUICK_TRAIN_CONFIG := $(if $(filter configs/stack.yml,$(CONFIG)),$(QUICK_CONFIG),$(CONFIG))
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
FEATURE_PROFILE_STAGE ?= true
FEATURE_PROFILE_STAGE_ARG := $(if $(filter false 0 no,$(FEATURE_PROFILE_STAGE)),--no-stage-profile,)
EXPERT_REPORT ?= artifacts/expert_report.md
EXPERT_REPORT_CSV ?= artifacts/expert_report.csv
EXPERT_REPORT_JSON ?= artifacts/expert_report.json
EXPERT_REPORT_TOP_N ?= 40
EXPERT_REPORT_MAX_WELLS ?=
EXPERT_REPORT_CONTEXT ?= fold-safe
EXPERT_REPORT_CONTEXT_ARG := $(if $(filter full full-context,$(EXPERT_REPORT_CONTEXT)),--full-context,)
EXPERT_REPORT_MAX_WELLS_ARG := $(if $(EXPERT_REPORT_MAX_WELLS),--max-wells $(EXPERT_REPORT_MAX_WELLS),)
PREDICTION_GUARD_ANCHOR ?= artifacts/cml_audit/ed4d9dc6c7cb479881f087fee1217253/submission.csv
PREDICTION_GUARD_CANDIDATE ?= $(SUBMISSION)
PREDICTION_GUARD_REPORT ?= artifacts/prediction_guard.md
PREDICTION_GUARD_JSON ?= artifacts/prediction_guard.json
PREDICTION_GUARD_MODE ?= strict
PREDICTION_GUARD_ALLOW_FAIL ?= false
PREDICTION_GUARD_ALLOW_FAIL_ARG := $(if $(filter true 1 yes,$(PREDICTION_GUARD_ALLOW_FAIL)),--allow-fail,)
DIRECT_SOLVER_OUTPUT ?= artifacts/direct_solver
DIRECT_SOLVER_ANCHOR ?= artifacts/cml_audit/ed4d9dc6c7cb479881f087fee1217253/submission.csv
DIRECT_SOLVER_VARIANT ?= stage12_raw
DIRECT_SOLVER_PROGRESS_INTERVAL ?= 5
DIRECT_SOLVER_MAX_WELLS ?=
DIRECT_SOLVER_MAX_WELLS_ARG := $(if $(DIRECT_SOLVER_MAX_WELLS),--max-wells $(DIRECT_SOLVER_MAX_WELLS),)
DIRECT_SOLVER_SAMPLE_SEED ?=
DIRECT_SOLVER_SAMPLE_SEED_ARG := $(if $(DIRECT_SOLVER_SAMPLE_SEED),--sample-seed $(DIRECT_SOLVER_SAMPLE_SEED),)
DIRECT_SOLVER_ANCHOR_ARG := $(if $(DIRECT_SOLVER_ANCHOR),--anchor-submission $(DIRECT_SOLVER_ANCHOR),)
DIRECT_SOLVER_PSEUDO_PUBLIC_TRIALS ?= 0
DIRECT_SOLVER_PSEUDO_PUBLIC_TRIPLE_SIZE ?= 3
DIRECT_SOLVER_PSEUDO_PUBLIC_ANCHOR ?= stage12_raw
DIRECT_SOLVER_PSEUDO_PUBLIC_MATCHED ?= true
DIRECT_SOLVER_PSEUDO_PUBLIC_CANDIDATE_K ?= 80
DIRECT_SOLVER_PSEUDO_PUBLIC_MATCHED_ARG := $(if $(filter true 1 yes,$(DIRECT_SOLVER_PSEUDO_PUBLIC_MATCHED)),--pseudo-public-matched --pseudo-public-candidate-k $(DIRECT_SOLVER_PSEUDO_PUBLIC_CANDIDATE_K),)
DIRECT_SOLVER_PSEUDO_PUBLIC_ARG := $(if $(filter-out 0,$(DIRECT_SOLVER_PSEUDO_PUBLIC_TRIALS)),--pseudo-public-trials $(DIRECT_SOLVER_PSEUDO_PUBLIC_TRIALS) --pseudo-public-triple-size $(DIRECT_SOLVER_PSEUDO_PUBLIC_TRIPLE_SIZE) --pseudo-public-anchor-variant $(DIRECT_SOLVER_PSEUDO_PUBLIC_ANCHOR) $(DIRECT_SOLVER_PSEUDO_PUBLIC_MATCHED_ARG),)
DIRECT_SOLVER_CROSS_WELL ?= true
DIRECT_SOLVER_CROSS_WELL_K ?= 8
DIRECT_SOLVER_CROSS_WELL_ARG := $(if $(filter true 1 yes,$(DIRECT_SOLVER_CROSS_WELL)),--cross-well-prior --cross-well-k $(DIRECT_SOLVER_CROSS_WELL_K),)
DIRECT_SOLVER_WORKERS ?= 1
DIRECT_SOLVER_WORKERS_ARG := --workers $(DIRECT_SOLVER_WORKERS)

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

CML_ID ?= 2b77f48a1a294304bf859ea798666d65
CML_SHORT := $(shell printf '%s' '$(CML_ID)' | cut -c1-12)
CML_MODEL_DIR ?= artifacts/clearml/$(CML_ID)
CML_ARG := $(if $(CML_ID),--clearml-task-id $(CML_ID),)
INFER_MODEL_DIR ?= $(CML_MODEL_DIR)
MODEL_DATASET ?= $(KAGGLE_USER)/rogii-baseline-artifacts$(if $(CML_SHORT),-$(CML_SHORT),)
MODEL_DATASET_DIR ?= artifacts/kaggle_model_dataset

INSTANCE ?=
SPACEBRIDGE_TASK ?= rogii
SERVER_NOTES ?= spacebridge_train
CLEARML_ENABLED ?= true
CLEARML_PROJECT ?= $(PROJECT_NAME)
CLEARML_TASK_NAME ?=
CLEARML_OUTPUT_URI ?= $(OUTPUT_URI)
CLEARML_TAGS ?= rogii,spacebridge,stack,gpu
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
BUNDLE_SOLUTION_FILES ?= pyproject.toml configs/stack.yml configs/stack_gpu.yml configs/quick.yml configs/quick_hmm.yml configs/stack_gpu_hmm.yml configs/direct_solver_policy.yml rogii/config.py rogii/clearml_data.py rogii/clearml_tracking.py rogii/features.py rogii/hmm_path.py rogii/direct_solver.py rogii/path_solver_extras.py rogii/path_features.py rogii/cross_well_prior.py rogii/top_signals.py rogii/spatial.py rogii/modeling.py rogii/pipeline.py rogii/submission.py rogii/inference.py rogii/kaggle_submit.py rogii/expert_report.py rogii/prediction_guard.py
BUNDLE_SOLUTION_ARGS := $(foreach file,$(BUNDLE_SOLUTION_FILES),--solution-file $(file))
BUNDLE_EXCLUDE ?= tests/
BUNDLE_EXCLUDE_ARGS := $(foreach item,$(BUNDLE_EXCLUDE),--exclude $(item))

.PHONY: install-deps download-data unzip-data ensure-data upload-clearml-data clearml-data-local-path fetch-clearml-model check-server-env train train-local train-server train-spacebridge quick-train expert-report prediction-guard direct-solver-train-eval direct-solver-test direct-solver-guard direct-solver-guard-bold direct-solver-use profile profile-quick profile-train profile-features profile-report train-kaggle train-kaggle-dry submit submit-dry status-train logs-train status-submit logs-submit mine-code mine-discussions research-brief best-public-solution model-bundle research-db format check

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

fetch-clearml-model:
	@test -n "$(CML_ID)" || (echo "Set CML_ID=<ClearML task id>" && exit 1)
	$(PYTHON) -m rogii.kaggle_submit fetch-clearml --cml-id $(CML_ID) --model-dir $(INFER_MODEL_DIR)

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
	$(PYTHON) -m rogii --config $(QUICK_TRAIN_CONFIG)

expert-report: ensure-data
	$(PYTHON) -m rogii.expert_report --config $(CONFIG) --output $(EXPERT_REPORT) --csv $(EXPERT_REPORT_CSV) --json $(EXPERT_REPORT_JSON) --top-n $(EXPERT_REPORT_TOP_N) $(EXPERT_REPORT_MAX_WELLS_ARG) $(EXPERT_REPORT_CONTEXT_ARG)

prediction-guard: ensure-data
	$(PYTHON) -m rogii.prediction_guard --candidate $(PREDICTION_GUARD_CANDIDATE) --anchor $(PREDICTION_GUARD_ANCHOR) --data-dir $(DATA_DIR) --output $(PREDICTION_GUARD_REPORT) --json $(PREDICTION_GUARD_JSON) --mode $(PREDICTION_GUARD_MODE) $(PREDICTION_GUARD_ALLOW_FAIL_ARG)

direct-solver-train-eval: ensure-data
	$(PYTHON) -m rogii.direct_solver --data-dir $(DATA_DIR) $(DIRECT_SOLVER_ANCHOR_ARG) --output-dir $(DIRECT_SOLVER_OUTPUT)/train_eval --progress-interval $(DIRECT_SOLVER_PROGRESS_INTERVAL) $(DIRECT_SOLVER_MAX_WELLS_ARG) $(DIRECT_SOLVER_SAMPLE_SEED_ARG) $(DIRECT_SOLVER_PSEUDO_PUBLIC_ARG) $(DIRECT_SOLVER_CROSS_WELL_ARG) $(DIRECT_SOLVER_WORKERS_ARG) --train-eval

direct-solver-test: ensure-data
	$(PYTHON) -m rogii.direct_solver --data-dir $(DATA_DIR) $(DIRECT_SOLVER_ANCHOR_ARG) --output-dir $(DIRECT_SOLVER_OUTPUT)/test --progress-interval $(DIRECT_SOLVER_PROGRESS_INTERVAL) $(DIRECT_SOLVER_MAX_WELLS_ARG) $(DIRECT_SOLVER_SAMPLE_SEED_ARG) $(DIRECT_SOLVER_CROSS_WELL_ARG) $(DIRECT_SOLVER_WORKERS_ARG)

direct-solver-guard: ensure-data
	$(MAKE) prediction-guard PREDICTION_GUARD_ANCHOR=$(DIRECT_SOLVER_ANCHOR) PREDICTION_GUARD_CANDIDATE=$(DIRECT_SOLVER_OUTPUT)/test/submission_direct_$(DIRECT_SOLVER_VARIANT).csv PREDICTION_GUARD_REPORT=$(DIRECT_SOLVER_OUTPUT)/prediction_guard_$(DIRECT_SOLVER_VARIANT).md PREDICTION_GUARD_JSON=$(DIRECT_SOLVER_OUTPUT)/prediction_guard_$(DIRECT_SOLVER_VARIANT).json PREDICTION_GUARD_MODE=$(PREDICTION_GUARD_MODE)

direct-solver-guard-bold: ensure-data
	$(MAKE) prediction-guard PREDICTION_GUARD_ANCHOR=$(DIRECT_SOLVER_ANCHOR) PREDICTION_GUARD_CANDIDATE=$(DIRECT_SOLVER_OUTPUT)/test/submission_direct_$(DIRECT_SOLVER_VARIANT).csv PREDICTION_GUARD_REPORT=$(DIRECT_SOLVER_OUTPUT)/prediction_guard_$(DIRECT_SOLVER_VARIANT)_bold.md PREDICTION_GUARD_JSON=$(DIRECT_SOLVER_OUTPUT)/prediction_guard_$(DIRECT_SOLVER_VARIANT)_bold.json PREDICTION_GUARD_MODE=bold

direct-solver-use:
	cp $(DIRECT_SOLVER_OUTPUT)/test/submission_direct_$(DIRECT_SOLVER_VARIANT).csv $(SUBMISSION)

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
	set -o pipefail; $(PYTHON) -m cProfile -o $(PROFILE_FILE) -m rogii.feature_profile --config $(CONFIG) --max-wells $(FEATURE_PROFILE_WELLS) --fold-id $(FEATURE_PROFILE_FOLD) --context $(FEATURE_PROFILE_CONTEXT) $(FEATURE_PROFILE_CACHE_ARG) $(FEATURE_PROFILE_STAGE_ARG) 2>&1 | tee $(PROFILE_LOG)
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
	$(PYTHON) -m rogii.kaggle_submit run --mode infer --competition $(COMPETITION) --user $(KAGGLE_USER) --kernel $(SUBMIT_KERNEL) --title "$(SUBMIT_TITLE)" --config $(CONFIG) --model-dir $(INFER_MODEL_DIR) $(CML_ARG) --model-dataset $(MODEL_DATASET) --model-dataset-dir $(MODEL_DATASET_DIR) --publish-model-dataset --kernel-dir $(SUBMIT_KERNEL_DIR) --data-dir $(KAGGLE_DATA_DIR) --artifact-dir $(SUBMIT_ARTIFACT_DIR) --output-dir $(SUBMIT_OUTPUT_DIR) --submission-file $(SUBMISSION) --kernel-timeout $(KERNEL_TIMEOUT) --wait-timeout $(KERNEL_WAIT_TIMEOUT) --poll-interval $(KERNEL_POLL_INTERVAL) --message "$(MESSAGE)" --skip-competition-submit --download-all-output

submit-dry:
	$(PYTHON) -m rogii.kaggle_submit run --mode infer --competition $(COMPETITION) --user $(KAGGLE_USER) --kernel $(SUBMIT_KERNEL) --title "$(SUBMIT_TITLE)" --config $(CONFIG) --model-dir $(INFER_MODEL_DIR) $(CML_ARG) --model-dataset $(MODEL_DATASET) --model-dataset-dir $(MODEL_DATASET_DIR) --kernel-dir $(SUBMIT_KERNEL_DIR) --data-dir $(KAGGLE_DATA_DIR) --artifact-dir $(SUBMIT_ARTIFACT_DIR) --output-dir $(SUBMIT_OUTPUT_DIR) --submission-file $(SUBMISSION) --kernel-timeout $(KERNEL_TIMEOUT) --wait-timeout $(KERNEL_WAIT_TIMEOUT) --poll-interval $(KERNEL_POLL_INTERVAL) --message "$(MESSAGE)" --skip-competition-submit --download-all-output --dry-run

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
