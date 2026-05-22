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
DRIFT_NCC_CONFIG ?= configs/drift_ncc.yml
DRIFT_NCC_QUICK_CONFIG ?= configs/drift_ncc_quick.yml
FORMATION_PLANE_OUTPUT ?= artifacts/formation_plane_knn
FORMATION_PLANE_QUICK_OUTPUT ?= artifacts/formation_plane_knn_quick
FORMATION_PLANE_BOOTSTRAP ?= 64
FORMATION_PLANE_QUICK_BOOTSTRAP ?= 4
FORMATION_SELECTOR_INPUT ?= $(FORMATION_PLANE_OUTPUT)/oof_candidates.parquet
FORMATION_SELECTOR_QUICK_INPUT ?= $(FORMATION_PLANE_QUICK_OUTPUT)/oof_candidates.parquet
FORMATION_SELECTOR_OUTPUT ?= artifacts/formation_selector
FORMATION_SELECTOR_QUICK_OUTPUT ?= artifacts/formation_selector_quick
FORMATION_SELECTOR_SCHEMA10 ?=
FORMATION_SELECTOR_SCHEMA10_ARG := $(if $(FORMATION_SELECTOR_SCHEMA10),--schema10-oof $(FORMATION_SELECTOR_SCHEMA10),)
FORMATION_B_LITE_INPUT ?= $(FORMATION_PLANE_OUTPUT)/oof_candidates.parquet
FORMATION_B_LITE_QUICK_INPUT ?= $(FORMATION_PLANE_QUICK_OUTPUT)/oof_candidates.parquet
FORMATION_B_LITE_OUTPUT ?= artifacts/formation_b_lite
FORMATION_B_LITE_QUICK_OUTPUT ?= artifacts/formation_b_lite_quick
FORMATION_B_LITE_SCHEMA10 ?=
FORMATION_B_LITE_SCHEMA10_ARG := $(if $(FORMATION_B_LITE_SCHEMA10),--schema10-oof $(FORMATION_B_LITE_SCHEMA10),)
FORMATION_B_LITE_PROGRESS_INTERVAL ?= 25
FORMATION_B2_INPUT ?= $(FORMATION_PLANE_OUTPUT)/oof_candidates.parquet
FORMATION_B2_QUICK_INPUT ?= $(FORMATION_PLANE_QUICK_OUTPUT)/oof_candidates.parquet
FORMATION_B2_B_SCORES ?= $(FORMATION_B_LITE_OUTPUT)/b_candidate_scores.parquet
FORMATION_B2_QUICK_B_SCORES ?= $(FORMATION_B_LITE_QUICK_OUTPUT)/b_candidate_scores.parquet
FORMATION_B2_OUTPUT ?= artifacts/formation_b2_constrained
FORMATION_B2_QUICK_OUTPUT ?= artifacts/formation_b2_constrained_quick
FORMATION_B2_SCHEMA10 ?=
FORMATION_B2_SCHEMA10_COLUMN ?=
FORMATION_B2_SCHEMA10_COLUMN_ARG := $(if $(FORMATION_B2_SCHEMA10_COLUMN),--schema10-column $(FORMATION_B2_SCHEMA10_COLUMN),)
FORMATION_B2_SCHEMA10_ARG := $(if $(FORMATION_B2_SCHEMA10),--schema10-oof $(FORMATION_B2_SCHEMA10) $(FORMATION_B2_SCHEMA10_COLUMN_ARG),)
FORMATION_B2_PROGRESS_INTERVAL ?= 25
FORMATION_B2_GUARDED_INPUT ?= $(FORMATION_B2_INPUT)
FORMATION_B2_GUARDED_CHOICES ?= $(FORMATION_B2_OUTPUT)/b2_selector_choices.csv
FORMATION_B2_GUARDED_METADATA ?= $(FORMATION_B2_OUTPUT)/b2_candidate_metadata.parquet
FORMATION_B2_GUARDED_OUTPUT ?= artifacts/formation_b2_guarded
FORMATION_B2_GUARDED_SCHEMA10 ?= $(FORMATION_B2_SCHEMA10)
FORMATION_B2_GUARDED_SCHEMA10_COLUMN ?= $(FORMATION_B2_SCHEMA10_COLUMN)
FORMATION_B2_GUARDED_SELECTORS ?= A_among_B_top10,B_among_A_top10,A_among_B_top20,B_among_A_top20,rank_0_7A_0_3B,rank_A_plus_B
FORMATION_B2_GUARDED_CV_POLICY_LIMIT ?= 40
FORMATION_B2_GUARDED_SCHEMA10_ARG := $(if $(FORMATION_B2_GUARDED_SCHEMA10),--schema10-oof $(FORMATION_B2_GUARDED_SCHEMA10),)
FORMATION_B2_GUARDED_SCHEMA10_COLUMN_ARG := $(if $(FORMATION_B2_GUARDED_SCHEMA10_COLUMN),--schema10-column $(FORMATION_B2_GUARDED_SCHEMA10_COLUMN),)
FORMATION_B2_CONFIG ?= configs/formation_b2_guarded_submit.yml
FORMATION_B2_INFER_INPUT ?= $(FORMATION_B2_GUARDED_INPUT)
FORMATION_B2_INFER_CHOICES ?= $(FORMATION_B2_GUARDED_CHOICES)
FORMATION_B2_INFER_METADATA ?= $(FORMATION_B2_GUARDED_METADATA)
FORMATION_B2_INFER_OUTPUT ?= artifacts/formation_b2_infer_oof_replay
FORMATION_B2_INFER_SCHEMA10 ?= $(FORMATION_B2_GUARDED_SCHEMA10)
FORMATION_B2_INFER_SCHEMA10_COLUMN ?= $(FORMATION_B2_GUARDED_SCHEMA10_COLUMN)
FORMATION_B2_INFER_REFERENCE ?= $(FORMATION_B2_GUARDED_OUTPUT)/guarded_predictions.parquet
FORMATION_B2_INFER_REFERENCE_COLUMN ?= b2_guarded_submit
FORMATION_B2_INFER_TOLERANCE ?= 1e-5
FORMATION_B2_INFER_SCHEMA10_COLUMN_ARG := $(if $(FORMATION_B2_INFER_SCHEMA10_COLUMN),--schema10-column $(FORMATION_B2_INFER_SCHEMA10_COLUMN),)
FORMATION_B2_INFER_REFERENCE_ARG := $(if $(FORMATION_B2_INFER_REFERENCE),--reference $(FORMATION_B2_INFER_REFERENCE) --reference-column $(FORMATION_B2_INFER_REFERENCE_COLUMN),)
FORMATION_B2_TEST_OUTPUT ?= artifacts/formation_b2_test_submit
FORMATION_B2_TEST_BASE_SUBMISSION ?= artifacts/kaggle_submit_output/submission.csv
FORMATION_B2_TEST_BASE_COLUMN ?= tvt
FORMATION_B2_TEST_PROGRESS_INTERVAL ?= 1
OOF_BASELINE_MODEL_DIR ?= artifacts/clearml/c11ac4df327f49f3b91ad293c69bc91e
OOF_BASELINE_OUTPUT ?= artifacts/oof_baseline/schema10_oof.parquet
OOF_BASELINE_NUM_WORKERS ?= 1
OOF_BASELINE_PROGRESS_INTERVAL ?= 25
SUBMISSION ?= submission.csv
MESSAGE ?= Submission
PROJECT_NAME ?= ROGII/Wellbore
B2_SUBMIT_CONFIG ?= configs/formation_b2_guarded_submit.yml
B2_SUBMIT_ARG := $(if $(B2_SUBMIT_CONFIG),--b2-config $(B2_SUBMIT_CONFIG),)
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

CML_ID ?= c11ac4df327f49f3b91ad293c69bc91e
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
BUNDLE_SOLUTION_FILES ?= pyproject.toml configs/stack.yml configs/stack_gpu.yml configs/quick.yml configs/quick_hmm.yml configs/drift_ncc.yml configs/drift_ncc_quick.yml configs/stack_gpu_hmm.yml configs/direct_solver_policy.yml configs/formation_b2_guarded_submit.yml rogii/config.py rogii/clearml_data.py rogii/clearml_tracking.py rogii/features.py rogii/hmm_path.py rogii/direct_solver.py rogii/path_solver_extras.py rogii/path_features.py rogii/cross_well_prior.py rogii/formation_plane_knn.py rogii/formation_selector.py rogii/formation_b_lite.py rogii/formation_b2_constrained.py rogii/formation_b2_guarded.py rogii/formation_b2_inference.py rogii/export_oof_baseline.py rogii/top_signals.py rogii/spatial.py rogii/modeling.py rogii/pipeline.py rogii/submission.py rogii/inference.py rogii/kaggle_submit.py rogii/expert_report.py rogii/prediction_guard.py
BUNDLE_SOLUTION_ARGS := $(foreach file,$(BUNDLE_SOLUTION_FILES),--solution-file $(file))
BUNDLE_EXCLUDE ?= tests/
BUNDLE_EXCLUDE_ARGS := $(foreach item,$(BUNDLE_EXCLUDE),--exclude $(item))

.PHONY: install-deps download-data unzip-data ensure-data upload-clearml-data clearml-data-local-path fetch-clearml-model check-server-env train train-local train-server train-spacebridge quick-train drift-ncc-train drift-ncc-quick formation-plane-knn formation-plane-knn-quick formation-selector formation-selector-quick formation-b-lite formation-b-lite-quick formation-b2 formation-b2-quick formation-b2-guarded formation-b2-infer-oof-replay formation-b2-infer-test export-oof-baseline expert-report prediction-guard direct-solver-train-eval direct-solver-test direct-solver-guard direct-solver-guard-bold direct-solver-use profile profile-quick profile-train profile-features profile-report train-kaggle train-kaggle-dry submit submit-dry status-train logs-train status-submit logs-submit mine-code mine-discussions research-brief best-public-solution model-bundle research-db format check

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

drift-ncc-train: ensure-data
	$(PYTHON) -m rogii --config $(DRIFT_NCC_CONFIG)

drift-ncc-quick:
	$(PYTHON) -m rogii --config $(DRIFT_NCC_QUICK_CONFIG)

formation-plane-knn: ensure-data
	$(PYTHON) -m rogii.formation_plane_knn --data-dir $(DATA_DIR) --output-dir $(FORMATION_PLANE_OUTPUT) --bootstrap-samples $(FORMATION_PLANE_BOOTSTRAP)

formation-plane-knn-quick:
	$(PYTHON) -m rogii.formation_plane_knn --data-dir $(DATA_DIR) --train-dir $(DATA_DIR)/public_train --output-dir $(FORMATION_PLANE_QUICK_OUTPUT) --n-splits 3 --bootstrap-samples $(FORMATION_PLANE_QUICK_BOOTSTRAP) --sample-rows-per-well 30 --min-points 20 --dense-k 30

formation-selector:
	$(PYTHON) -m rogii.formation_selector --input $(FORMATION_SELECTOR_INPUT) --output-dir $(FORMATION_SELECTOR_OUTPUT) $(FORMATION_SELECTOR_SCHEMA10_ARG)

formation-selector-quick:
	$(PYTHON) -m rogii.formation_selector --input $(FORMATION_SELECTOR_QUICK_INPUT) --output-dir $(FORMATION_SELECTOR_QUICK_OUTPUT) $(FORMATION_SELECTOR_SCHEMA10_ARG)

formation-b-lite:
	$(PYTHON) -m rogii.formation_b_lite --input $(FORMATION_B_LITE_INPUT) --data-dir $(DATA_DIR) --output-dir $(FORMATION_B_LITE_OUTPUT) --progress-interval $(FORMATION_B_LITE_PROGRESS_INTERVAL) $(FORMATION_B_LITE_SCHEMA10_ARG)

formation-b-lite-quick:
	$(PYTHON) -m rogii.formation_b_lite --input $(FORMATION_B_LITE_QUICK_INPUT) --data-dir $(DATA_DIR) --train-dir $(DATA_DIR)/public_train --output-dir $(FORMATION_B_LITE_QUICK_OUTPUT) --progress-interval $(FORMATION_B_LITE_PROGRESS_INTERVAL) $(FORMATION_B_LITE_SCHEMA10_ARG)

formation-b2:
	$(PYTHON) -m rogii.formation_b2_constrained --input $(FORMATION_B2_INPUT) --b-scores $(FORMATION_B2_B_SCORES) --output-dir $(FORMATION_B2_OUTPUT) --progress-interval $(FORMATION_B2_PROGRESS_INTERVAL) $(FORMATION_B2_SCHEMA10_ARG)

formation-b2-quick:
	$(PYTHON) -m rogii.formation_b2_constrained --input $(FORMATION_B2_QUICK_INPUT) --b-scores $(FORMATION_B2_QUICK_B_SCORES) --output-dir $(FORMATION_B2_QUICK_OUTPUT) --progress-interval $(FORMATION_B2_PROGRESS_INTERVAL) $(FORMATION_B2_SCHEMA10_ARG)

formation-b2-guarded:
	$(PYTHON) -m rogii.formation_b2_guarded --input $(FORMATION_B2_GUARDED_INPUT) --choices $(FORMATION_B2_GUARDED_CHOICES) --metadata $(FORMATION_B2_GUARDED_METADATA) --output-dir $(FORMATION_B2_GUARDED_OUTPUT) $(FORMATION_B2_GUARDED_SCHEMA10_ARG) $(FORMATION_B2_GUARDED_SCHEMA10_COLUMN_ARG) --selectors $(FORMATION_B2_GUARDED_SELECTORS) --cross-fold-policy-limit $(FORMATION_B2_GUARDED_CV_POLICY_LIMIT)

formation-b2-infer-oof-replay:
	$(PYTHON) -m rogii.formation_b2_inference oof-replay --config $(FORMATION_B2_CONFIG) --input $(FORMATION_B2_INFER_INPUT) --choices $(FORMATION_B2_INFER_CHOICES) --metadata $(FORMATION_B2_INFER_METADATA) --output-dir $(FORMATION_B2_INFER_OUTPUT) --schema10-oof $(FORMATION_B2_INFER_SCHEMA10) $(FORMATION_B2_INFER_SCHEMA10_COLUMN_ARG) $(FORMATION_B2_INFER_REFERENCE_ARG) --tolerance $(FORMATION_B2_INFER_TOLERANCE)

formation-b2-infer-test: ensure-data
	$(PYTHON) -m rogii.formation_b2_inference test --config $(FORMATION_B2_CONFIG) --data-dir $(DATA_DIR) --base-submission $(FORMATION_B2_TEST_BASE_SUBMISSION) --base-column $(FORMATION_B2_TEST_BASE_COLUMN) --output-dir $(FORMATION_B2_TEST_OUTPUT) --progress-interval $(FORMATION_B2_TEST_PROGRESS_INTERVAL)

export-oof-baseline: ensure-data
	$(PYTHON) -m rogii.export_oof_baseline --model-dir $(OOF_BASELINE_MODEL_DIR) --data-dir $(DATA_DIR) --output $(OOF_BASELINE_OUTPUT) --num-workers $(OOF_BASELINE_NUM_WORKERS) --progress-interval $(OOF_BASELINE_PROGRESS_INTERVAL)

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
	$(PYTHON) -m rogii.kaggle_submit run --mode infer --competition $(COMPETITION) --user $(KAGGLE_USER) --kernel $(SUBMIT_KERNEL) --title "$(SUBMIT_TITLE)" --config $(CONFIG) $(B2_SUBMIT_ARG) --model-dir $(INFER_MODEL_DIR) $(CML_ARG) --model-dataset $(MODEL_DATASET) --model-dataset-dir $(MODEL_DATASET_DIR) --publish-model-dataset --kernel-dir $(SUBMIT_KERNEL_DIR) --data-dir $(KAGGLE_DATA_DIR) --artifact-dir $(SUBMIT_ARTIFACT_DIR) --output-dir $(SUBMIT_OUTPUT_DIR) --submission-file $(SUBMISSION) --kernel-timeout $(KERNEL_TIMEOUT) --wait-timeout $(KERNEL_WAIT_TIMEOUT) --poll-interval $(KERNEL_POLL_INTERVAL) --message "$(MESSAGE)" --skip-competition-submit --download-all-output

submit-dry:
	$(PYTHON) -m rogii.kaggle_submit run --mode infer --competition $(COMPETITION) --user $(KAGGLE_USER) --kernel $(SUBMIT_KERNEL) --title "$(SUBMIT_TITLE)" --config $(CONFIG) $(B2_SUBMIT_ARG) --model-dir $(INFER_MODEL_DIR) $(CML_ARG) --model-dataset $(MODEL_DATASET) --model-dataset-dir $(MODEL_DATASET_DIR) --kernel-dir $(SUBMIT_KERNEL_DIR) --data-dir $(KAGGLE_DATA_DIR) --artifact-dir $(SUBMIT_ARTIFACT_DIR) --output-dir $(SUBMIT_OUTPUT_DIR) --submission-file $(SUBMISSION) --kernel-timeout $(KERNEL_TIMEOUT) --wait-timeout $(KERNEL_WAIT_TIMEOUT) --poll-interval $(KERNEL_POLL_INTERVAL) --message "$(MESSAGE)" --skip-competition-submit --download-all-output --dry-run

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
