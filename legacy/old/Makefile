SHELL := /bin/bash
.DEFAULT_GOAL := help

UV ?= uv
PYTHON ?= $(UV) run python
KAGGLE ?= $(UV) run kaggle

COMPETITION ?= rogii-wellbore-geology-prediction
KAGGLE_USER ?= sleep3r
DATA_DIR ?= data

CONFIG ?= configs/stack.yml
GPU_CONFIG ?= configs/stack_gpu.yml
QUICK_CONFIG ?= configs/quick.yml
SERVER_CONFIG ?= $(GPU_CONFIG)

OUTPUT_DIR ?= artifacts/stack
SUBMISSION ?= submission.csv
MESSAGE ?= Submission

PROFILE_DIR ?= artifacts/profiles
PROFILE_NAME ?= stack
PROFILE_FILE ?= $(PROFILE_DIR)/$(PROFILE_NAME).prof
PROFILE_LOG ?= $(PROFILE_DIR)/$(PROFILE_NAME).log
PROFILE_REPORT ?= $(PROFILE_DIR)/$(PROFILE_NAME).md

KAGGLE_DATA_DIR ?= /kaggle/input/$(COMPETITION)
KERNEL ?= rogii-baseline-infer
KERNEL_TITLE ?= ROGII Baseline Infer
KERNEL_DIR ?= artifacts/kaggle_kernel
KAGGLE_OUTPUT_DIR ?= artifacts/kaggle_output
KAGGLE_ARTIFACT_DIR ?= artifacts/kaggle_artifacts
MODEL_DIR ?= $(OUTPUT_DIR)
TESTS ?= tests

INSTANCE ?=
SPACEBRIDGE_TASK ?= rogii
SERVER_NOTES ?= spacebridge_train
PROJECT_NAME ?= ROGII/Wellbore
OUTPUT_URI ?= s3://s3-basket-cold.wb.ru/ds-experiments
CLEARML_ENABLED ?= true
CLEARML_PROJECT ?= $(PROJECT_NAME)
CLEARML_TASK_NAME ?=
CLEARML_OUTPUT_URI ?= $(OUTPUT_URI)
CLEARML_TAGS ?= rogii,spacebridge,stack,gpu
CLEARML_LOG_MODEL ?= true
CLEARML_DATA_PROJECT ?= $(PROJECT_NAME)
CLEARML_DATA_NAME ?= rogii-wellbore-geology-prediction
CLEARML_DATA_VERSION ?= 20260519_s3
CLEARML_DATA_CACHE_DIR ?= ~/.cache/clearml/rogii
SPACEBRIDGE_CMD_ARGS ?= config=$(SERVER_CONFIG) clearml_enabled=$(CLEARML_ENABLED) clearml_project=$(CLEARML_PROJECT) clearml_task_name=$(CLEARML_TASK_NAME) clearml_output_uri=$(CLEARML_OUTPUT_URI) clearml_tags=$(CLEARML_TAGS) clearml_log_model=$(CLEARML_LOG_MODEL) data_clearml_enabled=true data_clearml_project=$(CLEARML_DATA_PROJECT) data_clearml_name=$(CLEARML_DATA_NAME) data_clearml_version=$(CLEARML_DATA_VERSION) data_clearml_cache_dir=$(CLEARML_DATA_CACHE_DIR) notes=$(SERVER_NOTES)

.PHONY: help install data download-data unzip-data ensure-data doctor server-doctor train train-gpu train-server train-spacebridge quick profile submit submit-dry test check format clean-cache

help:
	@echo "ROGII old standalone"
	@echo ""
	@echo "Setup"
	@echo "  make install        Sync runtime and dev dependencies"
	@echo "  make data           Download and unzip Kaggle data into $(DATA_DIR)"
	@echo "  make doctor         Print environment and GPU preflight status for CONFIG"
	@echo ""
	@echo "Training"
	@echo "  make train          Train CONFIG=$(CONFIG)"
	@echo "  make train-gpu      Fail-fast GPU preflight, then train GPU_CONFIG=$(GPU_CONFIG)"
	@echo "  make train-server   Run SERVER_CONFIG=$(SERVER_CONFIG) through spacebridge"
	@echo "  make quick          Train QUICK_CONFIG=$(QUICK_CONFIG)"
	@echo "  make profile        Profile CONFIG=$(CONFIG) into $(PROFILE_REPORT)"
	@echo ""
	@echo "Submission"
	@echo "  make submit-dry     Build Kaggle inference kernel without pushing"
	@echo "  make submit         Push/run Kaggle inference kernel using MODEL_DIR=$(MODEL_DIR)"
	@echo ""
	@echo "Quality"
	@echo "  make test           Run pytest TESTS=$(TESTS)"
	@echo "  make check          Run ruff and pytest"
	@echo "  make format         Ruff import fixes, lint fixes, and formatting"
	@echo "  make clean-cache    Remove local generated caches"

install:
	$(UV) sync --all-groups

download-data:
	mkdir -p $(DATA_DIR)
	$(KAGGLE) competitions download -c $(COMPETITION) -p $(DATA_DIR)

unzip-data: download-data
	unzip -q -o $(DATA_DIR)/$(COMPETITION).zip -d $(DATA_DIR)

data: unzip-data

ensure-data:
	@if [ ! -d "$(DATA_DIR)/train" ] || [ ! -d "$(DATA_DIR)/test" ]; then \
		$(MAKE) data; \
	fi

doctor:
	$(PYTHON) --version
	$(UV) --version
	$(PYTHON) -m rogii.gpu_preflight --config $(CONFIG)

server-doctor:
	docker version
	docker context show
	docker buildx version
	$(PYTHON) -m rogii.spacebridge_preflight $(if $(INSTANCE),--instance "$(INSTANCE)",)

train: ensure-data
	$(PYTHON) -m rogii --config $(CONFIG)

train-gpu: ensure-data
	$(PYTHON) -m rogii.gpu_preflight --config $(GPU_CONFIG)
	$(PYTHON) -m rogii --config $(GPU_CONFIG)

train-server train-spacebridge: server-doctor
	@test -n "$(INSTANCE)" || (echo "Set INSTANCE=<portainer instance from portainer.yml>" && exit 1)
	$(UV) run spacebridge train $(SPACEBRIDGE_TASK) --instance $(INSTANCE) --cmd-args "$(SPACEBRIDGE_CMD_ARGS)"

quick: ensure-data
	$(PYTHON) -m rogii --config $(QUICK_CONFIG)

profile: ensure-data
	mkdir -p $(PROFILE_DIR)
	set -o pipefail; $(PYTHON) -m cProfile -o $(PROFILE_FILE) -m rogii --config $(CONFIG) 2>&1 | tee $(PROFILE_LOG)
	$(PYTHON) -m rogii.profile_report --profile $(PROFILE_FILE) --log $(PROFILE_LOG) --output $(PROFILE_REPORT) --title "ROGII profile: $(PROFILE_NAME)"

submit:
	$(PYTHON) -m rogii.kaggle_submit run --mode infer --competition $(COMPETITION) --user $(KAGGLE_USER) --kernel $(KERNEL) --title "$(KERNEL_TITLE)" --config $(CONFIG) --model-dir $(MODEL_DIR) --kernel-dir $(KERNEL_DIR) --data-dir $(KAGGLE_DATA_DIR) --artifact-dir $(KAGGLE_ARTIFACT_DIR) --output-dir $(KAGGLE_OUTPUT_DIR) --submission-file $(SUBMISSION) --message "$(MESSAGE)" --download-all-output

submit-dry:
	$(PYTHON) -m rogii.kaggle_submit run --mode infer --competition $(COMPETITION) --user $(KAGGLE_USER) --kernel $(KERNEL) --title "$(KERNEL_TITLE)" --config $(CONFIG) --model-dir $(MODEL_DIR) --kernel-dir $(KERNEL_DIR) --data-dir $(KAGGLE_DATA_DIR) --artifact-dir $(KAGGLE_ARTIFACT_DIR) --output-dir $(KAGGLE_OUTPUT_DIR) --submission-file $(SUBMISSION) --message "$(MESSAGE)" --download-all-output --skip-competition-submit --dry-run

test:
	$(UV) run --with pytest python -m pytest $(TESTS)

check:
	$(UV) run --with ruff python -m ruff check .
	$(UV) run --with pytest python -m pytest $(TESTS)

format:
	$(UV) run --with ruff python -m ruff check --fix --select I .
	$(UV) run --with ruff python -m ruff check --fix .
	$(UV) run --with ruff python -m ruff format .

clean-cache:
	rm -rf .pytest_cache .ruff_cache catboost_info artifacts/feature_cache artifacts/feature_cache_quick
