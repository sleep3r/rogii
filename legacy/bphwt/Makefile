# BPHWT v1 — training, inference, Docker, and remote-GPU orchestration.
# Mirrors the layout of old/Makefile (uv + spacebridge + portainer).
#
# This Makefile lives inside bphwt/ but always runs from the repo root
# (parent of bphwt/) so that `python -m bphwt.train` resolves the
# `bphwt` package correctly.  Invoke either way:
#
#     cd bphwt && make train-sanity
#     make -C bphwt train-sanity        # from repo root

SHELL := /bin/bash

# Locate repo root regardless of where `make` is invoked from.
HERE := $(realpath $(dir $(lastword $(MAKEFILE_LIST))))

UV ?= uv
# Every python command first cd's to this repo root so that bphwt/ is importable.
PYTHON ?= cd $(HERE) && $(UV) --project $(HERE) run python

# ---------------------------------------------------------------------------
# Data + configs (paths relative to repo root)
# ---------------------------------------------------------------------------
DATA_DIR        ?= data
LITE_CONFIG     ?= configs/bphwt_lite.yml
OVERFIT_CONFIG  ?= configs/bphwt_overfit.yml
SANITY_CONFIG   ?= $(LITE_CONFIG)
SMOKE_CONFIG    ?= $(OVERFIT_CONFIG)
V1_CONFIG       ?= $(LITE_CONFIG)
CONFIG          ?= $(LITE_CONFIG)
DEFAULT_SERVER_CONFIG ?= configs/bphwt_server.yml
ifeq ($(origin SERVER_CONFIG), undefined)
  ifeq ($(origin CONFIG), command line)
    SERVER_CONFIG := $(CONFIG)
  else
    SERVER_CONFIG := $(DEFAULT_SERVER_CONFIG)
  endif
endif

OUTPUT_DIR      ?= artifacts/bphwt_lite
SUBMISSION      ?= submission.csv
OOF_PATH        ?= artifacts/bphwt_oof.csv
OOF_DIAG_PATH   ?= artifacts/bphwt_oof_diagnostics.csv
OOF_SUMMARY_PATH ?= artifacts/bphwt_oof_diagnostics_summary.json

# Codebase snapshot for review / LLM context.  Excludes data, artifacts, caches.
CODEBASE_DIR      ?= artifacts/codebase
CODEBASE_STAMP    ?= $(shell date +%Y%m%d_%H%M%S)
CODEBASE_NAME     ?= bphwt_codebase_$(CODEBASE_STAMP)
CODEBASE_TXT      ?= $(CODEBASE_DIR)/$(CODEBASE_NAME).txt
CODEBASE_TAR      ?= $(CODEBASE_DIR)/$(CODEBASE_NAME).tar.gz
CODEBASE_SRC_DIRS ?= bphwt configs tests
CODEBASE_ROOT_FILES ?= Makefile Dockerfile Dockerfile.full pyproject.toml COMPETITION.md

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------
DOCKER_IMAGE    ?= bphwt:latest
DOCKER_PLATFORM ?= linux/amd64
BPHWT_BASE_IMAGE ?= harbor.wildberries.ru/tns-backoffice-security/spacebrigde:bphwt-base-aleksandr-kalashnikov
TORCH_CUDA      ?= cu124
TORCH_VERSION   ?= 2.5.1
DOCKER_GPU_FLAG ?= --gpus all

# No volume mounts by default — data comes from ClearML (data_clearml.enabled=true)
# and outputs go to S3 via tracking.output_uri.  If you need local file access
# for dev (e.g. running without ClearML), override on the command line:
#   make docker-run DOCKER_MOUNTS='-v $$(pwd)/../MTPNet/data:/app/MTPNet/data:ro'
DOCKER_MOUNTS   ?=

# ---------------------------------------------------------------------------
# Spacebridge / portainer (remote GPU)
# ---------------------------------------------------------------------------
INSTANCE            ?=
SPACEBRIDGE_TASK    ?= bphwt
SERVER_NOTES        ?= bphwt_sanity_run
SERVER_KEEP_ALIVE_ON_FAIL ?= 3600
SERVER_PREFLIGHT_SLEEP ?= 0
SERVER_SLEEP_ONLY ?= false
SERVER_REBUILD_CACHE ?= $(if $(BPHWT_REBUILD_CACHE),$(BPHWT_REBUILD_CACHE),false)

# ---- ClearML experiment tracking ----
CLEARML_ENABLED     ?= true
CLEARML_PROJECT     ?= ROGII/Wellbore/BPHWT
CLEARML_TASK_NAME   ?=
CLEARML_OUTPUT_URI  ?= s3://s3-basket-cold.wb.ru/ds-experiments
CLEARML_TAGS        ?= bphwt,gpu
CLEARML_LOG_MODEL   ?= true

# ---- ClearML dataset (already uploaded as ROGII/Wellbore/rogii-wellbore-... 20260519_s3) ----
CLEARML_DATA_ENABLED      ?= true
CLEARML_DATA_PROJECT      ?= ROGII/Wellbore
CLEARML_DATA_NAME         ?= rogii-wellbore-geology-prediction
CLEARML_DATA_VERSION      ?= 20260519_s3
CLEARML_DATA_CACHE_DIR    ?= ~/.cache/clearml/rogii
CLEARML_DATA_MAX_WORKERS  ?= 8

# spacebridge accepts key=value pairs and converts them to --key=value before
# baking them into Docker build arg CMD_ARGS.
SPACEBRIDGE_CMD_ARGS ?= config=$(SERVER_CONFIG) rebuild-cache=$(SERVER_REBUILD_CACHE) keep-alive-on-fail=$(SERVER_KEEP_ALIVE_ON_FAIL)
SPACEBRIDGE_DEBUG_CMD_ARGS ?= config=$(SERVER_CONFIG) rebuild-cache=$(SERVER_REBUILD_CACHE) keep-alive-on-fail=$(SERVER_KEEP_ALIVE_ON_FAIL) preflight-sleep=$(SERVER_PREFLIGHT_SLEEP) sleep-only=$(SERVER_SLEEP_ONLY)

# ---------------------------------------------------------------------------
# .PHONY
# ---------------------------------------------------------------------------
.PHONY: help install lock format check codebase collect-code \
        train train-local train-recipe train-overfit train-smoke train-sanity train-v1 \
        infer oof oof-diagnostics error-analysis \
        clearml-check clearml-upload-data clearml-data-local-path \
        docker-build docker-build-full docker-build-base docker-push-base docker-base ensure-docker-base docker-build-no-cache docker-run docker-run-cpu docker-shell \
        check-server-env check-server-config train-server train-spacebridge train-server-debug \
        clean clean-artifacts

help:
	@echo "BPHWT v1 targets (run from inside bphwt/ or via -C bphwt):"
	@echo ""
	@echo "  Local training:"
	@echo "    make install              uv sync (CPU torch on Mac, CUDA torch in Docker)"
	@echo "    make train-recipe         Small Karpathy-recipe BPHWT-lite loop"
	@echo "    make train-overfit        Two-well overfit/debug loop"
	@echo "    make train-smoke          Alias for train-overfit"
	@echo "    make train-sanity         Same as train-smoke unless CONFIG is overridden"
	@echo "    make train-v1             Same as train-smoke unless CONFIG is overridden"
	@echo "    make train CONFIG=...     Run a custom config (path from repo root)"
	@echo ""
	@echo "  Inference / OOF:"
	@echo "    make infer                Build submission.csv from a trained run"
	@echo "    make oof                  Generate OOF predictions"
	@echo "    make oof-diagnostics      Generate full-length OOF error analysis"
	@echo "    make error-analysis       Alias for oof-diagnostics"
	@echo ""
	@echo "  ClearML:"
	@echo "    make clearml-check             Check ClearML import/config and optional dataset"
	@echo "    make clearml-upload-data       Upload local data dir as a ClearML Dataset"
	@echo "    make clearml-data-local-path   Resolve / download the dataset locally"
	@echo ""
	@echo "  Docker:"
	@echo "    make docker-build         Fast overlay build: base image + current code"
	@echo "    make docker-base          Full dependency rebuild + push reusable base"
	@echo "    make docker-build-full    Full dependency rebuild (CUDA $(TORCH_CUDA))"
	@echo "    make docker-run           Run training inside Docker (mounts data + GPU)"
	@echo "    make docker-shell         Drop into interactive shell in image"
	@echo ""
	@echo "  Remote GPU (spacebridge / portainer):"
	@echo "    make check-server-env"
	@echo "    make train-server INSTANCE=el   # or xc, uses SERVER_CONFIG=$(SERVER_CONFIG)"
	@echo "    make train-server-debug INSTANCE=el SERVER_SLEEP_ONLY=true SERVER_PREFLIGHT_SLEEP=7200"
	@echo ""
	@echo "  Maintenance:"
	@echo "    make format / make check  ruff format + lint"
	@echo "    make codebase             Collect source/configs into txt + tar.gz"
	@echo "    make clean / clean-artifacts"

# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------
install:
	$(UV) --project $(HERE) sync

lock:
	$(UV) --project $(HERE) lock

format:
	cd $(HERE) && $(UV) run ruff check --fix --select I .
	cd $(HERE) && $(UV) run ruff check --fix .
	cd $(HERE) && $(UV) run ruff format .

check:
	cd $(HERE) && $(UV) run ruff check .

codebase collect-code:
	@cd "$(HERE)" && \
	TXT="$(CODEBASE_TXT)" && \
	TAR="$(CODEBASE_TAR)" && \
	mkdir -p "$$(dirname "$$TXT")" "$$(dirname "$$TAR")" && \
	FILES="$$( \
		find $(CODEBASE_SRC_DIRS) -type f \
			\( -name '*.py' -o -name '*.yml' -o -name '*.yaml' -o -name '*.toml' -o -name '*.md' \) \
			-not -path '*/__pycache__/*' \
			-not -path '*/.pytest_cache/*' \
			-print; \
		for f in $(CODEBASE_ROOT_FILES); do test -f "$$f" && printf '%s\n' "$$f"; done \
	)" && \
	FILES="$$(printf '%s\n' "$$FILES" | sort -u)" && \
	{ \
		echo "# BPHWT codebase snapshot"; \
		echo "# generated_utc: $$(date -u '+%Y-%m-%dT%H:%M:%SZ')"; \
		echo "# root: $(HERE)"; \
		echo ""; \
		echo "## Files"; \
		printf '%s\n' "$$FILES"; \
		for f in $$FILES; do \
			echo ""; \
			echo "================================================================"; \
			echo "FILE: $$f"; \
			echo "================================================================"; \
			cat "$$f"; \
		done; \
	} > "$$TXT" && \
	tar -czf "$$TAR" $$FILES && \
	echo "Codebase text:    $$TXT" && \
	echo "Codebase archive: $$TAR" && \
	wc -l "$$TXT"

# ---------------------------------------------------------------------------
# Local training (all paths relative to repo root)
# ---------------------------------------------------------------------------
train-local: ; $(PYTHON) -m bphwt.train $(CONFIG)
train:        train-local

train-recipe:
	$(PYTHON) -m bphwt.train $(LITE_CONFIG) --rebuild-cache

train-overfit:
	$(PYTHON) -m bphwt.train $(OVERFIT_CONFIG) --rebuild-cache

train-smoke:
	$(PYTHON) -m bphwt.train $(SMOKE_CONFIG)

train-sanity:
	$(PYTHON) -m bphwt.train $(SANITY_CONFIG)

train-v1:
	$(PYTHON) -m bphwt.train $(V1_CONFIG)

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
infer:
	$(PYTHON) -m bphwt.infer $(CONFIG) \
		--output_dir $(OUTPUT_DIR) \
		--submission_path $(SUBMISSION)

oof:
	$(PYTHON) -m bphwt.infer $(CONFIG) \
		--output_dir $(OUTPUT_DIR) \
		--oof \
		--oof_path $(OOF_PATH)

oof-diagnostics:
	$(PYTHON) -m bphwt.infer $(CONFIG) \
		--output_dir $(OUTPUT_DIR) \
		--oof_diagnostics \
		--oof_diag_path $(OOF_DIAG_PATH) \
		--oof_summary_path $(OOF_SUMMARY_PATH)

error-analysis: oof-diagnostics

# ---------------------------------------------------------------------------
# ClearML data management
# ---------------------------------------------------------------------------
clearml-check:
	$(PYTHON) -m bphwt.clearml_data check \
		--project "$(CLEARML_DATA_PROJECT)" \
		--name "$(CLEARML_DATA_NAME)" \
		--version "$(CLEARML_DATA_VERSION)"

clearml-upload-data:
	$(PYTHON) -m bphwt.clearml_data upload \
		--data-dir $(DATA_DIR) \
		--project "$(CLEARML_DATA_PROJECT)" \
		--name "$(CLEARML_DATA_NAME)" \
		--version "$(CLEARML_DATA_VERSION)" \
		--output-uri "$(CLEARML_OUTPUT_URI)" \
		--max-workers $(CLEARML_DATA_MAX_WORKERS)

clearml-data-local-path:
	$(PYTHON) -m bphwt.clearml_data download \
		--project "$(CLEARML_DATA_PROJECT)" \
		--name "$(CLEARML_DATA_NAME)" \
		--version "$(CLEARML_DATA_VERSION)" \
		--cache-dir "$(CLEARML_DATA_CACHE_DIR)"

# ---------------------------------------------------------------------------
# Docker (context = bphwt/ only; data/artifacts are runtime mounts or ClearML)
# ---------------------------------------------------------------------------
docker-build:
	cd $(HERE) && docker build \
		--platform $(DOCKER_PLATFORM) \
		-f Dockerfile \
		--build-arg BPHWT_BASE_IMAGE=$(BPHWT_BASE_IMAGE) \
		--build-arg CMD_ARGS=--config=$(CONFIG) \
		-t $(DOCKER_IMAGE) .

docker-build-full:
	cd $(HERE) && docker build \
		--platform $(DOCKER_PLATFORM) \
		-f Dockerfile.full \
		--build-arg TORCH_CUDA=$(TORCH_CUDA) \
		--build-arg TORCH_VERSION=$(TORCH_VERSION) \
		-t $(BPHWT_BASE_IMAGE) .

docker-build-base: docker-build-full

docker-push-base:
	docker push $(BPHWT_BASE_IMAGE)

docker-base: docker-build-base docker-push-base

ensure-docker-base:
	@if docker manifest inspect "$(BPHWT_BASE_IMAGE)" >/dev/null 2>&1; then \
		echo "Docker base image exists: $(BPHWT_BASE_IMAGE)"; \
	else \
		echo "Docker base image missing: $(BPHWT_BASE_IMAGE)"; \
		echo "Building and pushing dependency base image..."; \
		$(MAKE) --no-print-directory docker-base; \
	fi

docker-build-no-cache:
	cd $(HERE) && docker build --no-cache \
		--platform $(DOCKER_PLATFORM) \
		-f Dockerfile.full \
		--build-arg TORCH_CUDA=$(TORCH_CUDA) \
		--build-arg TORCH_VERSION=$(TORCH_VERSION) \
		-t $(BPHWT_BASE_IMAGE) .

docker-run:
	docker run --rm $(DOCKER_GPU_FLAG) \
		$(DOCKER_MOUNTS) \
		-e CMD_ARGS="--config=$(CONFIG)" \
		$(DOCKER_IMAGE)

docker-run-cpu:
	docker run --rm \
		$(DOCKER_MOUNTS) \
		-e CMD_ARGS="--config=$(SMOKE_CONFIG)" \
		$(DOCKER_IMAGE)

docker-shell:
	docker run --rm -it $(DOCKER_GPU_FLAG) \
		$(DOCKER_MOUNTS) \
		--entrypoint /bin/bash \
		$(DOCKER_IMAGE)

# ---------------------------------------------------------------------------
# Remote GPU via spacebridge (mirrors old/Makefile pattern)
# ---------------------------------------------------------------------------
check-server-env:
	docker version
	docker context show
	docker buildx version
	@test -f $(HERE)/portainer.yml || (echo "bphwt/portainer.yml not found" && exit 1)
	@echo "Available instances:"
	@grep -E "^  [a-zA-Z]+:" $(HERE)/portainer.yml || true

check-server-config:
	@test -n "$(SERVER_CONFIG)" || (echo "SERVER_CONFIG is empty" && exit 1)
	@if [ ! -f "$(HERE)/$(SERVER_CONFIG)" ]; then \
		echo "SERVER_CONFIG not found: $(SERVER_CONFIG)"; \
		case "$(SERVER_CONFIG)" in \
			bphwt/configs/*) echo "Hint: run from this repo with SERVER_CONFIG=configs/$$(basename "$(SERVER_CONFIG)")";; \
			*) echo "Hint: use SERVER_CONFIG=configs/bphwt_server.yml";; \
		esac; \
		exit 1; \
	fi
	@echo "SERVER_CONFIG=$(SERVER_CONFIG)"
	@echo "SPACEBRIDGE_CMD_ARGS=$(SPACEBRIDGE_CMD_ARGS)"

train-server train-spacebridge: check-server-env check-server-config ensure-docker-base
	@test -n "$(INSTANCE)" || (echo "Set INSTANCE=<el|xc> (see bphwt/portainer.yml)" && exit 1)
	cd $(HERE) && $(UV) run spacebridge train $(SPACEBRIDGE_TASK) \
		--instance $(INSTANCE) \
		--cmd-args "$(SPACEBRIDGE_CMD_ARGS)"
	@echo ""
	@echo "Note: ClearML settings are read from the YAML config (tracking.* and"
	@echo "data_clearml.*).  To override on the remote, edit the YAML or export"
	@echo "BPHWT_CLEARML_* / BPHWT_DATA_CLEARML_* in the portainer instance."

train-server-debug: check-server-env check-server-config
	@test -n "$(INSTANCE)" || (echo "Set INSTANCE=<el|xc> (see bphwt/portainer.yml)" && exit 1)
	cd $(HERE) && $(UV) run spacebridge train $(SPACEBRIDGE_TASK)-debug \
		--instance $(INSTANCE) \
		--cmd-args "$(SPACEBRIDGE_DEBUG_CMD_ARGS)"

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
clean:
	cd $(HERE) && find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	cd $(HERE) && find . -type f -name "*.pyc" -delete

clean-artifacts:
	rm -rf $(HERE)/artifacts/bphwt_*
