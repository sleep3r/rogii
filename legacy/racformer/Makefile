# RAC-Former v1 — training, inference, Docker, and remote-GPU orchestration.
# Mirrors the layout of old/Makefile (uv + spacebridge + portainer).
#
# This Makefile lives inside racformer/ but always runs from the repo root
# (parent of racformer/) so that `python -m racformer.train` resolves the
# `racformer` package correctly.  Invoke either way:
#
#     cd racformer && make train-sanity
#     make -C racformer train-sanity        # from repo root

SHELL := /bin/bash

# Locate repo root regardless of where `make` is invoked from.
ROOT := $(realpath $(dir $(lastword $(MAKEFILE_LIST)))/..)
HERE := $(realpath $(dir $(lastword $(MAKEFILE_LIST))))

UV ?= uv
# Every python command first cd's to repo root so that racformer/ is importable.
PYTHON ?= cd $(ROOT) && $(UV) --project $(HERE) run python

# ---------------------------------------------------------------------------
# Data + configs (paths relative to repo root)
# ---------------------------------------------------------------------------
DATA_DIR        ?= MTPNet/data
SANITY_CONFIG   ?= racformer/configs/racformer_sanity.yml
SANITY_DIRECT_CONFIG ?= racformer/configs/racformer_sanity_direct.yml
SANITY_SEGMENT_CONFIG ?= racformer/configs/racformer_sanity_segment.yml
SANITY_DIRECT_SEGMENT_CONFIG ?= racformer/configs/racformer_sanity_direct_segment.yml
SANITY_DIRECT_TEACHER_CONFIG ?= racformer/configs/racformer_sanity_direct_teacher.yml
SANITY_DIRECT_SEGMENT_TEACHER_CONFIG ?= racformer/configs/racformer_sanity_direct_segment_teacher.yml
V1_CONFIG       ?= racformer/configs/racformer_v1.yml
SMOKE_CONFIG    ?= racformer/configs/racformer_smoke.yml
OVERFIT8_CONFIG ?= racformer/configs/racformer_overfit8.yml
OVERFIT8_DIRECT_CONFIG ?= racformer/configs/racformer_overfit8_direct.yml
OVERFIT8_SEGMENT_CONFIG ?= racformer/configs/racformer_overfit8_segment.yml
CONFIG          ?= $(SANITY_CONFIG)

OUTPUT_DIR      ?= artifacts/racformer_sanity
SUBMISSION      ?= submission.csv
OOF_PATH        ?= artifacts/racformer_oof.csv
OOF_DIAG_PATH   ?= artifacts/racformer_oof_diagnostics.csv
OOF_SUMMARY_PATH ?= artifacts/racformer_oof_diagnostics_summary.json

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------
DOCKER_IMAGE    ?= racformer:latest
DOCKER_PLATFORM ?= linux/amd64
RACFORMER_BASE_IMAGE ?= harbor.wildberries.ru/tns-backoffice-security/spacebrigde:racformer-base-aleksandr-kalashnikov
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
SPACEBRIDGE_TASK    ?= racformer
SERVER_NOTES        ?= racformer_sanity_run

# ---- ClearML experiment tracking ----
CLEARML_ENABLED     ?= true
CLEARML_PROJECT     ?= ROGII/Wellbore/RACFormer
CLEARML_TASK_NAME   ?=
CLEARML_OUTPUT_URI  ?= s3://s3-basket-cold.wb.ru/ds-experiments
CLEARML_TAGS        ?= racformer,gpu
CLEARML_LOG_MODEL   ?= true

# ---- ClearML dataset (already uploaded as ROGII/Wellbore/rogii-wellbore-... 20260519_s3) ----
CLEARML_DATA_ENABLED      ?= true
CLEARML_DATA_PROJECT      ?= ROGII/Wellbore
CLEARML_DATA_NAME         ?= rogii-wellbore-geology-prediction
CLEARML_DATA_VERSION      ?= 20260519_s3
CLEARML_DATA_CACHE_DIR    ?= ~/.cache/clearml/rogii
CLEARML_DATA_MAX_WORKERS  ?= 8

# spacebridge accepts key=value pairs and converts them to --key=value before
# baking them into Docker build arg CMD_ARGS.  Dockerfile strips --config= so
# train.py still receives a positional YAML path.
SPACEBRIDGE_CMD_ARGS ?= config=$(CONFIG)

# ---------------------------------------------------------------------------
# .PHONY
# ---------------------------------------------------------------------------
.PHONY: help install lock format check \
        train train-local train-smoke train-sanity train-v1 \
        train-sanity-direct train-sanity-segment \
        train-sanity-direct-segment train-sanity-direct-teacher train-sanity-direct-segment-teacher \
        train-overfit8 train-overfit8-direct train-overfit8-segment \
        infer oof oof-diagnostics \
        clearml-upload-data clearml-data-local-path \
        docker-build docker-build-full docker-build-no-cache docker-run docker-run-cpu docker-shell \
        check-server-env train-server train-spacebridge \
        clean clean-artifacts

help:
	@echo "RAC-Former v1 targets (run from inside racformer/ or via -C racformer):"
	@echo ""
	@echo "  Local training:"
	@echo "    make install              uv sync (CPU torch on Mac, CUDA torch in Docker)"
	@echo "    make train-smoke          CPU smoke (5 wells, 2 folds, 3 epochs)"
	@echo "    make train-overfit8       Debug overfit on the same first 8 train wells"
	@echo "    make train-overfit8-direct  Same debug run, direct residual path only"
	@echo "    make train-overfit8-segment Same debug run, segment slope path only"
	@echo "    make train-sanity         5-fold OOF sanity, direct-first baseline"
	@echo "    make train-sanity-direct  5-fold OOF sanity, direct residual path only"
	@echo "    make train-sanity-segment 5-fold OOF sanity, segment slope path only"
	@echo "    make train-sanity-direct-segment         Direct baseline + segment path"
	@echo "    make train-sanity-direct-teacher         Direct baseline + ANCC teacher"
	@echo "    make train-sanity-direct-segment-teacher Direct + segment + ANCC teacher"
	@echo "    make train-v1             Current v1 direct-first baseline"
	@echo "    make train CONFIG=...     Run a custom config (path from repo root)"
	@echo ""
	@echo "  Inference / OOF:"
	@echo "    make infer                Build submission.csv from a trained run"
	@echo "    make oof                  Generate OOF predictions"
	@echo "    make oof-diagnostics      Generate OOF ablation + bucket diagnostics"
	@echo ""
	@echo "  ClearML:"
	@echo "    make clearml-upload-data       Upload local data dir as a ClearML Dataset"
	@echo "    make clearml-data-local-path   Resolve / download the dataset locally"
	@echo ""
	@echo "  Docker:"
	@echo "    make docker-build         Fast overlay build: base image + current code"
	@echo "    make docker-build-full    Full dependency rebuild (CUDA $(TORCH_CUDA))"
	@echo "    make docker-run           Run training inside Docker (mounts data + GPU)"
	@echo "    make docker-shell         Drop into interactive shell in image"
	@echo ""
	@echo "  Remote GPU (spacebridge / portainer):"
	@echo "    make check-server-env"
	@echo "    make train-server INSTANCE=el   # or xc"
	@echo ""
	@echo "  Maintenance:"
	@echo "    make format / make check  ruff format + lint"
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

# ---------------------------------------------------------------------------
# Local training (all paths relative to repo root)
# ---------------------------------------------------------------------------
train-local: ; $(PYTHON) -m racformer.train $(CONFIG)
train:        train-local

train-smoke:
	$(PYTHON) -m racformer.train $(SMOKE_CONFIG)

train-overfit8:
	$(PYTHON) -m racformer.train $(OVERFIT8_CONFIG)

train-overfit8-direct:
	$(PYTHON) -m racformer.train $(OVERFIT8_DIRECT_CONFIG)

train-overfit8-segment:
	$(PYTHON) -m racformer.train $(OVERFIT8_SEGMENT_CONFIG)

train-sanity:
	$(PYTHON) -m racformer.train $(SANITY_CONFIG)

train-sanity-direct:
	$(PYTHON) -m racformer.train $(SANITY_DIRECT_CONFIG)

train-sanity-segment:
	$(PYTHON) -m racformer.train $(SANITY_SEGMENT_CONFIG)

train-sanity-direct-segment:
	$(PYTHON) -m racformer.train $(SANITY_DIRECT_SEGMENT_CONFIG)

train-sanity-direct-teacher:
	$(PYTHON) -m racformer.train $(SANITY_DIRECT_TEACHER_CONFIG)

train-sanity-direct-segment-teacher:
	$(PYTHON) -m racformer.train $(SANITY_DIRECT_SEGMENT_TEACHER_CONFIG)

train-v1:
	$(PYTHON) -m racformer.train $(V1_CONFIG)

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
infer:
	$(PYTHON) -m racformer.infer $(CONFIG) \
		--output_dir $(OUTPUT_DIR) \
		--submission_path $(SUBMISSION)

oof:
	$(PYTHON) -m racformer.infer $(CONFIG) \
		--output_dir $(OUTPUT_DIR) \
		--oof \
		--oof_path $(OOF_PATH)

oof-diagnostics:
	$(PYTHON) -m racformer.infer $(CONFIG) \
		--output_dir $(OUTPUT_DIR) \
		--oof_diagnostics \
		--oof_diag_path $(OOF_DIAG_PATH) \
		--oof_summary_path $(OOF_SUMMARY_PATH)

# ---------------------------------------------------------------------------
# ClearML data management
# ---------------------------------------------------------------------------
clearml-upload-data:
	$(PYTHON) -m racformer.clearml_data upload \
		--data-dir $(DATA_DIR) \
		--project "$(CLEARML_DATA_PROJECT)" \
		--name "$(CLEARML_DATA_NAME)" \
		--version "$(CLEARML_DATA_VERSION)" \
		--output-uri "$(CLEARML_OUTPUT_URI)" \
		--max-workers $(CLEARML_DATA_MAX_WORKERS)

clearml-data-local-path:
	$(PYTHON) -m racformer.clearml_data download \
		--project "$(CLEARML_DATA_PROJECT)" \
		--name "$(CLEARML_DATA_NAME)" \
		--version "$(CLEARML_DATA_VERSION)" \
		--cache-dir "$(CLEARML_DATA_CACHE_DIR)"

# ---------------------------------------------------------------------------
# Docker (context = racformer/ only; data/artifacts are runtime mounts or ClearML)
# ---------------------------------------------------------------------------
docker-build:
	cd $(HERE) && docker build \
		--platform $(DOCKER_PLATFORM) \
		-f Dockerfile \
		--build-arg RACFORMER_BASE_IMAGE=$(RACFORMER_BASE_IMAGE) \
		--build-arg CMD_ARGS=--config=$(CONFIG) \
		-t $(DOCKER_IMAGE) .

docker-build-full:
	cd $(HERE) && docker build \
		--platform $(DOCKER_PLATFORM) \
		-f Dockerfile.full \
		--build-arg TORCH_CUDA=$(TORCH_CUDA) \
		--build-arg TORCH_VERSION=$(TORCH_VERSION) \
		-t $(RACFORMER_BASE_IMAGE) .

docker-build-no-cache:
	cd $(HERE) && docker build --no-cache \
		--platform $(DOCKER_PLATFORM) \
		-f Dockerfile.full \
		--build-arg TORCH_CUDA=$(TORCH_CUDA) \
		--build-arg TORCH_VERSION=$(TORCH_VERSION) \
		-t $(DOCKER_IMAGE) .

docker-run:
	docker run --rm $(DOCKER_GPU_FLAG) \
		$(DOCKER_MOUNTS) \
		-e CMD_ARGS=$(CONFIG) \
		$(DOCKER_IMAGE)

docker-run-cpu:
	docker run --rm \
		$(DOCKER_MOUNTS) \
		-e CMD_ARGS=$(SMOKE_CONFIG) \
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
	@test -f $(HERE)/portainer.yml || (echo "racformer/portainer.yml not found" && exit 1)
	@echo "Available instances:"
	@grep -E "^  [a-zA-Z]+:" $(HERE)/portainer.yml || true

train-server train-spacebridge: check-server-env
	@test -n "$(INSTANCE)" || (echo "Set INSTANCE=<el|xc> (see racformer/portainer.yml)" && exit 1)
	cd $(HERE) && $(UV) run spacebridge train $(SPACEBRIDGE_TASK) \
		--instance $(INSTANCE) \
		--cmd-args "$(SPACEBRIDGE_CMD_ARGS)"
	@echo ""
	@echo "Note: ClearML settings are read from the YAML config (tracking.* and"
	@echo "data_clearml.*).  To override on the remote, edit the YAML or export"
	@echo "RACFORMER_CLEARML_* / RACFORMER_DATA_CLEARML_* in the portainer instance."

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
clean:
	cd $(ROOT) && find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	cd $(ROOT) && find . -type f -name "*.pyc" -delete

clean-artifacts:
	rm -rf $(ROOT)/artifacts/racformer_*
