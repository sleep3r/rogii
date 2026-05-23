SHELL := /bin/bash
PYTHON ?= python3
UV ?= uv
CONFIG ?= configs/mtp_v0.yml
SMOKE_CONFIG ?= configs/mtp_smoke.yml
RUN_DIR ?= artifacts/mtp_v0
OUTPUT ?=
N_FOLDS ?= 5
SEED ?= 42
RANKER_VARIANT ?= conservative_regression
LOGIT_SOURCE ?= ranker
RANKER_LOGITS ?=
TAU_FT ?= 5.0
RANKER_BETA ?= 0.5
COPY_SOURCE ?= ../old/data
COPY_TARGET ?= data
CODEBASE_BUNDLE ?= artifacts/codebase_bundle.md

.PHONY: install test copy-data smoke train eval stitch rank rank-crossfit mtp-ranker-crossfit track track-audit codebase-bundle collect-code

install:
	$(UV) sync --extra dev

test:
	$(UV) run --extra dev pytest -q

copy-data:
	$(UV) run --extra dev python -m mtpnet.cli copy-data --source $(COPY_SOURCE) --target $(COPY_TARGET)

smoke:
	$(UV) run --extra dev python -m mtpnet.cli train --config $(SMOKE_CONFIG)

train:
	$(UV) run --extra dev python -m mtpnet.cli train --config $(CONFIG)

eval:
	$(UV) run --extra dev python -m mtpnet.cli eval --run-dir $(RUN_DIR)

stitch:
	$(UV) run --extra dev python -m mtpnet.cli stitch --run-dir $(RUN_DIR)

rank:
	$(UV) run --extra dev python -m mtpnet.cli rank --run-dir $(RUN_DIR)

rank-crossfit:
	$(UV) run --extra dev python -m mtpnet.cli rank-crossfit --run-dir $(RUN_DIR) --n-folds $(N_FOLDS) --seed $(SEED) --ranker-variant $(RANKER_VARIANT) $(if $(OUTPUT),--output-dir $(OUTPUT),)

mtp-ranker-crossfit: rank-crossfit

track:
	$(UV) run --extra dev python -m mtpnet.cli track --run-dir $(RUN_DIR) --logit-source $(LOGIT_SOURCE) --tau-ft $(TAU_FT) --ranker-beta $(RANKER_BETA) $(if $(RANKER_LOGITS),--ranker-logits $(RANKER_LOGITS),)

track-audit:
	$(UV) run --extra dev python -m mtpnet.cli track-audit --run-dir $(RUN_DIR)

codebase-bundle:
	$(UV) run --extra dev python -m mtpnet.codebase_bundle --root . --output $(CODEBASE_BUNDLE)

collect-code: codebase-bundle
