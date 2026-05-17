COMPETITION := rogii-wellbore-geology-prediction
KAGGLE_USER ?= sleep3r
UV ?= uv
PYTHON ?= $(UV) run python
KAGGLE ?= $(UV) run kaggle
DATA_DIR ?= data
CONFIG ?= configs/hgb.yml
SUBMIT_CONFIG ?= configs/submit.yml
BEST_CONFIG ?= configs/best.yml
SUBMISSION ?= submission.csv
KAGGLE_PACKAGE ?= artifacts/rogii_source.zip
KERNEL ?= rogii-hgb-submit
KERNEL_TITLE ?= ROGII HGB Submit
KERNEL_DIR ?= artifacts/kaggle_kernel
KERNEL_OUTPUT_DIR ?= artifacts/kaggle_output
KAGGLE_DATA_DIR ?= /kaggle/input/$(COMPETITION)
KAGGLE_ARTIFACT_DIR ?= artifacts/submit
KERNEL_TIMEOUT ?= 32400
KERNEL_WAIT_TIMEOUT ?= 36000
KERNEL_POLL_INTERVAL ?= 60
NOTEBOOK ?=
VERSION ?=
MESSAGE ?= Submission
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

.PHONY: install-deps download-data unzip-data ensure-data train quick-train train-submit mine-code mine-discussions research-brief research-db package-kaggle prepare-kaggle-kernel submit submit-kaggle submit-kaggle-dry submit-version

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
	$(UV) run python -m rogii --config $(CONFIG)

quick-train:
	$(UV) run python -m rogii --config configs/quick.yml

train-submit: ensure-data
	$(UV) run python -m rogii --config $(SUBMIT_CONFIG)

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
	$(PYTHON) -m rogii.kaggle_submit prepare --competition $(COMPETITION) --user $(KAGGLE_USER) --kernel $(KERNEL) --title "$(KERNEL_TITLE)" --config $(BEST_CONFIG) --kernel-dir $(KERNEL_DIR) --data-dir $(KAGGLE_DATA_DIR) --artifact-dir $(KAGGLE_ARTIFACT_DIR) --submission-file $(SUBMISSION)

submit: submit-kaggle

submit-kaggle:
	$(PYTHON) -m rogii.kaggle_submit run --competition $(COMPETITION) --user $(KAGGLE_USER) --kernel $(KERNEL) --title "$(KERNEL_TITLE)" --config $(BEST_CONFIG) --kernel-dir $(KERNEL_DIR) --data-dir $(KAGGLE_DATA_DIR) --artifact-dir $(KAGGLE_ARTIFACT_DIR) --output-dir $(KERNEL_OUTPUT_DIR) --submission-file $(SUBMISSION) --kernel-timeout $(KERNEL_TIMEOUT) --wait-timeout $(KERNEL_WAIT_TIMEOUT) --poll-interval $(KERNEL_POLL_INTERVAL) --message "$(MESSAGE)"

submit-kaggle-dry:
	$(PYTHON) -m rogii.kaggle_submit run --competition $(COMPETITION) --user $(KAGGLE_USER) --kernel $(KERNEL) --title "$(KERNEL_TITLE)" --config $(BEST_CONFIG) --kernel-dir $(KERNEL_DIR) --data-dir $(KAGGLE_DATA_DIR) --artifact-dir $(KAGGLE_ARTIFACT_DIR) --output-dir $(KERNEL_OUTPUT_DIR) --submission-file $(SUBMISSION) --kernel-timeout $(KERNEL_TIMEOUT) --wait-timeout $(KERNEL_WAIT_TIMEOUT) --poll-interval $(KERNEL_POLL_INTERVAL) --message "$(MESSAGE)" --dry-run

submit-version:
	@test -n "$(NOTEBOOK)" || (echo "Usage: make submit-version NOTEBOOK=<NOTEBOOK> VERSION=<VERSION> MESSAGE=\"Message\"" && exit 1)
	@test -n "$(VERSION)" || (echo "Usage: make submit-version NOTEBOOK=<NOTEBOOK> VERSION=<VERSION> MESSAGE=\"Message\"" && exit 1)
	$(KAGGLE) competitions submit -c $(COMPETITION) -f $(SUBMISSION) -k $(KAGGLE_USER)/$(NOTEBOOK) -v $(VERSION) -m "$(MESSAGE)"
