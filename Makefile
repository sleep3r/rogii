COMPETITION := rogii-wellbore-geology-prediction
KAGGLE_USER ?= sleep3r
UV ?= uv
PYTHON ?= $(UV) run python
KAGGLE ?= $(UV) run kaggle
DATA_DIR ?= data
CONFIG ?= configs/hgb.yml
SUBMISSION ?= submission.csv
NOTEBOOK ?=
VERSION ?=
MESSAGE ?= Submission

.PHONY: install-deps download-data unzip-data ensure-data baseline train quick-train validate-baseline submit

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

baseline:
	$(PYTHON) scripts/baseline.py --data-dir $(DATA_DIR) --output $(SUBMISSION)

train: ensure-data
	$(PYTHON) train.py --config $(CONFIG)

quick-train:
	$(PYTHON) train.py --config configs/quick.yml

validate-baseline:
	$(PYTHON) scripts/baseline.py --data-dir $(DATA_DIR) --output $(SUBMISSION) --truth-dir $(DATA_DIR)

submit:
	@test -n "$(NOTEBOOK)" || (echo "Usage: make submit NOTEBOOK=<NOTEBOOK> VERSION=<VERSION> MESSAGE=\"Message\"" && exit 1)
	@test -n "$(VERSION)" || (echo "Usage: make submit NOTEBOOK=<NOTEBOOK> VERSION=<VERSION> MESSAGE=\"Message\"" && exit 1)
	$(KAGGLE) competitions submit -c $(COMPETITION) -f $(SUBMISSION) -k $(KAGGLE_USER)/$(NOTEBOOK) -v $(VERSION) -m "$(MESSAGE)"
