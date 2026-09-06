# Same gates as CI. Usage: make install | make lint | make type | make test | make all | make results
PY ?= python

.PHONY: install lint format type test all clean results image local-train local-serve

install:
	$(PY) -m pip install -e ".[dev]"

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

type:
	mypy

test:
	pytest

all: lint type test

# Simulates the SageMaker container contract without Docker: lays out /opt/ml in a temp dir,
# runs `train`, then serves and invokes the model. Writes docs/experiments/.
results:
	bash scripts/run_experiments.sh

# The BYOC image (needs Docker). `docker run image train` / `docker run -p 8080:8080 image serve`.
image:
	docker build -t smdeploy:local -f container/Dockerfile .

local-train:
	smdeploy local-train --train-csv examples/credit_sample.csv --hyperparameters examples/hyperparameters.json --base-dir runs/opt-ml

local-serve:
	SMDEPLOY_BASE_DIR=runs/opt-ml smdeploy-serve

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml
