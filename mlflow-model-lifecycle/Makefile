# Same gates as CI. Usage: make install | make lint | make type | make test | make all | make results
PY ?= python

.PHONY: install lint format type test all clean results data mlflow-ui

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

# Rebuilds data/german_credit.csv from the raw UCI file (see data/README.md).
data:
	$(PY) scripts/prepare_data.py --raw data/raw/german.data --out data/german_credit.csv

# Trains baseline + challengers against a local SQLite tracking store, runs the promotion
# gate and writes docs/RESULTS.md inputs under docs/experiments/.
results:
	bash scripts/run_experiments.sh

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml
