# Same gates as CI. Usage: make install | make lint | make type | make test | make all | make results
PY ?= python

.PHONY: install lint format type test all clean results demo serve stack

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

# Simulated production traffic with injected drift, one monitoring run per window, and the
# detection-power / false-alarm evaluation of the monitor itself. Writes docs/experiments/.
results:
	bash scripts/run_experiments.sh

# Quick local demo: simulate, build a baseline, monitor one drifted window.
demo:
	mlwatch simulate --scenario covariate_shift --magnitude 1.0 --windows 3 --out runs/demo
	mlwatch baseline --capture runs/demo/baseline.jsonl --labels runs/demo/labels.jsonl --out runs/demo/baseline.json
	mlwatch run --baseline runs/demo/baseline.json --capture runs/demo/window_03.jsonl --labels runs/demo/labels.jsonl --policy policies/default.yaml --out runs/demo/report_03

serve:
	mlwatch serve --baseline runs/demo/baseline.json --policy policies/default.yaml --port 9108

# Prometheus + Grafana + the exporter (needs Docker).
stack:
	docker compose -f deploy/docker-compose.yml up --build

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml
