# Runs the same quality gates as CI for every project in this repository.
# Usage: make install | make lint | make type | make test | make all
#        make PROJECT=mlflow-model-lifecycle test      # one project only
PROJECTS := mlflow-model-lifecycle sagemaker-byoc-deploy model-monitoring-drift
PROJECT ?= $(PROJECTS)
PY ?= python

.PHONY: install lint format type test all clean

install:
	$(PY) -m pip install --upgrade pip
	@for p in $(PROJECTS); do echo "== install $$p"; $(PY) -m pip install -e "$$p[dev]" || exit 1; done

lint:
	@for p in $(PROJECT); do echo "== ruff $$p"; (cd $$p && ruff check . && ruff format --check .) || exit 1; done

format:
	@for p in $(PROJECT); do echo "== format $$p"; (cd $$p && ruff check --fix . && ruff format .) || exit 1; done

type:
	@for p in $(PROJECT); do echo "== mypy $$p"; (cd $$p && mypy) || exit 1; done

test:
	@for p in $(PROJECT); do echo "== pytest $$p"; (cd $$p && pytest) || exit 1; done

all: lint type test

clean:
	@for p in $(PROJECTS); do rm -rf $$p/.pytest_cache $$p/.mypy_cache $$p/.ruff_cache $$p/htmlcov $$p/.coverage; done
