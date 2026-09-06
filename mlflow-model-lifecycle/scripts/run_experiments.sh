#!/usr/bin/env bash
# Regenerates the inputs of docs/RESULTS.md in a fresh local MLflow store:
# baseline + two challengers, the promotion gate for each, paired comparisons, model cards.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-python}"
STORE="${MLREG_STORE:-mlruns}"
OUT="docs/experiments"
export MLFLOW_DISABLE_AGENT_HINT=1 MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR=false
export MLREG_TRACKING_URI="sqlite:///${STORE}/mlflow.db"

rm -rf "${STORE}" "${OUT}"
mkdir -p "${OUT}"

run() { echo; echo "\$ $*"; "$@"; }
gate() {  # gate <config> <version> ; records the exit code instead of aborting (HOLD = 2)
  local cfg="$1" v="$2" code=0
  run "$PY" -m mlreg.cli promote --config "$cfg" --policy gates/promotion.yaml \
      --candidate-version "$v" --out "${OUT}/gate_v${v}" --approved-by river || code=$?
  echo "exit_code=${code}" > "${OUT}/gate_v${v}/exit.txt"
  echo "gate v${v}: exit ${code}"
}

run "$PY" -m mlreg.cli validate --data data/german_credit.csv --schema data/schema.yaml | tee "${OUT}/validate.txt"

run "$PY" -m mlreg.cli train --config configs/baseline_logreg.yaml --register --alias challenger --out "${OUT}/baseline_logreg"
gate configs/baseline_logreg.yaml 1

run "$PY" -m mlreg.cli train --config configs/challenger_hgb.yaml --register --alias challenger --out "${OUT}/challenger_hgb"
gate configs/challenger_hgb.yaml 2

run "$PY" -m mlreg.cli train --config configs/challenger_rf.yaml --register --alias challenger --out "${OUT}/challenger_rf"
gate configs/challenger_rf.yaml 3

for pair in "1 2" "1 3" "2 3"; do
  set -- $pair
  run "$PY" -m mlreg.cli compare --config configs/baseline_logreg.yaml --versions "$1" "$2" --n-boot 2000 | tee "${OUT}/compare_v$1_v$2.txt"
done

run "$PY" -m mlreg.cli serve-check --config configs/baseline_logreg.yaml --alias champion | tee "${OUT}/serve_check.txt"
echo
echo "done: ${OUT}/ (store: ${STORE}/; browse with: mlflow ui --backend-store-uri ${MLREG_TRACKING_URI})"
