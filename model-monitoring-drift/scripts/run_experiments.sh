#!/usr/bin/env bash
# Regenerates docs/experiments/: a simulated production month with drift injected from a
# given window, one monitoring run per window with carried state, a stationary control, and
# the detection-power / false-alarm evaluation of the policy itself.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-python}"
OUT="docs/experiments"
SIM="runs/sim"
POLICY="policies/default.yaml"
rm -rf "${OUT}" "${SIM}"
mkdir -p "${OUT}"

run() { echo; echo "\$ $*"; "$@"; }

monitor_series() {  # monitor_series <name> <scenario> <magnitude> <windows>
  local name="$1" scenario="$2" magnitude="$3" windows="$4" dir="${SIM}/$1" code
  run "$PY" -m mlwatch.cli simulate --scenario "$scenario" --magnitude "$magnitude" --windows "$windows" --out "$dir" > /dev/null
  "$PY" -m mlwatch.cli baseline --capture "$dir/baseline.jsonl" --labels "$dir/labels.jsonl" --out "$dir/baseline.json" > "${OUT}/${name}_baseline.json"
  rm -f "$dir/monitor.state.json"
  : > "${OUT}/${name}_series.txt"
  for w in $(seq -f "%02g" 1 "$windows"); do
    code=0
    "$PY" -m mlwatch.cli run --baseline "$dir/baseline.json" --capture "$dir/window_${w}.jsonl" --labels "$dir/labels.jsonl" \
        --policy "$POLICY" --state "$dir/monitor.state.json" --out "${OUT}/${name}_window_${w}" --window-id "${name}-${w}" > /dev/null || code=$?
    status=$("$PY" -c "import json;d=json.load(open('${OUT}/${name}_window_${w}/report.json'));print(d['decision']['status'], d['decision']['action'])")
    echo "window ${w}: exit ${code} ${status}" | tee -a "${OUT}/${name}_series.txt"
  done
}

monitor_series stationary none 0.0 4
monitor_series covariate_shift covariate_shift 1.0 4
monitor_series concept_drift concept_drift 1.8 4
monitor_series quality_unseen quality_unseen 0.05 2
monitor_series score_shift score_shift 0.3 2

run "$PY" -m mlwatch.cli evaluate --policy "$POLICY" --seeds "${SEEDS:-20}" --out "${OUT}/monitor_evaluation.md"

echo
echo "done: ${OUT}/"
