#!/usr/bin/env bash
# Regenerates docs/experiments/: the container contract exercised end to end without Docker
# (train → serve → invoke in every content type), the rendered AWS request plan, the
# CloudFormation lint, and a small in-process latency benchmark of the hosting server.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-python}"
OUT="docs/experiments"
BASE="runs/opt-ml"
rm -rf "${OUT}" "${BASE}"
mkdir -p "${OUT}"

run() { echo; echo "\$ $*"; "$@"; }

run "$PY" scripts/make_example_data.py --out examples | tee "${OUT}/make_example_data.txt"

run "$PY" -m smdeploy.cli local-train --train-csv examples/credit_train.csv \
    --validation-csv examples/credit_validation.csv --hyperparameters examples/hyperparameters.json \
    --base-dir "${BASE}" | tee "${OUT}/local_train.txt"
cp "${BASE}/model/metadata.json" "${OUT}/model_metadata.json"
cp "${BASE}/output/data/evaluation.json" "${OUT}/evaluation.json"

run "$PY" -m smdeploy.cli local-invoke --model-dir "${BASE}/model" --payload examples/smoke_rows.csv \
    --content-type "text/csv; header=present" --accept application/json | tee "${OUT}/invoke_csv_json.txt"
run "$PY" -m smdeploy.cli local-invoke --model-dir "${BASE}/model" --payload examples/smoke_rows.csv \
    --content-type "text/csv; header=present" --accept text/csv | tee "${OUT}/invoke_csv_csv.txt"

export IMAGE_URI="123456789012.dkr.ecr.ap-southeast-2.amazonaws.com/credit-pd:abc1234"
export SAGEMAKER_EXECUTION_ROLE_ARN="arn:aws:iam::123456789012:role/credit-pd-sagemaker-execution"
export ARTIFACT_BUCKET="credit-pd-artifacts-123456789012-ap-southeast-2"
export PROJECT_NAME="credit-pd"
"$PY" -m smdeploy.cli plan --training configs/training.yaml --stage configs/endpoint.prod.yaml > "${OUT}/plan_prod.json"
"$PY" -m smdeploy.cli plan --stage configs/endpoint.staging.yaml > "${OUT}/plan_staging.json"
echo "plan written"

run cfn-lint infra/cloudformation/platform.yaml | tee "${OUT}/cfn_lint.txt" || true
echo "cfn-lint exit: ${PIPESTATUS[0]}" | tee -a "${OUT}/cfn_lint.txt"

run "$PY" scripts/bench_server.py --model-dir "${BASE}/model" --out "${OUT}/server_latency.md"

echo
echo "done: ${OUT}/"
