# smdeploy · bring-your-own-container ML on AWS SageMaker, testable offline

A credit-default classifier packaged the way SageMaker actually runs custom code: one image
that serves as both the **training container** (`docker run image train`, the `/opt/ml`
contract, metric regexes, spot checkpoints, failure files) and the **hosting container**
(`docker run image serve`, `/ping` and `/invocations` on 8080, content negotiation, payload
limits). Around it: an ECR + IAM + S3 platform in CloudFormation, a Model Registry with an
approval step, **blue/green endpoint updates with canary traffic and CloudWatch-alarm
auto-rollback**, target-tracking autoscaling, a smoke test through the runtime API, and a
GitHub Actions CD pipeline that assumes an OIDC role — no static keys anywhere.

Every AWS interaction is exercised without an account: the container contract runs in a
temporary directory, the API calls run against **moto**, and request shapes SageMaker does not
mock (`UpdateEndpoint` with a `DeploymentConfig`) are validated against the real botocore
service model with a `Stubber`.

| | |
|---|---|
| Quality gates | `ruff`, `mypy --strict` (with `boto3-stubs`), **70 tests** (offline, ≈ 10 s), **95 % branch coverage**, `cfn-lint` on the template, structural checks on the Dockerfile and both workflows |
| Contract coverage | training: hyperparameters as strings, channels, `SM_*` overrides, metric lines, `/opt/ml/model`, `/opt/ml/output/failure`, checkpoints · hosting: `/ping` 200/503, csv / json / jsonlines in and out, 400 / 406 / 413 / 415 / 500 mapping, custom-attributes echo, `/execution-parameters` |
| Release path | ECR (immutable tags, scan on push, lifecycle) → training job (spot + checkpoints) → model package (pending → approved) → staging (all-at-once + smoke) → **human approval** → production (canary 10 % · alarms · rollback · autoscaling) |
| Headline | In-process hosting latency **p50 8 ms for 1 row, 11 ms for 1 000 rows (≈ 89 k rows/s)**; the whole train → serve → invoke path and the CloudFormation template are verified in CI without Docker or AWS; the Docker image is built and driven end-to-end in the `container` CI job. Details in [docs/RESULTS.md](docs/RESULTS.md) |

Companion projects: [`mlflow-model-lifecycle`](../mlflow-model-lifecycle) (the model, its registry
and gated promotion) and [`model-monitoring-drift`](../model-monitoring-drift) (what happens after
deployment). Together: *track, ship, watch*. Azure ML equivalents of every piece:
[docs/AZURE_ML_MAPPING.md](docs/AZURE_ML_MAPPING.md).

---

## 1. Architecture

```
 container/Dockerfile ─ one image, two entrypoints on PATH ─────────────────────────────────────┐
   train  → smdeploy-train → paths.py (/opt/ml tree) → hyperparams.py (strings → types)         │
            → training.py: channels → sklearn pipeline → "validation:auc=0.697" lines           │
            → model_io.py: model.joblib + metadata.json (sha256) → /opt/ml/model, output/data   │
   serve  → smdeploy-serve → server.py (FastAPI): /ping · /invocations · /execution-parameters  │
            → codecs.py (csv/json/jsonlines ↔ frame) → inference.py (align, impute, threshold)  │
                                                                                                │
 aws/                                                                                           │
   ecr.py           ensure repo · login token · build/tag/push (runner injectable)              │
   training_job.py  TrainingJobSpec → CreateTrainingJob · wait (clock injectable) · outputs     │
   registry.py      model package group · register (metrics, evaluation S3) · approve · latest  │
   endpoint.py      content-addressed model/config names · create | blue/green update           │
                    (canary / linear / all-at-once, AutoRollbackConfiguration) · wait · rollback │
   operations.py    autoscaling target tracking · CloudWatch rollback alarms · smoke test        │
 release.py         stage YAML → alarms → deploy → wait → smoke → autoscaling (rollback on fail) │
 infra/cloudformation/platform.yaml   bucket · ECR · exec role · model package group · OIDC role │
 .github/workflows/cd.yml             build → train → register → staging → approve → production │
```

| Concern | Files | What is worth knowing |
|---|---|---|
| Filesystem contract | `paths.py` | `SageMakerPaths` resolves every path from an injectable base (`SMDEPLOY_BASE_DIR`) and honours `SM_CHANNEL_*`, `SM_MODEL_DIR`, `SM_OUTPUT_DATA_DIR`; the same object drives production and the local simulator |
| Hyperparameters | `hyperparams.py` | SageMaker transports `map<string,string>`; values are JSON-decoded and validated by pydantic, `sagemaker_*` keys ignored, `to_sagemaker_dict()` is the exact inverse (tested as a round trip) |
| Training | `training.py` | metrics printed as `validation:<name>=<value>` to match the `MetricDefinitions` regexes (tested against them), `evaluation.json` in `output/data`, failure reason to `output/failure` on any exception, checkpoint marker when `/opt/ml/checkpoints` exists |
| Artefact integrity | `model_io.py` | `metadata.json` carries the SHA-256 of `model.joblib`; hosting refuses to load (503 on `/ping`) if the bytes do not match — a truncated `model.tar.gz` fails the health check instead of scoring garbage |
| Hosting | `server.py`, `codecs.py`, `inference.py` | content negotiation on `Content-Type`/`Accept`, `SAGEMAKER_MAX_PAYLOAD_IN_MB` enforced before parsing, custom attributes and request ids echoed, JSON logs; batch transform hints on `/execution-parameters` |
| Deployment | `aws/endpoint.py` | endpoint configs are immutable, so names embed a hash of the spec → re-deploying the same thing is a no-op, a change becomes `UpdateEndpoint` with `BlueGreenUpdatePolicy` + `AutoRollbackConfiguration`; `wait_in_service` detects a SageMaker-initiated rollback (InService on the *old* config) and raises |
| Release | `release.py` | one function a CD job calls per stage; smoke failure after an update → explicit rollback to the previous config; Approved-only packages for production |
| Identity & infra | `infra/cloudformation/platform.yaml` | least-privilege execution role, GitHub OIDC provider + deploy role scoped to `repo:org/repo:*` refs/environments and to `${ProjectName}*` resources, `iam:PassRole` conditioned on `sagemaker.amazonaws.com`; validated with `cfn-lint` in the test suite |

---

## 2. Results

Everything below was produced on this machine with `make results` (no Docker, no AWS); the image
was then built and driven through `train`/`serve` on Docker Desktop with identical metrics
([RESULTS.md §6](docs/RESULTS.md)), which is also what the CI `container` job does on every push.

**Training contract, local simulation** (`examples/credit_train.csv`, 480 rows; validation 120 rows;
logistic regression, `threshold 0.3`):

| Metric | Value |
|---|---:|
| validation:auc | 0.697 |
| validation:log_loss | 0.594 |
| validation:brier | 0.205 |
| validation:accuracy | 0.608 |
| validation:precision / recall @0.3 | 0.453 / 0.571 |

The metric lines are captured by the default `MetricDefinitions` regexes (a test asserts each
regex recovers the printed value); the model directory holds `model.joblib`, `metadata.json`
(sha256 `80a2f858…`) and `output/data/evaluation.json`.

**Hosting latency, in-process** (`scripts/bench_server.py`, ASGI test client, no network —
isolates codec + model cost from SageMaker and network overhead):

| Batch rows | Content type | p50 ms | p95 ms | p99 ms | rows / s |
|---:|---|---:|---:|---:|---:|
| 1 | text/csv | 7.95 | 8.72 | 9.08 | 126 |
| 1 | application/json | 7.92 | 8.70 | 9.61 | 126 |
| 10 | text/csv | 8.39 | 9.76 | 11.49 | 1 192 |
| 100 | text/csv | 8.53 | 9.00 | 9.00 | 11 717 |
| 1 000 | text/csv | 11.18 | 11.61 | 11.61 | 89 408 |
| 1 000 | application/json | 12.16 | 13.29 | 13.29 | 82 264 |

The ≈ 8 ms floor is per-request overhead (framework + pandas frame construction); model cost is
≈ 3 µs/row, so batch transform with `MultiRecord` is the right mode for bulk scoring.

**Plan for production** (`smdeploy plan --stage configs/endpoint.prod.yaml`): canary 10 % of
capacity for 300 s, automatic rollback on the `credit-pd-prod-5xx-errors` / `-latency-p99`
alarms, then full shift, 2 → 6 instances tracking 60 invocations per instance. The rendered
`CreateTrainingJob` and `DeploymentConfig` requests are in `docs/experiments/plan_prod.json`.

---

## 3. Tests: what each one proves

| File | Proves |
|---|---|
| `test_contract_paths_hyperparams.py` | the `/opt/ml` layout and every `SM_*` override; channel files are discovered recursively and deterministically; string hyperparameters decode (`"null"`, JSON lists, comma lists) and round-trip |
| `test_training_contract.py` | a training run writes model, metadata, evaluation and checkpoint; metric lines match the CloudWatch regexes; a tampered `model.joblib` is refused; missing channel / single-class target / missing target write the failure file; validation channel vs split vs in-sample; all three estimators; `main()` exit codes |
| `test_codecs_inference.py` | every request shape (csv with/without header, five JSON layouts, jsonlines) and every response type; 400/406/415 with the right messages; column alignment, string coercion, missing values, threshold override, lazy loader states |
| `test_server_local_run.py` | `/ping` 200 vs 503, `/invocations` for each content type, 413 by body and by `Content-Length`, custom attributes echoed even on errors, 500 path, JSON logging, the simulator layout, `serve` settings from SageMaker env vars and the uvicorn factory |
| `test_aws_ecr_training_registry.py` | ECR repo idempotence and lifecycle policy; login token decoding; the docker command sequence never leaks the password; `CreateTrainingJob` request shape (strings, spot, checkpoints); wait loop transitions, failure reason, timeout with a fake clock; registry pending → approved → rejected and `latest` semantics; a Stubber validates `CreateModelPackage` against the service model |
| `test_aws_endpoint_release.py` | content-addressed names; `DeploymentConfig` for all three modes accepted by the real `UpdateEndpoint` model; create path under moto; blue/green update, SageMaker-initiated rollback and failure detection against a scripted fake; explicit rollback; autoscaling and alarm creation; smoke test outcomes; the release procedure end to end including rollback after a failed smoke test |
| `test_cli_infra.py` | every CLI command (local train/invoke, plan, push dry-run, register with a metric floor, approve, latest, deploy success/failure, status, rollback, teardown); `cfn-lint` passes; the Dockerfile has no `ENTRYPOINT`, a non-root user, port 8080 and the two entrypoints; CD uses OIDC and environments; example data generation |

---

## 4. Reproduce

```bash
python -m venv .venv && . .venv/Scripts/activate      # or source .venv/bin/activate
pip install -e ".[dev]"
make all                                              # ruff + mypy --strict + pytest (≈ 10 s)
make results                                          # simulate train → serve → invoke, plan, cfn-lint, latency → docs/experiments/

# the contract without Docker
python scripts/make_example_data.py
smdeploy local-train --train-csv examples/credit_train.csv --validation-csv examples/credit_validation.csv \
        --hyperparameters examples/hyperparameters.json --base-dir runs/opt-ml
smdeploy local-invoke --model-dir runs/opt-ml/model --payload examples/smoke_rows.csv \
        --content-type "text/csv; header=present" --accept application/json

# the real container (needs Docker)
make image
docker run --rm -v "$PWD/runs/opt-ml:/opt/ml" smdeploy:local train
docker run -d -p 8080:8080 -v "$PWD/runs/opt-ml/model:/opt/ml/model:ro" smdeploy:local serve
# plain "text/csv" is headerless (SageMaker's convention); a header row sent without the flag is a 400, not a prediction
curl -s localhost:8080/ping && curl -s -X POST -H "Content-Type: text/csv; header=present" \
     --data-binary @examples/smoke_rows.csv localhost:8080/invocations

# AWS (after `aws cloudformation deploy --template-file infra/cloudformation/platform.yaml ...`)
export IMAGE_URI=... SAGEMAKER_EXECUTION_ROLE_ARN=... ARTIFACT_BUCKET=... PROJECT_NAME=credit-pd
smdeploy plan --training configs/training.yaml --stage configs/endpoint.prod.yaml   # review before touching anything
smdeploy push --repository credit-pd --tag "$(git rev-parse --short HEAD)"
smdeploy train-job --config configs/training.yaml --out training.json
smdeploy register --group credit-pd --image-uri "$IMAGE_URI" --model-data-url s3://... --metrics-json training.json --min-auc 0.70
smdeploy deploy --stage configs/endpoint.staging.yaml --smoke-csv examples/smoke_rows.csv
smdeploy approve --package-arn arn:aws:sagemaker:... --by "$USER" --note "staging smoke passed"
smdeploy deploy --stage configs/endpoint.prod.yaml --smoke-csv examples/smoke_rows.csv
```

---

## 5. Design decisions

- **One image, two commands.** SageMaker starts BYOC containers with `train` or `serve` as the
  command, so there is no `ENTRYPOINT`; both are tiny shell shims over console scripts, which
  keeps the Python entrypoints unit-testable and the image layout obvious.
- **The contract is tested where it lives.** `SageMakerPaths` makes `/opt/ml` a parameter, so
  the exact code path SageMaker executes runs in a temp directory in every test and in the CI
  matrix; Docker adds packaging, not behaviour. The `container` CI job then proves the image.
- **Content-addressed endpoint configs.** Endpoint configs are immutable in SageMaker; naming
  them by a hash of the spec gives idempotent deployments (same spec → "unchanged") and a
  natural rollback target (the previous name).
- **Blue/green with alarms, not "update and hope".** `DeploymentConfig` shifts a canary first
  and lets SageMaker roll back on the 5xx / p99 alarms; the client additionally checks that the
  endpoint ended InService on the *new* config, because a rollback also ends InService.
- **Smoke test through the runtime API.** After every deployment real rows are sent with
  `InvokeEndpoint` and the response is checked for shape and probability range; a failure after
  an update triggers an explicit rollback to the previous config.
- **Approval is data.** Model packages are registered `PendingManualApproval`; the production
  stage only accepts `Approved`, and the approver, time and note are stored on the package
  itself — the CD `production` environment gate and the registry agree by construction.
- **OIDC, least privilege, IaC.** No access keys: GitHub assumes a role whose trust policy is
  pinned to this repository's refs and environments; the role can only touch `credit-pd*`
  resources and can only pass the execution role to SageMaker. The template is linted in the
  test suite.
- **moto + Stubber, honestly.** moto covers the create/describe/delete surface; where it does
  not implement an operation (`UpdateEndpoint`), a scripted fake models the status machine and
  botocore's `Stubber` validates the request shape against the real API schema — the two
  failure modes (wrong logic, wrong payload) are covered separately and explicitly.
- **Not built:** SageMaker Pipelines (CI is the orchestrator here and reviewable in a PR),
  multi-model endpoints, async/serverless inference beyond the config option, GPU images.

## Related projects

- [`mlflow-model-lifecycle`](../mlflow-model-lifecycle) — where the `@champion` model comes from:
  MLflow tracking, registry aliases and a statistically gated promotion.
- [`model-monitoring-drift`](../model-monitoring-drift) — consumes the endpoint's data capture:
  PSI/KS/chi-square drift, delayed-label performance, a retrain policy and a Prometheus exporter.
