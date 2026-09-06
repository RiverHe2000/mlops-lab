# Results

Produced by `make results` (`scripts/run_experiments.sh`) on 2026-09-06 on Windows 11,
Python 3.12, scikit-learn 1.9.0, no Docker daemon and no AWS account. Raw outputs are in
[`experiments/`](experiments/). The two things that need infrastructure — building the image
and running it — are done by the `container` job in `.github/workflows/ci.yml`; the two that
need AWS — a real training job and a real endpoint — are exercised against moto and a botocore
Stubber in the test suite and rendered as review-able requests by `smdeploy plan`.

## 1. Training contract (local simulation of `docker run image train`)

`write_sagemaker_layout` lays out `/opt/ml` under `runs/opt-ml/`, copies
`examples/credit_train.csv` (480 rows) and `examples/credit_validation.csv` (120 rows) into the
`train` and `validation` channels, writes `hyperparameters.json` with string values exactly as
SageMaker does, then `run_training` executes:

```
validation:auc=0.697192
validation:log_loss=0.593690
validation:brier=0.204551
validation:accuracy=0.608333
validation:precision=0.452830
validation:recall=0.571429
```

| Artefact | Content |
|---|---|
| `model/model.joblib` | fitted `Pipeline(ColumnTransformer(impute+scale, impute+one-hot), LogisticRegression(C=0.5))` |
| `model/metadata.json` | 10 feature columns (6 numeric, 4 categorical), target `default`, threshold 0.3, metrics on validation, `model_sha256 = 80a2f858…`, sklearn 1.9.0, Python 3.12 |
| `output/data/evaluation.json` | the same metrics + sha256 (uploaded by SageMaker as `output.tar.gz`, consumed by `smdeploy register`) |
| `checkpoints/checkpoint.json` | `{"status": "fitted"}` — the spot-training hook |

The synthetic data is deliberately noisy (AUC ≈ 0.70 on 120 validation rows); the point of
this project is the contract and the release path, the model quality story lives in
`mlflow-model-lifecycle`.

## 2. Hosting contract (local simulation of `docker run image serve`)

`smdeploy local-invoke` drives the same FastAPI app SageMaker would call:

```
$ smdeploy local-invoke --model-dir runs/opt-ml/model --payload examples/smoke_rows.csv \
      --content-type "text/csv; header=present" --accept application/json
HTTP 200 application/json
{"predictions": [{"probability": 0.1166, "label": 0}, {"probability": 0.5948, "label": 1},
                 {"probability": 0.1403, "label": 0}, {"probability": 0.3741, "label": 1},
                 {"probability": 0.3773, "label": 1}]}

$ smdeploy local-invoke ... --accept text/csv
HTTP 200 text/csv; charset=utf-8
0.116645,0
0.594784,1
0.140282,0
0.374082,1
0.377338,1
```

### Latency (in-process ASGI client, `scripts/bench_server.py`)

| Batch rows | Content type | p50 ms | p95 ms | p99 ms | rows / s |
|---:|---|---:|---:|---:|---:|
| 1 | text/csv | 7.95 | 8.72 | 9.08 | 126 |
| 1 | application/json | 7.92 | 8.70 | 9.61 | 126 |
| 10 | text/csv | 8.39 | 9.76 | 11.49 | 1 192 |
| 10 | application/json | 8.32 | 12.10 | 14.42 | 1 202 |
| 100 | text/csv | 8.53 | 9.00 | 9.00 | 11 717 |
| 100 | application/json | 8.64 | 9.49 | 9.49 | 11 574 |
| 1 000 | text/csv | 11.18 | 11.61 | 11.61 | 89 408 |
| 1 000 | application/json | 12.16 | 13.29 | 13.29 | 82 264 |

Reading: a flat ≈ 8 ms per request regardless of size (FastAPI/Starlette request handling,
pandas frame construction, sklearn pipeline dispatch), then ≈ 3 µs per row. For real-time
scoring the per-request floor dominates and two uvicorn workers per vCPU are plenty; for bulk
scoring use batch transform with `BatchStrategy: MultiRecord` and payloads near
`MaxPayloadInMB`, which the container advertises on `/execution-parameters`.

## 3. Requests the release path would send (`smdeploy plan`)

`docs/experiments/plan_prod.json` renders, without calling AWS, the `CreateTrainingJob`
request from `configs/training.yaml` and the production stage from `configs/endpoint.prod.yaml`
(placeholders filled from the environment):

```json
"HyperParameters": {"target": "default", "positive_label": "1", "id_columns": "application_id",
                    "model": "logistic_regression", "C": "0.5", "threshold": "0.3", "seed": "42"},
"StoppingCondition": {"MaxRuntimeInSeconds": 1800, "MaxWaitTimeInSeconds": 3600},
"EnableManagedSpotTraining": true,
"CheckpointConfig": {"S3Uri": "s3://credit-pd-artifacts-.../checkpoints/credit-pd-manual/"}
```

```json
"DeploymentConfig": {
  "BlueGreenUpdatePolicy": {
    "TrafficRoutingConfiguration": {"WaitIntervalInSeconds": 300, "Type": "CANARY",
                                    "CanarySize": {"Type": "CAPACITY_PERCENT", "Value": 10}},
    "TerminationWaitInSeconds": 300, "MaximumExecutionTimeoutInSeconds": 3600}
}
```

At release time `release.py` adds `AutoRollbackConfiguration` with the two alarms it creates
(`credit-pd-prod-5xx-errors`: `Invocation5XXErrors` sum ≥ 1 over 2 × 60 s;
`credit-pd-prod-latency-p99`: `ModelLatency` p99 > 1 000 ms) and, after the smoke test,
registers autoscaling 2 → 6 instances on 60 invocations per instance.

## 4. Infrastructure and packaging checks

| Check | Result |
|---|---|
| `cfn-lint infra/cloudformation/platform.yaml` | exit 0 (also asserted in `test_cloudformation_template_lints_clean`, regions `ap-southeast-2`, `us-east-1`) |
| Dockerfile structure (`dockerfile-parse`) | pinned base images, `USER sagemaker`, `EXPOSE 8080`, `HEALTHCHECK`, `train`/`serve` copied, **no `ENTRYPOINT`**, `CMD ["serve"]` |
| Workflows | `cd.yml` has `id-token: write`, uses `role-to-assume`, never `aws-access-key-id`; `deploy-staging` → `approve` (environment `production`) → `deploy-production` chain |
| Test suite | 70 tests, 95 % branch coverage, `mypy --strict` with typed boto3 clients |

## 5. What was verified against the real API schema without an account

| Operation | How |
|---|---|
| `CreateTrainingJob` | executed against moto (job completes), request built from a validated spec |
| `CreateModelPackageGroup` / `CreateModelPackage` / `UpdateModelPackage` / `ListModelPackages` | moto, full approval workflow; `CreateModelPackage` additionally validated by a botocore Stubber |
| `CreateModel` / `CreateEndpointConfig` / `CreateEndpoint` / `Describe*` / `Delete*` | moto |
| `UpdateEndpoint` with `DeploymentConfig` (all three routing modes + alarms) | botocore Stubber against the service model (moto does not implement the operation); status transitions against a scripted fake |
| `RegisterScalableTarget` / `PutScalingPolicy` / `PutMetricAlarm` | moto |
| `InvokeEndpoint` | Stubber with good / short / unparseable / out-of-range / error responses |
| ECR repository, lifecycle policy, authorization token | moto |

What this does **not** prove: instance provisioning times, real CloudWatch alarm latency, IAM
propagation delays, and image-pull behaviour — those need a sandbox account, and the CD workflow
is the first thing to run there.

## 6. The real container, run locally (Docker Desktop 4.89 / engine 29.7.2, 2026-09-06)

`make image`, then the two `docker run` lines from the README against a `/opt/ml` layout
written by `smdeploy local-train` (so the in-process simulation and the container saw the same
bytes):

| Step | Observed |
|---|---|
| `docker build` | `smdeploy:local`, 770 MB (python:3.12-slim + venv), runs as `sagemaker` (uid 10001), no `ENTRYPOINT` |
| `docker run -v …:/opt/ml smdeploy:local train` | read the string-typed `hyperparameters.json`, the `train` channel (600 rows) and the leftover checkpoint ("single-shot fit restarts from scratch"); printed the `validation:<name>=<value>` lines; wrote `model.joblib` + `metadata.json`; exit 0, no `/opt/ml/output/failure` |
| Same numbers as the simulation | AUC 0.7070, log-loss 0.5517, Brier 0.1849, accuracy 0.675, n = 120 — identical to `local-train` on the same layout |
| `docker run -p 8080:8080 … serve` | `/ping` 200 on the first probe, `/execution-parameters` → `MaxConcurrentTransforms 2, MultiRecord, 6 MB`, container `HEALTHCHECK` green |
| Content negotiation | csv → json (`X-Amzn-SageMaker-Custom-Attributes` echoed, `x-request-id` set), csv → jsonlines, json `instances` → csv; `application/xml` → 415; `Accept: image/png` → 406 |
| `SMDEPLOY_THRESHOLD=0.5` | identical probabilities, the two rows between 0.3 and 0.5 flip from 1 to 0 |
| Structured logs | one JSON line per invocation: `request_id`, `rows`, `latency_ms` (≈ 16 ms for 6 rows, 2 workers), `status`, content types |

One fix came out of the run. A `text/csv` request that *carries* a header but not the
`header=present` parameter used to be scored as one extra record — imputed numerics plus unseen
categories give a plausible-looking probability, which is the worst kind of wrong. The codec now
recognises a first row equal to the feature names and answers 400 with the remedy in the message
(`test_decode_csv_variants` covers it, and the rebuilt image was re-checked).
