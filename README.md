# mlops-lab

[![mlflow-model-lifecycle](https://github.com/RiverHe2000/mlops-lab/actions/workflows/mlflow-model-lifecycle-ci.yml/badge.svg)](https://github.com/RiverHe2000/mlops-lab/actions/workflows/mlflow-model-lifecycle-ci.yml)
[![sagemaker-byoc-deploy](https://github.com/RiverHe2000/mlops-lab/actions/workflows/sagemaker-byoc-deploy-ci.yml/badge.svg)](https://github.com/RiverHe2000/mlops-lab/actions/workflows/sagemaker-byoc-deploy-ci.yml)
[![model-monitoring-drift](https://github.com/RiverHe2000/mlops-lab/actions/workflows/model-monitoring-drift-ci.yml/badge.svg)](https://github.com/RiverHe2000/mlops-lab/actions/workflows/model-monitoring-drift-ci.yml)

Three independently runnable credit-model operations demonstrations — **track it, ship it,
watch it**: a statistically gated MLflow registry, a SageMaker-compatible container and
deployment workflow, and drift monitoring with a retraining recommendation. Docker components
have local run evidence; AWS deployment and a shared artefact flowing through all three
remain integration work.

| # | Project | What it demonstrates | Headline result |
|---|---|---|---|
| 01 | [mlflow-model-lifecycle](mlflow-model-lifecycle/) | Data contracts, held-out evaluation, registry aliases and promotion | German Credit baseline AUC **0.802 [0.744, 0.856]**; both challengers **HOLD**. New evaluations verify the training run’s actual train/holdout membership before scoring. [Evidence](mlflow-model-lifecycle/docs/RESULTS.md) |
| 02 | [sagemaker-byoc-deploy](sagemaker-byoc-deploy/) | Container contracts, canary configuration, rollback and OIDC workflow | Local container train/serve tested; **8 ms** p50 is in-process latency. AWS calls are validated with moto/Stubber; no cloud deployment result is claimed. [Evidence](sagemaker-byoc-deploy/docs/RESULTS.md) |
| 03 | [model-monitoring-drift](model-monitoring-drift/) | Drift/performance checks and a calibrated alert policy | Simulation: **1/20** stationary windows flagged; **20/20** one-sigma shifts detected. Local Prometheus/Grafana exercised; scheduled demo reports only, retraining dispatch requires manual opt-in. [Evidence](model-monitoring-drift/docs/RESULTS.md) |

Companion repositories: [`llm-engineering-lab`](https://github.com/RiverHe2000/llm-engineering-lab)
(Transformer internals, LoRA, an inference server) and [`genai-platform-lab`](https://github.com/RiverHe2000/genai-platform-lab)
(RAG, agents with guardrails, an LLM gateway).

---

## The through-line

The three stages address these operational questions. They currently use separate example
models/data: German Credit in MLflow, synthetic credit rows in BYOC, and a frozen synthetic
scorer in the monitor. A shared artefact adapter and SageMaker capture normalisation are
needed before claiming a single integrated end-to-end model lifecycle:

1. **"How do you manage experiments and model versions?"** (`mlreg`) — a contract on the
   data, lineage on every run, a registry alias that means something, and a promotion
   decision that is a *statistical* argument (paired bootstrap non-inferiority, calibration,
   score stability) rather than "the new AUC is bigger".
2. **"How do you deploy to the cloud and roll back?"** (`smdeploy`) — the container contract
   written to SageMaker's spec, infrastructure as code with OIDC instead of static keys,
   content-addressed endpoint configs so a rollback is deterministic, canary traffic with
   alarm-triggered automatic rollback, and a CD pipeline that stops for a human before
   production.
3. **"How do you know when it breaks?"** (`mlwatch`) — statistical drift tests with multiple
   testing correction, a policy calibrated against a simulator with known ground truth (so
   the false-alarm rate is a measured number, not a hope), and a scheduled workflow that
   opens an issue. A manually enabled dispatch calls the configured training workflow and
   fails visibly if dispatch fails; the scheduled simulation never starts AWS training.

Deliberately built on a classic tabular model rather than an LLM: it keeps the focus on the
platform, and it maps directly onto bank model-risk language (SR 11-7, APRA CPG 235).

---

## Engineering standard (identical across the three)

| Gate | Tooling |
|---|---|
| Lint + format | `ruff` with a broad rule set |
| Types | `mypy --strict` on `src/` **and** `tests/` (`boto3-stubs` for the AWS surface) |
| Tests | `pytest` with per-project branch-coverage gates (current counts in the linked CI runs); AWS is exercised offline with `moto` and `botocore` `Stubber`, so no account is needed to run the suite |
| Reproducibility | seeded data generation, deterministic gate arithmetic, evidence bundles per run |
| CI | one path-filtered workflow per project; the MLflow CI trains, registers and runs the gate as a dry run before a human approves the alias move; the SageMaker CI builds the image and exercises `train`/`serve` inside it |

```bash
python -m venv .venv && source .venv/bin/activate
make install                                      # install all three projects and dev extras
make all                                          # ruff + mypy + pytest for all three
make PROJECT=model-monitoring-drift test
```

Docker parts (MLflow server compose, the BYOC image, the Prometheus/Grafana stack) are
called out in each README; all three were run for real on Docker Desktop, and the problems
that only a real stack exposes are recorded in each `docs/RESULTS.md`.

**To run the CD workflow**: the `staging` and `production` environments are configured (with
a required reviewer on `production`, so the four-eyes gate is real), as are the `AWS_REGION`
and `PROJECT_NAME` variables. What is still needed — the deployment role, artefact bucket and
execution role, all outputs of the CloudFormation template — is in
[docs/DEPLOYMENT_SETUP.md](docs/DEPLOYMENT_SETUP.md), together with the teardown commands,
because endpoints bill per hour. Without any of it the CI workflow still runs in full: it
never touches AWS.

---

## Layout

```
mlops-lab/
├── mlflow-model-lifecycle/   mlreg: contract, tracking, registry, promotion gate, model card
├── sagemaker-byoc-deploy/    smdeploy: container contracts, CloudFormation, release, rollback
├── model-monitoring-drift/   mlwatch: drift tests, alert policy, simulator, exporter, dashboards
├── .github/workflows/        per-project CI + the SageMaker CD and the scheduled monitor
└── Makefile                  install / lint / type / test / all
```
