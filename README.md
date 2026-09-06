# mlops-lab

[![mlflow-model-lifecycle](https://github.com/ChuanHe-PhD/mlops-lab/actions/workflows/mlflow-model-lifecycle-ci.yml/badge.svg)](https://github.com/ChuanHe-PhD/mlops-lab/actions/workflows/mlflow-model-lifecycle-ci.yml)
[![sagemaker-byoc-deploy](https://github.com/ChuanHe-PhD/mlops-lab/actions/workflows/sagemaker-byoc-deploy-ci.yml/badge.svg)](https://github.com/ChuanHe-PhD/mlops-lab/actions/workflows/sagemaker-byoc-deploy-ci.yml)
[![model-monitoring-drift](https://github.com/ChuanHe-PhD/mlops-lab/actions/workflows/model-monitoring-drift-ci.yml/badge.svg)](https://github.com/ChuanHe-PhD/mlops-lab/actions/workflows/model-monitoring-drift-ci.yml)

One credit PD model followed through its whole operational life — **track it, ship it, watch
it**: experiment tracking and a registry with a statistical promotion gate, a
bring-your-own-container SageMaker deployment with blue/green rollout and automatic rollback,
and production drift monitoring with a calibrated alert policy that triggers retraining.

| # | Project | What it demonstrates | Headline result |
|---|---|---|---|
| 01 | [mlflow-model-lifecycle](mlflow-model-lifecycle/) — `mlreg` | Data contract → MLflow 3 tracking (lineage tags, nested CV runs) → pyfunc packaging with signature enforcement → registry aliases (champion/challenger/previous) → **paired-bootstrap non-inferiority gate** + calibration + score PSI + protected-attribute slices → automatic model card; Postgres + MinIO tracking server | German Credit: logistic-regression baseline AUC **0.802 [0.744, 0.856]** becomes champion; HGB challenger **HOLD** (ΔAUC −0.009 [−0.042, +0.025]); random forest HOLD (ECE 0.075, score PSI 0.74) — the gate held two challengers a point estimate would have promoted; **83 tests, 97.7 % coverage** |
| 02 | [sagemaker-byoc-deploy](sagemaker-byoc-deploy/) — `smdeploy` | One image, two entry points (`train`/`serve`) implementing SageMaker's container contracts; ECR, Spot training with checkpoints, model-package approval, **content-addressed endpoint configs + canary blue/green + CloudWatch alarm auto-rollback**, autoscaling, smoke tests; CloudFormation with GitHub OIDC least-privilege roles; CD with a human approval gate | Hosted latency p50 **8 ms** (1 row) / 11 ms (1 000 rows ≈ 89 k rows/s); cfn-lint clean; image built and run for real (770 MB, non-root): in-container `train` metrics bit-identical to the simulation, `serve` passes ping / execution-parameters / CSV·JSON·JSONLines / 415 / 406; **70 tests, 95.4 % coverage** |
| 03 | [model-monitoring-drift](model-monitoring-drift/) — `mlwatch` | PSI/KS/chi-square/JS/Wasserstein from their definitions + Benjamini-Hochberg correction + PSI bootstrap intervals; data-quality constraints; delayed-label performance; **alert policy** (drift *and* test agreement, consecutive-window escalation, INVESTIGATE/RETRAIN); a simulator with ground truth that measures the monitor itself; Prometheus exporter + rules + Grafana | **5 % false-alarm rate** on stationary traffic; **100 % detection** of 1σ covariate shift, localised to the right feature; concept drift caught only by the performance check (AUC −0.11/−0.23); compose stack run for real: `RetrainRecommended` fires, Grafana auto-provisions; **34 tests, 98.3 % coverage** |

Companion repositories: [`llm-engineering-lab`](https://github.com/ChuanHe-PhD/llm-engineering-lab)
(Transformer internals, LoRA, an inference server) and [`genai-platform-lab`](https://github.com/ChuanHe-PhD/genai-platform-lab)
(RAG, agents with guardrails, an LLM gateway).

---

## The through-line

The same model, the same artefact, three stages — which is exactly how the interview
questions come:

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
   opens an issue and can dispatch retraining.

Deliberately built on a classic tabular model rather than an LLM: it keeps the focus on the
platform, and it maps directly onto bank model-risk language (SR 11-7, APRA CPG 235).

---

## Engineering standard (identical across the three)

| Gate | Tooling |
|---|---|
| Lint + format | `ruff` with a broad rule set |
| Types | `mypy --strict` on `src/` **and** `tests/` (`boto3-stubs` for the AWS surface) |
| Tests | `pytest`, **187 tests**, branch-coverage gates ≥ 95 %; AWS is exercised offline with `moto` and `botocore` `Stubber`, so no account is needed to run the suite |
| Reproducibility | seeded data generation, deterministic gate arithmetic, evidence bundles per run |
| CI | one path-filtered workflow per project; the MLflow CI trains, registers and runs the gate as a dry run before a human approves the alias move; the SageMaker CI builds the image and exercises `train`/`serve` inside it |

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e "mlflow-model-lifecycle[dev]"     # and the other two
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
