# Results

Produced by `make results` (`scripts/run_experiments.sh`) on 2026-09-06. Raw outputs, gate
reports and model cards are under [`experiments/`](experiments/); the MLflow store it created is
`mlruns/` (not committed, regenerable).

| | |
|---|---|
| Machine | Windows 11, Python 3.12.0, CPU only (the pipeline never needs a GPU) |
| Libraries | mlflow 3.16.0, scikit-learn 1.9.0, pandas 3.0.5, numpy 2.5.2 |
| Data | `data/german_credit.csv`, fingerprint `df35f767300115b5…` (schema `german_credit` v1.0) |
| Split | stratified 25 % holdout (250 rows, 75 defaults), seed 42; 5-fold CV on 750 rows |
| Wall clock | ≈ 2 min for three trainings, three gates (2 000 paired bootstrap draws each) and three comparisons |

## 1. Contract validation

```
$ mlreg validate --data data/german_credit.csv --schema data/schema.yaml
OK: 1000 rows conform to the schema
```

## 2. Holdout performance (95 % bootstrap CIs, 1 000 draws)

| Version | Config | Threshold | AUC | KS | Brier | Log loss | ECE | Expected cost / applicant | Approval rate | Bad rate among approved |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v1 | `baseline_logreg` (C = 0.5) | 0.130 | 0.8024 [0.744, 0.856] | 0.533 | 0.1565 [0.131, 0.182] | 0.481 | 0.0589 | 0.604 | 0.288 | 0.111 |
| v2 | `challenger_hgb` (depth 3, lr 0.05, 300 iters) | 0.100 | 0.7937 [0.735, 0.847] | 0.470 | 0.1639 [0.137, 0.193] | 0.493 | 0.0484 | 0.492 | 0.304 | 0.053 |
| v3 | `challenger_rf` (400 trees, depth 8, leaf 5) | 0.250 | 0.8048 [0.746, 0.857] | 0.499 | 0.1632 [0.143, 0.184] | 0.495 | 0.0750 | 0.560 | 0.428 | 0.112 |

Cross-validation on the training part (mean ± std over 5 folds):

| Version | CV AUC | CV KS | CV Brier | CV ECE |
|---|---:|---:|---:|---:|
| v1 | 0.764 ± 0.022 | 0.461 ± 0.035 | 0.173 ± 0.008 | 0.082 ± 0.022 |
| v2 | 0.774 ± 0.030 | – | 0.169 ± 0.015 | – |
| v3 | 0.784 ± 0.013 | – | 0.170 ± 0.003 | – |

The holdout AUCs sit above the CV means for all three models; with 250 rows that is within the
interval, not a signal. The CV numbers are what the model card reports next to the holdout so a
reviewer sees both.

## 3. Gate decisions (`gates/promotion.yaml`)

### v1 `baseline_logreg` — no champion yet → absolute and slice checks only → **PROMOTE**

| Check | Value | Threshold | Result |
|---|---:|---:|:---:|
| holdout_size | 250 | 200 | PASS |
| auc_min | 0.8024 | 0.7200 | PASS |
| brier_max | 0.1565 | 0.2000 | PASS |
| ece_max | 0.0589 | 0.0600 | PASS |
| slice_auc_min[personal_status_sex] | 0.7837 (worst: male_single) | 0.6000 | PASS |
| slice_auc_min[foreign_worker] | 0.7973 (worst: yes) | 0.6000 | PASS |

### v2 `challenger_hgb` vs v1 → **HOLD** (exit code 2)

| Check | Value | Threshold | Result | Detail |
|---|---:|---:|:---:|---|
| holdout_size | 250 | 200 | PASS | |
| auc_min | 0.7937 | 0.7200 | PASS | |
| brier_max | 0.1639 | 0.2000 | PASS | |
| ece_max | 0.0484 | 0.0600 | PASS | |
| auc_noninferior | −0.0424 | −0.0100 | **FAIL** | ΔAUC −0.0087 [−0.0424, +0.0249] |
| brier_not_worse | 0.0202 | 0.0100 | **FAIL** | ΔBrier +0.0074 [−0.0057, +0.0202] |
| score_psi | 0.1053 | 0.2500 | PASS | |
| same_data_snapshot | – | – | PASS | |
| slice_auc_min[personal_status_sex] | 0.7829 | 0.6000 | PASS | worst: female_divorced_separated_married |
| slice_auc_min[foreign_worker] | 0.7854 | 0.6000 | PASS | worst: yes |

### v3 `challenger_rf` vs v1 → **HOLD** (exit code 2)

| Check | Value | Threshold | Result | Detail |
|---|---:|---:|:---:|---|
| holdout_size | 250 | 200 | PASS | |
| auc_min | 0.8048 | 0.7200 | PASS | |
| brier_max | 0.1632 | 0.2000 | PASS | |
| ece_max | 0.0750 | 0.0600 | **FAIL** | |
| auc_noninferior | −0.0294 | −0.0100 | **FAIL** | ΔAUC +0.0024 [−0.0294, +0.0348] |
| brier_not_worse | 0.0186 | 0.0100 | **FAIL** | ΔBrier +0.0066 [−0.0065, +0.0186] |
| score_psi | 0.7359 | 0.2500 | **FAIL** | PD distribution of the forest is on a different scale |
| same_data_snapshot | – | – | PASS | |
| slice_auc_min[personal_status_sex] | 0.7800 | 0.6000 | PASS | worst: female_divorced_separated_married |
| slice_auc_min[foreign_worker] | 0.7976 | 0.6000 | PASS | worst: yes |

Registry state after the run: v1 `@champion`; v2 and v3 tagged `gate.decision=HOLD`,
`gate.policy=promotion.yaml`, `gate.n_holdout=250`; v3 still holds `@challenger`.

## 4. Paired comparisons (`mlreg compare`, 2 000 paired bootstrap draws)

| Pair | ΔAUC | ΔKS | ΔBrier | ΔLog loss |
|---|---:|---:|---:|---:|
| v2 − v1 | −0.0087 [−0.0424, +0.0249] | −0.0629 [−0.1258, +0.0258] | +0.0074 [−0.0060, +0.0209] | +0.0127 [−0.0213, +0.0477] |
| v3 − v1 | +0.0024 [−0.0294, +0.0348] | −0.0343 [−0.0965, +0.0524] | +0.0066 [−0.0062, +0.0191] | +0.0147 [−0.0205, +0.0478] |
| v3 − v2 | +0.0111 [−0.0129, +0.0359] | +0.0286 [−0.0346, +0.0973] | −0.0008 [−0.0149, +0.0128] | +0.0020 [−0.0351, +0.0385] |

Every interval contains zero: on this holdout the three models are statistically
indistinguishable in discrimination, which is precisely why a non-inferiority margin narrower
than the interval half-width cannot be cleared. Doubling the holdout would shrink the widths by
about √2; the honest options are a larger validation set, a repeated-CV comparison, or a wider
margin agreed with the model owner.

## 5. Slice analysis of the champion (holdout)

| personal_status_sex | n | Default rate | AUC | Approval rate |
|---|---:|---:|---:|---:|
| female_divorced_separated_married | 78 | 0.346 | 0.831 | 0.269 |
| male_divorced_separated | 10 | 0.400 | 0.833 | 0.300 |
| male_married_widowed | 23 | 0.261 | 0.667 | 0.261 |
| male_single | 139 | 0.273 | 0.784 | 0.302 |

| foreign_worker | n | Default rate | AUC | Approval rate |
|---|---:|---:|---:|---:|
| no | 8 | 0.125 | 1.000 | 0.250 |
| yes | 242 | 0.306 | 0.797 | 0.289 |

Groups under the policy's `min_n = 30` (three of the six) are reported but not gated; their
AUCs are not interpretable at that size.

## 6. Serving check

```
$ mlreg serve-check --config configs/baseline_logreg.yaml --alias champion
models:/credit-pd@champion accepted a serving payload; sample output:
      pd decision                                                   reason_codes
0.050848  approve   checking_status=lt_0|installment_rate_pct|other_debtors=none
0.539054  decline duration_months|savings_status=lt_100|checking_status=0_to_200
0.053774  approve     purpose=education|savings_status=lt_100|other_debtors=none
```

`mlflow models serve -m models:/credit-pd@champion --env-manager local` exposes the same model
over REST; the signature makes the server reject payloads with a wrong dtype or missing column
before they reach the estimator.

## 7. Observations

- **The cost matrix drives the policy.** With missed defaults costing five times a lost good
  customer, cost-optimal thresholds land between 0.10 and 0.25 and approval rates below 45 %.
  The gate does not judge business cost; adding `max_expected_cost` to the policy is a one-line
  extension and a conversation with the model owner, not a modelling question.
- **Calibration and ranking are different properties.** The forest ranks as well as the
  baseline yet fails ECE and PSI; gradient boosting is the best calibrated yet cannot prove
  non-inferior ranking. A single-metric gate would have promoted one of them.
- **Everything above is regenerable** from `make results`; the seeds, split, config hashes and
  the data fingerprint are on every run and every registry version.

## 8. The team stack, verified end to end (Docker Desktop 4.89 / engine 29.7.2, 2026-09-06)

`docker compose -f deploy/docker-compose.yml up -d --build --wait` (MLflow 3.16 server image
built from `Dockerfile.mlflow`, Postgres 16, MinIO + bucket bootstrap), then the same CLI with
`MLREG_TRACKING_URI=http://localhost:5000` and nothing else changed:

| Step | Wall clock | Observed |
|---|---:|---|
| `train baseline_logreg --register --alias challenger` | 7 s | run + 5 nested CV runs, evidence pack and pyfunc model uploaded through the server (`--serve-artifacts`), version 1 created from the logged model |
| `promote --candidate-version 1` | 4 s | no champion yet → absolute + slice checks, all PASS; **PROMOTE**, `@champion` → v1 |
| `train challenger_hgb --register` | 7 s | version 2, `@challenger` |
| `promote --candidate-version 2` | 5 s | **HOLD (exit 2)** — ΔAUC −0.0087 [−0.0424, +0.0249] fails non-inferiority at −0.01, Brier +0.0074 [−0.0057, +0.0202] fails +0.01; the same numbers as the SQLite run in §3 (same seeds, same snapshot) |
| `compare --versions 1 2` | 4 s | paired deltas for AUC / KS / Brier / log-loss, all CIs straddle zero |
| `serve-check --alias champion` | 3 s | `models:/credit-pd@champion` downloaded through the server and scored the serving example |
| Registry (REST) | | `credit-pd`: `champion → 1`, `challenger → 2` |
| Postgres | | 12 runs, 2 model versions, 2 logged models, 110 metric rows |
| MinIO | | 74 objects under `mlflow-artifacts/1/…`, one `MLmodel` per logged model |

Two defects surfaced only against a real server, both fixed and covered by tests:

- **Presigned multipart downloads pointed inside the compose network.** A `--serve-artifacts`
  server on S3/MinIO advertises multipart downloads; the client then fetched chunks straight from
  `http://minio:9000`, which the host cannot resolve, and every `models:/` load sat in retry
  back-off for minutes before failing. `mlreg` now defaults `MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD=false`
  for HTTP tracking URIs (`tracking.prefer_proxied_downloads`, explicit settings win), which is
  also what keeps the S3 credentials on the server.
- **A GBK console killed `train --register` after training.** Against an HTTP server MLflow
  prints an emoji-prefixed "View run" line from `set_terminated`; on a Chinese-locale Windows
  console that raised `UnicodeEncodeError` before registration. The CLI now reconfigures
  non-UTF-8 stdio with `errors="backslashreplace"` (`test_console.py`).

Also fixed: the compose project was named after its directory (`deploy`), colliding with the
monitoring repo's stack; it is now `name: mlreg`.
