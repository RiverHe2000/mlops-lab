# Model card — credit-pd v3

| | |
|---|---|
| Model | `random_forest` (challenger_rf) |
| Registry | `credit-pd` version 3, aliases: challenger |
| MLflow run | `513809a75b834409b4b722606299f868` |
| Config hash | `ab70573cb76d3fcb` |
| Data fingerprint | `df35f767300115b5` (german_credit schema v1.0) |
| Code | `n/a` |
| Generated | 2026-09-06T00:07:50+00:00 |

## 1. Intended use

Probability of default (PD) for retail credit applications, used to rank and decline
applicants. Decision rule: decline when PD ≥ 0.250 (cost-optimal on out-of-fold predictions, cost matrix 5:1).
Not intended for pricing, limit setting or any use outside the training population.

## 2. Data

- Source file `german_credit.csv`, contract `german_credit` v1.0, fingerprint `df35f767300115b57a71d5e974c23c1b679067b2a4c93e21546479de03c14cb9`
- Split: stratified holdout 25% (seed 42); 5-fold CV on the training part
- Features: 7 numeric, 11 categorical
- Excluded protected attributes (reported as slices only): personal_status_sex, foreign_worker

## 3. Performance (holdout, 95 % bootstrap CIs)

| Metric | Holdout | CV mean ± std |
|---|---:|---:|
| AUC | 0.8048 [0.748, 0.853] | 0.7839 ± 0.0131 |
| KS | 0.4990 [0.419, 0.631] | 0.4603 ± 0.0264 |
| Brier | 0.1632 [0.143, 0.183] | 0.1701 ± 0.0027 |
| Log loss | 0.4953 | 0.5110 ± 0.0069 |
| ECE | 0.0750 | 0.0686 ± 0.0187 |
| Gini | 0.6096 | |
| Expected cost / applicant | 0.560 | |
| Approval rate | 0.428 | |
| Bad rate among approved | 0.112 | |

## 4. Slice analysis (protected attributes, holdout)

**personal_status_sex**

| Group | n | Default rate | AUC | Approval rate |
|---|---:|---:|---:|---:|
| female_divorced_separated_married | 78 | 0.346 | 0.780 | 0.397 |
| male_divorced_separated | 10 | 0.400 | 0.833 | 0.200 |
| male_married_widowed | 23 | 0.261 | 0.755 | 0.435 |
| male_single | 139 | 0.273 | 0.820 | 0.460 |

**foreign_worker**

| Group | n | Default rate | AUC | Approval rate |
|---|---:|---:|---:|---:|
| no | 8 | 0.125 | 1.000 | 0.750 |
| yes | 242 | 0.306 | 0.798 | 0.417 |

## 5. Promotion decision

**Decision: HOLD** (holdout n = 250)

| Check | Value | Threshold | Result | Detail |
|---|---:|---:|:---:|---|
| holdout_size | 250 | 200 | PASS |  |
| auc_min | 0.8048 | 0.7200 | PASS |  |
| brier_max | 0.1632 | 0.2000 | PASS |  |
| ece_max | 0.0750 | 0.0600 | FAIL |  |
| auc_noninferior | -0.0294 | -0.0100 | FAIL | delta +0.0024 [-0.0294, +0.0348] |
| brier_not_worse | 0.0186 | 0.0100 | FAIL | delta +0.0066 [-0.0065, +0.0186] |
| score_psi | 0.7359 | 0.2500 | FAIL | champion → challenger |
| same_data_snapshot | - | - | PASS | paired comparison requires both models trained on the gate's data snapshot |
| slice_auc_min[personal_status_sex] | 0.7800 | 0.6000 | PASS | worst group: female_divorced_separated_married |
| slice_auc_min[foreign_worker] | 0.7976 | 0.6000 | PASS | worst group: yes |

Paired deltas (challenger - champion):

- auc: +0.0024 [-0.0294, +0.0348]
- brier: +0.0066 [-0.0065, +0.0186]

## 6. Governance

- Owner: risk-modelling; use case: retail_credit_pd
- Approver: river
- Reproduce: `mlreg train --config <this config>` with the same data fingerprint reproduces the run (seeded split, CV and estimators).
- Evidence in MLflow: `config.json`, `schema.json`, `evaluation/*`, `plots/*`, `cv_folds.csv`, model signature with input example.

## 7. Limitations

- 1 000 applications from one lender in one period; no macro-cycle information, so point-in-time only.
- Reason codes are exact logit decompositions for linear models and unavailable for tree ensembles.
- Calibration is assessed on the holdout only; recalibrate before use on a population with a different base rate.
