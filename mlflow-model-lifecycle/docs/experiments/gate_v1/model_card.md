# Model card — credit-pd v1

| | |
|---|---|
| Model | `logistic_regression` (baseline_logreg) |
| Registry | `credit-pd` version 1, aliases: challenger |
| MLflow run | `b70fffab762f464c82cada39369af9ee` |
| Config hash | `91f772690405c9b0` |
| Data fingerprint | `df35f767300115b5` (german_credit schema v1.0) |
| Code | `n/a` |
| Generated | 2026-09-06T00:07:28+00:00 |

## 1. Intended use

Probability of default (PD) for retail credit applications, used to rank and decline
applicants. Decision rule: decline when PD ≥ 0.130 (cost-optimal on out-of-fold predictions, cost matrix 5:1).
Not intended for pricing, limit setting or any use outside the training population.

## 2. Data

- Source file `german_credit.csv`, contract `german_credit` v1.0, fingerprint `df35f767300115b57a71d5e974c23c1b679067b2a4c93e21546479de03c14cb9`
- Split: stratified holdout 25% (seed 42); 5-fold CV on the training part
- Features: 7 numeric, 11 categorical
- Excluded protected attributes (reported as slices only): personal_status_sex, foreign_worker

## 3. Performance (holdout, 95 % bootstrap CIs)

| Metric | Holdout | CV mean ± std |
|---|---:|---:|
| AUC | 0.8024 [0.747, 0.852] | 0.7637 ± 0.0217 |
| KS | 0.5333 [0.446, 0.658] | 0.4610 ± 0.0349 |
| Brier | 0.1565 [0.131, 0.182] | 0.1726 ± 0.0081 |
| Log loss | 0.4806 | 0.5172 ± 0.0161 |
| ECE | 0.0589 | 0.0824 ± 0.0222 |
| Gini | 0.6047 | |
| Expected cost / applicant | 0.604 | |
| Approval rate | 0.288 | |
| Bad rate among approved | 0.111 | |

## 4. Slice analysis (protected attributes, holdout)

**personal_status_sex**

| Group | n | Default rate | AUC | Approval rate |
|---|---:|---:|---:|---:|
| female_divorced_separated_married | 78 | 0.346 | 0.831 | 0.269 |
| male_divorced_separated | 10 | 0.400 | 0.833 | 0.300 |
| male_married_widowed | 23 | 0.261 | 0.667 | 0.261 |
| male_single | 139 | 0.273 | 0.784 | 0.302 |

**foreign_worker**

| Group | n | Default rate | AUC | Approval rate |
|---|---:|---:|---:|---:|
| no | 8 | 0.125 | 1.000 | 0.250 |
| yes | 242 | 0.306 | 0.797 | 0.289 |

## 5. Promotion decision

**Decision: PROMOTE** (holdout n = 250)

| Check | Value | Threshold | Result | Detail |
|---|---:|---:|:---:|---|
| holdout_size | 250 | 200 | PASS |  |
| auc_min | 0.8024 | 0.7200 | PASS |  |
| brier_max | 0.1565 | 0.2000 | PASS |  |
| ece_max | 0.0589 | 0.0600 | PASS |  |
| slice_auc_min[personal_status_sex] | 0.7837 | 0.6000 | PASS | worst group: male_single |
| slice_auc_min[foreign_worker] | 0.7973 | 0.6000 | PASS | worst group: yes |

## 6. Governance

- Owner: risk-modelling; use case: retail_credit_pd
- Approver: river
- Reproduce: `mlreg train --config <this config>` with the same data fingerprint reproduces the run (seeded split, CV and estimators).
- Evidence in MLflow: `config.json`, `schema.json`, `evaluation/*`, `plots/*`, `cv_folds.csv`, model signature with input example.

## 7. Limitations

- 1 000 applications from one lender in one period; no macro-cycle information, so point-in-time only.
- Reason codes are exact logit decompositions for linear models and unavailable for tree ensembles.
- Calibration is assessed on the holdout only; recalibrate before use on a population with a different base rate.
