# Model card — credit-pd v2

| | |
|---|---|
| Model | `hist_gradient_boosting` (challenger_hgb) |
| Registry | `credit-pd` version 2, aliases: challenger |
| MLflow run | `1e84a528aa3b41749b9ef243ae34c50e` |
| Config hash | `85c88b3bee6d59ce` |
| Data fingerprint | `df35f767300115b5` (german_credit schema v1.0) |
| Code | `n/a` |
| Generated | 2026-09-06T00:07:39+00:00 |

## 1. Intended use

Probability of default (PD) for retail credit applications, used to rank and decline
applicants. Decision rule: decline when PD ≥ 0.100 (cost-optimal on out-of-fold predictions, cost matrix 5:1).
Not intended for pricing, limit setting or any use outside the training population.

## 2. Data

- Source file `german_credit.csv`, contract `german_credit` v1.0, fingerprint `df35f767300115b57a71d5e974c23c1b679067b2a4c93e21546479de03c14cb9`
- Split: stratified holdout 25% (seed 42); 5-fold CV on the training part
- Features: 7 numeric, 11 categorical
- Excluded protected attributes (reported as slices only): personal_status_sex, foreign_worker

## 3. Performance (holdout, 95 % bootstrap CIs)

| Metric | Holdout | CV mean ± std |
|---|---:|---:|
| AUC | 0.7937 [0.743, 0.846] | 0.7737 ± 0.0304 |
| KS | 0.4705 [0.400, 0.597] | 0.4533 ± 0.0548 |
| Brier | 0.1639 [0.137, 0.192] | 0.1693 ± 0.0152 |
| Log loss | 0.4933 | 0.5154 ± 0.0340 |
| ECE | 0.0484 | 0.0834 ± 0.0166 |
| Gini | 0.5874 | |
| Expected cost / applicant | 0.492 | |
| Approval rate | 0.304 | |
| Bad rate among approved | 0.053 | |

## 4. Slice analysis (protected attributes, holdout)

**personal_status_sex**

| Group | n | Default rate | AUC | Approval rate |
|---|---:|---:|---:|---:|
| female_divorced_separated_married | 78 | 0.346 | 0.783 | 0.308 |
| male_divorced_separated | 10 | 0.400 | 0.792 | 0.100 |
| male_married_widowed | 23 | 0.261 | 0.696 | 0.217 |
| male_single | 139 | 0.273 | 0.801 | 0.331 |

**foreign_worker**

| Group | n | Default rate | AUC | Approval rate |
|---|---:|---:|---:|---:|
| no | 8 | 0.125 | 1.000 | 0.250 |
| yes | 242 | 0.306 | 0.785 | 0.306 |

## 5. Promotion decision

**Decision: HOLD** (holdout n = 250)

| Check | Value | Threshold | Result | Detail |
|---|---:|---:|:---:|---|
| holdout_size | 250 | 200 | PASS |  |
| auc_min | 0.7937 | 0.7200 | PASS |  |
| brier_max | 0.1639 | 0.2000 | PASS |  |
| ece_max | 0.0484 | 0.0600 | PASS |  |
| auc_noninferior | -0.0424 | -0.0100 | FAIL | delta -0.0087 [-0.0424, +0.0249] |
| brier_not_worse | 0.0202 | 0.0100 | FAIL | delta +0.0074 [-0.0057, +0.0202] |
| score_psi | 0.1053 | 0.2500 | PASS | champion → challenger |
| same_data_snapshot | - | - | PASS | paired comparison requires both models trained on the gate's data snapshot |
| slice_auc_min[personal_status_sex] | 0.7829 | 0.6000 | PASS | worst group: female_divorced_separated_married |
| slice_auc_min[foreign_worker] | 0.7854 | 0.6000 | PASS | worst group: yes |

Paired deltas (challenger - champion):

- auc: -0.0087 [-0.0424, +0.0249]
- brier: +0.0074 [-0.0057, +0.0202]

## 6. Governance

- Owner: risk-modelling; use case: retail_credit_pd
- Approver: river
- Reproduce: `mlreg train --config <this config>` with the same data fingerprint reproduces the run (seeded split, CV and estimators).
- Evidence in MLflow: `config.json`, `schema.json`, `evaluation/*`, `plots/*`, `cv_folds.csv`, model signature with input example.

## 7. Limitations

- 1 000 applications from one lender in one period; no macro-cycle information, so point-in-time only.
- Reason codes are exact logit decompositions for linear models and unavailable for tree ensembles.
- Calibration is assessed on the holdout only; recalibrate before use on a population with a different base rate.
