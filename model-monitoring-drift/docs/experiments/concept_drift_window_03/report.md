# Monitoring report — window `concept_drift-03`

**Status: CRITICAL · Action: RETRAIN** — 1000 rows, model versions {'v1': 1000}, generated 2026-09-06T00:57:22+00:00

## Findings

- performance/auc: AUC 0.591 on 1000 labels < floor 0.70, drop +0.204 vs baseline, Brier +0.089

## Feature drift

| Feature | Kind | n | Missing | PSI [95 % CI] | Test | p (BH) | JS | Shift |
|---|---|---:|---:|---:|---|---:|---:|---|
| x_income | numeric | 1000 | 0.0% | 0.019 [0.010, 0.047] | ks | 0.69 | 0.003 | mean +0.04 sd, W1 0.05 sd |
| x_utilisation | numeric | 1000 | 0.0% | 0.007 [0.005, 0.031] | ks | 0.918 | 0.001 | mean -0.00 sd, W1 0.04 sd |
| x_age | numeric | 1000 | 0.0% | 0.007 [0.007, 0.034] | ks | 0.829 | 0.001 | mean -0.00 sd, W1 0.06 sd |
| x_tenure | numeric | 1000 | 0.0% | 0.004 [0.004, 0.026] | ks | 0.918 | 0.001 | mean -0.01 sd, W1 0.05 sd |
| c_region | categorical | 1000 | 0.0% | 0.008 | chi2 | 0.69 | 0.001 | SA +2.4 pp |
| c_product | categorical | 1000 | 0.0% | 0.000 | chi2 | 0.918 | 0.000 | personal_loan +0.7 pp |

## Prediction drift

- score PSI 0.011, JS 0.002, mean 0.352 (-0.009), decline rate 27.4% (-2.1%)

## Data quality

- no issues

## Performance

- 1000 labels (100% coverage): AUC 0.591, KS 0.162, Brier 0.262, ECE 0.168, positive rate 36.4%
- vs baseline: AUC +0.204 drop, Brier +0.089
