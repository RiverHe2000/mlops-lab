# Monitoring report — window `concept_drift-01`

**Status: CRITICAL · Action: RETRAIN** — 1000 rows, model versions {'v1': 1000}, generated 2026-09-06T00:57:20+00:00

## Findings

- performance/auc: AUC 0.606 on 1000 labels < floor 0.70, drop +0.190 vs baseline, Brier +0.079

## Feature drift

| Feature | Kind | n | Missing | PSI [95 % CI] | Test | p (BH) | JS | Shift |
|---|---|---:|---:|---:|---|---:|---:|---|
| x_income | numeric | 1000 | 0.0% | 0.013 [0.009, 0.042] | ks | 0.293 | 0.002 | mean -0.05 sd, W1 0.06 sd |
| x_utilisation | numeric | 1000 | 0.0% | 0.019 [0.013, 0.052] | ks | 0.769 | 0.003 | mean +0.01 sd, W1 0.04 sd |
| x_age | numeric | 1000 | 0.0% | 0.005 [0.004, 0.024] | ks | 0.769 | 0.001 | mean +0.01 sd, W1 0.04 sd |
| x_tenure | numeric | 1000 | 0.0% | 0.011 [0.007, 0.039] | ks | 0.252 | 0.002 | mean -0.03 sd, W1 0.08 sd |
| c_region | categorical | 1000 | 0.0% | 0.003 | chi2 | 0.769 | 0.000 | VIC -1.7 pp |
| c_product | categorical | 1000 | 0.0% | 0.008 | chi2 | 0.223 | 0.001 | card -4.0 pp |

## Prediction drift

- score PSI 0.013, JS 0.002, mean 0.374 (+0.013), decline rate 31.2% (+1.7%)

## Data quality

- no issues

## Performance

- 1000 labels (100% coverage): AUC 0.606, KS 0.175, Brier 0.252, ECE 0.147, positive rate 34.9%
- vs baseline: AUC +0.190 drop, Brier +0.079
