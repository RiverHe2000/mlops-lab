# Monitoring report — window `concept_drift-04`

**Status: CRITICAL · Action: RETRAIN** — 1000 rows, model versions {'v1': 1000}, generated 2026-09-06T00:57:24+00:00

## Findings

- performance/auc: AUC 0.562 on 1000 labels < floor 0.70, drop +0.234 vs baseline, Brier +0.097

## Feature drift

| Feature | Kind | n | Missing | PSI [95 % CI] | Test | p (BH) | JS | Shift |
|---|---|---:|---:|---:|---|---:|---:|---|
| x_income | numeric | 1000 | 0.0% | 0.010 [0.007, 0.036] | ks | 0.881 | 0.002 | mean -0.01 sd, W1 0.03 sd |
| x_utilisation | numeric | 1000 | 0.0% | 0.013 [0.008, 0.044] | ks | 0.881 | 0.002 | mean +0.01 sd, W1 0.06 sd |
| x_age | numeric | 1000 | 0.0% | 0.015 [0.009, 0.042] | ks | 0.881 | 0.003 | mean -0.01 sd, W1 0.03 sd |
| x_tenure | numeric | 1000 | 0.0% | 0.003 [0.004, 0.023] | ks | 0.881 | 0.000 | mean +0.01 sd, W1 0.04 sd |
| c_region | categorical | 1000 | 0.0% | 0.011 | chi2 | 0.324 | 0.002 | NSW +4.1 pp |
| c_product | categorical | 1000 | 0.0% | 0.001 | chi2 | 0.881 | 0.000 | personal_loan +1.7 pp |

## Prediction drift

- score PSI 0.010, JS 0.002, mean 0.364 (+0.003), decline rate 29.6% (+0.1%)

## Data quality

- no issues

## Performance

- 1000 labels (100% coverage): AUC 0.562, KS 0.111, Brier 0.270, ECE 0.174, positive rate 35.9%
- vs baseline: AUC +0.234 drop, Brier +0.097
