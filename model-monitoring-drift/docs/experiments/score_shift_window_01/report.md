# Monitoring report — window `score_shift-01`

**Status: CRITICAL · Action: INVESTIGATE** — 1000 rows, model versions {'v1': 1000}, generated 2026-09-06T00:57:34+00:00

## Findings

- prediction/score: score PSI 0.872, decline rate 14.2% (-15.3% vs baseline)

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

- score PSI 0.872, JS 0.076, mean 0.262 (-0.099), decline rate 14.2% (-15.3%)

## Data quality

- no issues

## Performance

- 1000 labels (100% coverage): AUC 0.810, KS 0.469, Brier 0.181, ECE 0.100, positive rate 36.2%
- vs baseline: AUC -0.014 drop, Brier +0.008
