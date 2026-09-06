# Monitoring report — window `covariate_shift-02`

**Status: CRITICAL · Action: RETRAIN** — 1000 rows, model versions {'v1': 1000}, generated 2026-09-06T00:57:13+00:00

## Findings

- data/x_utilisation: PSI 0.762 (ks p_adj 1.35e-85), mean +0.89 sd, W1 0.92 sd; 2 consecutive window(s)
- prediction/score: score PSI 0.434, decline rate 55.9% (+26.4% vs baseline)

## Feature drift

| Feature | Kind | n | Missing | PSI [95 % CI] | Test | p (BH) | JS | Shift |
|---|---|---:|---:|---:|---|---:|---:|---|
| x_income | numeric | 1000 | 0.0% | 0.006 [0.005, 0.033] | ks | 0.684 | 0.001 | mean -0.02 sd, W1 0.04 sd |
| x_utilisation | numeric | 1000 | 0.0% | 0.762 [0.673, 0.897] | ks | 1.35e-85 | 0.128 | mean +0.89 sd, W1 0.92 sd |
| x_age | numeric | 1000 | 0.0% | 0.004 [0.005, 0.027] | ks | 0.684 | 0.001 | mean +0.02 sd, W1 0.05 sd |
| x_tenure | numeric | 1000 | 0.0% | 0.012 [0.007, 0.041] | ks | 0.684 | 0.002 | mean +0.03 sd, W1 0.05 sd |
| c_region | categorical | 1000 | 0.0% | 0.008 | chi2 | 0.459 | 0.001 | VIC -2.5 pp |
| c_product | categorical | 1000 | 0.0% | 0.001 | chi2 | 0.684 | 0.000 | card -1.5 pp |

## Prediction drift

- score PSI 0.434, JS 0.075, mean 0.533 (+0.172), decline rate 55.9% (+26.4%)

## Data quality

- no issues

## Performance

- 1000 labels (100% coverage): AUC 0.777, KS 0.442, Brier 0.193, ECE 0.047, positive rate 50.9%
- vs baseline: AUC +0.018 drop, Brier +0.020
