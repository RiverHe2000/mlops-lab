# Monitoring report — window `stationary-02`

**Status: OK · Action: NONE** — 1000 rows, model versions {'v1': 1000}, generated 2026-09-06T00:57:05+00:00

## Feature drift

| Feature | Kind | n | Missing | PSI [95 % CI] | Test | p (BH) | JS | Shift |
|---|---|---:|---:|---:|---|---:|---:|---|
| x_income | numeric | 1000 | 0.0% | 0.006 [0.005, 0.033] | ks | 0.684 | 0.001 | mean -0.02 sd, W1 0.04 sd |
| x_utilisation | numeric | 1000 | 0.0% | 0.035 [0.024, 0.072] | ks | 0.361 | 0.006 | mean -0.11 sd, W1 0.09 sd |
| x_age | numeric | 1000 | 0.0% | 0.004 [0.005, 0.027] | ks | 0.684 | 0.001 | mean +0.02 sd, W1 0.05 sd |
| x_tenure | numeric | 1000 | 0.0% | 0.012 [0.007, 0.041] | ks | 0.684 | 0.002 | mean +0.03 sd, W1 0.05 sd |
| c_region | categorical | 1000 | 0.0% | 0.008 | chi2 | 0.459 | 0.001 | VIC -2.5 pp |
| c_product | categorical | 1000 | 0.0% | 0.001 | chi2 | 0.684 | 0.000 | card -1.5 pp |

## Prediction drift

- score PSI 0.010, JS 0.002, mean 0.344 (-0.017), decline rate 27.5% (-2.0%)

## Data quality

- no issues

## Performance

- 1000 labels (100% coverage): AUC 0.772, KS 0.422, Brier 0.182, ECE 0.045, positive rate 33.3%
- vs baseline: AUC +0.024 drop, Brier +0.009
