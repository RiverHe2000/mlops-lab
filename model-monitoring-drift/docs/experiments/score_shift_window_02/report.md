# Monitoring report — window `score_shift-02`

**Status: CRITICAL · Action: RETRAIN** — 1000 rows, model versions {'v1': 1000}, generated 2026-09-06T00:57:35+00:00

## Findings

- prediction/score: score PSI 0.922, decline rate 10.7% (-18.8% vs baseline)

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

- score PSI 0.922, JS 0.085, mean 0.241 (-0.120), decline rate 10.7% (-18.8%)

## Data quality

- no issues

## Performance

- 1000 labels (100% coverage): AUC 0.772, KS 0.422, Brier 0.189, ECE 0.092, positive rate 33.3%
- vs baseline: AUC +0.024 drop, Brier +0.016
