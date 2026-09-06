# Monitoring report — window `covariate_shift-01`

**Status: CRITICAL · Action: INVESTIGATE** — 1000 rows, model versions {'v1': 1000}, generated 2026-09-06T00:57:12+00:00

## Findings

- data/x_utilisation: PSI 0.912 (ks p_adj 2.26e-90), mean +1.01 sd, W1 1.04 sd; 1 consecutive window(s)
- prediction/score: score PSI 0.592, decline rate 60.1% (+30.6% vs baseline)
- quality/x_utilisation: out_of_range: outside baseline [-3.408, 3.482]

## Feature drift

| Feature | Kind | n | Missing | PSI [95 % CI] | Test | p (BH) | JS | Shift |
|---|---|---:|---:|---:|---|---:|---:|---|
| x_income | numeric | 1000 | 0.0% | 0.013 [0.009, 0.042] | ks | 0.22 | 0.002 | mean -0.05 sd, W1 0.06 sd |
| x_utilisation | numeric | 1000 | 0.0% | 0.912 [0.796, 1.053] | ks | 2.26e-90 | 0.151 | mean +1.01 sd, W1 1.04 sd |
| x_age | numeric | 1000 | 0.0% | 0.005 [0.004, 0.024] | ks | 0.639 | 0.001 | mean +0.01 sd, W1 0.04 sd |
| x_tenure | numeric | 1000 | 0.0% | 0.011 [0.007, 0.039] | ks | 0.168 | 0.002 | mean -0.03 sd, W1 0.08 sd |
| c_region | categorical | 1000 | 0.0% | 0.003 | chi2 | 0.683 | 0.000 | VIC -1.7 pp |
| c_product | categorical | 1000 | 0.0% | 0.008 | chi2 | 0.112 | 0.001 | card -4.0 pp |

## Prediction drift

- score PSI 0.592, JS 0.100, mean 0.564 (+0.203), decline rate 60.1% (+30.6%)

## Data quality

- x_utilisation: out_of_range = 0.0130 (threshold 0.0100) outside baseline [-3.408, 3.482]

## Performance

- 1000 labels (100% coverage): AUC 0.809, KS 0.458, Brier 0.180, ECE 0.053, positive rate 52.0%
- vs baseline: AUC -0.013 drop, Brier +0.007
