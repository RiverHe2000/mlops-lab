# Monitoring report — window `covariate_shift-04`

**Status: CRITICAL · Action: RETRAIN** — 1000 rows, model versions {'v1': 1000}, generated 2026-09-06T00:57:15+00:00

## Findings

- data/x_utilisation: PSI 0.996 (ks p_adj 1.49e-93), mean +1.01 sd, W1 1.03 sd; 4 consecutive window(s)
- prediction/score: score PSI 0.528, decline rate 58.4% (+28.9% vs baseline)

## Feature drift

| Feature | Kind | n | Missing | PSI [95 % CI] | Test | p (BH) | JS | Shift |
|---|---|---:|---:|---:|---|---:|---:|---|
| x_income | numeric | 1000 | 0.0% | 0.010 [0.007, 0.036] | ks | 0.881 | 0.002 | mean -0.01 sd, W1 0.03 sd |
| x_utilisation | numeric | 1000 | 0.0% | 0.996 [0.873, 1.208] | ks | 1.49e-93 | 0.161 | mean +1.01 sd, W1 1.03 sd |
| x_age | numeric | 1000 | 0.0% | 0.015 [0.009, 0.042] | ks | 0.881 | 0.003 | mean -0.01 sd, W1 0.03 sd |
| x_tenure | numeric | 1000 | 0.0% | 0.003 [0.004, 0.023] | ks | 0.881 | 0.000 | mean +0.01 sd, W1 0.04 sd |
| c_region | categorical | 1000 | 0.0% | 0.011 | chi2 | 0.162 | 0.002 | NSW +4.1 pp |
| c_product | categorical | 1000 | 0.0% | 0.001 | chi2 | 0.881 | 0.000 | personal_loan +1.7 pp |

## Prediction drift

- score PSI 0.528, JS 0.090, mean 0.555 (+0.194), decline rate 58.4% (+28.9%)

## Data quality

- no issues

## Performance

- 1000 labels (100% coverage): AUC 0.786, KS 0.438, Brier 0.189, ECE 0.046, positive rate 52.8%
- vs baseline: AUC +0.010 drop, Brier +0.016
