# Monitoring report — window `covariate_shift-03`

**Status: CRITICAL · Action: RETRAIN** — 1000 rows, model versions {'v1': 1000}, generated 2026-09-06T00:57:14+00:00

## Findings

- data/x_utilisation: PSI 0.964 (ks p_adj 1.72e-87), mean +1.00 sd, W1 1.03 sd; 3 consecutive window(s)
- prediction/score: score PSI 0.504, decline rate 55.9% (+26.4% vs baseline)

## Feature drift

| Feature | Kind | n | Missing | PSI [95 % CI] | Test | p (BH) | JS | Shift |
|---|---|---:|---:|---:|---|---:|---:|---|
| x_income | numeric | 1000 | 0.0% | 0.019 [0.010, 0.047] | ks | 0.46 | 0.003 | mean +0.04 sd, W1 0.05 sd |
| x_utilisation | numeric | 1000 | 0.0% | 0.964 [0.857, 1.115] | ks | 1.72e-87 | 0.157 | mean +1.00 sd, W1 1.03 sd |
| x_age | numeric | 1000 | 0.0% | 0.007 [0.007, 0.034] | ks | 0.622 | 0.001 | mean -0.00 sd, W1 0.06 sd |
| x_tenure | numeric | 1000 | 0.0% | 0.004 [0.004, 0.026] | ks | 0.918 | 0.001 | mean -0.01 sd, W1 0.05 sd |
| c_region | categorical | 1000 | 0.0% | 0.008 | chi2 | 0.46 | 0.001 | SA +2.4 pp |
| c_product | categorical | 1000 | 0.0% | 0.000 | chi2 | 0.918 | 0.000 | personal_loan +0.7 pp |

## Prediction drift

- score PSI 0.504, JS 0.085, mean 0.543 (+0.181), decline rate 55.9% (+26.4%)

## Data quality

- no issues

## Performance

- 1000 labels (100% coverage): AUC 0.813, KS 0.488, Brier 0.176, ECE 0.029, positive rate 53.6%
- vs baseline: AUC -0.017 drop, Brier +0.003
