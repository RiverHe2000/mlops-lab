| Scenario | Magnitude | Target | Flagged (>= WARN) | Target hit | CRITICAL | RETRAIN | Mean target PSI | Mean AUC drop |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| none | 0 | - | 5% | - | 0% | 0% | - | -0.002 |
| covariate_shift | 0.25 | x_utilisation | 5% | 5% | 0% | 0% | 0.066 | -0.000 |
| covariate_shift | 0.5 | x_utilisation | 100% | 100% | 25% | 0% | 0.243 | 0.002 |
| covariate_shift | 1 | x_utilisation | 100% | 100% | 100% | 0% | 0.927 | 0.004 |
| prior_shift | 0.5 | performance | 10% | 10% | 5% | 5% | - | 0.002 |
| prior_shift | 1 | performance | 100% | 100% | 100% | 100% | - | 0.004 |
| concept_drift | 1 | performance | 100% | 100% | 100% | 100% | - | 0.114 |
| concept_drift | 1.8 | performance | 100% | 100% | 100% | 100% | - | 0.231 |
| quality_missing | 0.05 | x_income | 40% | 35% | 0% | 0% | 0.011 | 0.008 |
| quality_missing | 0.15 | x_income | 100% | 100% | 0% | 0% | 0.011 | 0.014 |
| quality_unseen | 0.02 | c_region | 40% | 40% | 0% | 0% | 0.005 | 0.005 |
| quality_unseen | 0.05 | c_region | 100% | 100% | 0% | 0% | 0.007 | 0.005 |
| score_shift | 0.1 | score | 100% | 100% | 0% | 0% | 0.148 | -0.002 |
| score_shift | 0.3 | score | 100% | 100% | 100% | 0% | 0.897 | -0.002 |
