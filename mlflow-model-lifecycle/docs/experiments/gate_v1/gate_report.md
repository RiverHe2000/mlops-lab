**Decision: PROMOTE** (holdout n = 250)

| Check | Value | Threshold | Result | Detail |
|---|---:|---:|:---:|---|
| holdout_size | 250 | 200 | PASS |  |
| auc_min | 0.8024 | 0.7200 | PASS |  |
| brier_max | 0.1565 | 0.2000 | PASS |  |
| ece_max | 0.0589 | 0.0600 | PASS |  |
| slice_auc_min[personal_status_sex] | 0.7837 | 0.6000 | PASS | worst group: male_single |
| slice_auc_min[foreign_worker] | 0.7973 | 0.6000 | PASS | worst group: yes |