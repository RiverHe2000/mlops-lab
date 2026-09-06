**Decision: HOLD** (holdout n = 250)

| Check | Value | Threshold | Result | Detail |
|---|---:|---:|:---:|---|
| holdout_size | 250 | 200 | PASS |  |
| auc_min | 0.8048 | 0.7200 | PASS |  |
| brier_max | 0.1632 | 0.2000 | PASS |  |
| ece_max | 0.0750 | 0.0600 | FAIL |  |
| auc_noninferior | -0.0294 | -0.0100 | FAIL | delta +0.0024 [-0.0294, +0.0348] |
| brier_not_worse | 0.0186 | 0.0100 | FAIL | delta +0.0066 [-0.0065, +0.0186] |
| score_psi | 0.7359 | 0.2500 | FAIL | champion → challenger |
| same_data_snapshot | - | - | PASS | paired comparison requires both models trained on the gate's data snapshot |
| slice_auc_min[personal_status_sex] | 0.7800 | 0.6000 | PASS | worst group: female_divorced_separated_married |
| slice_auc_min[foreign_worker] | 0.7976 | 0.6000 | PASS | worst group: yes |

Paired deltas (challenger - champion):

- auc: +0.0024 [-0.0294, +0.0348]
- brier: +0.0066 [-0.0065, +0.0186]