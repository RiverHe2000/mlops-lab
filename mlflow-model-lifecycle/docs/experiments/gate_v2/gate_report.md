**Decision: HOLD** (holdout n = 250)

| Check | Value | Threshold | Result | Detail |
|---|---:|---:|:---:|---|
| holdout_size | 250 | 200 | PASS |  |
| auc_min | 0.7937 | 0.7200 | PASS |  |
| brier_max | 0.1639 | 0.2000 | PASS |  |
| ece_max | 0.0484 | 0.0600 | PASS |  |
| auc_noninferior | -0.0424 | -0.0100 | FAIL | delta -0.0087 [-0.0424, +0.0249] |
| brier_not_worse | 0.0202 | 0.0100 | FAIL | delta +0.0074 [-0.0057, +0.0202] |
| score_psi | 0.1053 | 0.2500 | PASS | champion → challenger |
| same_data_snapshot | - | - | PASS | paired comparison requires both models trained on the gate's data snapshot |
| slice_auc_min[personal_status_sex] | 0.7829 | 0.6000 | PASS | worst group: female_divorced_separated_married |
| slice_auc_min[foreign_worker] | 0.7854 | 0.6000 | PASS | worst group: yes |

Paired deltas (challenger - champion):

- auc: -0.0087 [-0.0424, +0.0249]
- brier: +0.0074 [-0.0057, +0.0202]