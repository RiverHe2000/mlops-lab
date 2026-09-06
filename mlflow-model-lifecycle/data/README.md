# Data

`german_credit.csv` is the UCI **Statlog (German Credit Data)** set, 1 000 loan applications
with 20 attributes and a good/bad outcome, decoded from the raw coded file by
[`scripts/prepare_data.py`](../scripts/prepare_data.py).

| | |
|---|---|
| Source | Hofmann, H. (1994). Statlog (German Credit Data). UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/144 |
| Licence | CC BY 4.0 |
| Rows / columns | 1 000 / 22 (`application_id`, 20 attributes, `default`) |
| Target | `default` = 1 for the original class 2 ("bad"), 0 otherwise. Base rate 30 % |
| SHA-256 of the committed CSV | `e1c9e3d3ced2de74105987dd3600f55b148a349d079f17162d5864834d589da7` |

The contract that every run validates against is [`schema.yaml`](schema.yaml). Two columns are
marked `role: protected` (`personal_status_sex`, `foreign_worker`): they are never used as
model inputs but are kept so that performance can be reported per slice in the model card.

The dataset ships a cost matrix — misclassifying a bad customer as good costs 5, the reverse
costs 1 — which `mlreg` uses to report an expected cost per applicant and to pick a
cost-optimal decision threshold when the config asks for it.

Rebuild: `make data` (expects the raw `german.data` under `data/raw/`; the script asserts the
row count and prints the SHA-256 so a changed upstream file is noticed).
