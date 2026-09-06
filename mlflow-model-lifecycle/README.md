# mlreg · credit-PD model lifecycle on MLflow

Data contract → tracked training → Model Registry aliases → statistically gated
champion/challenger promotion → auto-generated model card. A retail credit probability-of-default
model built the way a bank's model-risk function expects it to be built: every run carries its
data fingerprint, config hash and code revision; the packaged model enforces its own input
schema; a challenger only becomes `@champion` when a paired bootstrap says it is not worse, its
calibration holds and its decisions are stable; and the decision is the CI exit code.

| | |
|---|---|
| Quality gates | `ruff`, `mypy --strict`, **74 tests** (offline, CPU, ≈ 40 s), **98 % branch coverage** |
| Stack | MLflow 3.16 (tracking + registry; SQLite locally, Postgres + MinIO via `deploy/docker-compose.yml`), scikit-learn, pydantic, GitHub Actions with an environment-gated promotion job |
| Data | UCI German Credit, 1 000 applications, 30 % default rate, decoded and contract-validated ([data/README.md](data/README.md)) |
| Headline | Baseline logistic regression **AUC 0.802 [0.744, 0.856]** on the 250-row holdout is promoted to `@champion`. Both challengers are **held**: gradient boosting because non-inferiority cannot be shown at n = 250 (ΔAUC −0.009 [−0.042, +0.025]), random forest because calibration (ECE 0.075) and score stability (PSI 0.74) fail. Full tables in [docs/RESULTS.md](docs/RESULTS.md) |

Companion projects: [`sagemaker-byoc-deploy`](../sagemaker-byoc-deploy) (ship the model as a
SageMaker container with blue/green rollout) and [`model-monitoring-drift`](../model-monitoring-drift)
(watch it in production). Together: *track, ship, watch*.

---

## 1. Architecture

```
 data/german_credit.csv ──► schema.py: validate against data/schema.yaml ──► coerce dtypes
                                    │  (nulls, ranges, category domains, id uniqueness, target domain)
                                    ▼
 data.py: fingerprint (sha256 of canonical CSV) · stratified split (seed) · protected columns kept aside
                                    │
 train.py ── K-fold CV (nested MLflow runs, out-of-fold PDs) ─► cost-optimal threshold (5:1 cost matrix)
          ── fit pipeline (schema-driven ColumnTransformer + estimator) ─► holdout metrics + bootstrap CIs
          ── slices on protected attributes · calibration table · decile lift · plots
          ── log pyfunc (signature + input example + params{threshold} + bundled code) ─► models:/m-…
                                    │
 registry.py: register ─► version N (+ lineage tags) ─► @challenger
                                    │
 gate.py: score @challenger and @champion on the SAME holdout ─► absolute · paired-bootstrap relative
          · score PSI · same-data-snapshot · protected-slice checks ─► PROMOTE (exit 0) | HOLD (exit 2)
                                    │
 model_card.py: evidence pack → Markdown card · registry.promote(): @champion moves, old → @previous
```

| Layer | Files | What is worth knowing |
|---|---|---|
| Contract | `config.py`, `schema.py` | One YAML describes columns, kinds, roles (`feature`/`protected`/`target`/`id`), category domains and ranges. It validates the frame, builds the preprocessing (categories fixed → identical design matrix across refits) and becomes the MLflow signature |
| Metrics | `metrics.py`, `stats.py` | AUC (rank form), KS, Brier, log loss, ECE, decile lift, expected cost and cost-optimal threshold written from the definitions and cross-checked against scikit-learn/SciPy in tests; percentile bootstrap, **paired** bootstrap for deltas, PSI |
| Tracking | `tracking.py`, `train.py` | Lineage tags (`data.fingerprint`, `mlreg.config_hash`, `git.sha`, schema fingerprint), flat params, CV as nested runs, an evidence pack of artefacts (`evaluation/*.csv`, `plots/*.png`, `config.json`, `schema.json`) |
| Packaging | `pyfunc.py`, `models.py` | `PDScorer` pyfunc returns `pd`, `decision`, `reason_codes`; the threshold is a signature *parameter* so callers override it per request; reason codes are exact logit decompositions for linear models; the package's own code is bundled with `code_paths` so the artefact loads anywhere |
| Registry | `registry.py` | Aliases (`@champion`, `@challenger`, `@previous`) instead of deprecated stages; promotion stamps `promoted_by`, `promoted_utc` and the gate report on the version, and retires the displaced version with a tag |
| Gate | `gate.py`, `gates/promotion.yaml` | Policy as YAML; both models re-scored from the registry on the current holdout, so the packaged artefact is what gets judged; `same_data_snapshot` blocks paired comparisons when either model was trained on a different data fingerprint |
| Governance | `model_card.py` | Intended use, data lineage, metrics with CIs, slice analysis, the full check table, approver, reproduction command — the fields an SR 11-7 / APRA CPG 235 reviewer asks for |
| Ops | `deploy/`, `.github/workflows/ci.yml` | Tracking server with Postgres backend store + MinIO artifact store behind `--serve-artifacts`; CI trains, gates in dry-run, writes the model card to the job summary and only a human-approved environment job moves the alias |

---

## 2. Results

Holdout = 250 applications (stratified 25 %, seed 42); 95 % percentile-bootstrap CIs; 5-fold CV on
the 750 training rows. Thresholds are cost-optimal on out-of-fold predictions under the dataset's
5 : 1 cost matrix, which is why they are low and approval rates conservative.

| Version | Model | AUC | KS | Brier | ECE | Threshold | Approval rate | Bad rate among approved | CV AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v1 → **@champion** | logistic regression (C = 0.5) | **0.802** [0.744, 0.856] | 0.533 | 0.157 [0.131, 0.182] | 0.059 | 0.130 | 0.288 | 0.111 | 0.764 ± 0.022 |
| v2 (HOLD) | hist gradient boosting | 0.794 [0.735, 0.847] | 0.470 | 0.164 [0.137, 0.193] | 0.048 | 0.100 | 0.304 | 0.053 | 0.774 ± 0.030 |
| v3 (HOLD) | random forest | 0.805 [0.746, 0.857] | 0.499 | 0.163 [0.143, 0.184] | 0.075 | 0.250 | 0.428 | 0.112 | 0.784 ± 0.013 |

**Gate decisions** (`gates/promotion.yaml`: AUC ≥ 0.72, Brier ≤ 0.20, ECE ≤ 0.06, n ≥ 200;
relative: lower bound of ΔAUC ≥ −0.01, upper bound of ΔBrier ≤ 0.01, score PSI ≤ 0.25, same data
snapshot; every protected slice with ≥ 30 rows AUC ≥ 0.60):

| Challenger vs champion | ΔAUC (paired, 95 % CI) | ΔBrier | Score PSI | Failed checks | Decision |
|---|---:|---:|---:|---|---|
| v1 vs none | – | – | – | none | **PROMOTE** |
| v2 vs v1 | −0.009 [−0.042, +0.025] | +0.007 [−0.006, +0.020] | 0.105 | `auc_noninferior`, `brier_not_worse` | **HOLD** |
| v3 vs v1 | +0.002 [−0.029, +0.035] | +0.007 [−0.007, +0.019] | 0.736 | `ece_max`, `auc_noninferior`, `brier_not_worse`, `score_psi` | **HOLD** |

What the numbers say: at n = 250 a ±0.03 AUC interval is the best any model can get, so a
challenger has to be *visibly* better to clear a 0.01 non-inferiority margin — the gate is
correctly saying "not enough evidence", not "worse". The forest ranks as well as the baseline
but its PDs live on a different scale (PSI 0.74), so swapping it in would change who gets
declined even at the same threshold; that is exactly what the stability check exists to catch.

---

## 3. Tests: what each one proves

| File | Proves |
|---|---|
| `test_metrics.py` | AUC equals scikit-learn's with and without ties; KS equals a hand computation and SciPy's two-sample statistic; Brier/log loss match; ECE is 0 for perfectly calibrated bins and 0.3 for a known miscalibration; decile capture is monotone and ends at 1; confusion/cost on a hand example; cost-optimal threshold is the grid minimum and falls when missed defaults get dearer; Hypothesis property: AUC(y, 1 − p) = 1 − AUC(y, p) |
| `test_stats.py` | bootstrap is seed-deterministic and brackets the estimate; the paired delta of a model with itself is exactly 0 with zero width; paired CI is tighter than the unpaired sum; PSI is 0 for identical, < 0.05 for same-distribution, > 0.25 for a one-sigma shift, ∞ for a constant reference vs a different constant |
| `test_schema.py`, `test_config.py` | every violation kind is detected (missing/unexpected column, null, non-numeric, range, unknown category, duplicate id, bad target); coercion yields contract dtypes; config hash ignores paths and tracking but not the model; env overrides; unknown keys rejected |
| `test_data.py` | the split is stratified, deterministic per seed, sizes 750/250; the data fingerprint is column-order-invariant, value-sensitive and pinned to the committed CSV |
| `test_features_models.py` | design-matrix width = numeric + Σ categories; unknown category → all-zero block; refit on a subset yields identical columns; reason codes are an exact decomposition of the logit; trees return none |
| `test_pyfunc_tracking.py` | the run carries lineage tags, params, CV metrics and CI bounds; K nested fold runs; the evidence-pack artefacts exist; the loaded pyfunc reproduces the in-memory pipeline to 1e-9; the `threshold` param flips decisions; **the signature rejects a wrong dtype and a missing column**; a serving payload validates |
| `test_registry_gate.py` | policy YAML loads; absolute-only, non-inferiority direction, same-snapshot, PSI and slice checks each flip the decision; register → alias → promote → `@previous` with retirement tags; promotion is idempotent; unknown model → `None` |
| `test_cli.py` | the whole lifecycle through the CLI: validate (exit 1 on a bad CSV), train + register, first promotion, no-op re-promotion, HOLD with exit 2 leaves the champion untouched, dry-run, real promotion moves aliases, compare, card, serve-check |
| `test_model_card.py` | every section renders; gate table, registry version, approver and fixed-threshold wording appear when supplied |

---

## 4. Reproduce

```bash
python -m venv .venv && . .venv/Scripts/activate      # or source .venv/bin/activate
pip install -e ".[dev]"
make all                                              # ruff + mypy --strict + pytest (≈ 40 s)

# the whole lifecycle against a local SQLite store (≈ 2 min, CPU)
make results                                          # → docs/experiments/*, mlruns/
make mlflow-ui                                        # browse runs, registry, aliases at :5000

# step by step
mlreg validate --data data/german_credit.csv --schema data/schema.yaml
mlreg train --config configs/baseline_logreg.yaml --register --alias challenger
mlreg promote --config configs/baseline_logreg.yaml --policy gates/promotion.yaml --candidate-version 1 --out runs/gate1 --approved-by you
mlreg train --config configs/challenger_hgb.yaml --register --alias challenger
mlreg promote --config configs/challenger_hgb.yaml --policy gates/promotion.yaml --candidate-version 2 --out runs/gate2 --dry-run   # exit 2 = HOLD
mlreg compare --config configs/baseline_logreg.yaml --versions 1 2
mlreg serve-check --config configs/baseline_logreg.yaml --alias champion
mlflow models serve -m models:/credit-pd@champion --env-manager local -p 5001               # REST scoring
```

Team setup: `cp deploy/.env.example deploy/.env`, `docker compose -f deploy/docker-compose.yml up -d`,
then `export MLREG_TRACKING_URI=http://localhost:5000` — the same commands log to the server. `mlreg` downloads models *through* the server (it defaults `MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD=false` for HTTP tracking URIs): the presigned URLs a `--serve-artifacts` server hands out point at `minio:9000`, which nothing outside the compose network can reach.

---

## 5. Design decisions

- **Aliases, not stages.** MLflow deprecated stages; an alias is a mutable pointer with a URI
  (`models:/credit-pd@champion`) so deployments never hard-code a version and promotion is one
  atomic move that leaves a tagged audit trail (`promoted_by`, `gate.report`, `retired_by_version`).
- **Paired bootstrap on the same rows.** Two independent CIs on a 250-row holdout overlap for
  almost any pair of reasonable models. Resampling the *same* rows for both removes the
  between-sample noise; the remaining width is the honest uncertainty of the difference.
- **Non-inferiority, not superiority.** The question a risk committee asks is "can this replace
  the incumbent without getting worse?", hence a margin on the lower bound rather than p < 0.05
  for an improvement. The margin (0.01 AUC) is policy, kept in YAML, not in code.
- **Threshold from out-of-fold predictions.** Choosing it on the holdout would leak; choosing it on
  training predictions would be optimistic. OOF PDs from the CV loop are the right population.
- **Score PSI as a decision-stability check.** Equal AUC does not mean equal decisions; a PD
  distribution on a different scale moves the accept/decline line for real customers.
- **`same_data_snapshot`.** A paired comparison is only valid when neither model has seen the
  holdout rows; both versions carry `data.fingerprint`, and the gate refuses to pair them otherwise.
- **The pyfunc is the deliverable.** Signature (with a `threshold` param) + input example +
  pinned requirements + bundled package code, so `mlflow models serve`, batch scoring and the
  gate all load the same artefact; the gate scores *from the registry*, not from memory.
- **Protected attributes are contract-level.** `role: protected` columns can never enter the
  design matrix (the transformer only sees `feature` columns) but are still sliced in every card.
- **Not used on purpose:** `mlflow.evaluate` (metrics needed CIs, pairing and credit-specific
  definitions), stages, autologging (explicit params are auditable), model-level feature
  importance for the forest as a governance artefact (impurity importance is not an explanation).

## Related projects

- [`sagemaker-byoc-deploy`](../sagemaker-byoc-deploy) — the registry's `@champion` becomes a SageMaker
  training/inference container with blue/green rollout, auto-rollback and an OIDC-secured CD pipeline.
- [`model-monitoring-drift`](../model-monitoring-drift) — inference logs from the deployed model
  are checked for data, prediction and performance drift with a calibrated alert policy.
