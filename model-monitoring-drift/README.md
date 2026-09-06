# mlwatch · production monitoring for a credit-risk classifier

Data drift, prediction drift, data quality and realised performance for a deployed binary
classifier, built from the definitions and tested against known values: PSI on baseline
quantile bins with a bootstrap interval, two-sample Kolmogorov–Smirnov and chi-square tests
with Benjamini–Hochberg control across features, Jensen–Shannon and Wasserstein as
scale-free companions, delayed-label AUC/Brier/ECE, and constraint checks for missing spikes,
unseen categories, out-of-range values and type errors. A YAML **alert and retrain policy**
turns measurements into `OK / WARN / CRITICAL` and `NONE / INVESTIGATE / RETRAIN`, with
consecutive-window escalation so a single odd batch never triggers a retrain. A simulator
with a known ground truth measures the monitor itself — **detection power and false-alarm
rate per scenario** — and the same code ships as a batch job with exit codes, a Prometheus
exporter with alert rules and a Grafana dashboard, and a scheduled GitHub Actions workflow
that opens an issue or dispatches retraining.

| | |
|---|---|
| Quality gates | `ruff`, `mypy --strict`, **34 tests** (offline, ≈ 5 s, incl. Hypothesis properties), **98 % branch coverage** |
| Detectors | numeric: PSI (+95 % bootstrap CI), KS, JS, Wasserstein/σ · categorical: PSI, two-sample χ², unseen share · score: PSI, JS, decline-rate change · quality: missing, unseen, range, type, duplicates, missing column · performance: AUC, KS, Brier, log loss, ECE vs baseline with `min_labels` |
| Policy | PSI **and** BH-adjusted test must agree; WARN → CRITICAL after 2 consecutive windows; RETRAIN on performance CRITICAL or sustained data/prediction CRITICAL; thin windows suppressed |
| Headline | With the shipped policy on 1 000-row windows: **5 % false alarms** on stationary traffic, **100 % detection of a 1σ covariate shift attributed to the right feature**, concept drift caught only by the performance check (AUC −0.11 / −0.23), a prior shift caught through calibration (Brier), a serving bug (scores × 0.7) caught as prediction drift with inputs unchanged. Full tables in [docs/RESULTS.md](docs/RESULTS.md) |

Companion projects: [`mlflow-model-lifecycle`](../mlflow-model-lifecycle) (where the model and its
promotion gate come from) and [`sagemaker-byoc-deploy`](../sagemaker-byoc-deploy) (the endpoint
whose data capture this monitors). Together: *track, ship, watch*.

---

## 1. Architecture

```
 capture JSONL (request_id, ts, model_version, features, score, decision)   labels JSONL (request_id, label, label_ts)
        │                                                                          │
        ▼                                                                          │
 baseline.py  ── trusted sample → Baseline: per-feature quantile bins + proportions + reference
                 sample (numeric), category proportions (categorical), score bins + decline rate,
                 labelled performance ─────────────────────────────────────────────┐          │
        │                                                                          ▼          ▼
 monitor.run_window(baseline, window frame, labels, policy, state)
   ├─ drift.feature_drifts     PSI [CI] · KS / χ² · JS · W1/σ · unseen share · BH-adjusted p
   ├─ drift.score_drift        PSI · JS · mean shift · decline-rate change
   ├─ quality.quality_checks   missing ↑ · unseen · out-of-range · type errors · duplicates · missing column
   ├─ performance.performance_check   join labels (as-of) → AUC/KS/Brier/ECE, drop vs baseline (≥ min_labels)
   └─ policy.decide            severities · consecutive-window state · action  ─► WindowReport (json/md, exit code)
        │
        ├─ cli.run   scheduled job (exit 0 / 2 / 3) ─► .github/workflows/monitor.yml → issue / retrain dispatch
        └─ exporter  FastAPI: /metrics (Prometheus gauges) · POST /run · /report ─► deploy/ (Prometheus rules, Grafana)
 simulate.py  DGP + frozen scorer + 7 drift scenarios ─► evaluate.py  detection power / false alarms per scenario
```

| Component | Files | What is worth knowing |
|---|---|---|
| Statistics | `stats.py`, `metrics.py` | PSI with quantile edges fixed on the baseline and an ε floor (an empty bin is a large number, not ∞); JS in bits (bounded [0, 1]); BH adjustment that skips NaNs; percentile bootstrap of PSI over the current window; rank-based AUC, KS, Brier, log loss, ECE |
| Baseline | `baseline.py` | JSON profile versioned with the model: bins, proportions, a capped reference sample for two-sample tests, categories, score bins, decline rate at the deployed threshold, labelled performance |
| Drift | `drift.py` | numeric: PSI + CI, KS vs reference sample, JS, Wasserstein in σ units, mean shift summary; categorical: PSI, **two-sample** χ² on a 2×k table (the baseline is a sample too), unseen share; score: PSI, JS, decline-rate change; missing columns reported as 100 % missing |
| Quality | `quality.py` | absolute missing-rate increase over baseline, unseen categories with examples, out-of-baseline-range share, non-numeric values in numeric columns (CRITICAL), duplicated request ids |
| Performance | `performance.py` | left-join on request id, `--as-of` for label latency, evaluated only above `min_labels` and with both classes present, drop vs baseline AUC / Brier |
| Policy | `config.py`, `policy.py`, `policies/default.yaml` | thresholds as data; per-subject consecutive counters persisted in a state file; the decision carries every finding with its value and threshold |
| Delivery | `cli.py`, `exporter.py`, `deploy/`, `.github/workflows/monitor.yml` | exit-code contract for schedulers; Prometheus gauges + counter, alert rules with `for:` windows, provisioned Grafana dashboard; the exporter only reads files under `--allowed-root` |
| Evaluation | `simulate.py`, `evaluate.py` | seven scenarios with a known target (feature, score or performance), N seeds each, flagged / target-hit / CRITICAL / RETRAIN rates and mean target PSI — the policy is validated like a model |

---

## 2. Results

**Monitor self-evaluation** (`mlwatch evaluate`, policy `policies/default.yaml`, baseline
4 000 rows, window 1 000 rows, 20 seeds per cell, labels available):

| Scenario | Magnitude | Target | Flagged (≥ WARN) | Target hit | CRITICAL | RETRAIN | Mean target PSI | Mean AUC drop |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| none | 0 | – | **5 %** | – | 0 % | 0 % | – | −0.002 |
| covariate_shift | 0.25 σ | x_utilisation | 5 % | 5 % | 0 % | 0 % | 0.066 | 0.000 |
| covariate_shift | 0.5 σ | x_utilisation | 100 % | 100 % | 25 % | 0 % | 0.243 | 0.002 |
| covariate_shift | 1 σ | x_utilisation | **100 %** | **100 %** | 100 % | 0 % | 0.927 | 0.004 |
| prior_shift | 0.5 | performance | 10 % | 10 % | 5 % | 5 % | – | 0.002 |
| prior_shift | 1.0 | performance | 100 % | 100 % | 100 % | 100 % | – | 0.004 |
| concept_drift | 1.0 | performance | 100 % | 100 % | 100 % | 100 % | – | 0.114 |
| concept_drift | 1.8 | performance | 100 % | 100 % | 100 % | 100 % | – | 0.231 |
| quality_missing | 5 % | x_income | 40 % | 35 % | 0 % | 0 % | 0.011 | 0.008 |
| quality_missing | 15 % | x_income | 100 % | 100 % | 0 % | 0 % | 0.011 | 0.014 |
| quality_unseen | 2 % | c_region | 40 % | 40 % | 0 % | 0 % | 0.005 | 0.005 |
| quality_unseen | 5 % | c_region | 100 % | 100 % | 0 % | 0 % | 0.007 | 0.005 |
| score_shift | ×0.9 | score | 100 % | 100 % | 0 % | 0 % | 0.148 | −0.002 |
| score_shift | ×0.7 | score | 100 % | 100 % | 100 % | 0 % | 0.897 | −0.002 |

What it says: the policy is quiet on stationary data (1 window in 20), blind to a quarter-sigma
shift by design (PSI 0.07 < 0.10), certain from half a sigma up, and it attributes the alarm to
the right feature every time. Concept drift and prior shift leave every feature statistic
flat — only the labelled performance check sees them, which is the argument for capturing
labels. RETRAIN needs either a performance CRITICAL or two consecutive CRITICAL windows, so
single-window rates for data/score drift are 0 % by construction; the series below shows the
escalation.

**Windowed series with carried state** (`make results`; four daily windows, same policy):

| Series | Window 1 | Window 2 | Window 3 | Window 4 |
|---|---|---|---|---|
| stationary | OK / NONE | OK / NONE | OK / NONE | OK / NONE |
| covariate_shift 1 σ | CRITICAL / INVESTIGATE (x_utilisation PSI 0.91, KS p_adj 10⁻⁹⁰, score PSI 0.59, decline rate +31 pp) | CRITICAL / **RETRAIN** (2nd consecutive) | RETRAIN | RETRAIN |
| concept_drift 1.8 | CRITICAL / **RETRAIN** (AUC 0.61 < 0.70 floor, −0.19 vs baseline, Brier +0.08; every feature PSI < 0.04) | RETRAIN | RETRAIN | RETRAIN |
| quality_unseen 5 % | WARN / INVESTIGATE (`c_region` unseen 4.1 %: `TAS`) | WARN / INVESTIGATE | | |
| score_shift ×0.7 | CRITICAL / INVESTIGATE (score PSI 0.87, decline rate 14 % vs 30 %; all feature PSI < 0.04) | CRITICAL / **RETRAIN** | | |

---

## 3. Tests: what each one proves

| File | Proves |
|---|---|
| `test_stats_schema_baseline.py` | quantile edges dedupe ties and bins sum to 1; PSI is 0 for identical, symmetric, > 0.25 for a reversal, finite for an empty bin, and (Hypothesis) non-negative for any proportion vectors; JS is 0 / 1 at the extremes; BH matches a hand example, skips NaNs, caps at 1; the bootstrap brackets the point estimate; records round-trip through JSONL with tz-aware timestamps; label de-duplication; window selection; feature-type inference; profile fields; baseline save/load equality |
| `test_drift_quality_performance.py` | stationary windows give PSI < 0.05 and p > 0.001, a 1σ shift gives PSI > 0.25, p < 10⁻⁶ and W1 ≈ 1σ; categorical unseen share and skew; score drift on a serving bug; BH-adjusted p ≥ raw p; every quality issue kind on a deliberately broken frame; performance evaluation only above `min_labels`, single-class guard, concept drift shows as AUC drop; the baseline compared with itself has PSI < 0.01 |
| `test_policy_monitor.py` | thin windows suppress verdicts; PSI alone or test alone does not fire; WARN → CRITICAL → RETRAIN across consecutive windows with the state file; prediction, quality and performance rules each map to the documented severity and action; end-to-end runs for five scenarios land on the expected status **and subject**; reports render and save; exit codes |
| `test_simulate_evaluate_exporter_cli.py` | the simulator is deterministic and each scenario changes what it claims to; dataset files and manifest; the evaluation grid and its table; target attribution; the exporter's gauges and counters; `POST /run` with root confinement (400 outside, 404 missing); the CLI lifecycle simulate → baseline → run (0 / 2 / 3) → evaluate → serve |

---

## 4. Reproduce

```bash
python -m venv .venv && . .venv/Scripts/activate      # or source .venv/bin/activate
pip install -e ".[dev]"
make all                                              # ruff + mypy --strict + pytest (≈ 5 s)
make results                                          # series + self-evaluation → docs/experiments/ (≈ 3 min)

make demo                                             # simulate → baseline → monitor one drifted window
mlwatch run --baseline runs/demo/baseline.json --capture runs/demo/window_02.jsonl \
            --labels runs/demo/labels.jsonl --policy policies/default.yaml \
            --state runs/demo/monitor.state.json --out runs/demo/report_02; echo "exit $?"   # 2 = investigate, 3 = retrain
mlwatch evaluate --policy policies/default.yaml --seeds 20 --out docs/experiments/monitor_evaluation.md

make serve                                            # exporter on :9108 → curl -X POST :9108/run -d '{"capture_path": ...}'
make stack                                            # Prometheus (:9090) + Grafana (:3000) + exporter, needs Docker
```

---

## 5. Design decisions

- **Two signals per feature.** PSI without a test flags every large window; a test without an
  effect size flags every tiny shift. Requiring both at a BH-adjusted α is the cheapest rule that
  gives a measured 5 % false-alarm rate here, and the evaluation harness is how the number is
  known rather than assumed.
- **The baseline is a sample.** The categorical test is a two-sample χ² on baseline counts vs
  window counts; treating baseline proportions as the population produced p ≈ 0.005 on
  stationary data during development — the stationary-fixture test is what caught it.
- **Escalation is stateful.** Consecutive-window counters live in a JSON state file the scheduler
  carries between runs (GitHub cache in the workflow), so "sustained" means what it says.
- **Labels are late, so two clocks.** Drift runs immediately; performance runs once enough
  labels have arrived (`--as-of`). Reports say "not evaluated" rather than reporting AUC on the
  early, biased slice.
- **Quality is not drift.** A missing spike or a new category is a pipeline incident; it is
  reported as such and never rationalised as a population change.
- **The monitor is validated like a model.** `mlwatch evaluate` is the equivalent of an ROC
  curve for the policy; a threshold change goes through it before it goes to production.
- **Not built (yet):** multivariate drift (domain classifier), per-segment slices, sequential
  change detection (CUSUM) instead of a consecutive count, remote data-capture ingestion.

## Related projects

- [`mlflow-model-lifecycle`](../mlflow-model-lifecycle) — the champion model, its baseline
  metrics and the promotion gate whose thresholds this policy mirrors.
- [`sagemaker-byoc-deploy`](../sagemaker-byoc-deploy) — the endpoint whose `DataCaptureConfig`
  JSONL is the input here; its CD workflow is what `monitor.yml` dispatches on RETRAIN.
