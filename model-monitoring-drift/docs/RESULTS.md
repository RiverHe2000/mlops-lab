# Results

Produced by `make results` (`scripts/run_experiments.sh`) on 2026-09-06, Windows 11, Python
3.12, CPU only. Raw reports (`report.json` / `report.md` per window), baselines and the
evaluation table are in [`experiments/`](experiments/). Everything is regenerable and seeded.

| | |
|---|---|
| Simulator | `simulate.py`: 4 numeric + 2 categorical features, a frozen logistic scorer (the "deployed model"), labels drawn from a slightly different truth, label latency 30 days |
| Baseline | 5 000 stationary rows scored by the model, with labels: AUC 0.796, KS 0.443, Brier 0.173, ECE 0.014, decline rate 29.5 % at threshold 0.5 |
| Windows | 1 000 rows per day; drift injected from window 1 |
| Policy | [`policies/default.yaml`](../policies/default.yaml): PSI 0.10 / 0.25 with BH-adjusted test at α = 0.01, 2 consecutive windows to escalate, AUC floor 0.70, drop 0.03 / 0.06, Brier +0.03 |
| Wall clock | ≈ 3 min for five series (16 windows) and the 14-cell × 20-seed evaluation |

## 1. Windowed series (state carried between windows)

### Stationary control — 4 × OK / NONE

| Window | x_utilisation PSI (KS p_adj) | c_region PSI | Score PSI | Decline rate | AUC (drop) |
|---|---:|---:|---:|---:|---:|
| 1 | 0.019 (0.77) | 0.003 | 0.013 | 31.2 % (+1.7 pp) | 0.810 (−0.014) |
| 2 | 0.035 (0.36) | 0.008 | 0.010 | 27.5 % (−2.0 pp) | 0.772 (+0.024) |
| 3 | 0.007 (0.92) | 0.008 | 0.011 | 27.4 % (−2.1 pp) | 0.818 (−0.022) |
| 4 | 0.013 (0.88) | 0.011 | 0.010 | 29.6 % (+0.1 pp) | 0.808 (−0.012) |

Window 2's AUC drop of +0.024 is within the 0.03 WARN margin: sampling noise on 1 000 labels
is ±0.03, which is exactly why the margin is where it is.

### Covariate shift, `x_utilisation` +1σ — INVESTIGATE, then RETRAIN

| Window | Status / action | x_utilisation PSI [detail] | Score PSI | Decline rate | AUC (drop) |
|---|---|---|---:|---:|---:|
| 1 | CRITICAL / INVESTIGATE | 0.912 (KS p_adj 2×10⁻⁹⁰, mean +1.01 sd, W1 1.04 sd), 1 consecutive | 0.592 | 60.1 % (+30.6 pp) | 0.809 (−0.013) |
| 2 | CRITICAL / **RETRAIN** | 0.762, 2 consecutive | 0.434 | 55.9 % (+26.4 pp) | 0.777 (+0.018) |
| 3 | CRITICAL / RETRAIN | 0.964, 3 consecutive | 0.504 | 55.9 % | 0.813 (−0.017) |
| 4 | CRITICAL / RETRAIN | 0.996, 4 consecutive | 0.528 | 58.4 % | 0.786 (+0.010) |

Window 1 also raised a quality finding (`x_utilisation` out of baseline range for 1.3 % of rows)
— a one-sigma shift pushes the tail past the baseline maximum. Note the AUC does not move: the
model still ranks correctly on a shifted population; what changed is *who gets declined*
(+30 pp), which is why prediction drift is reported next to the feature that caused it.

### Concept drift (income coefficient +1.8 in the truth) — RETRAIN from window 1

| Window | Status / action | Max feature PSI | Score PSI | AUC (drop) | Brier (increase) |
|---|---|---:|---:|---:|---:|
| 1 | CRITICAL / **RETRAIN** | 0.019 | 0.013 | 0.606 (+0.190) | +0.079 |
| 2 | CRITICAL / RETRAIN | 0.035 | 0.010 | 0.557 (+0.239) | +0.094 |
| 3 | CRITICAL / RETRAIN | 0.007 | 0.011 | 0.591 (+0.204) | +0.089 |
| 4 | CRITICAL / RETRAIN | 0.013 | 0.010 | 0.562 (+0.234) | +0.097 |

Every input and output statistic is indistinguishable from the stationary control; only the
labelled performance check sees the problem. With real 30-day label latency this alarm would
fire a month after the change — the reason `--as-of` exists and the reason feature-only
monitoring is not enough.

### Unseen category (`c_region = TAS`, 5 %) — WARN / INVESTIGATE

| Window | Status / action | Finding |
|---|---|---|
| 1 | WARN / INVESTIGATE | `c_region` unseen_category 4.1 % (new values `['TAS']`); PSI 0.005, AUC unchanged |
| 2 | WARN / INVESTIGATE | unseen 4.2 % |

A pipeline incident, not a population change: PSI stays flat because the one-hot encoder maps
`TAS` to all zeros, so the *model* is silently wrong for those rows while nothing else moves.

### Serving bug (scores × 0.7) — INVESTIGATE, then RETRAIN

| Window | Status / action | Score PSI | Decline rate | Max feature PSI | AUC |
|---|---|---:|---:|---:|---:|
| 1 | CRITICAL / INVESTIGATE | 0.872 | 14.2 % (−15.3 pp) | 0.019 | 0.810 |
| 2 | CRITICAL / RETRAIN | 0.922 | 10.7 % (−18.8 pp) | 0.035 | 0.772 |

Inputs unchanged, ranking unchanged (AUC), but half the declines disappeared: prediction drift
with no feature drift is the signature of a serving or post-processing bug, and the report says
so. (Whether the right action is "retrain" or "roll back the deployment" is a human call — hence
INVESTIGATE first.)

## 2. Monitor self-evaluation

`mlwatch evaluate --policy policies/default.yaml --seeds 20` — 20 independent worlds per cell,
each with its own 4 000-row baseline and one 1 000-row window; "target hit" means the finding
names the feature / score / performance the scenario actually perturbed.

| Scenario | Magnitude | Target | Flagged (≥ WARN) | Target hit | CRITICAL | RETRAIN | Mean target PSI | Mean AUC drop |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| none | 0 | – | 5 % | – | 0 % | 0 % | – | −0.002 |
| covariate_shift | 0.25 | x_utilisation | 5 % | 5 % | 0 % | 0 % | 0.066 | −0.000 |
| covariate_shift | 0.5 | x_utilisation | 100 % | 100 % | 25 % | 0 % | 0.243 | 0.002 |
| covariate_shift | 1 | x_utilisation | 100 % | 100 % | 100 % | 0 % | 0.927 | 0.004 |
| prior_shift | 0.5 | performance | 10 % | 10 % | 5 % | 5 % | – | 0.002 |
| prior_shift | 1 | performance | 100 % | 100 % | 100 % | 100 % | – | 0.004 |
| concept_drift | 1 | performance | 100 % | 100 % | 100 % | 100 % | – | 0.114 |
| concept_drift | 1.8 | performance | 100 % | 100 % | 100 % | 100 % | – | 0.231 |
| quality_missing | 0.05 | x_income | 40 % | 35 % | 0 % | 0 % | 0.011 | 0.008 |
| quality_missing | 0.15 | x_income | 100 % | 100 % | 0 % | 0 % | 0.011 | 0.014 |
| quality_unseen | 0.02 | c_region | 40 % | 40 % | 0 % | 0 % | 0.005 | 0.005 |
| quality_unseen | 0.05 | c_region | 100 % | 100 % | 0 % | 0 % | 0.007 | 0.005 |
| score_shift | 0.1 | score | 100 % | 100 % | 0 % | 0 % | 0.148 | −0.002 |
| score_shift | 0.3 | score | 100 % | 100 % | 100 % | 0 % | 0.897 | −0.002 |

Reading the table:

- **False alarms**: 1 of 20 stationary windows raised a WARN — the empirical size of the
  combined PSI + BH rule at these thresholds and n.
- **Power**: certain from half a sigma; a quarter sigma (PSI 0.07) is below the 0.10 line by
  design. Where the line goes is a business choice; the harness tells you what each choice buys.
- **Prior shift** is caught through calibration: AUC barely moves (+0.004) but Brier rises past
  +0.03, and since that is a performance CRITICAL the recommendation is RETRAIN (recalibrate).
- **Quality thresholds sit at the edge on purpose**: 5 % missing vs a 0.05 absolute-increase
  rule and 2 % unseen vs a 0.02 rule fire ≈ 40 % of the time, i.e. the rule is at its own
  detection boundary; three times the magnitude is caught every time.
- **RETRAIN is 0 % for single-window data/score drift** by construction (two consecutive
  CRITICAL windows are required); the series section shows the escalation on window 2.

## 3. Operational artefacts checked

| Artefact | Check |
|---|---|
| `deploy/prometheus/alerts.yml` | rules for stale monitor, feature/score PSI (with `for:` windows), AUC floor, quality issues, RETRAIN action; `promtool check rules` runs in CI |
| `deploy/grafana/dashboards/mlwatch.json` | status / action / AUC / label-coverage stats, PSI bar gauge, score PSI & decline rate, PSI over time, missing rates, rows & labels, runs by status |
| `deploy/docker-compose.yml`, `deploy/Dockerfile` | exporter (non-root, health-checked) + Prometheus + provisioned Grafana; image built in CI |
| `.github/workflows/monitor.yml` | daily schedule, cached state file, report to the job summary, issue on 2/3, training dispatch on 3 |

## 4. Docker stack, verified end to end (2026-09-06)

`docker compose -f deploy/docker-compose.yml up --build --wait` on Docker Desktop 4.89 (engine
29.7.2, WSL 2 backend), `make demo` data on the bind mount, then two windows posted to the
exporter — the same escalation as the series above, this time through the real HTTP path,
scrape, rule evaluation and dashboard provisioning:

| Step | Observed |
|---|---|
| Image build | `mlwatch/exporter:0.1.0`, 643 MB, non-root; `--wait` returned once all three services were healthy |
| `GET /health` | `{"status":"ok","baseline_model_version":"v1","windows_seen":0}` |
| `POST /run` window 3 (`x_utilisation` +1σ) | CRITICAL / INVESTIGATE — feature PSI 0.964 (KS p_adj 1.7×10⁻⁸⁷), score PSI 0.504, decline rate 55.9 %, AUC 0.813 |
| `POST /run` window 2 (second consecutive) | CRITICAL / **RETRAIN** — `consecutive.x_utilisation = 2` in `monitor.state.json` on the mount |
| Prometheus 2.53 | target `mlwatch` up; `mlwatch_feature_psi{feature="x_utilisation"}` scraped (0.762 after window 2) |
| Alert rules | `RetrainRecommended` **firing** (`for: 0m`); `FeatureDriftCritical`, `FeatureDriftWarning`, `ScoreDistributionShift` pending inside their `for:` windows; the rest inactive |
| Grafana 11.1 | Prometheus datasource, folder "ML monitoring" and dashboard "mlwatch · model monitoring" provisioned at start-up (`/api/search` finds it, `/api/health` ok) |

One fix came out of the run: Compose named the project after its directory (`deploy`), which is
also the directory name in the MLflow repo, so the two stacks shared one network and `down`
collided. Both compose files now declare an explicit `name:`.

## 5. Limitations

- Synthetic data with a known truth is what makes the power/false-alarm numbers possible; on
  a real portfolio the baseline would be a scored, labelled out-of-time sample and the
  thresholds would be re-validated with `mlwatch evaluate` against replayed history.
- Windows are independent days here; overlapping or weekly windows change the effective
  sample size and the consecutive-window semantics.
- No multivariate detector: correlated small shifts across many features can pass every
  univariate check. A domain-classifier AUC is the natural next addition.
