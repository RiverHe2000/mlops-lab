# Interview notes — production model monitoring

## Why monitor, and what exactly

**What can go wrong with a deployed model that tests did not catch?** Four different things,
each needing a different detector: the *inputs* change (covariate shift — PSI/KS per feature),
the *outputs* change without the inputs changing (a serving bug, a feature pipeline change —
score PSI, decline rate), the *relationship* changes (concept drift — only realised
performance on delayed labels can see it), and the *data breaks* (missing spikes, unseen
categories, type errors — quality constraints). `mlwatch` runs all four on every window and
reports which one fired; the evaluation table in `docs/RESULTS.md` shows that concept drift is
invisible to feature drift (PSI ≈ 0) and only the performance check catches it — which is
the whole argument for capturing labels.

**Why implement PSI/KS/chi-square yourself instead of Evidently or SageMaker Model Monitor?**
To know exactly what the numbers mean and to test them: PSI with fixed baseline quantile bins
and an ε-floor (so an empty bin does not become infinity), a two-sample KS against a stored
reference sample, a two-sample chi-square that includes the baseline's own sampling noise in the
null, and Jensen–Shannon for a bounded score. Each has a unit test against a known value or
property. The mapping to the managed products is one-to-one (`Model Monitor` statistics /
constraints ≈ `Baseline`; violations ≈ `QualityIssue`; a scheduled monitoring job ≈
`monitor.yml`), so switching is a plumbing change, not a modelling one.

**What does PSI actually measure and what are its failure modes?** Σ (c − r)·ln(c/r) over bins:
a symmetric divergence between two histograms, ≥ 0, unbounded. Failure modes: it depends on the
binning (I fix bins from the baseline quantiles so every bin starts at 10 %), it is undefined
for empty bins (floored at 1e-4), and with large n it flags tiny, irrelevant shifts — which is
why a feature only fires when PSI *and* a significance test agree, and why the test p-values are
Benjamini–Hochberg-adjusted across features.

**Why PSI and a test, not one of them?** They fail in opposite directions. PSI has no notion of
sample size: at n = 200 a PSI of 0.12 is noise, at n = 100 000 a PSI of 0.03 is real. The KS
p-value has no notion of effect size: at n = 100 000 it is significant for shifts nobody would
act on. Requiring both is the cheapest well-calibrated rule I know; the `none` row of the
evaluation (5 % flagged windows at the 0.01 BH level with six features) is the empirical
false-alarm rate.

## The policy

**How do you avoid paging people for a single odd batch?** Three mechanisms: `min_rows`
suppresses verdicts on thin windows; a WARN only becomes CRITICAL after
`consecutive_windows_for_critical` windows (state carried between runs); and the action
vocabulary separates *investigate* (a person looks) from *retrain* (a pipeline runs), with
retraining recommended only for a performance CRITICAL or a sustained data/prediction CRITICAL.
The stationary series in `docs/RESULTS.md` is four consecutive OK windows; the covariate-shift
series is INVESTIGATE on window 1 and RETRAIN from window 2.

**Where do the thresholds come from?** 0.10 / 0.25 are the credit-risk conventions for PSI
(model validation teams already use them for scorecard stability reports); AUC floor and drop
margins mirror the promotion gate in `mlflow-model-lifecycle` so that "worse than what we
promoted" is defined once. Everything is in `policies/default.yaml`, and `mlwatch evaluate`
reports what detection power and false-alarm rate a given file achieves before it is adopted —
the policy is validated like a model.

**What about label latency?** In credit, a default is observed months after the decision. The
performance check joins whatever labels exist (`--as-of` restricts to labels observed by a
date) and only evaluates above `min_labels`; until then the report says "not evaluated" rather
than reporting a metric on a biased early subset. Drift checks run immediately because they
need no labels — that is why both exist.

**Prior shift vs concept drift — how do they show up?** A prior shift (base rate moves) leaves
the features and scores alone but breaks calibration: the Brier score rises, so the performance
check fires (`prior_shift 1.0` → 100 % RETRAIN with AUC nearly unchanged). Concept drift moves
AUC (`concept_drift 1.0` → −0.11). Both need labels; neither is visible in PSI.

## Operations

**How does this run in production?** Two shapes. Batch: `mlwatch run` in a scheduled job
(`.github/workflows/monitor.yml` daily; in a bank it would be Airflow or a SageMaker
Processing job over the data-capture prefix), exit code 0/2/3 drives ticketing and the retrain
dispatch, the state file carries the consecutive-window counters. Streaming-ish: the
`mlwatch serve` exporter exposes the latest window as Prometheus gauges; Prometheus keeps the
history and evaluates the alert rules (`deploy/prometheus/alerts.yml`), Grafana shows the
provisioned dashboard. Same code path, two delivery mechanisms.

**Why an exporter with a POST /run endpoint?** So the monitor can be triggered by whatever
already orchestrates the platform (a webhook after the daily capture lands) and still be
scraped like any other service. The endpoint resolves paths under an `allowed_root` only —
a monitor should never become a file-read primitive.

**What would you add for a bank?** Slices (drift per segment / channel / product), a
multivariate detector (a domain classifier: train "baseline vs window" and use its AUC), a
proper sequential test for the consecutive-window rule (CUSUM / Page–Hinkley) instead of a
count, and evidence retention: every report with its baseline hash into the model inventory,
because CPS 230 / CPG 235 reviewers ask for the monitoring trail, not the dashboard.

## Pitfalls I hit

- A one-sample chi-square against baseline *proportions* treats the baseline as the population
  and produces p ≈ 0.005 on stationary data at n = 1 000; the two-sample contingency test fixes
  it. The tests caught this because the stationary fixture is compared with its own baseline.
- PSI on `score` for a covariate shift is large even when the model is "right" — the score
  *should* move when the population moves. The report therefore attributes the shift to the
  feature and reports the score PSI as context, and the policy does not retrain on score drift
  alone unless it is sustained.
- Windows paths and encodings: report Markdown contains non-ASCII (≥, —), so every read/write is
  explicit UTF-8.
- The Docker stack ran clean first time (RESULTS.md §4: exporter → scrape → `RetrainRecommended`
  firing → provisioned dashboard), but Compose had named the project after its directory
  (`deploy`), the same as the MLflow repo's stack, so the two shared a network and `down`
  collided. `name: mlwatch` fixes it; `for:` windows in the alert rules are why
  `FeatureDriftCritical` was *pending* rather than firing within the first minute — which is the
  intended behaviour, not a missing alert.
