# Interview notes — MLflow model lifecycle

## MLflow itself

**What does MLflow give you that a folder of JSON files does not?** Four things in one place:
a tracking store that makes runs queryable (params, metrics, tags, nested runs), an artifact
store that keeps the evidence next to the run, a model packaging format (the pyfunc flavour with
a signature) that any deployment target understands, and a registry that turns "which model is
live?" into a named pointer. My companion project `lora-finetune-eval` wrote run directories by
hand; this one shows what the managed version buys: lineage queries, aliases and a UI a
reviewer can open.

**How is a tracking server deployed for a team?** Two stores behind one server: a *backend
store* for metadata (Postgres — needs transactions and concurrent writers) and an *artifact
store* for files (S3/MinIO — cheap, large). `deploy/docker-compose.yml` runs exactly that with
`mlflow server --backend-store-uri postgresql://… --artifacts-destination s3://… --serve-artifacts`.
`--serve-artifacts` matters: clients upload through the server, so laptops and CI need only the
tracking URL, never S3 credentials. Locally the same code runs against
`sqlite:///mlruns/mlflow.db`; the client code does not change, only `MLREG_TRACKING_URI`.

**Aliases vs stages?** Stages (`Staging`/`Production`) were deprecated in 2.9: a fixed vocabulary,
one version per stage, no history. Aliases are arbitrary names pointing at one version each;
`models:/credit-pd@champion` is a stable URI for deployments and a promotion is one
`set_registered_model_alias` call. I keep `@champion`, `@challenger` and `@previous` (so a
rollback is one alias move too) and record who moved what on the version's tags.

**What is a model signature and why enforce it?** A typed description of inputs, outputs and
(since 2.6) inference *params*. When a pyfunc is loaded with a signature MLflow validates every
call: a string in a numeric column or a missing column raises before the estimator sees the
data — tested in `test_signature_is_enforced`. The `threshold` param lets callers move the
decision line per request without re-packaging the model; the default lives in the signature.

**What changed in MLflow 3 that bit you?** `log_model` takes `name=` instead of
`artifact_path=` and returns a *LoggedModel* (`models:/m-<id>`), which is what you register.
`validate_serving_input` is deprecated in favour of `mlflow.models.predict`, which spins up a
scoring server — correct but heavy for a unit test, so I kept the light check and silenced the
warning deliberately. And YAML: `yes`/`no` are booleans in PyYAML, so the German-credit
`telephone`/`foreign_worker` categories had to be quoted or the contract silently broke.

## The model side

**Why logistic regression as the champion when trees have better CV AUC?** Because the gate
could not show the trees are not worse on the holdout (ΔAUC intervals of ±0.03 at n = 250), the
forest's PDs are on a different scale (PSI 0.74) and its calibration fails, and a linear model
gives exact reason codes — which in credit is a regulatory expectation (adverse-action reasons),
not a nicety. "Better CV mean" is not the promotion criterion; the policy is.

**How do you pick the decision threshold?** Cost-optimal under the dataset's cost matrix
(missed default = 5, lost good = 1), evaluated on *out-of-fold* predictions from the CV loop:
holdout predictions would leak, in-sample ones are optimistic. The rule is in the config
(`decision_threshold: cost_optimal | <float>`), the chosen value is logged as a param and
stamped on the registry version.

**Reason codes — how exact are they?** For the logistic pipeline the logit is
`intercept + Σ_j coef_j · x_j` on the transformed design matrix, so per-feature contributions
are an exact decomposition, not an approximation; the top positive contributions are the
codes. The test reconstructs the PD from the contributions to 1e-9. For tree ensembles I
return nothing rather than something misleading; SHAP would be the honest next step.

**Protected attributes?** `personal_status_sex` and `foreign_worker` are `role: protected` in
the contract: the transformer cannot see them, but every card slices performance and approval
rates by them, and the policy gates the worst slice AUC where the slice is large enough.
Removing a column is not fairness (proxies remain) — it is the minimum, plus visibility.

## The promotion gate

**Why a paired bootstrap?** Both models are scored on the same 250 rows. Resampling rows once
and computing *both* metrics on each resample gives the distribution of the difference; the
shared sampling noise cancels. `test_paired_is_tighter_than_unpaired` shows the width gain, and
`test_paired_delta_of_identical_models_is_zero` shows that identical models give exactly zero
width — an unpaired test would give a wide interval around zero and never be able to promote a
retrained copy.

**Non-inferiority or superiority?** Non-inferiority: the lower bound of ΔAUC must clear −0.01
and the upper bound of ΔBrier must clear +0.01. Superiority ("p < 0.05 for an improvement")
would reject harmless refreshes; a point-estimate rule would promote noise. The margin is a
policy value in YAML, chosen with the model owner; the code just applies it.

**What do the results teach about sample size?** All three pairwise ΔAUC intervals contain
zero and are ≈ 0.06 wide. A margin of 0.01 cannot be cleared inside that; the right responses
are a larger validation set, a repeated-CV comparison, or an agreed wider margin — not a
looser statistic. The gate output states the interval so the conversation is about evidence.

**Why the PSI check?** Two models with equal AUC can decline different people. PSI between the
champion's and the challenger's PD distributions on the same rows measures how far the score
scale moved; above 0.25 the threshold no longer means the same thing and the operations team
would see a step change in approval rates.

**What is `same_data_snapshot` protecting against?** Pairing is only valid if neither model
trained on the holdout rows. Every version carries `data.fingerprint`; if the data file changed
between two trainings, the champion may have been trained on rows that are now in the holdout.
The gate refuses to pair in that case and says why.

**Why re-score from the registry instead of using the in-memory pipeline?** Because the
artefact is what ships. Loading `models:/credit-pd/2` through pyfunc exercises the signature,
the bundled code and the pickled pipeline — the gate would catch a packaging bug, not just a
modelling one.

## Governance and CI/CD

**How does this map to model-risk expectations (SR 11-7, APRA CPG 235)?** Conceptual soundness
→ config, schema and design decisions on the run; outcomes analysis → holdout metrics with
uncertainty, calibration, slices; ongoing monitoring → the companion monitoring project;
independent review → the gate report and the model card with an approver field; change
control → aliases with `promoted_by`, `gate.report`, `retired_by_version` tags. Everything is
addressable by run id, config hash and data fingerprint.

**Walk me through the CI workflow.** `quality` (ruff, mypy strict, tests, two Python versions)
→ `train-and-gate` (validate contract, train baseline, promote it as first champion, train the
challenger, run the gate in dry-run, write the gate table and model card to the job summary,
export the decision as a job output, upload the store and reports as evidence) → `promote`,
which only runs on `main`, only when the decision was PROMOTE, and only after a human approves
the `model-registry` environment. The four-eyes step is a GitHub feature, not a comment.

**What would change at bank scale?** The tracking URI becomes the team server (secret), the
artifact store S3 with SSE-KMS, auth in front of MLflow (it has none built in), the data
snapshot a versioned table rather than a CSV, the holdout a time-based out-of-time sample, the
evidence pack pushed to the model inventory system, and deployment handed to a serving
platform — which is what `sagemaker-byoc-deploy` does with this exact artefact.

## Pitfalls I hit

- PyYAML booleans (`yes`/`no`) inside category lists.
- MLflow default artifact root is `./mlruns` relative to the working directory even with a
  SQLite backend URI; the experiment is now created with an explicit `artifact_location`.
- scikit-learn clips `log_loss` with machine epsilon; my implementation clips at 1e-15, so
  cross-checks must avoid exact 0/1 predictions.
- Hypothesis found that AUC(y, 1 − p) ≠ 1 − AUC(y, p) when `1 − p` rounds sub-normal values into
  ties — the property test now rounds inputs to six decimals.
- **Only a real server shows this one:** with `--serve-artifacts` on S3/MinIO the server
  advertises presigned multipart downloads, and the presigned URLs carry the object store's
  *internal* hostname. From a laptop every `models:/` load hung in retry back-off. The fix is a
  client default (`MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD=false` for HTTP tracking URIs) — and
  the lesson is that "credentials never leave the server" and "clients never talk to the bucket"
  are two different guarantees; the compose file now promises both explicitly.
- MLflow prints an emoji "View run" line when the tracking URI is a server; on a GBK Windows
  console that is a `UnicodeEncodeError` *after* training and *before* registration. Stdio is
  reconfigured with `backslashreplace` at CLI start; a test drives it with a real GBK stream.
- Two repos with a `deploy/` directory get the same Compose project name and share a network;
  always set `name:` in a compose file that will live next to others.
