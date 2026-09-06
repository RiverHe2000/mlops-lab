# Interview notes — SageMaker BYOC, deployment strategies and MLOps CD

## The container contract

**What does "bring your own container" actually require?** For training: an image SageMaker
can start with the command `train`, which reads `/opt/ml/input/config/hyperparameters.json`
(all values strings), `inputdataconfig.json`, `resourceconfig.json` (hosts for distributed jobs),
the channel directories under `/opt/ml/input/data/<channel>/`, writes the model to
`/opt/ml/model/` (tarred to S3 as `model.tar.gz`), optional extras to `/opt/ml/output/data/`,
and on failure a reason to `/opt/ml/output/failure` and a non-zero exit. For hosting: the
command `serve`, an HTTP server on 8080 answering `GET /ping` (200 within the start-up window,
then every few seconds) and `POST /invocations` with whatever `ContentType`/`Accept` the client
sent, and the model extracted into `/opt/ml/model/`. That is the whole contract; the framework
containers just implement it for you.

**How did you test it without Docker?** By making `/opt/ml` a parameter. `SageMakerPaths` reads
its base from `SMDEPLOY_BASE_DIR`, the simulator lays out the exact tree in a temp directory,
and the same `train`/`serve` code runs in-process under pytest. The CI `container` job then
builds the image, runs `docker run image train` with the tree mounted, starts `serve` and
curls `/ping` and `/invocations` — the real thing, once the logic is already proven.

**Why does hosting verify a SHA-256?** `model.tar.gz` travels S3 → instance → extraction. A
truncated or wrong artefact would otherwise load and score nonsense. The training step records
the hash in `metadata.json`; `load_model` refuses a mismatch, so `/ping` returns 503 and the
deployment fails health checks instead of serving.

**Why metric lines instead of a metrics API?** Training jobs stream stdout to CloudWatch;
`MetricDefinitions` regexes turn matched lines into time series and into
`FinalMetricDataList` on the job. Printing `validation:auc=0.697` is the interface; the test
checks each regex recovers the value it should.

**Spot training?** `EnableManagedSpotTraining` with `MaxWaitTimeInSeconds ≥ MaxRuntimeInSeconds`
and a `CheckpointConfig.S3Uri` that SageMaker syncs to `/opt/ml/checkpoints`. The spec
validator refuses spot without those two; the trainer writes a checkpoint marker and logs when
it finds one (for a single-shot sklearn fit there is nothing to resume, for iterative training
this is where you would load state).

## Deployment

**Explain blue/green on SageMaker.** `UpdateEndpoint` with a `DeploymentConfig`: SageMaker
provisions the new fleet (green) next to the old one (blue), shifts traffic by the
`TrafficRoutingConfiguration` — `ALL_AT_ONCE`, `CANARY` (a slice first, then the rest after
`WaitIntervalInSeconds`) or `LINEAR` (steps) — watches the `AutoRollbackConfiguration` alarms
during the bake, and terminates blue after `TerminationWaitInSeconds`. If an alarm fires it
routes back to blue automatically. My `wait_in_service` also checks *which* config the endpoint
ended on, because "InService" is true after a rollback too.

**Why content-addressed names?** Endpoint configs are immutable; you always create a new one
and point the endpoint at it. Naming it by a hash of everything that defines it gives idempotent
deploys, a stable identity to roll back to, and a diff you can see in the console.

**Canary vs linear vs all-at-once — when?** Canary for production where you want a small blast
radius and a single decision point; linear when capacity must ramp gradually (large fleets,
warm-up sensitive models); all-at-once for staging or when the old and new models must not
serve simultaneously (e.g. incompatible feature contracts).

**What happens when the smoke test fails?** `release.py` distinguishes a fresh endpoint (nothing
to roll back to → fail loudly) from an update (roll back to the previous config, wait for
InService on it, then raise with the report). The alarms are for what happens *after* the
release; the smoke test is for what you can know immediately.

**Autoscaling?** Application Auto Scaling with target tracking on
`SageMakerVariantInvocationsPerInstance`; the target (60–70 % of a single instance's measured
capacity) and the cooldowns are policy. Scale-in is slower than scale-out on purpose. The
service-linked role is created lazily, which is why the deploy role gets a conditioned
`iam:CreateServiceLinkedRole`.

**Serverless / async endpoints?** The spec supports `ServerlessConfig` (memory + max
concurrency; no autoscaling policy, no instance type). Async inference (queue + S3 output) is
the right tool for large payloads or long-running scoring; not built here.

## CD and security

**Walk through the pipeline.** `build-push` (OIDC role, ECR login, immutable sha tag) → `train`
(`CreateTrainingJob` from `configs/training.yaml`, wait, collect `model.tar.gz` URI and metrics)
→ `register` (model package `PendingManualApproval`, refuses below `--min-auc`, carries git sha
and run id as metadata) → `deploy-staging` (all-at-once, 100 % data capture, smoke test) →
`approve` (GitHub `production` environment requires reviewers; the approval is then written on
the package with who/when/why) → `deploy-production` (canary + alarms + autoscaling). Every job
re-assumes the role; nothing is stored.

**Why OIDC instead of access keys in secrets?** Keys are long-lived, copied, leaked and never
rotated. With OIDC, GitHub issues a short-lived token whose `sub` claim names the repository,
branch or environment; the IAM trust policy pins to those claims, so only this repo's `main`,
`v*` tags and named environments can assume the role, and only for an hour.

**What is the execution role vs the deploy role?** The execution role is what SageMaker
*becomes* inside the job or endpoint: S3 read/write on the artifact bucket, ECR pull, logs,
metrics — nothing about endpoints. The deploy role is what CI *uses*: it may create/update
`credit-pd*` SageMaker resources and may pass the execution role to SageMaker, conditioned on
`iam:PassedToService`. Splitting them keeps the blast radius of a compromised runner small.

**Data capture and monitoring?** The endpoint config enables `DataCaptureConfig` (inputs and
outputs to S3, 100 % in staging, 20 % in production). That JSONL is the feed for the companion
`model-monitoring-drift` project (PSI/KS/chi-square, delayed-label AUC, retrain policy).

## Testing without an account

**moto and its limits.** moto implements most of the SageMaker control plane (models, endpoint
configs, endpoints, training jobs, model packages, autoscaling, alarms, ECR) and the tests run
the real boto3 calls against it. It does not implement `UpdateEndpoint`, so the blue/green
logic is tested against a scripted fake with explicit status transitions (Updating → InService
on new config / RollingBack → InService on old config / Failed), and the request payload is
validated with botocore's `Stubber`, which checks parameters against the same service model the
real client uses — if `DeploymentConfig` had a wrong key or type, that test fails.

**What can only be learned in a real account?** Provisioning times, alarm evaluation latency
during the canary bake, IAM eventual consistency, image pull time affecting the 4-minute
start-up window, and quotas. The CD workflow is designed to be the first thing run in a
sandbox, with `smdeploy plan` reviewed in the PR before it does.

## Azure ML

See [AZURE_ML_MAPPING.md](AZURE_ML_MAPPING.md): managed online endpoints with two named
deployments and a traffic split replace endpoint configs + `DeploymentConfig`; ACR replaces ECR;
workload identity federation replaces OIDC-to-IAM; Bicep replaces CloudFormation; the custom
container contract becomes a scoring route with liveness/readiness probes. The design — one
image, immutable artefacts, approval as data, canary with rollback, least privilege — is the
same.

## What running the real container changed

- The in-process simulation and `docker run … train` on the same `/opt/ml` layout produced
  identical metrics (RESULTS.md §6). That is the point of simulating the *contract* rather than
  mocking the *code*: the Dockerfile only adds packaging.
- The one behavioural fix came from a hand-written `curl`: a `text/csv` body that carries a
  header but not `header=present` was scored as one extra record. Imputation plus unknown
  categories made the header row look like a plausible applicant — a wrong answer with a 200.
  The codec now recognises a first row equal to the feature names and returns 400 with the
  remedy. In an interview this is the example for "fail loudly": SageMaker's own algorithms
  treat `text/csv` as headerless, so the convention is right, but a convention that can be
  violated silently needs a guard.
- Three parallel image builds over Wi-Fi produced `tls: bad record MAC` from pip inside
  BuildKit; sequential builds were clean. Worth knowing before blaming the Dockerfile.
