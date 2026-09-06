# SageMaker ↔ Azure ML: the same design on the other cloud

Australian banks split between AWS and Azure; the concepts in this project map one-to-one.
Every row names what `smdeploy` implements on SageMaker and the Azure ML equivalent an engineer
would reach for.

| Concern | This project (SageMaker) | Azure Machine Learning |
|---|---|---|
| Container image | ECR repository (immutable tags, scan on push) | Azure Container Registry (ACR) |
| Custom container contract | BYOC: `train` / `serve` executables, `/opt/ml` tree, `/ping` + `/invocations` on 8080 | Custom *environment* from a Dockerfile; for online endpoints a custom container serving `/score` (liveness/readiness routes configurable via `inference_config`) |
| Training run | `CreateTrainingJob` with channels, hyperparameters as strings, `MetricDefinitions` regexes, managed spot + checkpoints | `command` job with `inputs:` (URI folders), `command:` line, metrics logged through MLflow (Azure ML is MLflow-native), low-priority compute for spot |
| Metrics capture | Regex on stdout → CloudWatch + `FinalMetricDataList` | `mlflow.log_metric` inside the job → run metrics |
| Model registry | Model Package Group → versions with `ModelApprovalStatus` (`PendingManualApproval` → `Approved`) | Workspace / registry *models* with versions, tags and stages; approval via tags or Azure DevOps/GitHub environment gates |
| Real-time serving | Endpoint → endpoint config → production variant(s) | Managed online endpoint → deployments (blue/green named deployments) |
| Blue/green rollout | `UpdateEndpoint` with `DeploymentConfig`: canary/linear traffic shifting, `AutoRollbackConfiguration` on CloudWatch alarms | Two deployments under one endpoint; `az ml online-endpoint update --traffic "blue=90 green=10"`; mirrored traffic for shadow tests; rollback = traffic back to blue |
| Health checks | `/ping` must return 200 within the startup window; failing containers roll back | Liveness/readiness probes on the deployment (`liveness_probe`, `readiness_probe`) |
| Autoscaling | Application Auto Scaling target tracking on `SageMakerVariantInvocationsPerInstance` | Azure Monitor autoscale rules on the deployment (CPU, requests per second) |
| Rollback alarms | CloudWatch `Invocation5XXErrors`, `ModelLatency` p99 | Azure Monitor alerts on `RequestsPerMinute`, `RequestLatency`, HTTP 5xx |
| Batch scoring | Batch transform (`/execution-parameters`, `MaxPayloadInMB`, `BatchStrategy`) | Batch endpoints with a scoring script or custom container |
| Data capture | `DataCaptureConfig` on the endpoint config → S3 JSONL | Data collector on the deployment → Blob storage (input/output JSONL) |
| Drift monitoring | Model Monitor schedules over captured data | Model monitoring signals (data drift, prediction drift, data quality) |
| Identity for CI | IAM role assumed via GitHub OIDC (`AssumeRoleWithWebIdentity`) | Microsoft Entra workload identity federation (federated credential on an app registration / user-assigned managed identity) |
| Least privilege | Execution role for the job/endpoint + deploy role for CI, both in CloudFormation | Managed identity on the compute/endpoint + service principal for CI, both in Bicep |
| Infrastructure as code | CloudFormation (`infra/cloudformation/platform.yaml`, linted with `cfn-lint`) | Bicep / ARM templates (`az bicep build`, PSRule for Azure) |
| Offline testing | moto (mock AWS), botocore `Stubber` validating request shapes against the real service model | No moto equivalent with comparable coverage; typical approach is `azure-ai-ml` SDK objects + `MagicMock`ed `MLClient`, plus Bicep `what-if` |
| Certification | AWS Certified Machine Learning – Specialty / ML Engineer – Associate | DP-100 Designing and Implementing a Data Science Solution on Azure |

## What does *not* translate directly

- **Endpoint configs are immutable on SageMaker**; the content-addressed naming in
  `aws/endpoint.py` exists because of that. Azure deployments are mutable objects — the same
  intent is expressed by creating a new named deployment and moving traffic.
- **Approval is a first-class field on SageMaker model packages**; on Azure ML it is a
  convention (tags, stages) backed by pipeline gates. The audit trail therefore lives in the
  CI system rather than the registry unless you add it.
- **MLflow**: Azure ML workspaces *are* MLflow tracking servers, so the companion project
  `mlflow-model-lifecycle` would log straight into the workspace; on AWS you run MLflow
  yourself (or use SageMaker's managed MLflow) — that is what its `deploy/` stack is for.
