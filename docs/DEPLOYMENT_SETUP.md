# Enabling the CD workflow

`sagemaker-byoc-deploy` ships a deployment pipeline
(`.github/workflows/sagemaker-byoc-deploy-cd.yml`) that builds and pushes the image, runs a
SageMaker training job, registers the model package, deploys to staging, waits for a human,
and then rolls out to production as a canary with alarm-based rollback.

CI never touches AWS, so this repository is fully green without any of the configuration
below. This file records what the pipeline needs before it can actually run.

## Already configured in this repository

| Item | Value | Why |
|---|---|---|
| Environment `staging` | no protection rules | staging deploys automatically once the image and training job succeed |
| Environment `production` | **required reviewer** (`RiverHe2000`), protected branches only | the four-eyes gate: the `approve` job waits here, and nothing reaches production without a human |
| Variable `AWS_REGION` | `ap-southeast-2` | Sydney |
| Variable `PROJECT_NAME` | `credit-pd` | prefix for the model-package group, endpoint and image tag |

## Still required (needs an AWS account)

Deploy `sagemaker-byoc-deploy/infra/cloudformation/platform.yaml` first. It creates the
GitHub OIDC provider, the deployment role, the SageMaker execution role, the ECR repository,
the artefact bucket and the model-package group. **No static access keys anywhere** — GitHub
assumes the deployment role through OIDC.

Note the `GitHubRepo` parameter: its default is `sagemaker-byoc-deploy`, but the project now
lives in the `mlops-lab` repository, and the role's trust policy is scoped to that value.

```bash
aws cloudformation deploy \
  --template-file sagemaker-byoc-deploy/infra/cloudformation/platform.yaml \
  --stack-name credit-pd-platform \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides \
      ProjectName=credit-pd \
      GitHubOrg=RiverHe2000 \
      GitHubRepo=mlops-lab \
      GitHubBranch=main

aws cloudformation describe-stacks --stack-name credit-pd-platform \
  --query 'Stacks[0].Outputs' --output table
```

Then map three of the stack outputs onto the repository's settings:

| Stack output | Set as | Command |
|---|---|---|
| `GitHubDeployRoleArn` | secret `AWS_DEPLOY_ROLE_ARN` | `gh secret set AWS_DEPLOY_ROLE_ARN --repo RiverHe2000/mlops-lab --body "<value>"` |
| `ArtifactBucketName` | variable `ARTIFACT_BUCKET` | `gh variable set ARTIFACT_BUCKET --repo RiverHe2000/mlops-lab --body "<value>"` |
| `SageMakerExecutionRoleArn` | variable `SAGEMAKER_EXECUTION_ROLE_ARN` | `gh variable set SAGEMAKER_EXECUTION_ROLE_ARN --repo RiverHe2000/mlops-lab --body "<value>"` |

(`ImageRepositoryUri` and `ModelPackageGroupName` are derived from `PROJECT_NAME` by the
CLI, so they do not need to be stored.)

## Running it

```bash
gh workflow run sagemaker-byoc-deploy-cd.yml --repo RiverHe2000/mlops-lab
```

or push a `v*` tag. The run stops at the `approve` job; GitHub notifies the reviewer, and
`deploy-production` starts only after approval.

## Afterwards — this matters, endpoints bill per hour

The pipeline does **not** delete the endpoints it creates; a staging endpoint left running
keeps charging. Clean up explicitly:

```bash
smdeploy status   --endpoint-name credit-pd-staging --region ap-southeast-2
smdeploy teardown --endpoint-name credit-pd-staging --region ap-southeast-2
smdeploy teardown --endpoint-name credit-pd-prod    --region ap-southeast-2
```

To return production to the previous release without waiting for the CloudWatch alarm:

```bash
smdeploy rollback --endpoint-name credit-pd-prod --to-config <previous-endpoint-config-name>
```

Endpoint configs are content-addressed, so the previous name is stable and appears in the
`production-release.json` artefact of the earlier run.
