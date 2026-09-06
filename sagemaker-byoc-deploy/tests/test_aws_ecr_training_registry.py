from __future__ import annotations

import base64
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from botocore.stub import Stubber
from pydantic import ValidationError

from smdeploy.aws import ecr as ecr_mod
from smdeploy.aws.clients import AwsClients
from smdeploy.aws.ecr import (
    DockerUnavailableError,
    EcrRepository,
    build_and_push,
    ensure_repository,
    login_command,
)
from smdeploy.aws.registry import (
    RegistrationSpec,
    describe_package,
    ensure_model_package_group,
    latest_package,
    register_model_package,
    set_approval,
)
from smdeploy.aws.training_job import (
    Channel,
    TrainingJobError,
    TrainingJobSpec,
    start_training_job,
    training_outputs,
    wait_for_training_job,
)

from .conftest import ROLE, FakeClock


# --- ECR -------------------------------------------------------------------------------------
def test_ensure_repository_is_idempotent(aws: AwsClients) -> None:
    first = ensure_repository(aws.ecr, "credit-pd")
    second = ensure_repository(aws.ecr, "credit-pd")
    assert first == second
    assert first.uri.endswith("/credit-pd") and first.registry.endswith(".amazonaws.com")
    assert first.image_uri("abc") == first.uri + ":abc"
    desc = aws.ecr.describe_repositories(repositoryNames=["credit-pd"])["repositories"][0]
    assert desc["imageScanningConfiguration"]["scanOnPush"] is True
    assert desc["imageTagMutability"] == "IMMUTABLE"
    policy = json.loads(
        aws.ecr.get_lifecycle_policy(repositoryName="credit-pd")["lifecyclePolicyText"]
    )
    assert [r["rulePriority"] for r in policy["rules"]] == [1, 2]


def test_login_command_decodes_token(aws: AwsClients) -> None:
    cmd, password = login_command(aws.ecr)
    assert cmd[:4] == ["docker", "login", "--username", "AWS"] and "--password-stdin" in cmd
    assert not cmd[-1].startswith("https://")
    token = aws.ecr.get_authorization_token()["authorizationData"][0]["authorizationToken"]
    assert base64.b64decode(token).decode().split(":", 1)[1] == password


def test_build_and_push_command_sequence(tmp_path: Path) -> None:
    repo = EcrRepository(
        "credit-pd", "123.dkr.ecr.ap-southeast-2.amazonaws.com/credit-pd", "arn:ecr"
    )
    seen: list[tuple[list[str], str | None]] = []
    executed = build_and_push(
        repo,
        ["sha1", "latest-run"],
        context=tmp_path,
        dockerfile=tmp_path / "Dockerfile",
        login=(["docker", "login"], "secret"),
        runner=lambda cmd, stdin: seen.append((list(cmd), stdin)),
    )
    assert seen[0] == (["docker", "login"], "secret")
    assert executed[1][:3] == ["docker", "build", "--platform"] and executed[1][-1] == str(tmp_path)
    assert executed[2] == ["docker", "tag", repo.image_uri("sha1"), repo.image_uri("latest-run")]
    assert executed[3:] == [
        ["docker", "push", repo.image_uri("sha1")],
        ["docker", "push", repo.image_uri("latest-run")],
    ]
    assert all("secret" not in " ".join(c) for c in executed)
    with pytest.raises(ValueError, match="tag"):
        build_and_push(
            repo,
            [],
            context=tmp_path,
            dockerfile=tmp_path / "Dockerfile",
            runner=lambda _c, _s: None,
        )


def test_default_runner_without_docker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _: None)
    with pytest.raises(DockerUnavailableError):
        ecr_mod.default_runner(["docker", "version"], None)


# --- training jobs ---------------------------------------------------------------------------
def _spec(**over: Any) -> TrainingJobSpec:
    base: dict[str, Any] = {
        "job_name": "credit-pd-1",
        "image_uri": "123.dkr.ecr.ap-southeast-2.amazonaws.com/credit-pd:sha",
        "role_arn": ROLE,
        "hyperparameters": {"C": 0.5, "target": "default", "id_columns": ["a"], "flag": True},
        "channels": {
            "train": Channel(s3_uri="s3://b/train/"),
            "validation": Channel(s3_uri="s3://b/val/"),
        },
        "output_s3_uri": "s3://b/out/",
        "tags": {"project": "credit-pd"},
        "environment": {"TRAINING_JOB_NAME": "credit-pd-1"},
    }
    base.update(over)
    return TrainingJobSpec.model_validate(base)


def test_training_spec_request_shape() -> None:
    req = _spec().to_request()
    assert req["HyperParameters"] == {
        "C": "0.5",
        "target": "default",
        "id_columns": '["a"]',
        "flag": "true",
    }
    assert req["AlgorithmSpecification"]["MetricDefinitions"][0] == {
        "Name": "validation:auc",
        "Regex": "validation:auc=([0-9.]+)",
    }
    assert [c["ChannelName"] for c in req["InputDataConfig"]] == ["train", "validation"]
    assert req["EnableManagedSpotTraining"] is False and "CheckpointConfig" not in req
    assert req["Tags"] == [{"Key": "project", "Value": "credit-pd"}]
    spot = _spec(use_spot=True, max_wait_s=7200, checkpoint_s3_uri="s3://b/ckpt/").to_request()
    assert spot["StoppingCondition"] == {"MaxRuntimeInSeconds": 3600, "MaxWaitTimeInSeconds": 7200}
    assert spot["CheckpointConfig"] == {"S3Uri": "s3://b/ckpt/"}


@pytest.mark.parametrize(
    "over",
    [
        {"job_name": "bad_name!"},
        {"channels": {"validation": Channel(s3_uri="s3://b/v/")}},
        {"use_spot": True},
        {"use_spot": True, "max_wait_s": 100, "checkpoint_s3_uri": "s3://b/c/"},
        {"output_s3_uri": "/local/path"},
    ],
)
def test_training_spec_validation(over: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        _spec(**over)


def test_training_job_under_moto(aws: AwsClients) -> None:
    spec = _spec()
    arn = start_training_job(aws.sagemaker, spec)
    assert arn.endswith(":training-job/credit-pd-1")
    desc = wait_for_training_job(aws.sagemaker, spec.job_name, poll_s=0, sleep=lambda _s: None)
    out = training_outputs(desc)
    assert out.status == "Completed" and out.job_name == "credit-pd-1"
    assert out.model_artifact_uri is not None and out.model_artifact_uri.startswith("s3://b/out/")


class _ScriptedSageMaker:
    def __init__(self, statuses: list[str]) -> None:
        self.statuses = list(statuses)

    def describe_training_job(self, **kw: Any) -> dict[str, Any]:
        status = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
        desc: dict[str, Any] = {
            "TrainingJobName": kw["TrainingJobName"],
            "TrainingJobStatus": status,
        }
        if status == "Failed":
            desc["FailureReason"] = "AlgorithmError: boom"
        if status == "Completed":
            desc["FinalMetricDataList"] = [{"MetricName": "validation:auc", "Value": 0.81}]
            desc["ModelArtifacts"] = {"S3ModelArtifacts": "s3://b/out/model.tar.gz"}
            desc["BillableTimeInSeconds"] = 120
        return desc


def test_wait_transitions_failures_and_timeout() -> None:
    clock = FakeClock()
    sm: Any = _ScriptedSageMaker(["InProgress", "InProgress", "Completed"])
    desc = wait_for_training_job(sm, "j", poll_s=30, sleep=clock.sleep, clock=clock)
    assert clock.now == 60 and training_outputs(desc).metrics == {"validation:auc": 0.81}
    assert training_outputs(desc).billable_seconds == 120
    failed: Any = _ScriptedSageMaker(["Failed"])
    with pytest.raises(TrainingJobError, match="AlgorithmError"):
        wait_for_training_job(failed, "j", poll_s=0, sleep=clock.sleep, clock=clock)
    stuck: Any = _ScriptedSageMaker(["InProgress"])
    with pytest.raises(TimeoutError):
        wait_for_training_job(stuck, "j", poll_s=10, timeout_s=25, sleep=clock.sleep, clock=clock)


# --- registry --------------------------------------------------------------------------------
def test_registry_approval_workflow(aws: AwsClients) -> None:
    assert latest_package(aws.sagemaker, "credit-pd") is None
    group_arn = ensure_model_package_group(aws.sagemaker, "credit-pd")
    assert ensure_model_package_group(aws.sagemaker, "credit-pd") == group_arn
    spec = RegistrationSpec(
        group="credit-pd",
        image_uri="123.dkr.ecr.ap-southeast-2.amazonaws.com/credit-pd:sha",
        model_data_url="s3://b/out/model.tar.gz",
        metrics={"auc": 0.8123, "brier": 0.15},
        evaluation_s3_uri="s3://b/out/evaluation.json",
        customer_metadata={"git_sha": "abc"},
        tags={"project": "credit-pd"},
    )
    req = spec.to_request()
    assert req["CustomerMetadataProperties"] == {
        "metric_auc": "0.812300",
        "metric_brier": "0.150000",
        "git_sha": "abc",
    }
    assert (
        req["ModelMetrics"]["ModelQuality"]["Statistics"]["S3Uri"] == "s3://b/out/evaluation.json"
    )
    pending = register_model_package(aws.sagemaker, spec)
    assert pending.version == 1 and pending.approval_status == "PendingManualApproval"
    assert pending.image_uri == spec.image_uri and pending.model_data_url == spec.model_data_url
    assert pending.metadata["metric_auc"] == "0.812300"
    assert latest_package(aws.sagemaker, "credit-pd") is None
    assert latest_package(aws.sagemaker, "credit-pd", approval=None) == describe_package(
        aws.sagemaker, pending.arn
    )

    approved = set_approval(aws.sagemaker, pending.arn, "Approved", note="reviewer: staging passed")
    assert approved.approval_status == "Approved"
    newest = latest_package(aws.sagemaker, "credit-pd")
    assert newest is not None and newest.arn == pending.arn

    second = register_model_package(
        aws.sagemaker, spec.model_copy(update={"approval_status": "Approved"})
    )
    assert second.version == 2
    latest = latest_package(aws.sagemaker, "credit-pd")
    assert latest is not None and latest.version == 2
    rejected = set_approval(aws.sagemaker, second.arn, "Rejected", note="bad")
    assert rejected.approval_status == "Rejected"
    latest_after = latest_package(aws.sagemaker, "credit-pd")
    assert latest_after is not None and latest_after.version == 1


def test_registry_request_validation() -> None:
    with pytest.raises(ValidationError):
        RegistrationSpec(group="g", image_uri="img", model_data_url="not-s3")


def test_stubber_validates_create_model_package_shape(aws: AwsClients) -> None:
    """botocore validates our request against the real service model before any network call."""
    spec = RegistrationSpec(
        group="g", image_uri="img", model_data_url="s3://b/m.tar.gz", metrics={"auc": 0.8}
    )
    with Stubber(aws.sagemaker) as stub:
        stub.add_response(
            "create_model_package",
            {"ModelPackageArn": "arn:aws:sagemaker:ap-southeast-2:123456789012:model-package/g/1"},
            spec.to_request(),
        )
        assert aws.sagemaker.create_model_package(**spec.to_request())["ModelPackageArn"].endswith(
            "/g/1"
        )
