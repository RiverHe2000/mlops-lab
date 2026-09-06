"""SageMaker training jobs: a typed spec → CreateTrainingJob request, start, wait, collect."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .clients import tags_list

if TYPE_CHECKING:
    from mypy_boto3_sagemaker import SageMakerClient

JOB_NAME_RE = re.compile(r"^[a-zA-Z0-9](-*[a-zA-Z0-9]){0,62}$")
TERMINAL = {"Completed", "Failed", "Stopped"}


class Channel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    s3_uri: str = Field(pattern=r"^s3://")
    content_type: str = "text/csv"
    distribution: Literal["FullyReplicated", "ShardedByS3Key"] = "FullyReplicated"
    input_mode: Literal["File", "Pipe", "FastFile"] = "File"


class MetricDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    regex: str


def default_metric_definitions() -> list[MetricDefinition]:
    """Regexes matching the `validation:<name>=<value>` lines the container prints."""
    return [
        MetricDefinition(name=f"validation:{m}", regex=rf"validation:{m}=([0-9.]+)")
        for m in ("auc", "log_loss", "brier", "accuracy", "precision", "recall")
    ]


class TrainingJobSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_name: str
    image_uri: str
    role_arn: str = Field(pattern=r"^arn:aws:iam::\d{12}:role/")
    instance_type: str = "ml.m5.large"
    instance_count: int = Field(default=1, ge=1)
    volume_gb: int = Field(default=10, ge=1)
    max_runtime_s: int = Field(default=3600, ge=60)
    use_spot: bool = False
    max_wait_s: int | None = None
    checkpoint_s3_uri: str | None = None
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    channels: dict[str, Channel]
    output_s3_uri: str = Field(pattern=r"^s3://")
    metric_definitions: list[MetricDefinition] = Field(default_factory=default_metric_definitions)
    environment: dict[str, str] = Field(default_factory=dict)
    tags: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _rules(self) -> TrainingJobSpec:
        if not JOB_NAME_RE.match(self.job_name):
            raise ValueError(f"invalid training job name {self.job_name!r}")
        if "train" not in self.channels:
            raise ValueError("a 'train' channel is required")
        if self.use_spot:
            if self.max_wait_s is None or self.max_wait_s < self.max_runtime_s:
                raise ValueError("spot training needs max_wait_s >= max_runtime_s")
            if not self.checkpoint_s3_uri:
                raise ValueError(
                    "spot training needs checkpoint_s3_uri so interruptions can resume"
                )
        return self

    def hyperparameters_as_strings(self) -> dict[str, str]:
        return {
            k: (v if isinstance(v, str) else json.dumps(v)) for k, v in self.hyperparameters.items()
        }

    def to_request(self) -> dict[str, Any]:
        req: dict[str, Any] = {
            "TrainingJobName": self.job_name,
            "AlgorithmSpecification": {
                "TrainingImage": self.image_uri,
                "TrainingInputMode": "File",
                "MetricDefinitions": [
                    {"Name": m.name, "Regex": m.regex} for m in self.metric_definitions
                ],
            },
            "RoleArn": self.role_arn,
            "InputDataConfig": [
                {
                    "ChannelName": name,
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": ch.s3_uri,
                            "S3DataDistributionType": ch.distribution,
                        }
                    },
                    "ContentType": ch.content_type,
                    "InputMode": ch.input_mode,
                }
                for name, ch in self.channels.items()
            ],
            "OutputDataConfig": {"S3OutputPath": self.output_s3_uri},
            "ResourceConfig": {
                "InstanceType": self.instance_type,
                "InstanceCount": self.instance_count,
                "VolumeSizeInGB": self.volume_gb,
            },
            "StoppingCondition": {"MaxRuntimeInSeconds": self.max_runtime_s},
            "HyperParameters": self.hyperparameters_as_strings(),
            "EnableManagedSpotTraining": self.use_spot,
        }
        if self.use_spot:
            req["StoppingCondition"]["MaxWaitTimeInSeconds"] = self.max_wait_s
        if self.checkpoint_s3_uri:
            req["CheckpointConfig"] = {"S3Uri": self.checkpoint_s3_uri}
        if self.environment:
            req["Environment"] = dict(self.environment)
        if self.tags:
            req["Tags"] = tags_list(self.tags)
        return req


class TrainingJobError(RuntimeError):
    def __init__(self, name: str, status: str, reason: str) -> None:
        super().__init__(f"training job {name} ended {status}: {reason}")
        self.status = status
        self.reason = reason


@dataclass(frozen=True)
class TrainingOutputs:
    job_name: str
    status: str
    model_artifact_uri: str | None
    metrics: dict[str, float]
    billable_seconds: int | None


def start_training_job(sm: SageMakerClient, spec: TrainingJobSpec) -> str:
    return str(sm.create_training_job(**spec.to_request())["TrainingJobArn"])


def wait_for_training_job(
    sm: SageMakerClient,
    name: str,
    *,
    poll_s: float = 30.0,
    timeout_s: float = 4 * 3600,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    deadline = clock() + timeout_s
    while True:
        desc = dict(sm.describe_training_job(TrainingJobName=name))
        status = str(desc["TrainingJobStatus"])
        if status in TERMINAL:
            if status != "Completed":
                raise TrainingJobError(
                    name, status, str(desc.get("FailureReason", "no reason given"))
                )
            return desc
        if clock() >= deadline:
            raise TimeoutError(f"training job {name} still {status} after {timeout_s:.0f}s")
        sleep(poll_s)


def training_outputs(desc: dict[str, Any]) -> TrainingOutputs:
    metrics = {
        str(m["MetricName"]): float(m["Value"])
        for m in desc.get("FinalMetricDataList", [])
        if "MetricName" in m and "Value" in m
    }
    artifacts = desc.get("ModelArtifacts", {}) or {}
    return TrainingOutputs(
        job_name=str(desc["TrainingJobName"]),
        status=str(desc["TrainingJobStatus"]),
        model_artifact_uri=str(artifacts["S3ModelArtifacts"])
        if artifacts.get("S3ModelArtifacts")
        else None,
        metrics=metrics,
        billable_seconds=int(desc["BillableTimeInSeconds"])
        if desc.get("BillableTimeInSeconds") is not None
        else None,
    )
