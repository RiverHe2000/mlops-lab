"""SageMaker Model Registry: model package groups, versions with metrics, approval workflow.

Deployments only ever consume *Approved* packages; CI registers `PendingManualApproval` and a
person (or an automated gate with an audit trail) flips the status."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from botocore.exceptions import ClientError
from pydantic import BaseModel, ConfigDict, Field

from .clients import is_not_found, tags_list

if TYPE_CHECKING:
    from mypy_boto3_sagemaker import SageMakerClient

ApprovalStatus = Literal["Approved", "Rejected", "PendingManualApproval"]


@dataclass(frozen=True)
class ModelPackageInfo:
    arn: str
    group: str
    version: int
    status: str
    approval_status: str
    image_uri: str | None
    model_data_url: str | None
    created: datetime | None
    metadata: dict[str, str] = field(default_factory=dict)
    description: str = ""


class RegistrationSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    group: str
    image_uri: str
    model_data_url: str = Field(pattern=r"^s3://")
    metrics: dict[str, float] = Field(default_factory=dict)
    evaluation_s3_uri: str | None = None
    description: str = ""
    approval_status: ApprovalStatus = "PendingManualApproval"
    content_types: list[str] = Field(default_factory=lambda: ["text/csv", "application/json"])
    response_types: list[str] = Field(default_factory=lambda: ["application/json", "text/csv"])
    inference_instance_types: list[str] = Field(
        default_factory=lambda: ["ml.m5.large", "ml.m5.xlarge"]
    )
    transform_instance_types: list[str] = Field(default_factory=lambda: ["ml.m5.large"])
    customer_metadata: dict[str, str] = Field(default_factory=dict)
    tags: dict[str, str] = Field(default_factory=dict)

    def to_request(self) -> dict[str, Any]:
        metadata = {
            **{f"metric_{k}": f"{v:.6f}" for k, v in self.metrics.items()},
            **self.customer_metadata,
        }
        req: dict[str, Any] = {
            "ModelPackageGroupName": self.group,
            "ModelPackageDescription": self.description or f"smdeploy model from {self.image_uri}",
            "ModelApprovalStatus": self.approval_status,
            "InferenceSpecification": {
                "Containers": [{"Image": self.image_uri, "ModelDataUrl": self.model_data_url}],
                "SupportedContentTypes": list(self.content_types),
                "SupportedResponseMIMETypes": list(self.response_types),
                "SupportedRealtimeInferenceInstanceTypes": list(self.inference_instance_types),
                "SupportedTransformInstanceTypes": list(self.transform_instance_types),
            },
        }
        if metadata:
            req["CustomerMetadataProperties"] = metadata
        if self.evaluation_s3_uri:
            req["ModelMetrics"] = {
                "ModelQuality": {
                    "Statistics": {
                        "ContentType": "application/json",
                        "S3Uri": self.evaluation_s3_uri,
                    }
                }
            }
        if self.tags:
            req["Tags"] = tags_list(self.tags)
        return req


def ensure_model_package_group(sm: SageMakerClient, name: str, description: str = "") -> str:
    try:
        return str(
            sm.describe_model_package_group(ModelPackageGroupName=name)["ModelPackageGroupArn"]
        )
    except ClientError as exc:
        if not is_not_found(exc):
            raise
    created = sm.create_model_package_group(
        ModelPackageGroupName=name,
        ModelPackageGroupDescription=description or f"{name} model versions",
    )
    return str(created["ModelPackageGroupArn"])


def _info(desc: dict[str, Any]) -> ModelPackageInfo:
    containers = (desc.get("InferenceSpecification") or {}).get("Containers") or [{}]
    return ModelPackageInfo(
        arn=str(desc["ModelPackageArn"]),
        group=str(desc.get("ModelPackageGroupName", "")),
        version=int(desc.get("ModelPackageVersion", 0)),
        status=str(desc.get("ModelPackageStatus", "")),
        approval_status=str(desc.get("ModelApprovalStatus", "")),
        image_uri=containers[0].get("Image"),
        model_data_url=containers[0].get("ModelDataUrl"),
        created=desc.get("CreationTime"),
        metadata={
            str(k): str(v) for k, v in (desc.get("CustomerMetadataProperties") or {}).items()
        },
        description=str(desc.get("ModelPackageDescription", "")),
    )


def describe_package(sm: SageMakerClient, arn: str) -> ModelPackageInfo:
    return _info(dict(sm.describe_model_package(ModelPackageName=arn)))


def register_model_package(sm: SageMakerClient, spec: RegistrationSpec) -> ModelPackageInfo:
    ensure_model_package_group(sm, spec.group)
    arn = str(sm.create_model_package(**spec.to_request())["ModelPackageArn"])
    return describe_package(sm, arn)


def set_approval(
    sm: SageMakerClient, arn: str, status: ApprovalStatus, *, note: str
) -> ModelPackageInfo:
    sm.update_model_package(
        ModelPackageArn=arn, ModelApprovalStatus=status, ApprovalDescription=note[:1024]
    )
    return describe_package(sm, arn)


def latest_package(
    sm: SageMakerClient, group: str, *, approval: ApprovalStatus | None = "Approved"
) -> ModelPackageInfo | None:
    kwargs: dict[str, Any] = {
        "ModelPackageGroupName": group,
        "SortBy": "CreationTime",
        "SortOrder": "Descending",
        "MaxResults": 20,
    }
    if approval is not None:
        kwargs["ModelApprovalStatus"] = approval
    try:
        summaries = sm.list_model_packages(**kwargs)["ModelPackageSummaryList"]
    except ClientError as exc:
        if is_not_found(exc):
            return None
        raise
    if not summaries:
        return None
    newest = max(summaries, key=lambda s: int(s.get("ModelPackageVersion", 0)))
    return describe_package(sm, str(newest["ModelPackageArn"]))
