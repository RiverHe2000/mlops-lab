"""Typed boto3 client bundle. Everything downstream takes clients as arguments so tests can
pass moto-backed clients, botocore Stubbers or plain fakes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

if TYPE_CHECKING:
    from mypy_boto3_application_autoscaling import ApplicationAutoScalingClient
    from mypy_boto3_cloudwatch import CloudWatchClient
    from mypy_boto3_ecr import ECRClient
    from mypy_boto3_s3 import S3Client
    from mypy_boto3_sagemaker import SageMakerClient
    from mypy_boto3_sagemaker_runtime import SageMakerRuntimeClient
    from mypy_boto3_sts import STSClient

RETRY_CONFIG = Config(
    retries={"max_attempts": 8, "mode": "adaptive"}, connect_timeout=10, read_timeout=60
)


@dataclass
class AwsClients:
    region: str
    sagemaker: SageMakerClient
    sagemaker_runtime: SageMakerRuntimeClient
    ecr: ECRClient
    sts: STSClient
    cloudwatch: CloudWatchClient
    application_autoscaling: ApplicationAutoScalingClient
    s3: S3Client

    @classmethod
    def from_session(cls, *, region: str | None = None, profile: str | None = None) -> AwsClients:
        session = boto3.Session(profile_name=profile, region_name=region)
        resolved = session.region_name
        if not resolved:
            raise ValueError("no AWS region: pass --region or set AWS_REGION / AWS_DEFAULT_REGION")
        return cls(
            region=resolved,
            sagemaker=session.client("sagemaker", config=RETRY_CONFIG),
            sagemaker_runtime=session.client("sagemaker-runtime", config=RETRY_CONFIG),
            ecr=session.client("ecr", config=RETRY_CONFIG),
            sts=session.client("sts", config=RETRY_CONFIG),
            cloudwatch=session.client("cloudwatch", config=RETRY_CONFIG),
            application_autoscaling=session.client("application-autoscaling", config=RETRY_CONFIG),
            s3=session.client("s3", config=RETRY_CONFIG),
        )

    def account_id(self) -> str:
        return str(self.sts.get_caller_identity()["Account"])


def error_code(exc: ClientError) -> str:
    return str(exc.response.get("Error", {}).get("Code", ""))


def error_message(exc: ClientError) -> str:
    return str(exc.response.get("Error", {}).get("Message", ""))


def is_not_found(exc: ClientError) -> bool:
    """SageMaker reports missing resources as ValidationException('Could not find …')."""
    code = error_code(exc)
    if code in {"ResourceNotFound", "ResourceNotFoundException", "RepositoryNotFoundException"}:
        return True
    message = error_message(exc).lower()
    return code == "ValidationException" and (
        "could not find" in message or "does not exist" in message
    )


def tags_list(tags: dict[str, str]) -> list[dict[str, Any]]:
    return [{"Key": k, "Value": v} for k, v in sorted(tags.items())]
