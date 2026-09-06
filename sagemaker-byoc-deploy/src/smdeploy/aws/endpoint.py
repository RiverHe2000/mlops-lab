"""Real-time endpoints: idempotent model / endpoint-config creation, blue/green updates with
canary or linear traffic shifting and CloudWatch-alarm auto-rollback, explicit rollback.

Naming is content-addressed: the endpoint config name embeds a hash of everything that
defines it, so re-running a deployment with the same inputs is a no-op and a change produces a
new config that the endpoint is *updated* to (SageMaker endpoint configs are immutable)."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from botocore.exceptions import ClientError
from pydantic import BaseModel, ConfigDict, Field

from .clients import is_not_found, tags_list

if TYPE_CHECKING:
    from mypy_boto3_sagemaker import SageMakerClient

IN_SERVICE = "InService"
FAILED_STATES = {"Failed", "UpdateRollbackFailed", "OutOfService"}


class DataCapture(BaseModel):
    model_config = ConfigDict(extra="forbid")

    s3_uri: str = Field(pattern=r"^s3://")
    sampling_percent: int = Field(default=100, ge=0, le=100)
    capture_input: bool = True
    capture_output: bool = True


class ServerlessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_mb: Literal[1024, 2048, 3072, 4096, 5120, 6144] = 2048
    max_concurrency: int = Field(default=5, ge=1, le=200)


class EndpointSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    endpoint_name: str = Field(pattern=r"^[a-zA-Z0-9](-*[a-zA-Z0-9]){0,62}$")
    model_name_prefix: str = Field(pattern=r"^[a-zA-Z0-9](-*[a-zA-Z0-9]){0,40}$")
    image_uri: str
    model_data_url: str = Field(pattern=r"^s3://")
    role_arn: str = Field(pattern=r"^arn:aws:iam::\d{12}:role/")
    instance_type: str = "ml.m5.large"
    instance_count: int = Field(default=1, ge=1)
    variant_name: str = "AllTraffic"
    environment: dict[str, str] = Field(default_factory=dict)
    data_capture: DataCapture | None = None
    serverless: ServerlessConfig | None = None
    tags: dict[str, str] = Field(default_factory=dict)

    def fingerprint(self) -> str:
        payload = self.model_dump(mode="json", exclude={"tags"})
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:10]

    @property
    def model_name(self) -> str:
        return f"{self.model_name_prefix}-{self.fingerprint()}"

    @property
    def endpoint_config_name(self) -> str:
        return f"{self.endpoint_name}-{self.fingerprint()}"

    def production_variant(self) -> dict[str, Any]:
        variant: dict[str, Any] = {
            "VariantName": self.variant_name,
            "ModelName": self.model_name,
            "InitialVariantWeight": 1.0,
        }
        if self.serverless is not None:
            variant["ServerlessConfig"] = {
                "MemorySizeInMB": self.serverless.memory_mb,
                "MaxConcurrency": self.serverless.max_concurrency,
            }
        else:
            variant["InitialInstanceCount"] = self.instance_count
            variant["InstanceType"] = self.instance_type
        return variant


class RolloutStrategy(BaseModel):
    """How traffic moves from the old fleet (blue) to the new one (green)."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["all_at_once", "canary", "linear"] = "canary"
    canary_percent: int = Field(default=10, ge=1, le=50)
    linear_step_percent: int = Field(default=25, ge=1, le=50)
    wait_interval_s: int = Field(default=300, ge=0, le=3600)
    termination_wait_s: int = Field(default=120, ge=0, le=3600)
    max_timeout_s: int = Field(default=3600, ge=600, le=14400)
    rollback_alarms: list[str] = Field(default_factory=list)

    def to_deployment_config(self) -> dict[str, Any]:
        routing: dict[str, Any] = {"WaitIntervalInSeconds": self.wait_interval_s}
        if self.mode == "all_at_once":
            routing["Type"] = "ALL_AT_ONCE"
        elif self.mode == "canary":
            routing["Type"] = "CANARY"
            routing["CanarySize"] = {"Type": "CAPACITY_PERCENT", "Value": self.canary_percent}
        else:
            routing["Type"] = "LINEAR"
            routing["LinearStepSize"] = {
                "Type": "CAPACITY_PERCENT",
                "Value": self.linear_step_percent,
            }
        config: dict[str, Any] = {
            "BlueGreenUpdatePolicy": {
                "TrafficRoutingConfiguration": routing,
                "TerminationWaitInSeconds": self.termination_wait_s,
                "MaximumExecutionTimeoutInSeconds": self.max_timeout_s,
            }
        }
        if self.rollback_alarms:
            config["AutoRollbackConfiguration"] = {
                "Alarms": [{"AlarmName": a} for a in self.rollback_alarms]
            }
        return config


@dataclass(frozen=True)
class DeployResult:
    action: Literal["created", "updated", "unchanged"]
    endpoint_name: str
    endpoint_config_name: str
    model_name: str
    previous_config_name: str | None


class DeploymentError(RuntimeError):
    pass


class EndpointDeployer:
    def __init__(
        self,
        sm: SageMakerClient,
        *,
        poll_s: float = 15.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.sm = sm
        self.poll_s = poll_s
        self.sleep = sleep
        self.clock = clock

    # --- idempotent building blocks ------------------------------------------------------
    def ensure_model(self, spec: EndpointSpec) -> str:
        name = spec.model_name
        try:
            self.sm.describe_model(ModelName=name)
            return name
        except ClientError as exc:
            if not is_not_found(exc):
                raise
        container: dict[str, Any] = {"Image": spec.image_uri, "ModelDataUrl": spec.model_data_url}
        if spec.environment:
            container["Environment"] = dict(spec.environment)
        req: dict[str, Any] = {
            "ModelName": name,
            "PrimaryContainer": container,
            "ExecutionRoleArn": spec.role_arn,
        }
        if spec.tags:
            req["Tags"] = tags_list(spec.tags)
        self.sm.create_model(**req)
        return name

    def ensure_endpoint_config(self, spec: EndpointSpec) -> str:
        name = spec.endpoint_config_name
        try:
            self.sm.describe_endpoint_config(EndpointConfigName=name)
            return name
        except ClientError as exc:
            if not is_not_found(exc):
                raise
        req: dict[str, Any] = {
            "EndpointConfigName": name,
            "ProductionVariants": [spec.production_variant()],
        }
        if spec.data_capture is not None:
            dc = spec.data_capture
            options = []
            if dc.capture_input:
                options.append({"CaptureMode": "Input"})
            if dc.capture_output:
                options.append({"CaptureMode": "Output"})
            req["DataCaptureConfig"] = {
                "EnableCapture": True,
                "InitialSamplingPercentage": dc.sampling_percent,
                "DestinationS3Uri": dc.s3_uri,
                "CaptureOptions": options,
                "CaptureContentTypeHeader": {
                    "CsvContentTypes": ["text/csv"],
                    "JsonContentTypes": ["application/json", "application/jsonlines"],
                },
            }
        if spec.tags:
            req["Tags"] = tags_list(spec.tags)
        self.sm.create_endpoint_config(**req)
        return name

    def describe(self, endpoint_name: str) -> dict[str, Any] | None:
        try:
            return dict(self.sm.describe_endpoint(EndpointName=endpoint_name))
        except ClientError as exc:
            if is_not_found(exc):
                return None
            raise

    def _update(
        self, endpoint_name: str, config_name: str, deployment_config: dict[str, Any]
    ) -> None:
        req: dict[str, Any] = {
            "EndpointName": endpoint_name,
            "EndpointConfigName": config_name,
            "DeploymentConfig": deployment_config,
            "RetainAllVariantProperties": False,
        }
        self.sm.update_endpoint(**req)

    # --- deploy / wait / rollback ---------------------------------------------------------
    def deploy(self, spec: EndpointSpec, strategy: RolloutStrategy) -> DeployResult:
        model_name = self.ensure_model(spec)
        config_name = self.ensure_endpoint_config(spec)
        current = self.describe(spec.endpoint_name)
        if current is None:
            req: dict[str, Any] = {
                "EndpointName": spec.endpoint_name,
                "EndpointConfigName": config_name,
            }
            if spec.tags:
                req["Tags"] = tags_list(spec.tags)
            self.sm.create_endpoint(**req)
            return DeployResult("created", spec.endpoint_name, config_name, model_name, None)
        previous = str(current["EndpointConfigName"])
        if previous == config_name:
            return DeployResult("unchanged", spec.endpoint_name, config_name, model_name, previous)
        if current.get("EndpointStatus") != IN_SERVICE:
            raise DeploymentError(
                f"{spec.endpoint_name} is {current.get('EndpointStatus')}; "
                "only InService endpoints can be updated"
            )
        self._update(spec.endpoint_name, config_name, strategy.to_deployment_config())
        return DeployResult("updated", spec.endpoint_name, config_name, model_name, previous)

    def wait_in_service(
        self, endpoint_name: str, *, expected_config: str, timeout_s: float = 3600.0
    ) -> dict[str, Any]:
        """Poll until InService; a rollback by SageMaker surfaces as the *old* config name."""
        deadline = self.clock() + timeout_s
        while True:
            desc = self.describe(endpoint_name)
            if desc is None:
                raise DeploymentError(f"{endpoint_name} disappeared while waiting")
            status = str(desc.get("EndpointStatus"))
            if status in FAILED_STATES:
                raise DeploymentError(
                    f"{endpoint_name} is {status}: {desc.get('FailureReason', 'no reason')}"
                )
            if status == IN_SERVICE:
                actual = str(desc.get("EndpointConfigName"))
                if actual != expected_config:
                    raise DeploymentError(
                        f"{endpoint_name} is InService on {actual}, not {expected_config}: "
                        f"SageMaker rolled the update back "
                        f"({desc.get('FailureReason', 'alarm or health check')})"
                    )
                return desc
            if self.clock() >= deadline:
                raise TimeoutError(f"{endpoint_name} still {status} after {timeout_s:.0f}s")
            self.sleep(self.poll_s)

    def rollback(
        self, endpoint_name: str, to_config_name: str, *, timeout_s: float = 3600.0
    ) -> dict[str, Any]:
        rollback_strategy = RolloutStrategy(mode="all_at_once", wait_interval_s=0)
        self._update(endpoint_name, to_config_name, rollback_strategy.to_deployment_config())
        return self.wait_in_service(
            endpoint_name, expected_config=to_config_name, timeout_s=timeout_s
        )

    def delete(
        self, endpoint_name: str, *, delete_configs: bool = True, delete_models: bool = True
    ) -> list[str]:
        removed: list[str] = []
        desc = self.describe(endpoint_name)
        if desc is None:
            return removed
        config_name = str(desc["EndpointConfigName"])
        self.sm.delete_endpoint(EndpointName=endpoint_name)
        removed.append(f"endpoint:{endpoint_name}")
        if delete_configs:
            cfg: dict[str, Any] = dict(
                self.sm.describe_endpoint_config(EndpointConfigName=config_name)
            )
            self.sm.delete_endpoint_config(EndpointConfigName=config_name)
            removed.append(f"endpoint-config:{config_name}")
            if delete_models:
                for variant in cfg.get("ProductionVariants", []):
                    self.sm.delete_model(ModelName=str(variant["ModelName"]))
                    removed.append(f"model:{variant['ModelName']}")
        return removed
