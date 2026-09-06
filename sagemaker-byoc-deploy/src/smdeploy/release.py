"""The release procedure a CD job runs for one stage (staging, production):

    approved package ─► alarms ─► deploy (create | blue/green update) ─► wait InService
                    ─► smoke test through the runtime API ─► autoscaling
                    └─ smoke failure after an update ─► roll back to the previous config

Everything is driven by a YAML stage config and typed specs, so `smdeploy plan` can print the
exact API requests without touching AWS."""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field

from .aws.endpoint import (
    DeploymentError,
    DeployResult,
    EndpointDeployer,
    EndpointSpec,
    RolloutStrategy,
)
from .aws.operations import (
    AlarmSpec,
    AutoscalingSpec,
    SmokeResult,
    configure_autoscaling,
    ensure_rollback_alarms,
    smoke_test,
)
from .aws.registry import ModelPackageInfo

if TYPE_CHECKING:
    from .aws.clients import AwsClients

_VAR = re.compile(r"\$\{([A-Z0-9_]+)(?::-([^}]*))?\}")


def expand_env(text: str, env: dict[str, str] | None = None) -> str:
    """`${NAME}` / `${NAME:-default}` substitution; unset without a default is an error."""
    e = dict(os.environ) if env is None else env

    def repl(m: re.Match[str]) -> str:
        name, default = m.group(1), m.group(2)
        if name in e:
            return e[name]
        if default is not None:
            return default
        raise KeyError(f"environment variable {name} is required by the config")

    return _VAR.sub(repl, text)


def load_yaml(path: Path, env: dict[str, str] | None = None) -> dict[str, Any]:
    loaded = yaml.safe_load(expand_env(path.read_text(encoding="utf-8"), env))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    return loaded


class StageConfig(BaseModel):
    """One deployment stage. `endpoint.image_uri`/`model_data_url` are filled from the package."""

    model_config = ConfigDict(extra="forbid")

    stage: str
    model_package_group: str
    require_approved: bool = True
    endpoint: dict[str, Any]
    rollout: RolloutStrategy = Field(default_factory=RolloutStrategy)
    alarms: AlarmSpec | None = Field(default_factory=AlarmSpec)
    autoscaling: AutoscalingSpec | None = None
    smoke_rows: int = Field(default=5, ge=1, le=100)

    @classmethod
    def from_yaml(cls, path: Path, env: dict[str, str] | None = None) -> StageConfig:
        return cls.model_validate(load_yaml(path, env))

    def endpoint_spec(self, package: ModelPackageInfo) -> EndpointSpec:
        if not package.image_uri or not package.model_data_url:
            raise ValueError(f"model package {package.arn} has no container image / model data")
        merged = {
            **self.endpoint,
            "image_uri": package.image_uri,
            "model_data_url": package.model_data_url,
        }
        tags = {
            **merged.get("tags", {}),
            "smdeploy:stage": self.stage,
            "smdeploy:package": package.arn.rsplit("/", 1)[-1],
        }
        merged["tags"] = tags
        return EndpointSpec.model_validate(merged)


@dataclass
class ReleaseReport:
    stage: str
    package_arn: str
    deploy: DeployResult
    smoke: SmokeResult | None
    alarms: list[str] = field(default_factory=list)
    autoscaling_policy: str | None = None
    rolled_back_to: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "package_arn": self.package_arn,
            "deploy": asdict(self.deploy),
            "smoke": asdict(self.smoke) if self.smoke else None,
            "alarms": list(self.alarms),
            "autoscaling_policy": self.autoscaling_policy,
            "rolled_back_to": self.rolled_back_to,
        }


class ReleaseError(RuntimeError):
    def __init__(self, message: str, report: ReleaseReport) -> None:
        super().__init__(message)
        self.report = report


def release(
    clients: AwsClients,
    stage: StageConfig,
    package: ModelPackageInfo,
    smoke_rows: pd.DataFrame,
    feature_columns: list[str],
    *,
    deployer: EndpointDeployer | None = None,
    wait_timeout_s: float = 3600.0,
) -> ReleaseReport:
    if stage.require_approved and package.approval_status != "Approved":
        raise ValueError(
            f"{stage.stage}: package {package.arn} is {package.approval_status}, not Approved"
        )
    deployer = deployer or EndpointDeployer(clients.sagemaker)
    spec = stage.endpoint_spec(package)
    strategy = stage.rollout

    alarms: list[str] = []
    if stage.alarms is not None:
        alarms = ensure_rollback_alarms(
            clients.cloudwatch, spec.endpoint_name, spec.variant_name, stage.alarms
        )
        strategy = strategy.model_copy(update={"rollback_alarms": alarms})

    deploy = deployer.deploy(spec, strategy)
    report = ReleaseReport(stage.stage, package.arn, deploy, None, alarms)
    if deploy.action != "unchanged":
        try:
            deployer.wait_in_service(
                spec.endpoint_name,
                expected_config=deploy.endpoint_config_name,
                timeout_s=wait_timeout_s,
            )
        except (DeploymentError, TimeoutError) as exc:
            raise ReleaseError(str(exc), report) from exc

    smoke = smoke_test(
        clients.sagemaker_runtime,
        spec.endpoint_name,
        smoke_rows.head(stage.smoke_rows),
        feature_columns,
    )
    report.smoke = smoke
    if not smoke.ok:
        if deploy.action == "updated" and deploy.previous_config_name:
            deployer.rollback(
                spec.endpoint_name, deploy.previous_config_name, timeout_s=wait_timeout_s
            )
            report.rolled_back_to = deploy.previous_config_name
        raise ReleaseError(f"smoke test failed: {smoke.detail}", report)

    if stage.autoscaling is not None and spec.serverless is None:
        report.autoscaling_policy = configure_autoscaling(
            clients.application_autoscaling,
            spec.endpoint_name,
            spec.variant_name,
            stage.autoscaling,
        )
    return report
