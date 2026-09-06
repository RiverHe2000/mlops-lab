"""Day-2 operations around an endpoint: target-tracking autoscaling, the CloudWatch alarms
that drive auto-rollback, and an end-to-end smoke test through the runtime API."""

from __future__ import annotations

import io
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from mypy_boto3_application_autoscaling import ApplicationAutoScalingClient
    from mypy_boto3_cloudwatch import CloudWatchClient
    from mypy_boto3_sagemaker_runtime import SageMakerRuntimeClient


# --- autoscaling -----------------------------------------------------------------------------
class AutoscalingSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_capacity: int = Field(default=1, ge=1)
    max_capacity: int = Field(default=4, ge=1)
    target_invocations_per_instance: float = Field(default=70.0, gt=0)
    scale_in_cooldown_s: int = Field(default=300, ge=0)
    scale_out_cooldown_s: int = Field(default=60, ge=0)


def resource_id(endpoint_name: str, variant_name: str) -> str:
    return f"endpoint/{endpoint_name}/variant/{variant_name}"


def configure_autoscaling(
    aas: ApplicationAutoScalingClient, endpoint_name: str, variant_name: str, spec: AutoscalingSpec
) -> str:
    if spec.max_capacity < spec.min_capacity:
        raise ValueError("max_capacity must be >= min_capacity")
    rid = resource_id(endpoint_name, variant_name)
    aas.register_scalable_target(
        ServiceNamespace="sagemaker",
        ResourceId=rid,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        MinCapacity=spec.min_capacity,
        MaxCapacity=spec.max_capacity,
    )
    policy = aas.put_scaling_policy(
        PolicyName=f"{endpoint_name}-{variant_name}-invocations",
        ServiceNamespace="sagemaker",
        ResourceId=rid,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        PolicyType="TargetTrackingScaling",
        TargetTrackingScalingPolicyConfiguration={
            "TargetValue": spec.target_invocations_per_instance,
            "PredefinedMetricSpecification": {
                "PredefinedMetricType": "SageMakerVariantInvocationsPerInstance"
            },
            "ScaleInCooldown": spec.scale_in_cooldown_s,
            "ScaleOutCooldown": spec.scale_out_cooldown_s,
        },
    )
    return str(policy["PolicyARN"])


def describe_autoscaling(
    aas: ApplicationAutoScalingClient, endpoint_name: str, variant_name: str
) -> dict[str, Any]:
    rid = resource_id(endpoint_name, variant_name)
    targets = aas.describe_scalable_targets(ServiceNamespace="sagemaker", ResourceIds=[rid])[
        "ScalableTargets"
    ]
    policies = aas.describe_scaling_policies(ServiceNamespace="sagemaker", ResourceId=rid)[
        "ScalingPolicies"
    ]
    return {
        "resource_id": rid,
        "targets": [dict(t) for t in targets],
        "policies": [{"name": p["PolicyName"], "type": p["PolicyType"]} for p in policies],
    }


# --- alarms ----------------------------------------------------------------------------------
class AlarmSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error_5xx_threshold: float = Field(default=1.0, ge=0)
    latency_p99_ms: float = Field(default=2000.0, gt=0)
    period_s: int = Field(default=60, ge=10)
    evaluation_periods: int = Field(default=2, ge=1)


def alarm_names(endpoint_name: str) -> list[str]:
    return [f"{endpoint_name}-5xx-errors", f"{endpoint_name}-latency-p99"]


def ensure_rollback_alarms(
    cw: CloudWatchClient, endpoint_name: str, variant_name: str, spec: AlarmSpec
) -> list[str]:
    """Two alarms SageMaker watches during a blue/green update; if either fires it rolls back."""
    dims: list[Any] = [
        {"Name": "EndpointName", "Value": endpoint_name},
        {"Name": "VariantName", "Value": variant_name},
    ]
    names = alarm_names(endpoint_name)
    cw.put_metric_alarm(
        AlarmName=names[0],
        AlarmDescription=f"5xx responses from {endpoint_name}/{variant_name}",
        Namespace="AWS/SageMaker",
        MetricName="Invocation5XXErrors",
        Statistic="Sum",
        Dimensions=dims,
        Period=spec.period_s,
        EvaluationPeriods=spec.evaluation_periods,
        Threshold=spec.error_5xx_threshold,
        ComparisonOperator="GreaterThanOrEqualToThreshold",
        TreatMissingData="notBreaching",
    )
    cw.put_metric_alarm(
        AlarmName=names[1],
        AlarmDescription=f"p99 model latency of {endpoint_name}/{variant_name}",
        Namespace="AWS/SageMaker",
        MetricName="ModelLatency",
        ExtendedStatistic="p99",
        Dimensions=dims,
        Period=spec.period_s,
        EvaluationPeriods=spec.evaluation_periods,
        Threshold=spec.latency_p99_ms * 1000.0,  # ModelLatency is reported in microseconds
        ComparisonOperator="GreaterThanThreshold",
        TreatMissingData="notBreaching",
    )
    return names


# --- smoke test ------------------------------------------------------------------------------
@dataclass(frozen=True)
class SmokeResult:
    ok: bool
    status_code: int
    latency_ms: float
    n_predictions: int
    detail: str


def csv_payload(rows: pd.DataFrame, columns: list[str]) -> bytes:
    buf = io.StringIO()
    rows[columns].to_csv(buf, header=False, index=False, lineterminator="\n")
    return buf.getvalue().encode("utf-8")


def smoke_test(
    runtime: SageMakerRuntimeClient,
    endpoint_name: str,
    rows: pd.DataFrame,
    columns: list[str],
    *,
    target_variant: str | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> SmokeResult:
    """Send real rows through `InvokeEndpoint` and check the response is sane."""
    body = csv_payload(rows, columns)
    kwargs: dict[str, Any] = {
        "EndpointName": endpoint_name,
        "ContentType": "text/csv",
        "Accept": "application/json",
        "Body": body,
    }
    if target_variant:
        kwargs["TargetVariant"] = target_variant
    started = clock()
    try:
        response = runtime.invoke_endpoint(**kwargs)
    except Exception as exc:
        return SmokeResult(False, 0, (clock() - started) * 1000, 0, f"{type(exc).__name__}: {exc}")
    latency = (clock() - started) * 1000
    status = int(response.get("ResponseMetadata", {}).get("HTTPStatusCode", 200))
    raw = response["Body"].read()
    try:
        payload = json.loads(raw)
        predictions = payload["predictions"]
        probs = [float(p["probability"]) for p in predictions]
    except (ValueError, KeyError, TypeError) as exc:
        return SmokeResult(False, status, latency, 0, f"unparseable response: {exc}: {raw[:200]!r}")
    if len(predictions) != len(rows):
        return SmokeResult(
            False,
            status,
            latency,
            len(predictions),
            f"expected {len(rows)} predictions, got {len(predictions)}",
        )
    if any(not 0.0 <= p <= 1.0 for p in probs):
        return SmokeResult(False, status, latency, len(predictions), "probability outside [0, 1]")
    return SmokeResult(
        True,
        status,
        latency,
        len(predictions),
        f"{len(predictions)} predictions in {latency:.0f} ms",
    )
