from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from botocore.stub import Stubber
from pydantic import ValidationError

from smdeploy.aws.clients import AwsClients
from smdeploy.aws.endpoint import (
    DataCapture,
    DeploymentError,
    EndpointDeployer,
    EndpointSpec,
    RolloutStrategy,
    ServerlessConfig,
)
from smdeploy.aws.operations import (
    AlarmSpec,
    AutoscalingSpec,
    alarm_names,
    configure_autoscaling,
    describe_autoscaling,
    ensure_rollback_alarms,
    smoke_test,
)
from smdeploy.aws.registry import ModelPackageInfo
from smdeploy.release import ReleaseError, StageConfig, expand_env, release
from smdeploy.synthetic import FEATURES

from .conftest import ROLE, ROOT, FakeClock, FakeSageMaker

IMAGE = "123456789012.dkr.ecr.ap-southeast-2.amazonaws.com/credit-pd:sha1"
MODEL_DATA = "s3://bucket/out/model.tar.gz"


def _spec(**over: Any) -> EndpointSpec:
    base: dict[str, Any] = {
        "endpoint_name": "credit-pd-staging",
        "model_name_prefix": "credit-pd",
        "image_uri": IMAGE,
        "model_data_url": MODEL_DATA,
        "role_arn": ROLE,
        "environment": {"SAGEMAKER_MODEL_SERVER_WORKERS": "2"},
        "data_capture": DataCapture(s3_uri="s3://bucket/capture/", sampling_percent=50),
        "tags": {"project": "credit-pd"},
    }
    base.update(over)
    return EndpointSpec.model_validate(base)


def _package(**over: Any) -> ModelPackageInfo:
    base: dict[str, Any] = {
        "arn": "arn:aws:sagemaker:ap-southeast-2:123456789012:model-package/credit-pd/3",
        "group": "credit-pd",
        "version": 3,
        "status": "Completed",
        "approval_status": "Approved",
        "image_uri": IMAGE,
        "model_data_url": MODEL_DATA,
        "created": None,
    }
    base.update(over)
    return ModelPackageInfo(**base)


# --- specs -----------------------------------------------------------------------------------
def test_spec_names_are_content_addressed() -> None:
    a, b = _spec(), _spec()
    assert a.fingerprint() == b.fingerprint() and a.model_name == b.model_name
    assert a.endpoint_config_name == f"credit-pd-staging-{a.fingerprint()}"
    assert _spec(tags={"x": "y"}).fingerprint() == a.fingerprint()  # tags do not change identity
    assert _spec(instance_count=2).fingerprint() != a.fingerprint()
    assert _spec(image_uri=IMAGE + "0").fingerprint() != a.fingerprint()
    variant = a.production_variant()
    assert variant["InstanceType"] == "ml.m5.large" and variant["InitialInstanceCount"] == 1
    serverless = _spec(
        serverless=ServerlessConfig(memory_mb=2048, max_concurrency=3)
    ).production_variant()
    assert serverless["ServerlessConfig"] == {"MemorySizeInMB": 2048, "MaxConcurrency": 3}
    assert "InstanceType" not in serverless
    with pytest.raises(ValidationError):
        _spec(endpoint_name="bad name")


def test_rollout_strategy_deployment_config() -> None:
    canary = RolloutStrategy(
        mode="canary", canary_percent=10, rollback_alarms=["a", "b"]
    ).to_deployment_config()
    routing = canary["BlueGreenUpdatePolicy"]["TrafficRoutingConfiguration"]
    assert routing["Type"] == "CANARY" and routing["CanarySize"] == {
        "Type": "CAPACITY_PERCENT",
        "Value": 10,
    }
    assert canary["AutoRollbackConfiguration"] == {
        "Alarms": [{"AlarmName": "a"}, {"AlarmName": "b"}]
    }
    linear = RolloutStrategy(mode="linear", linear_step_percent=25).to_deployment_config()
    assert (
        linear["BlueGreenUpdatePolicy"]["TrafficRoutingConfiguration"]["LinearStepSize"]["Value"]
        == 25
    )
    assert "AutoRollbackConfiguration" not in linear
    all_at_once = RolloutStrategy(mode="all_at_once", wait_interval_s=0).to_deployment_config()
    assert all_at_once["BlueGreenUpdatePolicy"]["TrafficRoutingConfiguration"] == {
        "Type": "ALL_AT_ONCE",
        "WaitIntervalInSeconds": 0,
    }


def test_stubber_accepts_every_rollout_mode(aws: AwsClients) -> None:
    """The DeploymentConfig we build must be valid against the real UpdateEndpoint model."""
    for mode in ("all_at_once", "canary", "linear"):
        strategy = RolloutStrategy(mode=mode, rollback_alarms=["credit-pd-prod-5xx-errors"])
        params: dict[str, Any] = {
            "EndpointName": "credit-pd-prod",
            "EndpointConfigName": "credit-pd-prod-abc",
            "DeploymentConfig": strategy.to_deployment_config(),
            "RetainAllVariantProperties": False,
        }
        with Stubber(aws.sagemaker) as stub:
            stub.add_response(
                "update_endpoint",
                {
                    "EndpointArn": "arn:aws:sagemaker:ap-southeast-2:123456789012:endpoint/credit-pd-prod"
                },
                params,
            )
            assert aws.sagemaker.update_endpoint(**params)["EndpointArn"].endswith("credit-pd-prod")


# --- deployer under moto (create path) -------------------------------------------------------
def test_deploy_creates_then_is_idempotent(aws: AwsClients) -> None:
    deployer = EndpointDeployer(aws.sagemaker, poll_s=0, sleep=lambda _s: None)
    spec = _spec()
    result = deployer.deploy(spec, RolloutStrategy())
    assert result.action == "created" and result.previous_config_name is None
    desc = deployer.wait_in_service(spec.endpoint_name, expected_config=result.endpoint_config_name)
    assert desc["EndpointStatus"] == "InService"
    cfg = aws.sagemaker.describe_endpoint_config(EndpointConfigName=result.endpoint_config_name)
    assert cfg["DataCaptureConfig"]["InitialSamplingPercentage"] == 50
    assert cfg["ProductionVariants"][0]["ModelName"] == result.model_name
    model = aws.sagemaker.describe_model(ModelName=result.model_name)
    assert (
        model["PrimaryContainer"]["Image"] == IMAGE
        and model["PrimaryContainer"]["Environment"] == spec.environment
    )
    again = deployer.deploy(spec, RolloutStrategy())
    assert again.action == "unchanged" and again.endpoint_config_name == result.endpoint_config_name
    assert deployer.describe("nope") is None
    removed = deployer.delete(spec.endpoint_name)
    assert removed == [
        f"endpoint:{spec.endpoint_name}",
        f"endpoint-config:{result.endpoint_config_name}",
        f"model:{result.model_name}",
    ]
    assert deployer.delete(spec.endpoint_name) == []


# --- deployer against the fake (update path) --------------------------------------------------
def _deployer(fake: FakeSageMaker, clock: FakeClock) -> EndpointDeployer:
    return EndpointDeployer(fake, poll_s=10, sleep=clock.sleep, clock=clock)  # type: ignore[arg-type]


def test_blue_green_update_success() -> None:
    fake, clock = FakeSageMaker(["InService"]), FakeClock()
    deployer = _deployer(fake, clock)
    v1 = _spec()
    created = deployer.deploy(v1, RolloutStrategy())
    deployer.wait_in_service(v1.endpoint_name, expected_config=created.endpoint_config_name)
    v2 = _spec(image_uri=IMAGE.replace("sha1", "sha2"))
    strategy = RolloutStrategy(mode="canary", rollback_alarms=["x"])
    updated = deployer.deploy(v2, strategy)
    assert (
        updated.action == "updated" and updated.previous_config_name == created.endpoint_config_name
    )
    update_call = next(kw for name, kw in fake.calls if name == "update_endpoint")
    assert update_call["DeploymentConfig"] == strategy.to_deployment_config()
    assert update_call["RetainAllVariantProperties"] is False
    desc = deployer.wait_in_service(v2.endpoint_name, expected_config=updated.endpoint_config_name)
    assert desc["EndpointConfigName"] == updated.endpoint_config_name and clock.now > 0


def test_update_rolled_back_by_sagemaker_and_failed() -> None:
    fake, clock = FakeSageMaker(["RolledBack", "Failed"]), FakeClock()
    deployer = _deployer(fake, clock)
    v1 = _spec()
    created = deployer.deploy(v1, RolloutStrategy())
    deployer.wait_in_service(v1.endpoint_name, expected_config=created.endpoint_config_name)
    v2 = _spec(instance_count=2)
    updated = deployer.deploy(v2, RolloutStrategy())
    with pytest.raises(DeploymentError, match="rolled the update back"):
        deployer.wait_in_service(v2.endpoint_name, expected_config=updated.endpoint_config_name)
    v3 = _spec(instance_count=3)
    updated3 = deployer.deploy(v3, RolloutStrategy())
    with pytest.raises(DeploymentError, match="Failed"):
        deployer.wait_in_service(v3.endpoint_name, expected_config=updated3.endpoint_config_name)


def test_update_refused_when_not_in_service_and_timeout() -> None:
    fake, clock = FakeSageMaker(), FakeClock()
    deployer = _deployer(fake, clock)
    v1 = _spec()
    deployer.deploy(v1, RolloutStrategy())  # endpoint is still Creating
    with pytest.raises(DeploymentError, match="only InService"):
        deployer.deploy(_spec(instance_count=2), RolloutStrategy())
    fake.endpoints[v1.endpoint_name]["EndpointStatus"] = "Updating"
    fake._pending[v1.endpoint_name].clear()
    with pytest.raises(TimeoutError):
        deployer.wait_in_service(v1.endpoint_name, expected_config="x", timeout_s=25)


def test_explicit_rollback() -> None:
    fake, clock = FakeSageMaker(["InService", "InService"]), FakeClock()
    deployer = _deployer(fake, clock)
    v1 = _spec()
    created = deployer.deploy(v1, RolloutStrategy())
    deployer.wait_in_service(v1.endpoint_name, expected_config=created.endpoint_config_name)
    updated = deployer.deploy(_spec(instance_count=2), RolloutStrategy())
    deployer.wait_in_service(v1.endpoint_name, expected_config=updated.endpoint_config_name)
    desc = deployer.rollback(v1.endpoint_name, created.endpoint_config_name)
    assert desc["EndpointConfigName"] == created.endpoint_config_name
    last = fake.calls[-1]
    rollback_call = next(kw for name, kw in reversed(fake.calls) if name == "update_endpoint")
    assert (
        rollback_call["DeploymentConfig"]["BlueGreenUpdatePolicy"]["TrafficRoutingConfiguration"][
            "Type"
        ]
        == "ALL_AT_ONCE"
    )
    assert last[0] == "describe_endpoint"


# --- operations ------------------------------------------------------------------------------
def test_autoscaling_and_alarms(aws: AwsClients) -> None:
    arn = configure_autoscaling(
        aws.application_autoscaling,
        "credit-pd-prod",
        "AllTraffic",
        AutoscalingSpec(min_capacity=2, max_capacity=6),
    )
    assert arn.startswith("arn:aws:autoscaling")
    desc = describe_autoscaling(aws.application_autoscaling, "credit-pd-prod", "AllTraffic")
    assert (
        desc["targets"][0]["MinCapacity"] == 2
        and desc["policies"][0]["type"] == "TargetTrackingScaling"
    )
    with pytest.raises(ValueError, match="max_capacity"):
        configure_autoscaling(
            aws.application_autoscaling, "e", "v", AutoscalingSpec(min_capacity=3, max_capacity=1)
        )
    names = ensure_rollback_alarms(
        aws.cloudwatch, "credit-pd-prod", "AllTraffic", AlarmSpec(latency_p99_ms=1000)
    )
    assert names == alarm_names("credit-pd-prod")
    alarms = {
        a["AlarmName"]: a for a in aws.cloudwatch.describe_alarms(AlarmNames=names)["MetricAlarms"]
    }
    assert (
        alarms[names[0]]["MetricName"] == "Invocation5XXErrors"
        and alarms[names[0]]["Statistic"] == "Sum"
    )
    assert (
        alarms[names[1]]["ExtendedStatistic"] == "p99"
        and alarms[names[1]]["Threshold"] == 1_000_000.0
    )
    assert {d["Value"] for d in alarms[names[1]]["Dimensions"]} == {"credit-pd-prod", "AllTraffic"}


def _runtime_stub(aws: AwsClients, body: bytes, status: int = 200) -> Stubber:
    stub = Stubber(aws.sagemaker_runtime)
    stub.add_response(
        "invoke_endpoint",
        {
            "Body": _Streaming(body),
            "ContentType": "application/json",
            "ResponseMetadata": {"HTTPStatusCode": status},
        },
        {
            "EndpointName": "e",
            "ContentType": "text/csv",
            "Accept": "application/json",
            "Body": _AnyBytes(),
        },
    )
    return stub


class _Streaming:
    def __init__(self, data: bytes) -> None:
        self.data = data

    def read(self) -> bytes:
        return self.data


class _AnyBytes:
    def __eq__(self, other: object) -> bool:
        return isinstance(other, bytes | bytearray)


def test_smoke_test_outcomes(aws: AwsClients, frame: pd.DataFrame) -> None:
    rows = frame.head(3)
    good = json.dumps({"predictions": [{"probability": 0.2, "label": 0}] * 3}).encode()
    with _runtime_stub(aws, good):
        result = smoke_test(aws.sagemaker_runtime, "e", rows, FEATURES)
    assert result.ok and result.n_predictions == 3 and result.status_code == 200
    short = json.dumps({"predictions": [{"probability": 0.2, "label": 0}]}).encode()
    with _runtime_stub(aws, short):
        assert not smoke_test(aws.sagemaker_runtime, "e", rows, FEATURES).ok
    with _runtime_stub(aws, b"not json"):
        assert "unparseable" in smoke_test(aws.sagemaker_runtime, "e", rows, FEATURES).detail
    bad_prob = json.dumps({"predictions": [{"probability": 1.5, "label": 1}] * 3}).encode()
    with _runtime_stub(aws, bad_prob):
        assert "outside" in smoke_test(aws.sagemaker_runtime, "e", rows, FEATURES).detail
    with Stubber(aws.sagemaker_runtime) as stub:
        stub.add_client_error("invoke_endpoint", "ModelError", "container returned 500")
        failed = smoke_test(aws.sagemaker_runtime, "e", rows, FEATURES, target_variant="AllTraffic")
    assert not failed.ok and "ModelError" in failed.detail


# --- release ---------------------------------------------------------------------------------
def test_stage_config_env_expansion(tmp_path: Path) -> None:
    assert expand_env("${A}-${B:-dflt}", {"A": "x"}) == "x-dflt"
    with pytest.raises(KeyError, match="MISSING"):
        expand_env("${MISSING}", {})
    env = {
        "PROJECT_NAME": "credit-pd",
        "SAGEMAKER_EXECUTION_ROLE_ARN": ROLE,
        "ARTIFACT_BUCKET": "bkt",
    }
    for name in ("endpoint.staging.yaml", "endpoint.prod.yaml"):
        stage = StageConfig.from_yaml(ROOT / "configs" / name, env)
        spec = stage.endpoint_spec(_package())
        assert (
            spec.image_uri == IMAGE
            and spec.tags["smdeploy:stage"] == stage.stage
            and spec.tags["smdeploy:package"] == "3"
        )
        assert spec.data_capture is not None and spec.data_capture.s3_uri.startswith(
            "s3://bkt/capture/"
        )
    prod = StageConfig.from_yaml(ROOT / "configs" / "endpoint.prod.yaml", env)
    assert prod.require_approved and prod.rollout.mode == "canary" and prod.autoscaling is not None
    with pytest.raises(ValueError, match="no container image"):
        prod.endpoint_spec(_package(image_uri=None))


def _stage(**over: Any) -> StageConfig:
    base: dict[str, Any] = {
        "stage": "staging",
        "model_package_group": "credit-pd",
        "require_approved": True,
        "endpoint": {"endpoint_name": "e", "model_name_prefix": "credit-pd", "role_arn": ROLE},
        "rollout": {"mode": "all_at_once", "wait_interval_s": 0},
        "alarms": {"latency_p99_ms": 800},
        "autoscaling": {"min_capacity": 1, "max_capacity": 2},
    }
    base.update(over)
    return StageConfig.model_validate(base)


def test_release_create_smoke_autoscale(aws: AwsClients, frame: pd.DataFrame) -> None:
    fake, clock = FakeSageMaker(), FakeClock()
    deployer = _deployer(fake, clock)
    good = json.dumps({"predictions": [{"probability": 0.3, "label": 0}] * 5}).encode()
    with _runtime_stub(aws, good):
        report = release(aws, _stage(), _package(), frame, FEATURES, deployer=deployer)
    assert report.deploy.action == "created" and report.smoke is not None and report.smoke.ok
    assert report.alarms == alarm_names("e") and report.autoscaling_policy is not None
    assert report.rolled_back_to is None
    payload = report.to_dict()
    assert payload["deploy"]["action"] == "created" and payload["smoke"]["ok"] is True
    update_calls = [kw for name, kw in fake.calls if name == "update_endpoint"]
    assert update_calls == []
    with pytest.raises(ValueError, match="not Approved"):
        release(
            aws,
            _stage(),
            _package(approval_status="PendingManualApproval"),
            frame,
            FEATURES,
            deployer=deployer,
        )


def test_release_update_smoke_failure_rolls_back(aws: AwsClients, frame: pd.DataFrame) -> None:
    fake, clock = FakeSageMaker(["InService", "InService"]), FakeClock()
    deployer = _deployer(fake, clock)
    good = json.dumps({"predictions": [{"probability": 0.3, "label": 0}] * 5}).encode()
    with _runtime_stub(aws, good):
        first = release(aws, _stage(), _package(), frame, FEATURES, deployer=deployer)
    v2 = _package(image_uri=IMAGE.replace("sha1", "sha2"), version=4, arn=_package().arn[:-1] + "4")
    with (
        _runtime_stub(aws, b"garbage"),
        pytest.raises(ReleaseError, match="smoke test failed") as exc,
    ):
        release(aws, _stage(), v2, frame, FEATURES, deployer=deployer)
    report = exc.value.report
    assert (
        report.deploy.action == "updated"
        and report.rolled_back_to == first.deploy.endpoint_config_name
    )
    assert fake.endpoints["e"]["EndpointConfigName"] == first.deploy.endpoint_config_name
    strategy_used = next(kw for name, kw in fake.calls if name == "update_endpoint")[
        "DeploymentConfig"
    ]
    assert strategy_used["AutoRollbackConfiguration"]["Alarms"] == [
        {"AlarmName": a} for a in alarm_names("e")
    ]


def test_release_unchanged_skips_wait_and_failed_wait(aws: AwsClients, frame: pd.DataFrame) -> None:
    fake, clock = FakeSageMaker(["Failed"]), FakeClock()
    deployer = _deployer(fake, clock)
    good = json.dumps({"predictions": [{"probability": 0.3, "label": 0}] * 5}).encode()
    with _runtime_stub(aws, good):
        release(
            aws,
            _stage(autoscaling=None, alarms=None),
            _package(),
            frame,
            FEATURES,
            deployer=deployer,
        )
    with _runtime_stub(aws, good):
        again = release(
            aws,
            _stage(autoscaling=None, alarms=None),
            _package(),
            frame,
            FEATURES,
            deployer=deployer,
        )
    assert (
        again.deploy.action == "unchanged"
        and again.alarms == []
        and again.autoscaling_policy is None
    )
    v2 = _package(image_uri=IMAGE.replace("sha1", "sha2"))
    with pytest.raises(ReleaseError, match="Failed"):
        release(aws, _stage(autoscaling=None, alarms=None), v2, frame, FEATURES, deployer=deployer)
