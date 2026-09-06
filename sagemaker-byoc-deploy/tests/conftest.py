from __future__ import annotations

import warnings
from collections import deque
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from botocore.exceptions import ClientError
from moto import mock_aws

from smdeploy.aws.clients import AwsClients
from smdeploy.hyperparams import TrainingHyperparameters
from smdeploy.local_run import simulate_training
from smdeploy.synthetic import FEATURES, TARGET, make_credit_frame
from smdeploy.training import TrainingReport

warnings.filterwarnings("ignore")
REGION = "ap-southeast-2"
ROLE = "arn:aws:iam::123456789012:role/credit-pd-sagemaker-execution"
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def frame() -> pd.DataFrame:
    return make_credit_frame(500, seed=7)


@pytest.fixture(scope="session")
def csv_dir(tmp_path_factory: pytest.TempPathFactory, frame: pd.DataFrame) -> Path:
    d = tmp_path_factory.mktemp("csv")
    frame.iloc[:400].to_csv(d / "train.csv", index=False)
    frame.iloc[400:].to_csv(d / "validation.csv", index=False)
    frame.iloc[400:405][FEATURES].to_csv(d / "smoke.csv", index=False)
    return d


@pytest.fixture(scope="session")
def hp() -> TrainingHyperparameters:
    return TrainingHyperparameters(
        target=TARGET, id_columns=["application_id"], C=0.5, threshold=0.3
    )


@pytest.fixture(scope="session")
def trained(
    tmp_path_factory: pytest.TempPathFactory, csv_dir: Path, hp: TrainingHyperparameters
) -> TrainingReport:
    base = tmp_path_factory.mktemp("opt-ml")
    return simulate_training(
        base,
        train_csv=csv_dir / "train.csv",
        validation_csv=csv_dir / "validation.csv",
        hyperparameters=hp,
    )


@pytest.fixture
def aws(monkeypatch: pytest.MonkeyPatch) -> Iterator[AwsClients]:
    for key in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SECURITY_TOKEN",
        "AWS_SESSION_TOKEN",
    ):
        monkeypatch.setenv(key, "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", REGION)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    with mock_aws():
        yield AwsClients.from_session(region=REGION)


def not_found(what: str) -> ClientError:
    return ClientError(
        {"Error": {"Code": "ValidationException", "Message": f"Could not find {what}"}}, "Describe"
    )


class FakeSageMaker:
    """In-memory SageMaker with endpoint status transitions (moto has no update_endpoint).

    `update_outcomes` scripts what happens after each update: "InService" (green takes over),
    "RolledBack" (SageMaker reverted to the old config) or "Failed".
    """

    def __init__(self, update_outcomes: list[str] | None = None) -> None:
        self.models: dict[str, dict[str, Any]] = {}
        self.configs: dict[str, dict[str, Any]] = {}
        self.endpoints: dict[str, dict[str, Any]] = {}
        self._pending: dict[str, deque[dict[str, Any]]] = {}
        self.outcomes = deque(update_outcomes or [])
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def _record(self, name: str, kwargs: dict[str, Any]) -> None:
        self.calls.append((name, kwargs))

    # models / configs
    def describe_model(self, **kw: Any) -> dict[str, Any]:
        self._record("describe_model", kw)
        if kw["ModelName"] not in self.models:
            raise not_found("model")
        return self.models[kw["ModelName"]]

    def create_model(self, **kw: Any) -> dict[str, Any]:
        self._record("create_model", kw)
        self.models[kw["ModelName"]] = dict(kw)
        return {"ModelArn": f"arn:model/{kw['ModelName']}"}

    def delete_model(self, **kw: Any) -> None:
        self._record("delete_model", kw)
        self.models.pop(kw["ModelName"])

    def describe_endpoint_config(self, **kw: Any) -> dict[str, Any]:
        self._record("describe_endpoint_config", kw)
        if kw["EndpointConfigName"] not in self.configs:
            raise not_found("endpoint configuration")
        return self.configs[kw["EndpointConfigName"]]

    def create_endpoint_config(self, **kw: Any) -> dict[str, Any]:
        self._record("create_endpoint_config", kw)
        self.configs[kw["EndpointConfigName"]] = dict(kw)
        return {"EndpointConfigArn": f"arn:cfg/{kw['EndpointConfigName']}"}

    def delete_endpoint_config(self, **kw: Any) -> None:
        self._record("delete_endpoint_config", kw)
        self.configs.pop(kw["EndpointConfigName"])

    # endpoints
    def describe_endpoint(self, **kw: Any) -> dict[str, Any]:
        self._record("describe_endpoint", kw)
        name = kw["EndpointName"]
        if name not in self.endpoints:
            raise not_found("endpoint")
        queue = self._pending.get(name)
        if queue:
            self.endpoints[name].update(queue.popleft())
        return dict(self.endpoints[name])

    def create_endpoint(self, **kw: Any) -> dict[str, Any]:
        self._record("create_endpoint", kw)
        name = kw["EndpointName"]
        self.endpoints[name] = {
            "EndpointName": name,
            "EndpointConfigName": kw["EndpointConfigName"],
            "EndpointStatus": "Creating",
        }
        self._pending[name] = deque(
            [{"EndpointStatus": "Creating"}, {"EndpointStatus": "InService"}]
        )
        return {"EndpointArn": f"arn:endpoint/{name}"}

    def update_endpoint(self, **kw: Any) -> dict[str, Any]:
        self._record("update_endpoint", kw)
        name = kw["EndpointName"]
        if name not in self.endpoints:
            raise not_found("endpoint")
        old = self.endpoints[name]["EndpointConfigName"]
        new = kw["EndpointConfigName"]
        outcome = self.outcomes.popleft() if self.outcomes else "InService"
        if outcome == "InService":
            steps = [
                {"EndpointStatus": "Updating"},
                {"EndpointStatus": "InService", "EndpointConfigName": new},
            ]
        elif outcome == "RolledBack":
            steps = [
                {"EndpointStatus": "Updating"},
                {"EndpointStatus": "RollingBack"},
                {
                    "EndpointStatus": "InService",
                    "EndpointConfigName": old,
                    "FailureReason": "alarm fired",
                },
            ]
        else:
            steps = [
                {"EndpointStatus": "Updating"},
                {"EndpointStatus": "Failed", "FailureReason": "container unhealthy"},
            ]
        self.endpoints[name]["EndpointStatus"] = "Updating"
        self._pending[name] = deque(steps)
        return {"EndpointArn": f"arn:endpoint/{name}"}

    def delete_endpoint(self, **kw: Any) -> None:
        self._record("delete_endpoint", kw)
        self.endpoints.pop(kw["EndpointName"])


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds
