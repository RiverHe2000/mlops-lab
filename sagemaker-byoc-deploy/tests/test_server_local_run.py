from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import uvicorn
from fastapi.testclient import TestClient

from smdeploy import serve
from smdeploy.inference import ModelHandler
from smdeploy.local_run import simulate_invocation, write_sagemaker_layout
from smdeploy.server import (
    CUSTOM_ATTRIBUTES,
    REQUEST_ID,
    JsonFormatter,
    configure_logging,
    create_app,
)
from smdeploy.synthetic import FEATURES, TARGET
from smdeploy.training import TrainingReport


@pytest.fixture
def client(trained: TrainingReport) -> Any:
    app = create_app(ModelHandler(trained.model_dir), max_payload_mb=1, workers=2)
    with TestClient(app) as c:
        yield c


def _csv(frame: pd.DataFrame, n: int = 3, header: bool = False) -> bytes:
    return frame.head(n)[FEATURES].to_csv(header=header, index=False, lineterminator="\n").encode()


def test_ping_and_execution_parameters(client: TestClient) -> None:
    r = client.get("/ping")
    assert (
        r.status_code == 200
        and r.json()["status"] == "ok"
        and r.json()["model_kind"] == "logistic_regression"
    )
    assert client.get("/execution-parameters").json() == {
        "MaxConcurrentTransforms": 2,
        "BatchStrategy": "MultiRecord",
        "MaxPayloadInMB": 1,
    }


def test_invocations_content_types(client: TestClient, frame: pd.DataFrame) -> None:
    r = client.post("/invocations", content=_csv(frame), headers={"Content-Type": "text/csv"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/json")
    assert len(r.json()["predictions"]) == 3 and REQUEST_ID in r.headers
    r = client.post(
        "/invocations",
        content=_csv(frame, header=True),
        headers={"Content-Type": "text/csv; header=present", "Accept": "text/csv"},
    )
    assert r.status_code == 200 and r.text.count("\n") == 3
    payload = json.dumps({"instances": frame.head(2)[FEATURES].to_dict(orient="records")}).encode()
    r = client.post(
        "/invocations",
        content=payload,
        headers={"Content-Type": "application/json", "Accept": "application/jsonlines"},
    )
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/jsonlines")
    assert len(r.text.strip().splitlines()) == 2
    r = client.post("/invocations", content=_csv(frame), headers={"Content-Type": "text/csv"})
    assert r.status_code == 200  # default Accept → json


def test_error_mapping(client: TestClient, frame: pd.DataFrame) -> None:
    r = client.post("/invocations", content=b"<x/>", headers={"Content-Type": "application/xml"})
    assert r.status_code == 415 and "error" in r.json()
    r = client.post(
        "/invocations",
        content=_csv(frame),
        headers={"Content-Type": "text/csv", "Accept": "image/png"},
    )
    assert r.status_code == 406
    r = client.post("/invocations", content=b"1,2\n", headers={"Content-Type": "text/csv"})
    assert r.status_code == 400 and "width" in r.json()["error"]
    missing = frame.head(2)[FEATURES].drop(columns=["age"])
    body = json.dumps({"instances": missing.to_dict(orient="records")}).encode()
    r = client.post("/invocations", content=body, headers={"Content-Type": "application/json"})
    assert r.status_code == 400 and "missing feature" in r.json()["error"]
    big = b"x" * (1024 * 1024 + 1)
    r = client.post("/invocations", content=big, headers={"Content-Type": "text/csv"})
    assert r.status_code == 413
    r = client.post(
        "/invocations",
        content=b"1",
        headers={"Content-Type": "text/csv", "Content-Length": str(10**9)},
    )
    assert r.status_code == 413


def test_custom_attributes_and_request_id_echoed(client: TestClient, frame: pd.DataFrame) -> None:
    headers = {"Content-Type": "text/csv", CUSTOM_ATTRIBUTES: "trace=abc", REQUEST_ID: "req-1"}
    r = client.post("/invocations", content=_csv(frame), headers=headers)
    assert r.headers[CUSTOM_ATTRIBUTES] == "trace=abc" and r.headers[REQUEST_ID] == "req-1"
    r = client.post(
        "/invocations",
        content=b"<x/>",
        headers={"Content-Type": "application/xml", CUSTOM_ATTRIBUTES: "t"},
    )
    assert r.headers[CUSTOM_ATTRIBUTES] == "t"  # echoed on errors too


def test_unloaded_model_reports_503(tmp_path: Path) -> None:
    app = create_app(ModelHandler(tmp_path / "missing"))
    with TestClient(app) as c:
        r = c.get("/ping")
        assert r.status_code == 503 and "FileNotFoundError" in r.json()["error"]
        assert (
            c.post(
                "/invocations", content=b"1,2,3\n", headers={"Content-Type": "text/csv"}
            ).status_code
            == 503
        )


def test_internal_error_is_500(
    trained: TrainingReport, frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    handler = ModelHandler(trained.model_dir)
    app = create_app(handler)
    with TestClient(app) as c:
        monkeypatch.setattr(
            handler.predictor, "predict", lambda _f: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        r = c.post("/invocations", content=_csv(frame), headers={"Content-Type": "text/csv"})
        assert r.status_code == 500 and "internal error" in r.json()["error"]


def test_json_logging() -> None:
    configure_logging("DEBUG")
    record = logging.LogRecord(
        "smdeploy.server", logging.INFO, __file__, 1, "invocation", None, None
    )
    record.request_id = "r1"
    record.rows = 3
    payload = json.loads(JsonFormatter().format(record))
    assert payload["msg"] == "invocation" and payload["request_id"] == "r1" and payload["rows"] == 3
    assert logging.getLogger("smdeploy").level == logging.DEBUG


def test_local_run_layout_and_invocation(
    tmp_path: Path, csv_dir: Path, frame: pd.DataFrame, trained: TrainingReport
) -> None:
    paths = write_sagemaker_layout(
        tmp_path / "ml",
        train_csv=csv_dir / "train.csv",
        validation_csv=csv_dir / "validation.csv",
        hyperparameters={"target": TARGET, "C": 0.5},
        hosts=("algo-1", "algo-2"),
    )
    hp = json.loads(paths.hyperparameters_file.read_text())
    assert hp == {"target": TARGET, "C": "0.5"}  # transported as strings
    assert set(json.loads(paths.input_data_config_file.read_text())) == {"train", "validation"}
    assert paths.read_resource_config()["hosts"] == ["algo-1", "algo-2"]
    assert paths.channel_files("train")[0].name == "train.csv"

    status, content, _ = simulate_invocation(
        trained.model_dir, _csv(frame), content_type="text/csv"
    )
    assert status == 200 and len(json.loads(content)["predictions"]) == 3
    status, _, _ = simulate_invocation(
        trained.model_dir, _csv(frame), threshold=0.5, extra_headers={CUSTOM_ATTRIBUTES: "x"}
    )
    assert status == 200
    status, content, _ = simulate_invocation(tmp_path / "missing", b"1\n")
    assert status == 503


def test_serve_settings_and_factory(
    trained: TrainingReport, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = {
        "SMDEPLOY_BASE_DIR": str(trained.model_dir.parent),
        "SAGEMAKER_BIND_TO_PORT": "9000",
        "SAGEMAKER_MODEL_SERVER_WORKERS": "3",
        "SAGEMAKER_MAX_PAYLOAD_IN_MB": "12",
        "SMDEPLOY_THRESHOLD": "0.4",
    }
    settings = serve.ServeSettings.from_env(env)
    assert (settings.port, settings.workers, settings.max_payload_mb, settings.threshold) == (
        9000,
        3,
        12,
        0.4,
    )
    assert settings.model_dir == trained.model_dir
    defaults = serve.ServeSettings.from_env(
        {"SMDEPLOY_BASE_DIR": str(trained.model_dir.parent), "SM_NUM_CPUS": "16"}
    )
    assert defaults.port == 8080 and defaults.workers == 4 and defaults.threshold is None

    for k, v in env.items():
        monkeypatch.setenv(k, v)
    app = serve.app_factory()
    with TestClient(app) as c:
        assert c.get("/ping").status_code == 200
        assert c.get("/execution-parameters").json()["MaxPayloadInMB"] == 12

    captured: dict[str, Any] = {}
    monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: captured.update(kw, target=a[0]))
    assert serve.main() == 0
    assert (
        captured["target"] == "smdeploy.serve:app_factory"
        and captured["port"] == 9000
        and captured["workers"] == 3
    )
