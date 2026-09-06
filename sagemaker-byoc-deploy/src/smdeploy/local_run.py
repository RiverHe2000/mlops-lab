"""Simulate `docker run <image> train` and `docker run <image> serve` without Docker.

Lays out the exact /opt/ml tree SageMaker would mount under a directory of your choice,
runs the same `train` entrypoint in-process, and drives the same FastAPI app through an
in-process HTTP client. CI uses this to prove the container contract before an image exists;
the Dockerfile only adds packaging around the very same code.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from .hyperparams import TrainingHyperparameters
from .inference import ModelHandler
from .paths import SageMakerPaths
from .server import create_app
from .training import TrainingReport, run_training


def write_sagemaker_layout(
    base: Path,
    *,
    train_csv: Path,
    hyperparameters: dict[str, Any] | TrainingHyperparameters | None = None,
    validation_csv: Path | None = None,
    hosts: tuple[str, ...] = ("algo-1",),
) -> SageMakerPaths:
    paths = SageMakerPaths(base=Path(base), env={})
    paths.input_config.mkdir(parents=True, exist_ok=True)
    if isinstance(hyperparameters, TrainingHyperparameters):
        hp_dict = hyperparameters.to_sagemaker_dict()
    else:
        hp_dict = {
            k: (v if isinstance(v, str) else json.dumps(v))
            for k, v in (hyperparameters or {}).items()
        }
    paths.hyperparameters_file.write_text(json.dumps(hp_dict, indent=2), encoding="utf-8")

    channels: dict[str, dict[str, str]] = {}
    for name, src in (("train", train_csv), ("validation", validation_csv)):
        if src is None:
            continue
        dest = paths.channel(name)
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dest / Path(src).name)
        channels[name] = {
            "ContentType": "text/csv",
            "TrainingInputMode": "File",
            "S3DistributionType": "FullyReplicated",
        }
    paths.input_data_config_file.write_text(json.dumps(channels, indent=2), encoding="utf-8")
    paths.resource_config_file.write_text(
        json.dumps(
            {"current_host": hosts[0], "hosts": list(hosts), "network_interface_name": "eth0"}
        ),
        encoding="utf-8",
    )
    paths.ensure_output_dirs()
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    return paths


def simulate_training(
    base: Path,
    *,
    train_csv: Path,
    hyperparameters: dict[str, Any] | TrainingHyperparameters | None = None,
    validation_csv: Path | None = None,
) -> TrainingReport:
    paths = write_sagemaker_layout(
        base, train_csv=train_csv, hyperparameters=hyperparameters, validation_csv=validation_csv
    )
    return run_training(paths, job_name="local-simulation")


def simulate_invocation(
    model_dir: Path,
    body: bytes,
    *,
    content_type: str = "text/csv",
    accept: str = "application/json",
    threshold: float | None = None,
    extra_headers: dict[str, str] | None = None,
) -> tuple[int, bytes, dict[str, str]]:
    app = create_app(ModelHandler(model_dir, threshold=threshold))
    with TestClient(app) as client:
        ping = client.get("/ping")
        if ping.status_code != 200:
            return ping.status_code, ping.content, dict(ping.headers)
        headers = {"Content-Type": content_type, "Accept": accept, **(extra_headers or {})}
        response = client.post("/invocations", content=body, headers=headers)
        return response.status_code, response.content, dict(response.headers)
