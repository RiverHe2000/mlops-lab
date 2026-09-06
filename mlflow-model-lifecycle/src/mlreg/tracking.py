"""Thin MLflow layer: configuration, lineage tags and artefact helpers."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

from . import __version__
from .config import TrackingConfig, TrainingConfig

PROXY_MULTIPART_ENV = "MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD"


def prefer_proxied_downloads(uri: str, env: MutableMapping[str, str]) -> bool:
    """Default clients of an HTTP tracking server to download artefacts *through* the server.

    A `mlflow server --serve-artifacts` backed by S3/MinIO advertises presigned multipart
    downloads, and the presigned URLs point at the object store's own endpoint — `minio:9000`
    in `deploy/docker-compose.yml`, which nothing outside the compose network can resolve.
    Every `models:/` load then stalls in retries for minutes before failing. Streaming through
    the server is what the compose file promises anyway (the S3 credentials never leave it).
    An explicit setting in the environment is respected; returns True when this set the default.
    """
    if not uri.startswith(("http://", "https://")) or PROXY_MULTIPART_ENV in env:
        return False
    env[PROXY_MULTIPART_ENV] = "false"
    return True


def configure(tracking: TrackingConfig) -> str:
    """Point MLflow at the store (env override wins) and select the experiment; returns the URI."""
    uri = tracking.resolved_uri()
    if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
        # relative sqlite path: make sure the directory exists so MLflow can create the DB
        db = Path(uri[len("sqlite:///") :])
        if not db.is_absolute():
            db.parent.mkdir(parents=True, exist_ok=True)
    prefer_proxied_downloads(uri, os.environ)
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    if mlflow.get_experiment_by_name(tracking.experiment) is None:
        mlflow.create_experiment(
            tracking.experiment, artifact_location=tracking.resolved_artifact_location()
        )
    mlflow.set_experiment(tracking.experiment)
    return uri


def git_sha(cwd: Path | None = None) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    sha = out.stdout.strip()
    return sha if out.returncode == 0 and sha else None


def lineage_tags(
    cfg: TrainingConfig, data_fingerprint: str, schema_fingerprint: str
) -> dict[str, str]:
    """Everything needed to answer 'which code, config and data produced this model?'."""
    tags = {
        "mlreg.version": __version__,
        "mlreg.config_name": cfg.name,
        "mlreg.config_hash": cfg.config_hash(),
        "mlreg.model_kind": cfg.model.kind,
        "data.fingerprint": data_fingerprint,
        "data.schema_fingerprint": schema_fingerprint,
        "data.path": cfg.data_path.name,
        "split.seed": str(cfg.split.seed),
        "split.test_size": str(cfg.split.test_size),
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    sha = git_sha()
    if sha:
        tags["git.sha"] = sha
    tags.update(cfg.tags)
    return tags


@contextmanager
def start_run(run_name: str, tags: dict[str, str], *, nested: bool = False) -> Iterator[Any]:
    with mlflow.start_run(run_name=run_name, tags=tags, nested=nested) as run:
        yield run


def log_params_flat(params: dict[str, Any], prefix: str = "") -> None:
    flat = {f"{prefix}{k}": _param_str(v) for k, v in params.items()}
    if flat:
        mlflow.log_params(flat)


def _param_str(v: Any) -> str:
    if isinstance(v, dict | list):
        return json.dumps(v, sort_keys=True)
    return str(v)


def log_metrics_prefixed(metrics: dict[str, float], prefix: str, step: int | None = None) -> None:
    clean = {f"{prefix}{k}": float(v) for k, v in metrics.items() if v == v}  # drops NaN
    if clean:
        mlflow.log_metrics(clean, step=step)


def series_to_metrics(s: pd.Series) -> dict[str, float]:
    return {str(k): float(v) for k, v in s.items()}


def log_json(obj: Any, name: str) -> None:
    mlflow.log_dict(json.loads(json.dumps(obj, default=str)), name)


def log_frame(df: pd.DataFrame, name: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / Path(name).name
        df.to_csv(path, index=False, lineterminator="\n")
        parent = str(Path(name).parent)
        mlflow.log_artifact(str(path), artifact_path=None if parent in {"", "."} else parent)


def log_text(text: str, name: str) -> None:
    mlflow.log_text(text, name)


def log_file(path: Path, artifact_dir: str | None = None) -> None:
    mlflow.log_artifact(str(path), artifact_path=artifact_dir)


def active_run_id() -> str:
    run = mlflow.active_run()
    if run is None:
        raise RuntimeError("no active MLflow run")
    return str(run.info.run_id)
