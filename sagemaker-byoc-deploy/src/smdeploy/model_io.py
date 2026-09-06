"""Model artefact layout inside /opt/ml/model with integrity metadata.

    model.joblib     the fitted sklearn Pipeline
    metadata.json    columns, hyperparameters, metrics, versions and the SHA-256 of model.joblib

Hosting verifies the hash before serving: a truncated upload, a mismatched model.tar.gz or a
hand-edited artefact fails the `/ping` health check instead of silently scoring garbage.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import sklearn
from pydantic import BaseModel, ConfigDict, Field

from . import __version__

MODEL_FILE = "model.joblib"
METADATA_FILE = "metadata.json"


class ModelIntegrityError(RuntimeError):
    """The artefact on disk does not match its metadata."""


class ModelMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    smdeploy_version: str = __version__
    created_utc: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds")
    )
    target: str
    positive_label: int | str
    threshold: float
    feature_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    model_kind: str
    hyperparameters: dict[str, Any]
    metrics: dict[str, float]
    metrics_on: str
    n_train: int
    n_validation: int
    model_sha256: str = ""
    python_version: str = Field(default_factory=platform.python_version)
    sklearn_version: str = sklearn.__version__
    training_job_name: str | None = None


@dataclass
class LoadedModel:
    pipeline: Any
    metadata: ModelMetadata
    model_dir: Path


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_model(pipeline: Any, metadata: ModelMetadata, model_dir: Path) -> ModelMetadata:
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / MODEL_FILE
    joblib.dump(pipeline, model_path)
    stamped = metadata.model_copy(update={"model_sha256": file_sha256(model_path)})
    (model_dir / METADATA_FILE).write_text(
        json.dumps(stamped.model_dump(mode="json"), indent=2, sort_keys=True), encoding="utf-8"
    )
    return stamped


def load_metadata(model_dir: Path) -> ModelMetadata:
    path = model_dir / METADATA_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing: not a smdeploy model directory")
    return ModelMetadata.model_validate_json(path.read_text(encoding="utf-8"))


def load_model(model_dir: Path, *, verify: bool = True) -> LoadedModel:
    metadata = load_metadata(model_dir)
    model_path = model_dir / MODEL_FILE
    if not model_path.is_file():
        raise FileNotFoundError(f"{model_path} missing")
    if verify:
        actual = file_sha256(model_path)
        if actual != metadata.model_sha256:
            raise ModelIntegrityError(
                f"{MODEL_FILE} sha256 {actual[:12]}… does not match metadata "
                f"{metadata.model_sha256[:12]}…"
            )
    return LoadedModel(pipeline=joblib.load(model_path), metadata=metadata, model_dir=model_dir)
