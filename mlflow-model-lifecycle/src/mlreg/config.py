"""Typed configuration: data contract, training config and tracking settings."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ColumnKind = Literal["numeric", "categorical"]
ColumnRole = Literal["feature", "protected", "target", "id"]
ModelKind = Literal["logistic_regression", "hist_gradient_boosting", "random_forest"]

ENV_TRACKING_URI = "MLREG_TRACKING_URI"
ENV_ARTIFACT_LOCATION = "MLREG_ARTIFACT_LOCATION"


class SchemaColumn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    kind: ColumnKind
    role: ColumnRole = "feature"
    categories: list[str] | None = None
    min: float | None = None
    max: float | None = None
    description: str = ""

    @model_validator(mode="after")
    def _consistent(self) -> SchemaColumn:
        if self.kind == "numeric" and self.categories is not None:
            raise ValueError(f"{self.name}: numeric columns cannot list categories")
        if self.kind == "categorical" and (self.min is not None or self.max is not None):
            raise ValueError(f"{self.name}: categorical columns cannot have min/max")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(f"{self.name}: min > max")
        if self.categories is not None and len(set(self.categories)) != len(self.categories):
            raise ValueError(f"{self.name}: duplicate categories")
        return self


class DataSchema(BaseModel):
    """The data contract. Also the single source of truth for feature construction."""

    model_config = ConfigDict(extra="forbid")

    name: str
    version: str
    target: str
    id_column: str
    positive_label: int = 1
    columns: list[SchemaColumn]

    @model_validator(mode="after")
    def _roles(self) -> DataSchema:
        names = [c.name for c in self.columns]
        if len(set(names)) != len(names):
            raise ValueError("duplicate column names in schema")
        by_name = {c.name: c for c in self.columns}
        if self.target not in by_name or by_name[self.target].role != "target":
            raise ValueError(f"target {self.target!r} must be a column with role 'target'")
        if self.id_column not in by_name or by_name[self.id_column].role != "id":
            raise ValueError(f"id_column {self.id_column!r} must be a column with role 'id'")
        if sum(c.role == "target" for c in self.columns) != 1:
            raise ValueError("exactly one target column is required")
        if not self.feature_columns:
            raise ValueError("schema has no feature columns")
        return self

    @property
    def feature_columns(self) -> list[str]:
        return [c.name for c in self.columns if c.role == "feature"]

    @property
    def numeric_features(self) -> list[str]:
        return [c.name for c in self.columns if c.role == "feature" and c.kind == "numeric"]

    @property
    def categorical_features(self) -> list[str]:
        return [c.name for c in self.columns if c.role == "feature" and c.kind == "categorical"]

    @property
    def protected_columns(self) -> list[str]:
        return [c.name for c in self.columns if c.role == "protected"]

    def column(self, name: str) -> SchemaColumn:
        for c in self.columns:
            if c.name == name:
                return c
        raise KeyError(name)

    @classmethod
    def from_yaml(cls, path: str | Path) -> DataSchema:
        with Path(path).open(encoding="utf-8") as fh:
            return cls.model_validate(yaml.safe_load(fh))

    def fingerprint(self) -> str:
        return _sha256_json(self.model_dump(mode="json"))


class SplitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    test_size: float = Field(default=0.25, gt=0.0, lt=1.0)
    seed: int = 42
    stratify: bool = True


class CVConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    folds: int = Field(default=5, ge=2)
    seed: int = 42


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: ModelKind
    params: dict[str, Any] = Field(default_factory=dict)


class TrackingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tracking_uri: str = "sqlite:///mlruns/mlflow.db"
    artifact_location: str | None = None
    experiment: str = "credit-pd"
    registered_model: str = "credit-pd"

    def resolved_uri(self) -> str:
        """`MLREG_TRACKING_URI` overrides the file so CI and laptops can share configs."""
        return os.environ.get(ENV_TRACKING_URI) or self.tracking_uri

    def resolved_artifact_location(self) -> str | None:
        return os.environ.get(ENV_ARTIFACT_LOCATION) or self.artifact_location


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    data_path: Path
    schema_path: Path
    split: SplitConfig = Field(default_factory=SplitConfig)
    cv: CVConfig = Field(default_factory=CVConfig)
    model: ModelConfig
    decision_threshold: float | Literal["cost_optimal"] = "cost_optimal"
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    tags: dict[str, str] = Field(default_factory=dict)

    @field_validator("decision_threshold")
    @classmethod
    def _threshold_range(cls, v: float | str) -> float | str:
        if isinstance(v, float | int) and not 0.0 < float(v) < 1.0:
            raise ValueError("decision_threshold must be in (0, 1) or 'cost_optimal'")
        return v

    @classmethod
    def from_yaml(cls, path: str | Path, *, base_dir: Path | None = None) -> TrainingConfig:
        p = Path(path)
        with p.open(encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        cfg = cls.model_validate(raw)
        root = base_dir if base_dir is not None else Path.cwd()
        return cfg.model_copy(
            update={
                "data_path": _resolve(cfg.data_path, root),
                "schema_path": _resolve(cfg.schema_path, root),
            }
        )

    def config_hash(self) -> str:
        """Stable hash of everything that influences the model (paths and tracking excluded)."""
        payload = self.model_dump(mode="json", exclude={"data_path", "schema_path", "tracking"})
        return _sha256_json(payload)


def _resolve(p: Path, root: Path) -> Path:
    return p if p.is_absolute() else (root / p)


def _sha256_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()
