"""Typed hyperparameters with SageMaker's stringly-typed transport handled at the edge.

`CreateTrainingJob.HyperParameters` is `map<string, string>`, so the file the container reads
contains `{"C": "0.5", "max_depth": "null", "id_columns": "[\\"a\\"]"}`. Values are JSON-decoded
where possible and then validated by pydantic; unknown keys (SageMaker adds `sagemaker_*`
entries for framework containers) are ignored instead of failing the job.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ModelKind = Literal["logistic_regression", "random_forest", "hist_gradient_boosting"]


class TrainingHyperparameters(BaseModel):
    model_config = ConfigDict(extra="ignore")

    target: str = "default"
    positive_label: int | str = 1
    id_columns: list[str] = Field(default_factory=list)
    model: ModelKind = "logistic_regression"
    C: float = Field(default=1.0, gt=0)
    n_estimators: int = Field(default=200, ge=1)
    max_depth: int | None = None
    max_iter: int = Field(default=500, ge=1)
    threshold: float = Field(default=0.5, gt=0, lt=1)
    seed: int = 42
    validation_split: float = Field(default=0.2, ge=0, lt=1)

    @field_validator("id_columns", mode="before")
    @classmethod
    def _split_ids(cls, v: Any) -> Any:
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return []
            if s.startswith("["):
                return json.loads(s)
            return [part.strip() for part in s.split(",") if part.strip()]
        return v

    @classmethod
    def from_sagemaker_dict(cls, raw: dict[str, Any]) -> TrainingHyperparameters:
        return cls.model_validate({k: _decode(v) for k, v in raw.items()})

    @classmethod
    def from_sagemaker_file(cls, path: Path) -> TrainingHyperparameters:
        """Missing file → defaults (running `train` with no hyperparameters is legal)."""
        if not path.is_file():
            return cls()
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"{path}: hyperparameters must be a JSON object")
        return cls.from_sagemaker_dict(loaded)

    def to_sagemaker_dict(self) -> dict[str, str]:
        """Inverse transport: every value becomes a string the container can decode again."""
        out: dict[str, str] = {}
        for key, value in self.model_dump(mode="json").items():
            out[key] = value if isinstance(value, str) else json.dumps(value)
        return out


def _decode(value: Any) -> Any:
    """Turn a transported string back into a JSON value when it parses; keep raw strings."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text.lower() in {"none", "null"}:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value
