"""The alert and retrain policy, as YAML. Thresholds are the model owner's decision; the code
only applies them and reports which one fired."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

Severity = Literal["OK", "WARN", "CRITICAL"]
Action = Literal["NONE", "INVESTIGATE", "RETRAIN"]
SEVERITY_RANK: dict[str, int] = {"OK": 0, "WARN": 1, "CRITICAL": 2}


class WindowRules(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_rows: int = Field(default=200, ge=1)


class DataDriftRules(BaseModel):
    model_config = ConfigDict(extra="forbid")

    psi_warn: float = Field(default=0.10, ge=0)
    psi_critical: float = Field(default=0.25, ge=0)
    test_alpha: float = Field(default=0.01, gt=0, lt=1)
    require_test_and_psi: bool = True
    consecutive_windows_for_critical: int = Field(default=2, ge=1)


class PredictionDriftRules(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score_psi_warn: float = Field(default=0.10, ge=0)
    score_psi_critical: float = Field(default=0.25, ge=0)
    decline_rate_change_warn: float = Field(default=0.10, ge=0)


class QualityRules(BaseModel):
    model_config = ConfigDict(extra="forbid")

    missing_rate_increase_warn: float = Field(default=0.05, ge=0)
    unseen_category_share_warn: float = Field(default=0.02, ge=0)
    out_of_range_share_warn: float = Field(default=0.01, ge=0)
    type_error_share_critical: float = Field(default=0.001, ge=0)
    duplicate_share_warn: float = Field(default=0.01, ge=0)


class PerformanceRules(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_labels: int = Field(default=200, ge=2)
    auc_min: float = Field(default=0.70, ge=0.5, le=1.0)
    auc_drop_warn: float = Field(default=0.03, ge=0)
    auc_drop_critical: float = Field(default=0.06, ge=0)
    brier_increase_critical: float = Field(default=0.03, ge=0)


def _default_investigate() -> list[Severity]:
    return ["WARN", "CRITICAL"]


class ActionRules(BaseModel):
    model_config = ConfigDict(extra="forbid")

    investigate_on: list[Severity] = Field(default_factory=_default_investigate)
    retrain_on_performance_critical: bool = True
    retrain_on_sustained_data_critical: bool = True


class AlertPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    window: WindowRules = Field(default_factory=WindowRules)
    data_drift: DataDriftRules = Field(default_factory=DataDriftRules)
    prediction_drift: PredictionDriftRules = Field(default_factory=PredictionDriftRules)
    quality: QualityRules = Field(default_factory=QualityRules)
    performance: PerformanceRules = Field(default_factory=PerformanceRules)
    actions: ActionRules = Field(default_factory=ActionRules)

    @classmethod
    def from_yaml(cls, path: Path) -> AlertPolicy:
        with Path(path).open(encoding="utf-8") as fh:
            return cls.model_validate(yaml.safe_load(fh) or {})
