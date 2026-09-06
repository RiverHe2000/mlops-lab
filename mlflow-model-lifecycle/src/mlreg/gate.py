"""Champion/challenger promotion gate.

The challenger and the incumbent are scored on the *same* holdout rows (paired), so the
relative checks are confidence intervals on the paired delta, not two separate point
estimates. The decision is the CLI exit code: PROMOTE → 0, HOLD → 2.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field

from . import metrics as M
from .stats import BootstrapResult, paired_bootstrap_delta, psi

Decision = Literal["PROMOTE", "HOLD"]


class AbsoluteThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_auc: float = 0.70
    max_brier: float = 0.25
    max_ece: float = 0.10
    min_n_holdout: int = 100


class RelativeThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    auc_noninferiority_margin: float = Field(default=0.01, ge=0.0)
    brier_max_degradation: float = Field(default=0.01, ge=0.0)
    max_score_psi: float = Field(default=0.25, ge=0.0)
    n_boot: int = Field(default=1000, ge=100)
    level: float = Field(default=0.95, gt=0.5, lt=1.0)
    seed: int = 0


class SliceThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_auc: float = 0.55
    min_n: int = 30


class PromotionPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    absolute: AbsoluteThresholds = Field(default_factory=AbsoluteThresholds)
    relative: RelativeThresholds = Field(default_factory=RelativeThresholds)
    slices: SliceThresholds | None = None
    require_same_data_snapshot: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> PromotionPolicy:
        with Path(path).open(encoding="utf-8") as fh:
            return cls.model_validate(yaml.safe_load(fh))


@dataclass
class Check:
    name: str
    passed: bool
    value: float | None
    threshold: float | None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GateResult:
    decision: Decision
    checks: list[Check]
    challenger: dict[str, float]
    champion: dict[str, float] | None
    deltas: dict[str, BootstrapResult] = field(default_factory=dict)
    n_holdout: int = 0

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if not c.passed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "n_holdout": self.n_holdout,
            "checks": [c.to_dict() for c in self.checks],
            "challenger": self.challenger,
            "champion": self.champion,
            "deltas": {k: v.to_dict() for k, v in self.deltas.items()},
        }

    def to_markdown(self) -> str:
        lines = [f"**Decision: {self.decision}** (holdout n = {self.n_holdout})", ""]
        lines.append("| Check | Value | Threshold | Result | Detail |")
        lines.append("|---|---:|---:|:---:|---|")
        for c in self.checks:
            result = "PASS" if c.passed else "FAIL"
            lines.append(
                f"| {c.name} | {_fmt(c.value)} | {_fmt(c.threshold)} | {result} | {c.detail} |"
            )
        if self.deltas:
            lines += ["", "Paired deltas (challenger - champion):", ""]
            for k, v in self.deltas.items():
                lines.append(f"- {k}: {v}")
        return "\n".join(lines)


def _fmt(v: float | None) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    return f"{v:.4f}" if abs(v) < 100 else f"{v:.0f}"


def evaluate_gate(
    policy: PromotionPolicy,
    y: np.ndarray,
    p_challenger: np.ndarray,
    p_champion: np.ndarray | None,
    threshold: float,
    *,
    slices: dict[str, pd.Series] | None = None,
    same_snapshot: bool | None = None,
) -> GateResult:
    y = np.asarray(y)
    p_challenger = np.asarray(p_challenger)
    checks: list[Check] = []
    deltas: dict[str, BootstrapResult] = {}

    chal = M.summarize(y, p_challenger, threshold)
    a = policy.absolute
    checks.append(
        Check("holdout_size", len(y) >= a.min_n_holdout, float(len(y)), float(a.min_n_holdout))
    )
    checks.append(Check("auc_min", chal["auc"] >= a.min_auc, chal["auc"], a.min_auc))
    checks.append(Check("brier_max", chal["brier"] <= a.max_brier, chal["brier"], a.max_brier))
    checks.append(Check("ece_max", chal["ece"] <= a.max_ece, chal["ece"], a.max_ece))

    champ: dict[str, float] | None = None
    if p_champion is not None:
        p_champion = np.asarray(p_champion)
        r = policy.relative
        champ = M.summarize(y, p_champion, threshold)
        d_auc = paired_bootstrap_delta(
            y, p_champion, p_challenger, M.roc_auc, n_boot=r.n_boot, seed=r.seed, level=r.level
        )
        d_brier = paired_bootstrap_delta(
            y, p_champion, p_challenger, M.brier, n_boot=r.n_boot, seed=r.seed + 1, level=r.level
        )
        deltas = {"auc": d_auc, "brier": d_brier}
        checks.append(
            Check(
                "auc_noninferior",
                d_auc.lower >= -r.auc_noninferiority_margin,
                d_auc.lower,
                -r.auc_noninferiority_margin,
                f"delta {d_auc}",
            )
        )
        checks.append(
            Check(
                "brier_not_worse",
                d_brier.upper <= r.brier_max_degradation,
                d_brier.upper,
                r.brier_max_degradation,
                f"delta {d_brier}",
            )
        )
        stability = psi(p_champion, p_challenger)
        checks.append(
            Check(
                "score_psi",
                stability <= r.max_score_psi,
                stability,
                r.max_score_psi,
                "champion → challenger",
            )
        )
        if policy.require_same_data_snapshot:
            ok = bool(same_snapshot) if same_snapshot is not None else False
            checks.append(
                Check(
                    "same_data_snapshot",
                    ok,
                    None,
                    None,
                    "paired comparison requires both models trained on the gate's data snapshot",
                )
            )

    if policy.slices is not None and slices:
        s = policy.slices
        for col, groups in slices.items():
            table = M.slice_metrics(y, p_challenger, groups, threshold)
            eligible = table[(table["n"] >= s.min_n) & table["auc"].notna()]
            if len(eligible):
                aucs = eligible["auc"].to_numpy(dtype=float)
                k = int(np.argmin(aucs))
                worst = float(aucs[k])
                worst_group = str(eligible["group"].to_numpy()[k])
                passed = worst >= s.min_auc
            else:
                worst, worst_group, passed = float("nan"), "-", True
            checks.append(
                Check(
                    f"slice_auc_min[{col}]", passed, worst, s.min_auc, f"worst group: {worst_group}"
                )
            )

    decision: Decision = "PROMOTE" if all(c.passed for c in checks) else "HOLD"
    return GateResult(decision, checks, chal, champ, deltas, n_holdout=len(y))
