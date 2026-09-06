"""Turn measurements into a decision: OK / WARN / CRITICAL and NONE / INVESTIGATE / RETRAIN.

Design choices that keep the pager quiet without going blind:
  * a feature only reaches WARN when the PSI threshold *and* the BH-adjusted test agree
    (PSI alone flags every large-n window; the test alone flags every tiny effect);
  * WARN escalates to CRITICAL only after `consecutive_windows_for_critical` windows, so a
    one-off batch anomaly is investigated rather than triggering a retrain;
  * a performance CRITICAL (labels say the model got worse) recommends retraining immediately,
    because that is the only signal that measures the thing we care about;
  * quality faults are reported as incidents in their own right, since drift statistics on
    broken data are meaningless.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import SEVERITY_RANK, Action, AlertPolicy, Severity
from .drift import FeatureDrift, ScoreDrift
from .performance import PerformanceResult
from .quality import QualityIssue


@dataclass
class Finding:
    area: str
    subject: str
    severity: Severity
    message: str
    value: float | None = None
    threshold: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MonitorState:
    """Carried between windows: how many consecutive windows each subject has been drifting."""

    consecutive: dict[str, int] = field(default_factory=dict)
    windows_seen: int = 0
    last_status: str = "OK"

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> MonitorState:
        p = Path(path)
        if not p.is_file():
            return cls()
        raw = json.loads(p.read_text(encoding="utf-8"))
        return cls(
            consecutive={str(k): int(v) for k, v in raw.get("consecutive", {}).items()},
            windows_seen=int(raw.get("windows_seen", 0)),
            last_status=str(raw.get("last_status", "OK")),
        )


@dataclass
class Decision:
    status: Severity
    action: Action
    findings: list[Finding]
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "action": self.action,
            "reasons": list(self.reasons),
            "findings": [f.to_dict() for f in self.findings],
        }


def _max(a: Severity, b: Severity) -> Severity:
    return a if SEVERITY_RANK[a] >= SEVERITY_RANK[b] else b


def _data_severity(d: FeatureDrift, policy: AlertPolicy) -> Severity:
    rules = policy.data_drift
    if math.isnan(d.psi):
        return "OK"
    tested = d.p_adjusted is not None and not math.isnan(d.p_adjusted)
    significant = (
        (not rules.require_test_and_psi)
        or (not tested)
        or (d.p_adjusted is not None and d.p_adjusted < rules.test_alpha)
    )
    if d.psi >= rules.psi_critical and significant:
        return "CRITICAL"
    if d.psi >= rules.psi_warn and significant:
        return "WARN"
    return "OK"


def decide(
    policy: AlertPolicy,
    *,
    n_rows: int,
    feature_drifts: list[FeatureDrift],
    score_drift: ScoreDrift,
    quality_issues: list[QualityIssue],
    performance: PerformanceResult,
    state: MonitorState,
) -> Decision:
    findings: list[Finding] = []
    status: Severity = "OK"
    sustained_critical = False
    enough_rows = n_rows >= policy.window.min_rows

    if not enough_rows:
        findings.append(
            Finding(
                "window",
                "*",
                "WARN",
                f"only {n_rows} rows (< {policy.window.min_rows}); drift verdicts suppressed",
                float(n_rows),
                float(policy.window.min_rows),
            )
        )
        status = "WARN"

    rules = policy.data_drift
    for d in feature_drifts:
        sev = _data_severity(d, policy) if enough_rows else "OK"
        streak = state.consecutive.get(d.name, 0) + 1 if sev != "OK" else 0
        state.consecutive[d.name] = streak
        if sev == "WARN" and streak >= rules.consecutive_windows_for_critical:
            sev = "CRITICAL"
        if sev == "CRITICAL" and streak >= rules.consecutive_windows_for_critical:
            sustained_critical = True
        if sev != "OK":
            adj = "n/a" if d.p_adjusted is None else f"{d.p_adjusted:.3g}"
            findings.append(
                Finding(
                    "data",
                    d.name,
                    sev,
                    f"PSI {d.psi:.3f} ({d.test} p_adj {adj}), {d.shift_summary}; "
                    f"{streak} consecutive window(s)",
                    d.psi,
                    rules.psi_critical if sev == "CRITICAL" else rules.psi_warn,
                )
            )
            status = _max(status, sev)

    prules = policy.prediction_drift
    score_sev: Severity = "OK"
    if enough_rows and not math.isnan(score_drift.psi):
        if score_drift.psi >= prules.score_psi_critical:
            score_sev = "CRITICAL"
        elif (
            score_drift.psi >= prules.score_psi_warn
            or abs(score_drift.decline_rate_change) >= prules.decline_rate_change_warn
        ):
            score_sev = "WARN"
    streak = state.consecutive.get("score", 0) + 1 if score_sev != "OK" else 0
    state.consecutive["score"] = streak
    if score_sev != "OK":
        if score_sev == "CRITICAL" and streak >= rules.consecutive_windows_for_critical:
            sustained_critical = True
        findings.append(
            Finding(
                "prediction",
                "score",
                score_sev,
                f"score PSI {score_drift.psi:.3f}, decline rate {score_drift.decline_rate:.1%} "
                f"({score_drift.decline_rate_change:+.1%} vs baseline)",
                score_drift.psi,
                prules.score_psi_critical if score_sev == "CRITICAL" else prules.score_psi_warn,
            )
        )
        status = _max(status, score_sev)

    for issue in quality_issues:
        q_sev: Severity = "CRITICAL" if issue.kind in {"type_error", "missing_column"} else "WARN"
        findings.append(
            Finding(
                "quality",
                issue.feature,
                q_sev,
                f"{issue.kind}: {issue.detail}",
                issue.value,
                issue.threshold,
            )
        )
        status = _max(status, q_sev)

    perf_rules = policy.performance
    performance_critical = False
    if performance.metrics is None:
        findings.append(
            Finding(
                "performance",
                "labels",
                "OK",
                f"{performance.n_labelled} labels ({performance.label_coverage:.0%} coverage) "
                f"< {perf_rules.min_labels}: not evaluated",
                float(performance.n_labelled),
                float(perf_rules.min_labels),
            )
        )
    else:
        auc = performance.metrics["auc"]
        sev = "OK"
        message = f"AUC {auc:.3f} on {performance.n_labelled} labels"
        if auc < perf_rules.auc_min:
            sev = "CRITICAL"
            message += f" < floor {perf_rules.auc_min:.2f}"
        if performance.auc_drop is not None:
            message += f", drop {performance.auc_drop:+.3f} vs baseline"
            if performance.auc_drop >= perf_rules.auc_drop_critical:
                sev = "CRITICAL"
            elif performance.auc_drop >= perf_rules.auc_drop_warn:
                sev = _max(sev, "WARN")
        if (
            performance.brier_increase is not None
            and performance.brier_increase >= perf_rules.brier_increase_critical
        ):
            sev = "CRITICAL"
            message += f", Brier {performance.brier_increase:+.3f}"
        performance_critical = sev == "CRITICAL"
        findings.append(Finding("performance", "auc", sev, message, auc, perf_rules.auc_min))
        status = _max(status, sev)

    reasons: list[str] = [
        f"{f.area}/{f.subject}: {f.message}" for f in findings if f.severity != "OK"
    ]
    action: Action = "NONE"
    if status in policy.actions.investigate_on:
        action = "INVESTIGATE"
    if (performance_critical and policy.actions.retrain_on_performance_critical) or (
        sustained_critical and policy.actions.retrain_on_sustained_data_critical
    ):
        action = "RETRAIN"
    state.windows_seen += 1
    state.last_status = status
    return Decision(status, action, findings, reasons)
