from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from mlwatch.baseline import Baseline
from mlwatch.config import AlertPolicy
from mlwatch.drift import FeatureDrift, ScoreDrift
from mlwatch.monitor import WindowReport, load_capture_frame, load_labels_frame, run_window
from mlwatch.performance import PerformanceResult
from mlwatch.policy import Decision, MonitorState, decide
from mlwatch.quality import QualityIssue
from mlwatch.schema import CaptureRecord, LabelRecord, write_jsonl

from .conftest import make_window


def _drift(name: str = "x", psi: float = 0.0, p_adj: float | None = 0.5) -> FeatureDrift:
    return FeatureDrift(
        name, "numeric", 1000, 0.0, psi, psi, psi, "ks", 0.1, 0.5, 0.01, p_adjusted=p_adj
    )


def _score(psi: float = 0.0, decline_change: float = 0.0) -> ScoreDrift:
    return ScoreDrift(1000, psi, 0.0, 0.3, 0.0, 0.3 + decline_change, decline_change)


def _perf(
    metrics: dict[str, float] | None = None,
    auc_drop: float | None = None,
    brier_increase: float | None = None,
) -> PerformanceResult:
    return PerformanceResult(
        1000,
        500 if metrics else 10,
        0.5,
        200,
        metrics,
        {"auc": 0.8, "brier": 0.15},
        auc_drop,
        brier_increase,
    )


def _decide(
    policy: AlertPolicy, state: MonitorState | None = None, **over: object
) -> tuple[Decision, MonitorState]:
    state = state or MonitorState()
    kwargs: dict[str, object] = {
        "n_rows": 1000,
        "feature_drifts": [_drift()],
        "score_drift": _score(),
        "quality_issues": [],
        "performance": _perf(),
    }
    kwargs.update(over)
    return decide(policy, state=state, **kwargs), state  # type: ignore[arg-type]


def test_quiet_window_is_ok(policy: AlertPolicy) -> None:
    decision, state = _decide(policy)
    assert decision.status == "OK" and decision.action == "NONE" and decision.reasons == []
    assert state.windows_seen == 1 and state.consecutive == {"x": 0, "score": 0}
    assert any(f.area == "performance" and "not evaluated" in f.message for f in decision.findings)


def test_insufficient_rows_suppresses_verdicts(policy: AlertPolicy) -> None:
    decision, _ = _decide(policy, n_rows=50, feature_drifts=[_drift(psi=0.9, p_adj=1e-9)])
    assert decision.status == "WARN" and decision.action == "INVESTIGATE"
    assert [f.area for f in decision.findings if f.severity != "OK"] == ["window"]


def test_psi_needs_test_agreement_and_escalates(policy: AlertPolicy) -> None:
    only_psi, _ = _decide(policy, feature_drifts=[_drift(psi=0.3, p_adj=0.5)])
    assert only_psi.status == "OK"
    only_test, _ = _decide(policy, feature_drifts=[_drift(psi=0.02, p_adj=1e-9)])
    assert only_test.status == "OK"
    state = MonitorState()
    first, state = _decide(policy, state, feature_drifts=[_drift(psi=0.15, p_adj=1e-9)])
    assert first.status == "WARN" and first.action == "INVESTIGATE" and state.consecutive["x"] == 1
    second, state = _decide(policy, state, feature_drifts=[_drift(psi=0.15, p_adj=1e-9)])
    assert (
        second.status == "CRITICAL" and second.action == "RETRAIN" and state.consecutive["x"] == 2
    )
    calm, state = _decide(policy, state)
    assert calm.status == "OK" and state.consecutive["x"] == 0
    untested, _ = _decide(policy, feature_drifts=[_drift(psi=0.3, p_adj=None)])
    assert untested.status == "CRITICAL"
    relaxed = policy.model_copy(
        update={"data_drift": policy.data_drift.model_copy(update={"require_test_and_psi": False})}
    )
    assert _decide(relaxed, feature_drifts=[_drift(psi=0.3, p_adj=0.5)])[0].status == "CRITICAL"
    no_retrain = policy.model_copy(
        update={
            "actions": policy.actions.model_copy(
                update={"retrain_on_sustained_data_critical": False}
            )
        }
    )
    st = MonitorState(consecutive={"x": 1})
    sustained, _ = _decide(no_retrain, st, feature_drifts=[_drift(psi=0.3, p_adj=1e-9)])
    assert sustained.status == "CRITICAL" and sustained.action == "INVESTIGATE"


def test_prediction_quality_and_performance_rules(policy: AlertPolicy) -> None:
    warn, _ = _decide(policy, score_drift=_score(psi=0.15))
    assert warn.status == "WARN" and warn.findings[0].area == "prediction"
    rate, _ = _decide(policy, score_drift=_score(psi=0.02, decline_change=0.2))
    assert rate.status == "WARN"
    crit, _ = _decide(policy, score_drift=_score(psi=0.4))
    assert crit.status == "CRITICAL" and crit.action == "INVESTIGATE"
    q_warn, _ = _decide(policy, quality_issues=[QualityIssue("x", "missing_rate", 0.2, 0.05)])
    assert q_warn.status == "WARN"
    q_crit, _ = _decide(policy, quality_issues=[QualityIssue("x", "type_error", 0.01, 0.001)])
    assert q_crit.status == "CRITICAL" and q_crit.action == "INVESTIGATE"
    fine, _ = _decide(
        policy, performance=_perf({"auc": 0.79, "brier": 0.16}, auc_drop=0.01, brier_increase=0.01)
    )
    assert fine.status == "OK"
    slip, _ = _decide(
        policy, performance=_perf({"auc": 0.76, "brier": 0.16}, auc_drop=0.04, brier_increase=0.01)
    )
    assert slip.status == "WARN"
    floor, _ = _decide(
        policy, performance=_perf({"auc": 0.65, "brier": 0.2}, auc_drop=0.15, brier_increase=0.05)
    )
    assert floor.status == "CRITICAL" and floor.action == "RETRAIN" and "floor" in floor.reasons[0]
    payload = floor.to_dict()
    assert payload["action"] == "RETRAIN" and payload["findings"][-1]["area"] == "performance"


def test_state_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "s.json"
    assert MonitorState.load(path) == MonitorState()
    state = MonitorState(consecutive={"a": 2}, windows_seen=3, last_status="WARN")
    state.save(path)
    assert MonitorState.load(path) == state


@pytest.mark.parametrize(
    ("scenario", "magnitude", "expect_status", "expect_subject"),
    [
        ("none", 0.0, "OK", None),
        ("covariate_shift", 1.0, "CRITICAL", "x_utilisation"),
        ("quality_unseen", 0.05, "WARN", "c_region"),
        ("score_shift", 0.3, "CRITICAL", "score"),
        ("concept_drift", 1.8, "CRITICAL", "auc"),
    ],
)
def test_run_window_end_to_end(
    baseline: Baseline,
    policy: AlertPolicy,
    scenario: str,
    magnitude: float,
    expect_status: str,
    expect_subject: str | None,
) -> None:
    frame, labels = make_window(scenario, magnitude, seed=11)  # type: ignore[arg-type]
    report = run_window(baseline, frame, labels, policy, MonitorState(), window_id="w", n_boot=30)
    assert report.decision.status == expect_status, report.decision.reasons
    if expect_subject:
        assert any(
            f.subject == expect_subject and f.severity != "OK" for f in report.decision.findings
        ), report.decision.reasons
    assert report.n_rows == 1000 and report.model_versions == {"v1": 1000}
    assert report.performance.metrics is not None
    md = report.to_markdown()
    assert f"Status: {expect_status}" in md and "## Feature drift" in md and "## Performance" in md
    payload = report.to_dict()
    assert payload["decision"]["status"] == expect_status and len(payload["features"]) == 6
    assert report.exit_code == {"NONE": 0, "INVESTIGATE": 2, "RETRAIN": 3}[report.decision.action]


def test_run_window_without_labels_and_save(
    baseline: Baseline, policy: AlertPolicy, tmp_path: Path
) -> None:
    frame, _ = make_window("none", 0.0, seed=12)
    report = run_window(
        baseline, frame, None, policy, MonitorState(), window_id="nolabels", n_boot=20
    )
    assert report.performance.metrics is None and "not evaluated" in report.to_markdown()
    json_path, md_path = report.save(tmp_path / "out")
    assert json.loads(json_path.read_text(encoding="utf-8"))[
        "window_id"
    ] == "nolabels" and md_path.read_text(encoding="utf-8").startswith("# Monitoring report")
    tiny = run_window(
        baseline, frame.head(50), None, policy, MonitorState(), window_id="tiny", n_boot=20
    )
    assert tiny.decision.status == "WARN" and "only 50 rows" in tiny.decision.reasons[0]
    assert tiny.window_start is not None and tiny.window_end is not None
    assert isinstance(tiny, WindowReport)


def test_load_helpers(tmp_path: Path) -> None:
    now = datetime(2026, 5, 1, tzinfo=UTC)
    caps = [
        CaptureRecord(
            request_id=f"r{i}", ts=now + timedelta(hours=i), features={"a": float(i)}, score=0.5
        )
        for i in range(5)
    ]
    write_jsonl(tmp_path / "c.jsonl", caps)
    write_jsonl(tmp_path / "l.jsonl", [LabelRecord(request_id="r1", label=1)])
    frame = load_capture_frame(
        tmp_path / "c.jsonl", now + timedelta(hours=1), now + timedelta(hours=3)
    )
    assert frame["request_id"].tolist() == ["r1", "r2"]
    labels = load_labels_frame(tmp_path / "l.jsonl")
    assert labels is not None and len(labels) == 1
    assert load_labels_frame(None) is None
    assert isinstance(frame, pd.DataFrame)
