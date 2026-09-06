"""Prometheus exporter + on-demand runner, so drift lands on the same dashboards and alert
rules as latency and error rates. Gauges hold the latest window; Prometheus keeps the history.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    generate_latest,
)
from pydantic import BaseModel, Field

from .baseline import Baseline
from .config import SEVERITY_RANK, AlertPolicy
from .monitor import EXIT_BY_ACTION, WindowReport, load_capture_frame, load_labels_frame, run_window
from .policy import MonitorState

log = logging.getLogger("mlwatch.exporter")


class Exporter:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        r = self.registry
        self.feature_psi = Gauge(
            "mlwatch_feature_psi", "PSI of a feature vs baseline", ["feature"], registry=r
        )
        self.feature_p_adj = Gauge(
            "mlwatch_feature_test_p_adjusted",
            "BH-adjusted drift test p-value",
            ["feature"],
            registry=r,
        )
        self.feature_missing = Gauge(
            "mlwatch_feature_missing_rate", "Missing rate of a feature", ["feature"], registry=r
        )
        self.score_psi = Gauge(
            "mlwatch_score_psi", "PSI of the score distribution vs baseline", registry=r
        )
        self.decline_rate = Gauge(
            "mlwatch_decline_rate", "Share of scores at or above the threshold", registry=r
        )
        self.status = Gauge("mlwatch_window_status", "0 OK, 1 WARN, 2 CRITICAL", registry=r)
        self.action = Gauge("mlwatch_window_action", "0 NONE, 2 INVESTIGATE, 3 RETRAIN", registry=r)
        self.rows = Gauge("mlwatch_window_rows", "Rows in the last window", registry=r)
        self.labels = Gauge("mlwatch_window_labels", "Labelled rows in the last window", registry=r)
        self.label_coverage = Gauge(
            "mlwatch_label_coverage", "Share of rows with a label", registry=r
        )
        self.auc = Gauge(
            "mlwatch_auc", "Realised AUC on labelled rows (NaN if not evaluated)", registry=r
        )
        self.quality_issues = Gauge(
            "mlwatch_quality_issues", "Number of data-quality issues", registry=r
        )
        self.last_run = Gauge(
            "mlwatch_last_run_timestamp_seconds", "Unix time of the last run", registry=r
        )
        self.runs = Counter("mlwatch_runs_total", "Monitoring runs", ["status"], registry=r)
        self.last_report: WindowReport | None = None

    def update(self, report: WindowReport) -> None:
        for f in report.features:
            self.feature_psi.labels(feature=f.name).set(f.psi)
            self.feature_p_adj.labels(feature=f.name).set(
                float("nan") if f.p_adjusted is None else f.p_adjusted
            )
            self.feature_missing.labels(feature=f.name).set(f.missing_rate)
        self.score_psi.set(report.score.psi)
        self.decline_rate.set(report.score.decline_rate)
        self.status.set(SEVERITY_RANK[report.decision.status])
        self.action.set(EXIT_BY_ACTION[report.decision.action])
        self.rows.set(report.n_rows)
        self.labels.set(report.performance.n_labelled)
        self.label_coverage.set(report.performance.label_coverage)
        self.auc.set(
            report.performance.metrics["auc"] if report.performance.metrics else float("nan")
        )
        self.quality_issues.set(len(report.quality))
        self.last_run.set(datetime.fromisoformat(report.generated_utc).timestamp())
        self.runs.labels(status=report.decision.status).inc()
        self.last_report = report

    def render(self) -> bytes:
        return bytes(generate_latest(self.registry))


class RunRequest(BaseModel):
    capture_path: str
    labels_path: str | None = None
    window_id: str | None = None
    start: datetime | None = None
    end: datetime | None = None
    n_boot: int = Field(default=200, ge=10, le=2000)


def create_app(
    baseline_path: Path,
    policy_path: Path,
    *,
    state_path: Path | None = None,
    allowed_root: Path | None = None,
) -> FastAPI:
    baseline = Baseline.load(baseline_path)
    policy = AlertPolicy.from_yaml(policy_path)
    state = MonitorState.load(state_path) if state_path else MonitorState()
    exporter = Exporter()
    root = Path(allowed_root).resolve() if allowed_root else None
    app = FastAPI(title="mlwatch exporter")
    app.state.exporter = exporter

    def _resolve(raw: str) -> Path:
        p = Path(raw).resolve()
        if root is not None and root not in p.parents and p != root:
            raise HTTPException(status_code=400, detail=f"{raw} is outside the allowed root")
        if not p.is_file():
            raise HTTPException(status_code=404, detail=f"{raw} not found")
        return p

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "baseline_model_version": baseline.model_version,
            "windows_seen": state.windows_seen,
        }

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(content=exporter.render(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/report")
    async def report() -> dict[str, Any]:
        if exporter.last_report is None:
            raise HTTPException(status_code=404, detail="no run yet")
        return exporter.last_report.to_dict()

    @app.post("/run")
    async def run(req: RunRequest) -> dict[str, Any]:
        capture = _resolve(req.capture_path)
        labels = _resolve(req.labels_path) if req.labels_path else None
        frame = load_capture_frame(capture, req.start, req.end)
        window_id = req.window_id or capture.stem
        rep = run_window(
            baseline,
            frame,
            load_labels_frame(labels),
            policy,
            state,
            window_id=window_id,
            n_boot=req.n_boot,
        )
        exporter.update(rep)
        if state_path:
            state.save(state_path)
        log.info("window %s: %s / %s", window_id, rep.decision.status, rep.decision.action)
        return rep.to_dict()

    return app
