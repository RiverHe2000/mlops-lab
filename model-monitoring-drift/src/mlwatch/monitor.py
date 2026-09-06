"""One monitoring run over one window: drift, quality, performance, decision, report."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from . import __version__
from .baseline import Baseline
from .config import AlertPolicy
from .drift import FeatureDrift, ScoreDrift, feature_drifts, score_drift
from .performance import PerformanceResult, performance_check
from .policy import Decision, MonitorState, decide
from .quality import QualityIssue, quality_checks
from .schema import (
    CaptureRecord,
    LabelRecord,
    captures_to_frame,
    labels_to_frame,
    read_jsonl,
    select_window,
)

EXIT_BY_ACTION = {"NONE": 0, "INVESTIGATE": 2, "RETRAIN": 3}


@dataclass
class WindowReport:
    window_id: str
    generated_utc: str
    n_rows: int
    model_versions: dict[str, int]
    features: list[FeatureDrift]
    score: ScoreDrift
    quality: list[QualityIssue]
    performance: PerformanceResult
    decision: Decision
    baseline_model_version: str = ""
    mlwatch_version: str = __version__
    window_start: str | None = None
    window_end: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def exit_code(self) -> int:
        return EXIT_BY_ACTION[self.decision.action]

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "generated_utc": self.generated_utc,
            "mlwatch_version": self.mlwatch_version,
            "baseline_model_version": self.baseline_model_version,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "n_rows": self.n_rows,
            "model_versions": self.model_versions,
            "decision": self.decision.to_dict(),
            "features": [f.to_dict() for f in self.features],
            "score": self.score.to_dict(),
            "quality": [q.to_dict() for q in self.quality],
            "performance": self.performance.to_dict(),
            **self.extra,
        }

    def to_markdown(self) -> str:
        d = self.decision
        lines = [
            f"# Monitoring report — window `{self.window_id}`",
            "",
            f"**Status: {d.status} · Action: {d.action}** — {self.n_rows} rows, "
            f"model versions {self.model_versions}, generated {self.generated_utc}",
            "",
        ]
        if d.reasons:
            lines += ["## Findings", ""] + [f"- {r}" for r in d.reasons] + [""]
        lines += [
            "## Feature drift",
            "",
            "| Feature | Kind | n | Missing | PSI [95 % CI] | Test | p (BH) | JS | Shift |",
            "|---|---|---:|---:|---:|---|---:|---:|---|",
        ]
        for f in self.features:
            ci = "" if math.isnan(f.psi_ci_low) else f" [{f.psi_ci_low:.3f}, {f.psi_ci_high:.3f}]"
            p_adj = "-" if f.p_adjusted is None else f"{f.p_adjusted:.3g}"
            lines.append(
                f"| {f.name} | {f.kind} | {f.n} | {f.missing_rate:.1%} | {_f(f.psi)}{ci} | "
                f"{f.test} | {p_adj} | {_f(f.js)} | {f.shift_summary} |"
            )
        s = self.score
        lines += [
            "",
            "## Prediction drift",
            "",
            f"- score PSI {_f(s.psi)}, JS {_f(s.js)}, mean {_f(s.mean)} ({s.mean_change:+.3f}), "
            f"decline rate {s.decline_rate:.1%} ({s.decline_rate_change:+.1%})",
            "",
            "## Data quality",
            "",
        ]
        lines += [
            f"- {q.feature}: {q.kind} = {q.value:.4f} (threshold {q.threshold:.4f}) {q.detail}"
            for q in self.quality
        ] or ["- no issues"]
        p = self.performance
        lines += ["", "## Performance", ""]
        if p.metrics is None:
            lines.append(
                f"- {p.n_labelled} labels of {p.n_scored} rows ({p.label_coverage:.0%}); "
                f"below min_labels {p.min_labels}, not evaluated"
            )
        else:
            m = p.metrics
            lines.append(
                f"- {p.n_labelled} labels ({p.label_coverage:.0%} coverage): "
                f"AUC {m['auc']:.3f}, KS {m['ks']:.3f}, "
                f"Brier {m['brier']:.3f}, ECE {m['ece']:.3f}, "
                f"positive rate {m['positive_rate']:.1%}"
            )
            if p.auc_drop is not None:
                lines.append(
                    f"- vs baseline: AUC {p.auc_drop:+.3f} drop, Brier {p.brier_increase:+.3f}"
                )
        return "\n".join(lines) + "\n"

    def save(self, directory: Path) -> tuple[Path, Path]:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        json_path = directory / "report.json"
        md_path = directory / "report.md"
        json_path.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
        md_path.write_text(self.to_markdown(), encoding="utf-8")
        return json_path, md_path


def _f(v: float) -> str:
    return "-" if v != v else f"{v:.3f}"


def run_window(
    baseline: Baseline,
    frame: pd.DataFrame,
    labels: pd.DataFrame | None,
    policy: AlertPolicy,
    state: MonitorState,
    *,
    window_id: str,
    n_boot: int = 200,
    seed: int = 0,
) -> WindowReport:
    drifts = feature_drifts(baseline, frame, n_boot=n_boot, seed=seed)
    scores = score_drift(
        baseline.score, frame["score"] if "score" in frame.columns else pd.Series(dtype="float64")
    )
    issues = quality_checks(baseline, frame, policy.quality)
    perf = performance_check(baseline, frame, labels, min_labels=policy.performance.min_labels)
    decision = decide(
        policy,
        n_rows=len(frame),
        feature_drifts=drifts,
        score_drift=scores,
        quality_issues=issues,
        performance=perf,
        state=state,
    )
    versions = (
        {str(k): int(v) for k, v in frame["model_version"].value_counts().items()}
        if "model_version" in frame.columns
        else {}
    )
    ts = pd.to_datetime(frame["ts"], utc=True) if "ts" in frame.columns and len(frame) else None
    return WindowReport(
        window_id=window_id,
        generated_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        n_rows=len(frame),
        model_versions=versions,
        features=drifts,
        score=scores,
        quality=issues,
        performance=perf,
        decision=decision,
        baseline_model_version=baseline.model_version,
        window_start=None if ts is None else str(ts.min()),
        window_end=None if ts is None else str(ts.max()),
    )


def load_capture_frame(
    path: Path, start: datetime | None = None, end: datetime | None = None
) -> pd.DataFrame:
    records = select_window(read_jsonl(path, CaptureRecord), start, end)
    return captures_to_frame(records)


def load_labels_frame(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    return labels_to_frame(read_jsonl(path, LabelRecord))
