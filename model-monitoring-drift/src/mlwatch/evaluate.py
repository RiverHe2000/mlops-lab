"""Monitor-the-monitor: detection power and false-alarm rate of the policy on simulated drift.

For every scenario x magnitude, `seeds` independent worlds are generated (clean baseline with
labels, one drifted window), the monitor runs once with a fresh state, and we record whether
it raised anything, whether it pointed at the right subject, and what it recommended. The
`none` rows are the false-alarm rate at the chosen thresholds.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np

from .baseline import build_baseline
from .config import AlertPolicy
from .monitor import run_window
from .policy import MonitorState
from .schema import captures_to_frame, labels_to_frame
from .simulate import FEATURES, TARGET_BY_SCENARIO, Scenario, WindowSpec, generate_window

DEFAULT_GRID: dict[Scenario, list[float]] = {
    "none": [0.0],
    "covariate_shift": [0.25, 0.5, 1.0],
    "prior_shift": [0.5, 1.0],
    "concept_drift": [1.0, 1.8],
    "quality_missing": [0.05, 0.15],
    "quality_unseen": [0.02, 0.05],
    "score_shift": [0.1, 0.3],
}


@dataclass
class EvalRow:
    scenario: str
    magnitude: float
    seeds: int
    target: str
    flagged_rate: float
    target_hit_rate: float
    critical_rate: float
    retrain_rate: float
    mean_target_psi: float
    mean_auc_drop: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _target_hit(report_findings: list[Any], target: str) -> bool:
    if not target:
        return False
    for f in report_findings:
        if f.severity == "OK":
            continue
        if target == "performance" and f.area == "performance":
            return True
        if target == "score" and f.area == "prediction":
            return True
        if f.subject == target:
            return True
    return False


def evaluate_monitor(
    policy: AlertPolicy,
    *,
    grid: dict[Scenario, list[float]] | None = None,
    seeds: int = 20,
    n_baseline: int = 4000,
    n_window: int = 1000,
    with_labels: bool = True,
    n_boot: int = 50,
) -> list[EvalRow]:
    grid = grid or DEFAULT_GRID
    start = datetime(2026, 3, 1, tzinfo=UTC)
    rows: list[EvalRow] = []
    for scenario, magnitudes in grid.items():
        target = TARGET_BY_SCENARIO[scenario]
        for magnitude in magnitudes:
            flagged = hits = critical = retrain = 0
            psis: list[float] = []
            drops: list[float] = []
            for seed in range(seeds):
                base_caps, base_labels = generate_window(
                    WindowSpec(
                        "none",
                        0.0,
                        n_baseline,
                        10_000 + seed,
                        start - timedelta(days=30),
                        hours=24 * 30,
                    )
                )
                base_frame = captures_to_frame(base_caps)
                base_lab = labels_to_frame(base_labels).set_index("request_id")["label"]
                labels_series = base_frame["request_id"].map(base_lab) if with_labels else None
                baseline = build_baseline(
                    base_frame,
                    feature_columns=FEATURES,
                    threshold=0.5,
                    labels=labels_series,
                    seed=seed,
                )
                caps, labels = generate_window(
                    WindowSpec(scenario, magnitude, n_window, 20_000 + seed, start)
                )
                frame = captures_to_frame(caps)
                report = run_window(
                    baseline,
                    frame,
                    labels_to_frame(labels) if with_labels else None,
                    policy,
                    MonitorState(),
                    window_id=f"{scenario}-{magnitude}-{seed}",
                    n_boot=n_boot,
                    seed=seed,
                )
                status = report.decision.status
                flagged += status != "OK"
                critical += status == "CRITICAL"
                retrain += report.decision.action == "RETRAIN"
                hits += _target_hit(report.decision.findings, target)
                if target in FEATURES:
                    psis.append(next(f.psi for f in report.features if f.name == target))
                elif target == "score":
                    psis.append(report.score.psi)
                if report.performance.auc_drop is not None:
                    drops.append(report.performance.auc_drop)
            rows.append(
                EvalRow(
                    scenario=scenario,
                    magnitude=magnitude,
                    seeds=seeds,
                    target=target or "-",
                    flagged_rate=flagged / seeds,
                    target_hit_rate=hits / seeds if target else float("nan"),
                    critical_rate=critical / seeds,
                    retrain_rate=retrain / seeds,
                    mean_target_psi=float(np.mean(psis)) if psis else float("nan"),
                    mean_auc_drop=float(np.mean(drops)) if drops else float("nan"),
                )
            )
    return rows


def to_markdown(rows: list[EvalRow]) -> str:
    lines = [
        "| Scenario | Magnitude | Target | Flagged (>= WARN) | Target hit | CRITICAL | RETRAIN "
        "| Mean target PSI | Mean AUC drop |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r.scenario} | {r.magnitude:g} | {r.target} | {r.flagged_rate:.0%} | "
            f"{_pct(r.target_hit_rate)} | "
            f"{r.critical_rate:.0%} | {r.retrain_rate:.0%} | "
            f"{_num(r.mean_target_psi)} | {_num(r.mean_auc_drop)} |"
        )
    return "\n".join(lines) + "\n"


def _pct(v: float) -> str:
    return "-" if v != v else f"{v:.0%}"


def _num(v: float) -> str:
    return "-" if v != v else f"{v:.3f}"
