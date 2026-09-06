"""Model card rendered from the run's evidence pack, the gate result and the registry state.

Structured after the fields a model-risk reviewer asks for (SR 11-7 / APRA CPG 235 style):
purpose and scope, data lineage, performance with uncertainty, slice behaviour, the promotion
decision with the exact checks, and governance identifiers that link back to MLflow.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd

from .config import DataSchema, TrainingConfig
from .gate import GateResult
from .registry import RegisteredVersion


def _f(v: Any, nd: int = 4) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    if x != x:
        return "-"
    return f"{x:.{nd}f}"


def _ci(ci: dict[str, float] | None, nd: int = 3) -> str:
    if not ci:
        return ""
    return f" [{_f(ci['lower'], nd)}, {_f(ci['upper'], nd)}]"


def render_model_card(
    *,
    cfg: TrainingConfig,
    schema: DataSchema,
    metrics_test: dict[str, float],
    test_ci: dict[str, dict[str, float]],
    cv_mean: dict[str, float],
    cv_std: dict[str, float],
    slices: dict[str, pd.DataFrame],
    threshold: float,
    run_id: str,
    data_fingerprint: str,
    gate: GateResult | None = None,
    registered: RegisteredVersion | None = None,
    approved_by: str | None = None,
    git_sha: str | None = None,
) -> str:
    now = datetime.now(UTC).isoformat(timespec="seconds")
    lines: list[str] = []
    title = f"{cfg.tracking.registered_model}" + (f" v{registered.version}" if registered else "")
    registry_cell = f"`{cfg.tracking.registered_model}`"
    if registered:
        registry_cell += (
            f" version {registered.version}, aliases: {', '.join(registered.aliases) or '-'}"
        )
    rule = (
        "(cost-optimal on out-of-fold predictions, cost matrix 5:1)."
        if cfg.decision_threshold == "cost_optimal"
        else "(fixed in config)."
    )
    lines += [f"# Model card — {title}", ""]
    lines += [
        "| | |",
        "|---|---|",
        f"| Model | `{cfg.model.kind}` ({cfg.name}) |",
        f"| Registry | {registry_cell} |",
        f"| MLflow run | `{run_id}` |",
        f"| Config hash | `{cfg.config_hash()[:16]}` |",
        f"| Data fingerprint | `{data_fingerprint[:16]}` "
        f"({schema.name} schema v{schema.version}) |",
        f"| Code | `{git_sha or 'n/a'}` |",
        f"| Generated | {now} |",
        "",
        "## 1. Intended use",
        "",
        "Probability of default (PD) for retail credit applications, used to rank and decline",
        f"applicants. Decision rule: decline when PD ≥ {threshold:.3f} {rule}",
        "Not intended for pricing, limit setting or any use outside the training population.",
        "",
        "## 2. Data",
        "",
        f"- Source file `{cfg.data_path.name}`, contract `{schema.name}` v{schema.version}, "
        f"fingerprint `{data_fingerprint}`",
        f"- Split: stratified holdout {cfg.split.test_size:.0%} (seed {cfg.split.seed}); "
        f"{cfg.cv.folds}-fold CV on the training part",
        f"- Features: {len(schema.numeric_features)} numeric, "
        f"{len(schema.categorical_features)} categorical",
        "- Excluded protected attributes (reported as slices only): "
        f"{', '.join(schema.protected_columns) or 'none'}",
        "",
        "## 3. Performance (holdout, 95 % bootstrap CIs)",
        "",
        "| Metric | Holdout | CV mean ± std |",
        "|---|---:|---:|",
    ]
    for key, label in [
        ("auc", "AUC"),
        ("ks", "KS"),
        ("brier", "Brier"),
        ("log_loss", "Log loss"),
        ("ece", "ECE"),
    ]:
        hold = _f(metrics_test.get(key)) + _ci(test_ci.get(key))
        cvm = f"{_f(cv_mean.get(key))} ± {_f(cv_std.get(key))}" if key in cv_mean else "-"
        lines.append(f"| {label} | {hold} | {cvm} |")
    lines += [
        f"| Gini | {_f(metrics_test.get('gini'))} | |",
        f"| Expected cost / applicant | {_f(metrics_test.get('expected_cost'), 3)} | |",
        f"| Approval rate | {_f(metrics_test.get('approval_rate'), 3)} | |",
        f"| Bad rate among approved | {_f(metrics_test.get('bad_rate_among_approved'), 3)} | |",
        "",
    ]
    if slices:
        lines += ["## 4. Slice analysis (protected attributes, holdout)", ""]
        for col, table in slices.items():
            lines += [
                f"**{col}**",
                "",
                "| Group | n | Default rate | AUC | Approval rate |",
                "|---|---:|---:|---:|---:|",
            ]
            for _, r in table.iterrows():
                lines.append(
                    f"| {r['group']} | {int(r['n'])} | {_f(r['positive_rate'], 3)} | "
                    f"{_f(r['auc'], 3)} | {_f(r['approval_rate'], 3)} |"
                )
            lines.append("")
    lines += ["## 5. Promotion decision", ""]
    if gate is None:
        lines += ["Not yet evaluated against a promotion policy.", ""]
    else:
        lines += [gate.to_markdown(), ""]
    lines += [
        "## 6. Governance",
        "",
        f"- Owner: {cfg.tags.get('owner', 'n/a')}; use case: {cfg.tags.get('use_case', 'n/a')}",
        f"- Approver: {approved_by or 'pending'}",
        "- Reproduce: `mlreg train --config <this config>` with the same data fingerprint "
        "reproduces the run (seeded split, CV and estimators).",
        "- Evidence in MLflow: `config.json`, `schema.json`, `evaluation/*`, `plots/*`, "
        "`cv_folds.csv`, model signature with input example.",
        "",
        "## 7. Limitations",
        "",
        "- 1 000 applications from one lender in one period; no macro-cycle information, "
        "so point-in-time only.",
        "- Reason codes are exact logit decompositions for linear models and unavailable for "
        "tree ensembles.",
        "- Calibration is assessed on the holdout only; recalibrate before use on a population "
        "with a different base rate.",
        "",
    ]
    return "\n".join(lines)
