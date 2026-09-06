from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from mlreg.config import DataSchema
from mlreg.gate import PromotionPolicy, evaluate_gate
from mlreg.model_card import _ci, _f, render_model_card
from mlreg.registry import RegisteredVersion

HEADINGS = [
    "# Model card",
    "## 1. Intended use",
    "## 2. Data",
    "## 3. Performance",
    "## 4. Slice analysis",
    "## 5. Promotion decision",
    "## 6. Governance",
    "## 7. Limitations",
]


def _render(t: SimpleNamespace, schema: DataSchema, **extra: object) -> str:
    return render_model_card(
        cfg=t.cfg,
        schema=schema,
        metrics_test=t.result.metrics_test,
        test_ci=t.result.test_ci,
        cv_mean=t.result.cv_mean(),
        cv_std=t.result.cv_std(),
        slices=t.result.slices,
        threshold=t.result.threshold,
        run_id=t.result.run_id,
        data_fingerprint=t.result.data_fingerprint,
        **extra,  # type: ignore[arg-type]
    )


def test_card_without_gate(trained: SimpleNamespace, schema: DataSchema) -> None:
    card = _render(trained, schema)
    for heading in HEADINGS:
        assert heading in card
    assert "Not yet evaluated" in card
    assert "Approver: pending" in card
    assert "personal_status_sex" in card and "foreign_worker" in card
    assert f"{trained.result.metrics_test['auc']:.4f}" in card
    assert "cost-optimal" in card
    assert trained.result.run_id in card


def test_card_with_gate_and_registration(trained: SimpleNamespace, schema: DataSchema) -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    p = np.clip(0.3 + 0.3 * y + rng.normal(0, 0.2, 300), 0, 1)
    gate = evaluate_gate(PromotionPolicy(), y, p, None, 0.5)
    registered = RegisteredVersion("credit-pd-test", 3, "run", "src", aliases=["champion"])
    fixed_cfg = trained.cfg.model_copy(update={"decision_threshold": 0.5})
    t = SimpleNamespace(cfg=fixed_cfg, result=trained.result)
    card = _render(
        t, schema, gate=gate, registered=registered, approved_by="reviewer", git_sha="abc123"
    )
    assert "v3" in card and "champion" in card
    assert f"Decision: {gate.decision}" in card
    assert "Approver: reviewer" in card and "`abc123`" in card
    assert "fixed in config" in card


def test_helpers() -> None:
    assert _f("x") == "x"
    assert _f(float("nan")) == "-"
    assert _f(1.23456, 2) == "1.23"
    assert _ci(None) == ""
    assert _ci({"lower": 0.1, "upper": 0.2}) == " [0.100, 0.200]"
