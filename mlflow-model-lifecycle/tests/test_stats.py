from __future__ import annotations

import math

import numpy as np
import pytest

from mlreg import metrics as M
from mlreg.stats import bootstrap_ci, paired_bootstrap_delta, psi


def _scores(n: int = 2000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    p = np.clip(rng.normal(0.3 + 0.3 * y, 0.25), 0, 1)
    return y, p


def test_bootstrap_ci_is_deterministic_and_brackets_estimate() -> None:
    y, p = _scores()
    r1 = bootstrap_ci(y, p, M.roc_auc, n_boot=200, seed=3)
    r2 = bootstrap_ci(y, p, M.roc_auc, n_boot=200, seed=3)
    assert r1 == r2
    assert r1.lower <= r1.estimate <= r1.upper
    assert r1.estimate == pytest.approx(M.roc_auc(y, p))
    assert str(r1).count("[") == 1
    assert set(r1.to_dict()) == {"estimate", "lower", "upper", "level", "n_boot"}


def test_paired_delta_of_identical_models_is_zero() -> None:
    y, p = _scores(n=300)
    d = paired_bootstrap_delta(y, p, p, M.roc_auc, n_boot=100)
    assert (d.estimate, d.lower, d.upper) == (0.0, 0.0, 0.0)


def test_paired_delta_detects_improvement_and_is_tighter_than_unpaired() -> None:
    y, p_good = _scores()
    rng = np.random.default_rng(1)
    p_bad = np.clip(p_good + rng.normal(0, 0.12, len(p_good)), 0, 1)
    d = paired_bootstrap_delta(y, p_bad, p_good, M.roc_auc, n_boot=300, seed=2)
    assert d.estimate > 0
    assert d.lower > 0  # the improvement is resolvable at 95 %
    ci_a = bootstrap_ci(y, p_bad, M.roc_auc, n_boot=300, seed=2)
    ci_b = bootstrap_ci(y, p_good, M.roc_auc, n_boot=300, seed=2)
    unpaired_width = (ci_a.upper - ci_a.lower) + (ci_b.upper - ci_b.lower)
    assert d.upper - d.lower < unpaired_width


def test_paired_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="equal length"):
        paired_bootstrap_delta(np.array([0, 1]), np.array([0.1, 0.2]), np.array([0.1]), M.brier)


def test_bootstrap_filters_degenerate_resamples() -> None:
    y = np.array([0] * 30 + [1] * 2)
    p = np.linspace(0, 1, 32)
    r = bootstrap_ci(y, p, M.roc_auc, n_boot=200, seed=0)
    assert math.isfinite(r.lower) and math.isfinite(r.upper)


def test_psi_properties() -> None:
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    assert psi(ref, ref) == pytest.approx(0.0, abs=1e-12)
    assert psi(ref, rng.normal(0, 1, 5000)) < 0.05
    assert psi(ref, ref + 1.0) > 0.25
    assert psi(ref, ref * 2.0) > 0.10
    assert psi(np.ones(10), np.ones(5)) == 0.0
    assert psi(np.ones(10), np.zeros(5)) == math.inf
    with pytest.raises(ValueError, match="empty"):
        psi(np.array([]), ref)
