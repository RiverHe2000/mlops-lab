from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mlreg import registry as R
from mlreg.config import DataSchema
from mlreg.gate import (
    AbsoluteThresholds,
    PromotionPolicy,
    RelativeThresholds,
    SliceThresholds,
    _fmt,
    evaluate_gate,
)

from .conftest import log_quick_model, tracking_for


def _scores(n: int = 600, seed: int = 0, sep: float = 1.6) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    z = sep * y + rng.normal(0, 1, n)
    return y, 1 / (1 + np.exp(-(z - 0.8)))


LOOSE = AbsoluteThresholds(min_auc=0.7, max_brier=0.25, max_ece=0.2, min_n_holdout=100)


def test_policy_yaml_loads(root: Path) -> None:
    pol = PromotionPolicy.from_yaml(root / "gates" / "promotion.yaml")
    assert pol.absolute.min_auc == 0.72
    assert pol.relative.n_boot == 2000
    assert pol.slices is not None and pol.slices.min_n == 30
    assert pol.require_same_data_snapshot


def test_absolute_only_gate() -> None:
    y, p = _scores()
    g = evaluate_gate(PromotionPolicy(absolute=LOOSE), y, p, None, 0.5)
    assert g.decision == "PROMOTE"
    assert g.champion is None and not g.deltas
    assert [c.name for c in g.checks] == ["holdout_size", "auc_min", "brier_max", "ece_max"]
    assert "PROMOTE" in g.to_markdown()
    assert g.to_dict()["decision"] == "PROMOTE" and g.to_dict()["n_holdout"] == 600


def test_absolute_failures_hold() -> None:
    y, p = _scores()
    strict = PromotionPolicy(
        absolute=AbsoluteThresholds(min_auc=0.99, min_n_holdout=10_000, max_brier=1.0, max_ece=1.0)
    )
    g = evaluate_gate(strict, y, p, None, 0.5)
    assert g.decision == "HOLD"
    assert {c.name for c in g.failed} == {"holdout_size", "auc_min"}
    assert "FAIL" in g.to_markdown()


def test_relative_noninferiority_and_direction() -> None:
    y, p = _scores(n=2000)
    rng = np.random.default_rng(5)
    worse = np.clip(p + rng.normal(0, 0.35, len(p)), 0, 1)
    # PSI is tested separately; here the noisy challenger has a very different score distribution
    pol = PromotionPolicy(
        absolute=LOOSE, relative=RelativeThresholds(n_boot=300, max_score_psi=100.0)
    )
    hold = evaluate_gate(pol, y, worse, p, 0.5, same_snapshot=True)
    assert hold.decision == "HOLD"
    assert "auc_noninferior" in {c.name for c in hold.failed}
    assert hold.deltas["auc"].estimate < 0
    promote = evaluate_gate(pol, y, p, worse, 0.5, same_snapshot=True)
    assert promote.decision == "PROMOTE"
    assert promote.deltas["auc"].lower > 0
    assert promote.champion is not None and promote.champion["auc"] < promote.challenger["auc"]
    assert "score_psi" in {c.name for c in promote.checks}
    assert "Paired deltas" in promote.to_markdown()


def test_same_snapshot_requirement() -> None:
    y, p = _scores()
    pol = PromotionPolicy(absolute=LOOSE, relative=RelativeThresholds(n_boot=200))
    g = evaluate_gate(pol, y, p, p, 0.5, same_snapshot=False)
    assert g.decision == "HOLD" and [c.name for c in g.failed] == ["same_data_snapshot"]
    g2 = evaluate_gate(pol, y, p, p, 0.5, same_snapshot=None)
    assert g2.decision == "HOLD"
    relaxed = pol.model_copy(update={"require_same_data_snapshot": False})
    assert evaluate_gate(relaxed, y, p, p, 0.5).decision == "PROMOTE"


def test_score_psi_check_catches_distribution_shift() -> None:
    y, p = _scores()
    shrunk = np.clip(p * 0.2, 0, 1)  # same ranking, very different score distribution
    pol = PromotionPolicy(absolute=LOOSE, relative=RelativeThresholds(n_boot=200))
    g = evaluate_gate(pol, y, shrunk, p, 0.5, same_snapshot=True)
    assert "score_psi" in {c.name for c in g.failed}
    assert g.deltas["auc"].estimate == pytest.approx(0.0, abs=1e-12)


def test_slice_checks() -> None:
    y, p = _scores(n=400)
    rng = np.random.default_rng(9)
    p_mixed = p.copy()
    p_mixed[200:] = rng.uniform(0, 1, 200)  # group b carries no signal
    groups = pd.Series(["a"] * 200 + ["b"] * 200)
    pol = PromotionPolicy(absolute=LOOSE, slices=SliceThresholds(min_auc=0.6, min_n=30))
    g = evaluate_gate(pol, y, p_mixed, None, 0.5, slices={"grp": groups})
    failed = {c.name: c for c in g.failed}
    assert (
        "slice_auc_min[grp]" in failed and "worst group: b" in failed["slice_auc_min[grp]"].detail
    )
    tiny = PromotionPolicy(absolute=LOOSE, slices=SliceThresholds(min_auc=0.6, min_n=10_000))
    g2 = evaluate_gate(tiny, y, p_mixed, None, 0.5, slices={"grp": groups})
    check = next(c for c in g2.checks if c.name == "slice_auc_min[grp]")
    assert check.passed and check.value is not None and math.isnan(check.value)
    assert "| - |" in g2.to_markdown()


def test_fmt() -> None:
    assert _fmt(None) == "-" and _fmt(float("nan")) == "-"
    assert _fmt(0.12345) == "0.1235" and _fmt(1234.0) == "1234"


@pytest.fixture
def store(tmp_path: Path, schema: DataSchema, frame: pd.DataFrame) -> SimpleNamespace:
    tracking = tracking_for(tmp_path, registered_model="reg-test")
    uris = [log_quick_model(schema, frame, tracking, c=c) for c in (1.0, 0.1)]
    return SimpleNamespace(tracking=tracking, uris=uris)


def test_register_alias_promote_flow(store: SimpleNamespace) -> None:
    name = "reg-test"
    assert R.list_versions(name) == []
    assert R.get_alias(name, R.CHAMPION) is None
    v1 = R.register(store.uris[0], name, tags={"data.fingerprint": "abc"})
    assert (v1.version, v1.tags["data.fingerprint"], v1.uri) == (1, "abc", "models:/reg-test/1")
    assert v1.run_id
    R.set_alias(name, R.CHALLENGER, 1)
    challenger = R.get_alias(name, R.CHALLENGER)
    assert challenger is not None and challenger.version == 1

    promoted, displaced = R.promote(
        name, 1, approved_by="tester", evidence={"gate.report": "x.json"}
    )
    assert displaced is None
    assert R.CHAMPION in promoted.aliases and R.CHALLENGER not in promoted.aliases
    assert promoted.tags["promoted_by"] == "tester" and promoted.tags["gate.report"] == "x.json"

    v2 = R.register(store.uris[1], name)
    assert v2.version == 2
    promoted2, displaced2 = R.promote(name, 2, approved_by="tester")
    assert displaced2 is not None and displaced2.version == 1
    previous = R.get_alias(name, R.PREVIOUS)
    assert previous is not None and previous.version == 1
    assert "retired_utc" in previous.tags and previous.tags["retired_by_version"] == "2"
    champion = R.get_alias(name, R.CHAMPION)
    assert champion is not None and champion.version == 2 == promoted2.version

    again = R.promote(name, 2, approved_by="x")  # idempotent
    assert again[1] is None and again[0].version == 2

    versions = R.list_versions(name)
    assert [v.version for v in versions] == [1, 2]
    assert set(versions[0].to_dict()) == {
        "name",
        "version",
        "run_id",
        "source",
        "aliases",
        "tags",
        "uri",
    }
    assert hasattr(R.load_by_alias(name, R.CHAMPION), "predict")
    assert hasattr(R.load_version(name, 1), "predict")


def test_get_alias_for_unknown_model(store: SimpleNamespace) -> None:
    assert R.get_alias("does-not-exist", R.CHAMPION) is None
    assert R.list_versions("does-not-exist") == []
