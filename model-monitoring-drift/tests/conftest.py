from __future__ import annotations

import warnings
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from mlwatch.baseline import Baseline, build_baseline
from mlwatch.config import AlertPolicy
from mlwatch.schema import captures_to_frame, labels_to_frame
from mlwatch.simulate import FEATURES, Scenario, WindowSpec, generate_window

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "policies" / "default.yaml"
START = datetime(2026, 3, 1, tzinfo=UTC)


@pytest.fixture(scope="session")
def policy() -> AlertPolicy:
    return AlertPolicy.from_yaml(POLICY_PATH)


def make_window(
    scenario: Scenario, magnitude: float, *, n: int = 1000, seed: int = 1
) -> tuple[pd.DataFrame, pd.DataFrame]:
    caps, labels = generate_window(WindowSpec(scenario, magnitude, n, seed, START))
    return captures_to_frame(caps), labels_to_frame(labels)


@pytest.fixture(scope="session")
def reference() -> SimpleNamespace:
    caps, labels = generate_window(
        WindowSpec("none", 0.0, 3000, 7, START - timedelta(days=30), hours=24 * 30)
    )
    frame = captures_to_frame(caps)
    label_frame = labels_to_frame(labels)
    lookup = label_frame.set_index("request_id")["label"]
    return SimpleNamespace(
        frame=frame, labels=label_frame, label_series=frame["request_id"].map(lookup)
    )


@pytest.fixture(scope="session")
def baseline(reference: SimpleNamespace) -> Baseline:
    return build_baseline(
        reference.frame,
        feature_columns=FEATURES,
        threshold=0.5,
        labels=reference.label_series,
        model_version="v1",
    )
