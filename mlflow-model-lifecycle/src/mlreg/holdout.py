"""Bind evaluation to the held-out rows recorded by the model's training run.

A CSV fingerprint alone cannot detect a changed split. Persist membership hashes for both
partitions, so a comparison rejects training overlap before loading or scoring any model.
The hashes avoid putting raw application identifiers in this additional evidence file;
they are identifiers for integrity checks, not a privacy guarantee for low-entropy IDs.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Literal

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from pydantic import BaseModel, ConfigDict

from .data import Dataset

ARTIFACT = "evaluation/split_manifest.json"
TAG = "data.split_manifest_sha256"


class HoldoutMismatchError(ValueError):
    """The stored training evidence does not establish an untouched holdout."""


class SplitManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal[1] = 1
    data_fingerprint: str
    train_ids: tuple[str, ...]
    test_ids: tuple[str, ...]

    @classmethod
    def from_dataset(cls, ds: Dataset) -> SplitManifest:
        def hashed(ids: object) -> str:
            return hashlib.sha256(str(ids).encode("utf-8")).hexdigest()

        return cls(
            data_fingerprint=ds.fingerprint,
            train_ids=tuple(sorted(hashed(i) for i in ds.ids_train)),
            test_ids=tuple(sorted(hashed(i) for i in ds.ids_test)),
        )

    def fingerprint(self) -> str:
        encoded = json.dumps(self.model_dump(mode="json"), sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def check(self, current: SplitManifest) -> None:
        for label, partition in (("training", self.train_ids), ("holdout", self.test_ids)):
            if not partition or len(partition) != len(set(partition)):
                raise HoldoutMismatchError(f"stored {label} membership is empty or duplicated")
        if set(self.train_ids) & set(self.test_ids):
            raise HoldoutMismatchError("stored training and holdout memberships overlap")
        if self.data_fingerprint != current.data_fingerprint:
            raise HoldoutMismatchError("data snapshot differs from the training run")
        if set(self.train_ids) & set(current.test_ids):
            raise HoldoutMismatchError("current holdout includes rows used to train this model")
        if self.test_ids != current.test_ids or self.train_ids != current.train_ids:
            raise HoldoutMismatchError("train/holdout membership differs from the training run")


def verify_run_holdout(run_id: str | None, current: SplitManifest) -> None:
    """Fail closed for legacy, missing or modified evidence, including the first promotion."""
    if not run_id:
        raise HoldoutMismatchError("model has no training run with split evidence")
    client = MlflowClient()
    try:
        digest = client.get_run(run_id).data.tags.get(TAG)
        if not digest:
            raise HoldoutMismatchError("training run has no split manifest; retrain to record it")
        with tempfile.TemporaryDirectory() as tmp:
            path = client.download_artifacts(run_id, ARTIFACT, tmp)
            saved = SplitManifest.model_validate_json(Path(path).read_text(encoding="utf-8"))
    except (MlflowException, OSError, ValueError) as exc:
        if isinstance(exc, HoldoutMismatchError):
            raise
        raise HoldoutMismatchError(f"cannot verify training split evidence: {exc}") from exc
    if saved.fingerprint() != digest:
        raise HoldoutMismatchError("split manifest digest does not match the training run")
    saved.check(current)
