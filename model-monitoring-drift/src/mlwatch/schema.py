"""Inference capture and label records (JSONL), the same shape a SageMaker data-capture or
an application log would produce after light normalisation.

    capture: {"request_id", "ts", "model_version", "features": {...}, "score", "decision"}
    labels:  {"request_id", "label", "label_ts"}   — arrive later, joined on request_id
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

FeatureValue = float | int | str | bool | None

META_COLUMNS = ["request_id", "ts", "model_version", "score", "decision"]


class CaptureRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    request_id: str
    ts: datetime
    model_version: str = "unknown"
    features: dict[str, FeatureValue]
    score: float = Field(ge=0.0, le=1.0)
    decision: str | None = None
    latency_ms: float | None = None

    @field_validator("ts")
    @classmethod
    def _aware(cls, v: datetime) -> datetime:
        return v if v.tzinfo is not None else v.replace(tzinfo=UTC)


class LabelRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    request_id: str
    label: int = Field(ge=0, le=1)
    label_ts: datetime | None = None

    @field_validator("label_ts")
    @classmethod
    def _aware(cls, v: datetime | None) -> datetime | None:
        if v is None or v.tzinfo is not None:
            return v
        return v.replace(tzinfo=UTC)


def read_jsonl[T: BaseModel](path: Path, model: type[T]) -> list[T]:
    out: list[T] = []
    with Path(path).open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                out.append(model.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"{path}:{n}: {exc}") from exc
    return out


def write_jsonl(path: Path, records: Iterable[BaseModel]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r.model_dump(mode="json"), separators=(",", ":")) + "\n")
            n += 1
    return n


def captures_to_frame(records: Sequence[CaptureRecord]) -> pd.DataFrame:
    """One row per request: metadata columns first, then one column per feature."""
    rows: list[dict[str, Any]] = []
    for r in records:
        row: dict[str, Any] = {
            "request_id": r.request_id,
            "ts": r.ts,
            "model_version": r.model_version,
            "score": r.score,
            "decision": r.decision,
        }
        row.update(r.features)
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=META_COLUMNS)
    frame = pd.DataFrame(rows)
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
    return frame


def labels_to_frame(records: Sequence[LabelRecord]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=["request_id", "label", "label_ts"])
    frame = pd.DataFrame([r.model_dump() for r in records])
    frame = frame.drop_duplicates(subset="request_id", keep="last")
    frame["label_ts"] = pd.to_datetime(frame["label_ts"], utc=True)
    return frame


def select_window(
    records: Sequence[CaptureRecord], start: datetime | None, end: datetime | None
) -> list[CaptureRecord]:
    """[start, end) on the capture timestamp; None bounds are open."""
    out: list[CaptureRecord] = []
    for r in records:
        if start is not None and r.ts < start:
            continue
        if end is not None and r.ts >= end:
            continue
        out.append(r)
    return out


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [str(c) for c in frame.columns if c not in META_COLUMNS]
