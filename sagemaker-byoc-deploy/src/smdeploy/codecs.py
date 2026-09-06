"""Request/response codecs for `/invocations`.

SageMaker passes the caller's `ContentType` and `Accept` headers straight through to the
container, so the container owns content negotiation. Supported:

    request   text/csv (no header by default, as SageMaker's own algorithms expect;
                        `text/csv; header=present` to include one — a headerless request whose
                        first row is the feature names is rejected with 400, never scored)
              application/json  {"instances": [...]}, {"dataframe_records": [...]},
                                {"dataframe_split": {...}}, {"inputs": {col: [...]}}, or a bare list
              application/jsonlines  one JSON object per line
    response  application/json (default), text/csv, application/jsonlines

Errors carry the HTTP status SageMaker expects: 400 for a bad body, 415 for an unsupported
request type, 406 for an unsatisfiable Accept.
"""

from __future__ import annotations

import csv
import io
import json
from typing import Any

import pandas as pd

CSV = "text/csv"
JSON = "application/json"
JSONLINES = "application/jsonlines"
SUPPORTED_REQUEST = (CSV, JSON, JSONLINES)
SUPPORTED_RESPONSE = (JSON, CSV, JSONLINES)


class CodecError(ValueError):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


def parse_media_type(header: str | None, default: str = CSV) -> tuple[str, dict[str, str]]:
    if not header or not header.strip():
        return default, {}
    media, _, rest = header.partition(";")
    params: dict[str, str] = {}
    for part in rest.split(";"):
        key, _, value = part.strip().partition("=")
        if key:
            params[key.strip().lower()] = value.strip().strip('"').lower()
    return media.strip().lower(), params


def decode_request(body: bytes, content_type: str | None, columns: list[str]) -> pd.DataFrame:
    if not body or not body.strip():
        raise CodecError("empty request body")
    media, params = parse_media_type(content_type)
    text = body.decode(params.get("charset", "utf-8"))
    if media == CSV:
        return _decode_csv(text, columns, header_present=params.get("header") == "present")
    if media == JSON:
        return _decode_json(text, columns)
    if media == JSONLINES:
        return _decode_jsonlines(text, columns)
    raise CodecError(f"unsupported content type {media!r}; use one of {SUPPORTED_REQUEST}", 415)


def _decode_csv(text: str, columns: list[str], *, header_present: bool) -> pd.DataFrame:
    try:
        if header_present:
            frame = pd.read_csv(io.StringIO(text))
        else:
            frame = pd.read_csv(io.StringIO(text), header=None)
            if frame.shape[1] != len(columns):
                raise CodecError(
                    f"csv row width {frame.shape[1]} does not match the model's "
                    f"{len(columns)} feature columns {columns}"
                )
            if len(frame) and _is_feature_header(frame.iloc[0].tolist(), columns):
                # Scoring the header line as a row would silently return a prediction for a
                # record made of imputed numerics and unseen categories. Refuse instead.
                raise CodecError(
                    "the first csv row is the feature header; send it as "
                    "'text/csv; header=present' or drop the header line"
                )
            frame.columns = pd.Index(columns)
    except pd.errors.ParserError as exc:
        raise CodecError(f"malformed csv: {exc}") from exc
    return frame


def _is_feature_header(cells: list[Any], columns: list[str]) -> bool:
    """True when a row is exactly the model's feature names (a header sent without the flag)."""
    return sorted(str(c).strip() for c in cells) == sorted(columns)


def _decode_json(text: str, columns: list[str]) -> pd.DataFrame:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CodecError(f"malformed json: {exc.msg}") from exc
    if isinstance(payload, dict):
        if "instances" in payload:
            return _records_or_rows(payload["instances"], columns)
        if "dataframe_records" in payload:
            return _records_or_rows(payload["dataframe_records"], columns)
        if "dataframe_split" in payload:
            split = payload["dataframe_split"]
            if not isinstance(split, dict) or "data" not in split:
                raise CodecError("dataframe_split needs 'columns' and 'data'")
            return pd.DataFrame(split["data"], columns=split.get("columns", columns))
        if "inputs" in payload and isinstance(payload["inputs"], dict):
            return pd.DataFrame(payload["inputs"])
        raise CodecError(
            "json object must contain one of: instances, dataframe_records, dataframe_split, inputs"
        )
    if isinstance(payload, list):
        return _records_or_rows(payload, columns)
    raise CodecError("json body must be an object or a list")


def _records_or_rows(items: Any, columns: list[str]) -> pd.DataFrame:
    if not isinstance(items, list) or not items:
        raise CodecError("expected a non-empty list of records")
    if all(isinstance(r, dict) for r in items):
        return pd.DataFrame.from_records(items)
    if all(isinstance(r, list) for r in items):
        widths = {len(r) for r in items}
        if widths != {len(columns)}:
            raise CodecError(f"row width {sorted(widths)} does not match {len(columns)} columns")
        return pd.DataFrame(items, columns=columns)
    raise CodecError("records must all be objects or all be arrays")


def _decode_jsonlines(text: str, columns: list[str]) -> pd.DataFrame:
    rows: list[Any] = []
    for n, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise CodecError(f"malformed jsonlines at line {n}: {exc.msg}") from exc
    return _records_or_rows(rows, columns)


def negotiate_accept(accept: str | None) -> str:
    if not accept or not accept.strip():
        return JSON
    offered: list[str] = []
    for part in accept.split(","):
        media, _ = parse_media_type(part, default="")
        offered.append(media)
    if "*/*" in offered or "application/*" in offered:
        return JSON
    for media in offered:
        if media in SUPPORTED_RESPONSE:
            return media
    raise CodecError(f"cannot produce {accept!r}; supported: {SUPPORTED_RESPONSE}", 406)


def encode_response(predictions: pd.DataFrame, accept: str) -> tuple[bytes, str]:
    records = [
        {"probability": float(p), "label": int(label)}
        for p, label in zip(predictions["probability"], predictions["label"], strict=True)
    ]
    if accept == JSON:
        return json.dumps({"predictions": records}).encode("utf-8"), JSON
    if accept == JSONLINES:
        return ("\n".join(json.dumps(r) for r in records) + "\n").encode("utf-8"), JSONLINES
    if accept == CSV:
        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator="\n")
        for r in records:
            writer.writerow([f"{r['probability']:.6f}", r["label"]])
        return buf.getvalue().encode("utf-8"), CSV
    raise CodecError(f"unsupported response type {accept!r}", 406)
