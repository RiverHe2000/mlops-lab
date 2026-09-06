from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from smdeploy import codecs
from smdeploy.codecs import (
    CodecError,
    decode_request,
    encode_response,
    negotiate_accept,
    parse_media_type,
)
from smdeploy.inference import InputError, ModelHandler, Predictor
from smdeploy.model_io import load_model
from smdeploy.synthetic import FEATURES
from smdeploy.training import TrainingReport

COLS = ["a", "b", "c"]


def test_parse_media_type() -> None:
    assert parse_media_type(None) == ("text/csv", {})
    assert parse_media_type("Application/JSON; charset=UTF-8") == (
        "application/json",
        {"charset": "utf-8"},
    )
    assert parse_media_type('text/csv; header="present"') == ("text/csv", {"header": "present"})


def test_decode_csv_variants() -> None:
    frame = decode_request(b"1,2,x\n3,4,y\n", "text/csv", COLS)
    assert frame.columns.tolist() == COLS and len(frame) == 2
    with_header = decode_request(b"a,b,c\n1,2,x\n", "text/csv; header=present", COLS)
    assert with_header.columns.tolist() == COLS
    with pytest.raises(CodecError, match="width") as exc:
        decode_request(b"1,2\n", "text/csv", COLS)
    assert exc.value.status_code == 400
    # A header sent without `header=present` must not be scored as a record.
    with pytest.raises(CodecError, match="header=present") as exc:
        decode_request(b"a,b,c\n1,2,x\n", "text/csv", COLS)
    assert exc.value.status_code == 400
    with pytest.raises(CodecError, match="header=present"):  # any column order
        decode_request(b"c,b,a\n1,2,x\n", "text/csv", COLS)
    with pytest.raises(CodecError, match="empty"):
        decode_request(b"  ", "text/csv", COLS)
    with pytest.raises(CodecError, match="malformed csv"):
        decode_request(b'1,"2,x\n3,4,y\n', "text/csv", COLS)


def test_decode_json_variants() -> None:
    records = [{"a": 1, "b": 2, "c": "x"}, {"a": 3, "b": 4, "c": "y"}]
    for payload in (
        {"instances": records},
        {"instances": [[1, 2, "x"], [3, 4, "y"]]},
        {"dataframe_records": records},
        {"dataframe_split": {"columns": COLS, "data": [[1, 2, "x"], [3, 4, "y"]]}},
        {"inputs": {"a": [1, 3], "b": [2, 4], "c": ["x", "y"]}},
        records,
        [[1, 2, "x"], [3, 4, "y"]],
    ):
        frame = decode_request(json.dumps(payload).encode(), "application/json", COLS)
        assert frame.columns.tolist() == COLS and len(frame) == 2, payload
    for bad, message in (
        ({"foo": 1}, "must contain"),
        ({"dataframe_split": {"columns": COLS}}, "dataframe_split"),
        ({"instances": []}, "non-empty"),
        ({"instances": [[1, 2]]}, "row width"),
        ({"instances": [1, 2]}, "objects or all be arrays"),
        (42, "object or a list"),
    ):
        with pytest.raises(CodecError, match=message):
            decode_request(json.dumps(bad).encode(), "application/json", COLS)
    with pytest.raises(CodecError, match="malformed json"):
        decode_request(b"{not json", "application/json", COLS)


def test_decode_jsonlines_and_unsupported() -> None:
    body = b'{"a": 1, "b": 2, "c": "x"}\n\n{"a": 3, "b": 4, "c": "y"}\n'
    assert len(decode_request(body, "application/jsonlines", COLS)) == 2
    with pytest.raises(CodecError, match="line 2"):
        decode_request(b'{"a": 1}\n{bad\n', "application/jsonlines", COLS)
    with pytest.raises(CodecError) as exc:
        decode_request(b"<xml/>", "application/xml", COLS)
    assert exc.value.status_code == 415


def test_negotiate_accept() -> None:
    assert negotiate_accept(None) == codecs.JSON
    assert negotiate_accept("*/*") == codecs.JSON
    assert negotiate_accept("text/csv") == codecs.CSV
    assert negotiate_accept("application/jsonlines, text/csv") == codecs.JSONLINES
    assert negotiate_accept("image/png, text/csv;q=0.9") == codecs.CSV
    with pytest.raises(CodecError) as exc:
        negotiate_accept("image/png")
    assert exc.value.status_code == 406


def test_encode_response() -> None:
    preds = pd.DataFrame({"probability": [0.25, 0.75], "label": [0, 1]})
    body, media = encode_response(preds, codecs.JSON)
    assert media == codecs.JSON and json.loads(body)["predictions"][1] == {
        "probability": 0.75,
        "label": 1,
    }
    body, media = encode_response(preds, codecs.CSV)
    assert body == b"0.250000,0\n0.750000,1\n"
    body, media = encode_response(preds, codecs.JSONLINES)
    assert body.count(b"\n") == 2 and json.loads(body.splitlines()[0])["label"] == 0
    with pytest.raises(CodecError):
        encode_response(preds, "image/png")


def test_predictor_alignment_and_threshold(trained: TrainingReport, frame: pd.DataFrame) -> None:
    predictor = Predictor(load_model(trained.model_dir))
    rows = frame.head(5)
    out = predictor.predict(rows)  # extra columns (id, target) are dropped
    assert out.columns.tolist() == ["probability", "label"] and len(out) == 5
    assert set(out["label"]) <= {0, 1}
    shuffled = rows[list(reversed(rows.columns))]
    np.testing.assert_allclose(predictor.predict(shuffled)["probability"], out["probability"])
    as_strings = rows.astype(str)  # csv payloads arrive stringly typed
    np.testing.assert_allclose(
        predictor.predict(as_strings)["probability"], out["probability"], atol=1e-9
    )
    with_missing = rows.copy()
    with_missing.loc[with_missing.index[0], "income"] = np.nan
    with_missing.loc[with_missing.index[1], "housing"] = np.nan
    assert len(predictor.predict(with_missing)) == 5
    with pytest.raises(InputError, match="missing feature"):
        predictor.predict(rows.drop(columns=["income"]))
    with pytest.raises(InputError, match="no rows"):
        predictor.predict(rows.head(0))
    strict = Predictor(load_model(trained.model_dir), threshold=0.999)
    assert (strict.predict(rows)["label"] == 0).all()
    with pytest.raises(ValueError, match="threshold"):
        Predictor(load_model(trained.model_dir), threshold=2.0)
    assert predictor.feature_columns == FEATURES


def test_model_handler(trained: TrainingReport, tmp_path: Path) -> None:
    fresh = ModelHandler(trained.model_dir)
    assert fresh.ready is False
    handler = ModelHandler(trained.model_dir)
    with pytest.raises(RuntimeError, match="not loaded"):
        _ = handler.predictor
    first = handler.load()
    assert handler.load() is first
    assert handler.ready is True
    assert handler.error is None
    broken = ModelHandler(tmp_path / "missing")
    with pytest.raises(FileNotFoundError):
        broken.load()
    assert broken.ready is False
    assert broken.error is not None and "FileNotFoundError" in broken.error
