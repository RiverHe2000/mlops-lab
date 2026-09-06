"""The CLI must survive consoles that cannot encode MLflow's emoji run URLs (GBK, cp1252).

Found against the compose server on a Chinese-locale Windows box: `train --register` trained,
then died in MLflow's `set_terminated` print, so the model version never existed.
"""

from __future__ import annotations

import io
import sys

import pytest

from mlreg.cli import _harden_console_encoding

RUN_LINE = "\U0001f3c3 View run baseline at: http://localhost:5000/#/experiments/1/runs/abc"


def test_gbk_console_no_longer_raises_on_emoji(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = io.BytesIO()
    gbk_stdout = io.TextIOWrapper(raw, encoding="gbk", errors="strict")
    monkeypatch.setattr(sys, "stdout", gbk_stdout)
    with pytest.raises(UnicodeEncodeError):
        print(RUN_LINE)

    _harden_console_encoding()
    print(RUN_LINE)
    gbk_stdout.flush()

    assert gbk_stdout.errors == "backslashreplace"
    assert b"View run baseline at: http://localhost:5000" in raw.getvalue()


def test_utf8_console_is_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = io.BytesIO()
    utf8_stdout = io.TextIOWrapper(raw, encoding="utf-8", errors="strict")
    monkeypatch.setattr(sys, "stdout", utf8_stdout)

    _harden_console_encoding()
    print(RUN_LINE)
    utf8_stdout.flush()

    assert utf8_stdout.errors == "strict"
    assert RUN_LINE.encode("utf-8") in raw.getvalue()


def test_streams_without_reconfigure_are_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    class Plain:
        encoding = "gbk"

    monkeypatch.setattr(sys, "stdout", Plain())
    _harden_console_encoding()  # must not raise
