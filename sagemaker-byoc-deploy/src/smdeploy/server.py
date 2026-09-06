"""The hosting server: SageMaker inference container contract on FastAPI.

    GET  /ping                   200 once the model is loaded and verified, 503 otherwise
    POST /invocations            content negotiation via Content-Type / Accept
    GET  /execution-parameters   batch-transform tuning hints

SageMaker also forwards `X-Amzn-SageMaker-Custom-Attributes`; it is echoed back so callers
can correlate. Errors are JSON with the status the platform expects (4xx = client, 5xx = us).
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from . import __version__
from .codecs import CodecError, decode_request, encode_response, negotiate_accept
from .inference import InputError, ModelHandler

log = logging.getLogger("smdeploy.server")
CUSTOM_ATTRIBUTES = "X-Amzn-SageMaker-Custom-Attributes"
REQUEST_ID = "X-Request-Id"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key in (
            "request_id",
            "rows",
            "latency_ms",
            "status",
            "content_type",
            "accept",
            "sha256",
        ):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        return json.dumps(payload)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger("smdeploy")
    root.handlers[:] = [handler]
    root.setLevel(level)
    root.propagate = False


def create_app(handler: ModelHandler, *, max_payload_mb: int = 6, workers: int = 1) -> FastAPI:
    max_bytes = max_payload_mb * 1024 * 1024

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        with contextlib.suppress(Exception):  # /ping stays 503 and reports the reason
            handler.load()
        yield

    app = FastAPI(title="smdeploy inference", version=__version__, lifespan=lifespan)

    @app.get("/ping")
    async def ping() -> Response:
        if handler.ready:
            meta = handler.predictor.meta
            return JSONResponse(
                {
                    "status": "ok",
                    "model_sha256": meta.model_sha256[:12],
                    "model_kind": meta.model_kind,
                }
            )
        return JSONResponse({"status": "loading", "error": handler.error}, status_code=503)

    @app.get("/execution-parameters")
    async def execution_parameters() -> Response:
        return JSONResponse(
            {
                "MaxConcurrentTransforms": workers,
                "BatchStrategy": "MultiRecord",
                "MaxPayloadInMB": max_payload_mb,
            }
        )

    @app.post("/invocations")
    async def invocations(request: Request) -> Response:
        started = time.perf_counter()
        request_id = request.headers.get(REQUEST_ID) or uuid.uuid4().hex
        content_type = request.headers.get("content-type")
        accept_header = request.headers.get("accept")
        extra_headers = {REQUEST_ID: request_id}
        custom = request.headers.get(CUSTOM_ATTRIBUTES)
        if custom is not None:
            extra_headers[CUSTOM_ATTRIBUTES] = custom

        def fail(status: int, message: str) -> Response:
            _log_invocation(request_id, 0, started, status, content_type, accept_header)
            return JSONResponse(
                {"error": message, "request_id": request_id},
                status_code=status,
                headers=extra_headers,
            )

        if not handler.ready:
            return fail(503, f"model not loaded: {handler.error}")
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > max_bytes:
            return fail(413, f"payload exceeds {max_payload_mb} MB")
        body = await request.body()
        if len(body) > max_bytes:
            return fail(413, f"payload exceeds {max_payload_mb} MB")
        try:
            accept = negotiate_accept(accept_header)
            frame = decode_request(body, content_type, handler.predictor.feature_columns)
            predictions = handler.predictor.predict(frame)
            payload, media = encode_response(predictions, accept)
        except CodecError as exc:
            return fail(exc.status_code, str(exc))
        except InputError as exc:
            return fail(400, str(exc))
        except Exception:
            log.exception("invocation failed", extra={"request_id": request_id})
            return fail(500, "internal error while scoring; see container logs")
        _log_invocation(request_id, len(predictions), started, 200, content_type, accept)
        return Response(content=payload, media_type=media, headers=extra_headers)

    return app


def _log_invocation(
    request_id: str,
    rows: int,
    started: float,
    status: int,
    content_type: str | None,
    accept: str | None,
) -> None:
    log.info(
        "invocation",
        extra={
            "request_id": request_id,
            "rows": rows,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "status": status,
            "content_type": content_type,
            "accept": accept,
        },
    )
