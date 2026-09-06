"""In-process latency benchmark of the hosting app (no network, no Docker).

Measures end-to-end /invocations latency through the ASGI test client for several batch
sizes and content types, so the numbers isolate codec + model cost from network and SageMaker
overhead. Writes a Markdown table.
"""

from __future__ import annotations

import argparse
import io
import json
import statistics
import time
from pathlib import Path

from fastapi.testclient import TestClient

from smdeploy.inference import ModelHandler
from smdeploy.model_io import load_metadata
from smdeploy.server import create_app
from smdeploy.synthetic import make_credit_frame


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repeats", type=int, default=200)
    args = ap.parse_args(argv)

    meta = load_metadata(args.model_dir)
    frame = make_credit_frame(1000, seed=11)[meta.feature_columns]
    app = create_app(ModelHandler(args.model_dir))
    rows_out = [
        "| Batch rows | Content type | p50 ms | p95 ms | p99 ms | rows / s |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    with TestClient(app) as client:
        assert client.get("/ping").status_code == 200
        for n in (1, 10, 100, 1000):
            sample = frame.head(n)
            payloads = {
                "text/csv": sample.to_csv(header=False, index=False, lineterminator="\n").encode(),
                "application/json": json.dumps(
                    {"instances": sample.to_dict(orient="records")}
                ).encode(),
            }
            for ctype, body in payloads.items():
                repeats = max(20, args.repeats // max(1, n // 10))
                lat: list[float] = []
                for _ in range(repeats):
                    t0 = time.perf_counter()
                    r = client.post(
                        "/invocations",
                        content=body,
                        headers={"Content-Type": ctype, "Accept": "application/json"},
                    )
                    lat.append((time.perf_counter() - t0) * 1000)
                    assert r.status_code == 200, r.text
                lat.sort()
                p50 = statistics.median(lat)
                p95 = lat[int(0.95 * (len(lat) - 1))]
                p99 = lat[int(0.99 * (len(lat) - 1))]
                rate = n / (p50 / 1000)
                rows_out.append(
                    f"| {n} | {ctype} | {p50:.2f} | {p95:.2f} | {p99:.2f} | {rate:,.0f} |"
                )
    text = "\n".join(rows_out) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    buf = io.StringIO()
    buf.write(text)
    print(buf.getvalue())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
