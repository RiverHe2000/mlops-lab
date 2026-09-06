"""Command line: baseline | run | simulate | evaluate | serve.

`run` exits 0 when no action is needed, 2 for INVESTIGATE and 3 for RETRAIN, so a scheduled
job can branch on it (open a ticket, trigger the training pipeline).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from .baseline import Baseline, build_baseline
from .config import AlertPolicy
from .evaluate import DEFAULT_GRID, evaluate_monitor, to_markdown
from .monitor import load_capture_frame, load_labels_frame, run_window
from .policy import MonitorState
from .schema import feature_columns
from .simulate import SCENARIOS, simulate_dataset


def _dt(value: str | None) -> datetime | None:
    return None if value is None else datetime.fromisoformat(value)


def cmd_baseline(args: argparse.Namespace) -> int:
    frame = load_capture_frame(Path(args.capture), _dt(args.start), _dt(args.end))
    labels_frame = load_labels_frame(Path(args.labels) if args.labels else None)
    labels: pd.Series | None = None
    if labels_frame is not None:
        lookup = labels_frame.set_index("request_id")["label"]
        labels = frame["request_id"].map(lookup)
    baseline = build_baseline(
        frame,
        feature_columns=args.features.split(",") if args.features else feature_columns(frame),
        threshold=args.threshold,
        labels=labels,
        model_version=args.model_version or str(frame["model_version"].mode().iat[0]),
    )
    baseline.save(Path(args.out))
    print(
        json.dumps(
            {
                "out": args.out,
                "n": baseline.n,
                "features": baseline.feature_types,
                "performance": baseline.performance,
            },
            indent=2,
        )
    )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    baseline = Baseline.load(Path(args.baseline))
    policy = AlertPolicy.from_yaml(Path(args.policy))
    state_path = Path(args.state) if args.state else None
    state = MonitorState.load(state_path) if state_path else MonitorState()
    frame = load_capture_frame(Path(args.capture), _dt(args.start), _dt(args.end))
    labels = load_labels_frame(Path(args.labels) if args.labels else None)
    as_of = _dt(args.as_of)
    if labels is not None and as_of is not None:
        cutoff = pd.Timestamp(as_of)
        labels = labels[labels["label_ts"].isna() | (labels["label_ts"] <= cutoff)]
    report = run_window(
        baseline,
        frame,
        labels,
        policy,
        state,
        window_id=args.window_id or Path(args.capture).stem,
        n_boot=args.n_boot,
    )
    if state_path:
        state.save(state_path)
    if args.out:
        report.save(Path(args.out))
    print(
        report.to_markdown()
        if args.format == "md"
        else json.dumps(report.to_dict(), indent=2, default=str)
    )
    return report.exit_code


def cmd_simulate(args: argparse.Namespace) -> int:
    manifest = simulate_dataset(
        Path(args.out),
        scenario=args.scenario,
        magnitude=args.magnitude,
        windows=args.windows,
        n_baseline=args.n_baseline,
        n_window=args.n_window,
        seed=args.seed,
        threshold=args.threshold,
    )
    print(json.dumps(manifest.to_dict(), indent=2))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    policy = AlertPolicy.from_yaml(Path(args.policy))
    grid = DEFAULT_GRID
    if args.scenario:
        grid = {s: DEFAULT_GRID[s] for s in args.scenario}
    rows = evaluate_monitor(
        policy,
        grid=grid,
        seeds=args.seeds,
        n_baseline=args.n_baseline,
        n_window=args.n_window,
        with_labels=not args.no_labels,
        n_boot=args.n_boot,
    )
    table = to_markdown(rows)
    print(table)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(table, encoding="utf-8")
        out.with_suffix(".json").write_text(
            json.dumps([r.to_dict() for r in rows], indent=2), encoding="utf-8"
        )
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .exporter import create_app

    app = create_app(
        Path(args.baseline),
        Path(args.policy),
        state_path=Path(args.state) if args.state else None,
        allowed_root=Path(args.allowed_root) if args.allowed_root else None,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="mlwatch", description="Production model monitoring.")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("baseline", help="build the reference profile from captured, scored data")
    p.add_argument("--capture", required=True)
    p.add_argument("--labels", default=None)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--features", default=None, help="comma-separated; default: every non-metadata column"
    )
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--model-version", default=None)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.set_defaults(func=cmd_baseline)

    p = sub.add_parser("run", help="monitor one window (exit 0 none / 2 investigate / 3 retrain)")
    p.add_argument("--baseline", required=True)
    p.add_argument("--capture", required=True)
    p.add_argument("--labels", default=None)
    p.add_argument("--policy", required=True)
    p.add_argument("--state", default=None, help="JSON file carrying consecutive-window counts")
    p.add_argument("--out", default=None, help="directory for report.json / report.md")
    p.add_argument("--window-id", default=None)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--as-of", default=None, help="only use labels observed up to this time")
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--format", choices=["md", "json"], default="md")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("simulate", help="generate baseline + drifted windows with labels")
    p.add_argument("--scenario", choices=SCENARIOS, default="covariate_shift")
    p.add_argument("--magnitude", type=float, default=1.0)
    p.add_argument("--windows", type=int, default=3)
    p.add_argument("--n-baseline", type=int, default=5000)
    p.add_argument("--n-window", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser(
        "evaluate", help="detection power / false alarms of the policy on simulations"
    )
    p.add_argument("--policy", required=True)
    p.add_argument("--scenario", nargs="*", choices=SCENARIOS, default=None)
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--n-baseline", type=int, default=4000)
    p.add_argument("--n-window", type=int, default=1000)
    p.add_argument("--n-boot", type=int, default=50)
    p.add_argument("--no-labels", action="store_true")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("serve", help="Prometheus exporter with POST /run")
    p.add_argument("--baseline", required=True)
    p.add_argument("--policy", required=True)
    p.add_argument("--state", default=None)
    p.add_argument("--allowed-root", default=None)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=9108)
    p.set_defaults(func=cmd_serve)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    code: int = args.func(args)
    return code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
