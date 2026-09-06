"""Command line: validate | train | register | promote | card | compare | serve-check."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

from . import metrics as M
from . import registry as R
from . import tracking as T
from .config import DataSchema, TrainingConfig
from .data import Dataset, load_frame, make_split
from .gate import GateResult, PromotionPolicy, evaluate_gate
from .model_card import render_model_card
from .pyfunc import score_with
from .schema import validate_frame
from .stats import bootstrap_ci, paired_bootstrap_delta
from .train import train

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_HOLD = 2


def _load(cfg_path: str) -> tuple[TrainingConfig, DataSchema]:
    cfg = TrainingConfig.from_yaml(cfg_path)
    return cfg, DataSchema.from_yaml(cfg.schema_path)


def _write(out: Path | None, name: str, payload: str) -> None:
    if out is None:
        return
    out.mkdir(parents=True, exist_ok=True)
    (out / name).write_text(payload, encoding="utf-8")


def cmd_validate(args: argparse.Namespace) -> int:
    schema = DataSchema.from_yaml(args.schema)
    dtypes = {
        schema.id_column: "str",
        **dict.fromkeys(schema.categorical_features + schema.protected_columns, "str"),
    }
    df = pd.read_csv(args.data, dtype=dtypes)
    report = validate_frame(df, schema)
    print(report.summary())
    return EXIT_OK if report.ok else EXIT_ERROR


def cmd_train(args: argparse.Namespace) -> int:
    cfg, _ = _load(args.config)
    result = train(cfg, n_boot=args.n_boot)
    summary = result.summary()
    if args.register:
        rv = R.register(
            result.model_uri,
            cfg.tracking.registered_model,
            tags={
                "data.fingerprint": result.data_fingerprint,
                "mlreg.config_hash": result.config_hash,
                "mlreg.config_name": cfg.name,
                "decision.threshold": f"{result.threshold:.4f}",
                "test.auc": f"{result.metrics_test['auc']:.4f}",
            },
        )
        if args.alias:
            R.set_alias(cfg.tracking.registered_model, args.alias, rv.version)
            rv = R.get_version(cfg.tracking.registered_model, rv.version)
        summary["registered"] = rv.to_dict()
    text = json.dumps(summary, indent=2, default=str)
    print(text)
    _write(Path(args.out) if args.out else None, "train_summary.json", text)
    return EXIT_OK


def cmd_register(args: argparse.Namespace) -> int:
    cfg, _ = _load(args.config)
    T.configure(cfg.tracking)
    tags = dict(kv.split("=", 1) for kv in args.tag)
    rv = R.register(args.model_uri, cfg.tracking.registered_model, tags=tags)
    if args.alias:
        R.set_alias(cfg.tracking.registered_model, args.alias, rv.version)
        rv = R.get_version(cfg.tracking.registered_model, rv.version)
    print(json.dumps(rv.to_dict(), indent=2))
    return EXIT_OK


def _holdout(cfg: TrainingConfig, schema: DataSchema) -> Dataset:
    return make_split(load_frame(cfg.data_path, schema), schema, cfg.split)


def _score_versions(
    cfg: TrainingConfig, ds: Dataset, versions: list[int]
) -> dict[int, pd.DataFrame]:
    return {
        v: score_with(R.load_version(cfg.tracking.registered_model, v), ds.X_test) for v in versions
    }


def _threshold_for(version: R.RegisteredVersion) -> float:
    tag = version.tags.get("decision.threshold")
    if tag is not None:
        return float(tag)
    if version.run_id:
        import mlflow

        param = mlflow.get_run(version.run_id).data.params.get("decision.threshold")
        if param is not None:
            return float(param)
    return 0.5


def cmd_promote(args: argparse.Namespace) -> int:
    cfg, schema = _load(args.config)
    T.configure(cfg.tracking)
    name = cfg.tracking.registered_model
    policy = PromotionPolicy.from_yaml(args.policy)
    candidate = R.get_version(name, args.candidate_version)
    champion = R.get_alias(name, args.champion_alias)
    if champion is not None and champion.version == candidate.version:
        print(f"v{candidate.version} already holds @{args.champion_alias}; nothing to do")
        return EXIT_OK
    ds = _holdout(cfg, schema)
    versions = [candidate.version] + ([champion.version] if champion else [])
    scores = _score_versions(cfg, ds, versions)
    threshold = _threshold_for(candidate)
    p_chal = scores[candidate.version]["pd"].to_numpy()
    p_champ = scores[champion.version]["pd"].to_numpy() if champion else None
    same: bool | None = None
    if champion is not None:
        same = (
            candidate.tags.get("data.fingerprint") == ds.fingerprint
            and champion.tags.get("data.fingerprint") == ds.fingerprint
        )
    slices = {c: ds.protected_test[c] for c in schema.protected_columns}
    gate = evaluate_gate(
        policy, ds.y_test, p_chal, p_champ, threshold, slices=slices, same_snapshot=same
    )

    out = Path(args.out) if args.out else None
    _write(out, "gate_report.json", json.dumps(gate.to_dict(), indent=2, default=str))
    _write(out, "gate_report.md", gate.to_markdown())
    card = _card_for(cfg, schema, candidate, gate, ds, p_chal, threshold, args.approved_by)
    _write(out, "model_card.md", card)
    print(gate.to_markdown())

    R.set_tags(
        name,
        candidate.version,
        {
            "gate.decision": gate.decision,
            "gate.policy": Path(args.policy).name,
            "gate.n_holdout": str(gate.n_holdout),
        },
    )
    if gate.decision == "PROMOTE" and not args.dry_run:
        evidence = {"gate.report": str(out / "gate_report.json") if out else "stdout"}
        promoted, displaced = R.promote(
            name,
            candidate.version,
            approved_by=args.approved_by,
            evidence=evidence,
            alias=args.champion_alias,
        )
        msg = f"\nPromoted {name} v{promoted.version} → @{args.champion_alias}"
        if displaced:
            msg += f" (displaced v{displaced.version} → @previous)"
        print(msg)
        return EXIT_OK
    if gate.decision == "PROMOTE":
        print("\nDry run: gate passed, alias not moved.")
        return EXIT_OK
    print(f"\nHOLD: {len(gate.failed)} check(s) failed: {', '.join(c.name for c in gate.failed)}")
    return EXIT_HOLD


def _card_for(
    cfg: TrainingConfig,
    schema: DataSchema,
    version: R.RegisteredVersion,
    gate: GateResult | None,
    ds: Dataset,
    p: Any,
    threshold: float,
    approved_by: str | None,
) -> str:
    y = ds.y_test
    metrics_test = M.summarize(y, p, threshold)
    cis = {
        "auc": bootstrap_ci(y, p, M.roc_auc, n_boot=500, seed=1).to_dict(),
        "brier": bootstrap_ci(y, p, M.brier, n_boot=500, seed=2).to_dict(),
        "ks": bootstrap_ci(y, p, M.ks_statistic, n_boot=500, seed=3).to_dict(),
    }
    slices = {
        c: M.slice_metrics(y, p, ds.protected_test[c], threshold) for c in schema.protected_columns
    }
    cv_mean, cv_std = _cv_from_run(version.run_id)
    return render_model_card(
        cfg=cfg,
        schema=schema,
        metrics_test=metrics_test,
        test_ci=cis,
        cv_mean=cv_mean,
        cv_std=cv_std,
        slices=slices,
        threshold=threshold,
        run_id=version.run_id or "n/a",
        data_fingerprint=ds.fingerprint,
        gate=gate,
        registered=version,
        approved_by=approved_by,
        git_sha=T.git_sha(),
    )


def _cv_from_run(run_id: str | None) -> tuple[dict[str, float], dict[str, float]]:
    if not run_id:
        return {}, {}
    import mlflow

    metrics = mlflow.get_run(run_id).data.metrics
    mean = {k[len("cv_mean_") :]: float(v) for k, v in metrics.items() if k.startswith("cv_mean_")}
    std = {k[len("cv_std_") :]: float(v) for k, v in metrics.items() if k.startswith("cv_std_")}
    return mean, std


def cmd_card(args: argparse.Namespace) -> int:
    cfg, schema = _load(args.config)
    T.configure(cfg.tracking)
    version = R.get_version(cfg.tracking.registered_model, args.version)
    ds = _holdout(cfg, schema)
    threshold = _threshold_for(version)
    p = _score_versions(cfg, ds, [version.version])[version.version]["pd"].to_numpy()
    card = _card_for(cfg, schema, version, None, ds, p, threshold, args.approved_by)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(card, encoding="utf-8")
    print(card)
    return EXIT_OK


def cmd_compare(args: argparse.Namespace) -> int:
    cfg, schema = _load(args.config)
    T.configure(cfg.tracking)
    a, b = args.versions
    ds = _holdout(cfg, schema)
    scores = _score_versions(cfg, ds, [a, b])
    pa = scores[a]["pd"].to_numpy()
    pb = scores[b]["pd"].to_numpy()
    rows = []
    for key, fn in [
        ("auc", M.roc_auc),
        ("ks", M.ks_statistic),
        ("brier", M.brier),
        ("log_loss", M.log_loss),
    ]:
        d = paired_bootstrap_delta(ds.y_test, pa, pb, fn, n_boot=args.n_boot, seed=0)
        rows.append(
            {
                "metric": key,
                f"v{a}": fn(ds.y_test, pa),
                f"v{b}": fn(ds.y_test, pb),
                "delta": d.estimate,
                "ci_lower": d.lower,
                "ci_upper": d.upper,
            }
        )
    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return EXIT_OK


def cmd_serve_check(args: argparse.Namespace) -> int:
    """Prove the registered artefact is servable: load by alias, validate a serving payload."""
    from mlflow.models import convert_input_example_to_serving_input, validate_serving_input

    cfg, schema = _load(args.config)
    T.configure(cfg.tracking)
    uri = f"models:/{cfg.tracking.registered_model}@{args.alias}"
    frame = load_frame(cfg.data_path, schema)
    example = frame[schema.feature_columns].head(3)
    payload = convert_input_example_to_serving_input(example)
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore", FutureWarning
        )  # deprecated in 3.13 in favour of mlflow.models.predict
        result = validate_serving_input(uri, payload)
    sample = pd.DataFrame(result).to_string(index=False)
    print(f"{uri} accepted a serving payload; sample output:\n{sample}")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="mlreg", description="Credit PD model lifecycle on MLflow.")
    sub = ap.add_subparsers(dest="command", required=True)

    v = sub.add_parser("validate", help="validate a CSV against the data contract")
    v.add_argument("--data", required=True)
    v.add_argument("--schema", required=True)
    v.set_defaults(func=cmd_validate)

    t = sub.add_parser("train", help="train, evaluate and log a model")
    t.add_argument("--config", required=True)
    t.add_argument("--n-boot", type=int, default=1000)
    t.add_argument("--register", action="store_true", help="also create a registry version")
    t.add_argument("--alias", default=None, help="alias to set on the new version, e.g. challenger")
    t.add_argument("--out", default=None, help="directory for train_summary.json")
    t.set_defaults(func=cmd_train)

    r = sub.add_parser("register", help="register a logged model URI")
    r.add_argument("--config", required=True)
    r.add_argument("--model-uri", required=True)
    r.add_argument("--alias", default=None)
    r.add_argument("--tag", action="append", default=[], help="key=value (repeatable)")
    r.set_defaults(func=cmd_register)

    p = sub.add_parser("promote", help="gate a candidate version against the champion")
    p.add_argument("--config", required=True)
    p.add_argument("--policy", required=True)
    p.add_argument("--candidate-version", type=int, required=True)
    p.add_argument("--champion-alias", default=R.CHAMPION)
    p.add_argument("--approved-by", default="ci")
    p.add_argument(
        "--out", default=None, help="directory for gate_report.{json,md} and model_card.md"
    )
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_promote)

    c = sub.add_parser("card", help="render the model card for a registry version")
    c.add_argument("--config", required=True)
    c.add_argument("--version", type=int, required=True)
    c.add_argument("--approved-by", default=None)
    c.add_argument("--out", default=None)
    c.set_defaults(func=cmd_card)

    cmp_ = sub.add_parser("compare", help="paired bootstrap comparison of two versions")
    cmp_.add_argument("--config", required=True)
    cmp_.add_argument("--versions", type=int, nargs=2, required=True)
    cmp_.add_argument("--n-boot", type=int, default=1000)
    cmp_.set_defaults(func=cmd_compare)

    s = sub.add_parser("serve-check", help="validate a serving payload against the aliased model")
    s.add_argument("--config", required=True)
    s.add_argument("--alias", default=R.CHAMPION)
    s.set_defaults(func=cmd_serve_check)
    return ap


def _harden_console_encoding() -> None:
    """Stop non-UTF-8 consoles (Windows GBK / cp1252) from crashing on MLflow's run URLs.

    Against an HTTP tracking server MLflow prints an emoji-prefixed "View run ... at:" line
    from `set_terminated`, i.e. after training but before registration. On a GBK console that
    is a UnicodeEncodeError that fails the whole `train --register`. Replacing unencodable
    characters is the least surprising behaviour; UTF-8 consoles are left untouched.
    """
    for stream in (sys.stdout, sys.stderr):
        encoding = str(getattr(stream, "encoding", None) or "").lower().replace("-", "")
        reconfigure = getattr(stream, "reconfigure", None)
        if encoding != "utf8" and reconfigure is not None:
            reconfigure(errors="backslashreplace")


def main(argv: list[str] | None = None) -> int:
    _harden_console_encoding()
    args = build_parser().parse_args(argv)
    code: int = args.func(args)
    return code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
