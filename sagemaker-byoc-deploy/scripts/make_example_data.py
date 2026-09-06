"""Write the example CSVs used by the README, the local simulator and the CD smoke test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from smdeploy.synthetic import FEATURES, TARGET, make_credit_frame


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("examples"))
    ap.add_argument("--rows", type=int, default=600)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    frame = make_credit_frame(args.rows, seed=7)
    split = int(len(frame) * 0.8)
    frame.iloc[:split].to_csv(args.out / "credit_train.csv", index=False, lineterminator="\n")
    frame.iloc[split:].to_csv(args.out / "credit_validation.csv", index=False, lineterminator="\n")
    frame.to_csv(args.out / "credit_sample.csv", index=False, lineterminator="\n")
    frame.iloc[split : split + 5][FEATURES].to_csv(
        args.out / "smoke_rows.csv", index=False, lineterminator="\n"
    )
    # SageMaker transports every hyperparameter as a string
    (args.out / "hyperparameters.json").write_text(
        json.dumps(
            {
                "target": TARGET,
                "model": "logistic_regression",
                "C": "0.5",
                "threshold": "0.3",
                "id_columns": "application_id",
                "validation_split": "0.2",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.out}/ ({len(frame)} rows, default rate {frame[TARGET].mean():.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
