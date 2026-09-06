"""Decode the raw UCI Statlog (German Credit) file into a readable, typed CSV.

Source: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data (CC BY 4.0).
The raw file uses opaque codes (A11, A34, ...). This script maps every code to a stable
snake_case token, names the 20 attributes, and turns the 1/2 target into `default` (1 = bad).
It is deterministic: the same raw file always yields byte-identical output, and the SHA-256
of the output is printed so it can be pinned in data/README.md.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

COLUMNS = [
    "checking_status",
    "duration_months",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_status",
    "employment_since",
    "installment_rate_pct",
    "personal_status_sex",
    "other_debtors",
    "residence_since_years",
    "property",
    "age_years",
    "other_installment_plans",
    "housing",
    "existing_credits_count",
    "job",
    "dependents_count",
    "telephone",
    "foreign_worker",
    "target_raw",
]

CODES: dict[str, dict[str, str]] = {
    "checking_status": {"A11": "lt_0", "A12": "0_to_200", "A13": "ge_200_or_salary", "A14": "none"},
    "credit_history": {
        "A30": "no_credits_all_paid",
        "A31": "all_paid_this_bank",
        "A32": "existing_paid_duly",
        "A33": "past_delays",
        "A34": "critical_or_other_credits",
    },
    "purpose": {
        "A40": "car_new",
        "A41": "car_used",
        "A42": "furniture_equipment",
        "A43": "radio_tv",
        "A44": "domestic_appliances",
        "A45": "repairs",
        "A46": "education",
        "A47": "vacation",
        "A48": "retraining",
        "A49": "business",
        "A410": "other",
    },
    "savings_status": {
        "A61": "lt_100",
        "A62": "100_to_500",
        "A63": "500_to_1000",
        "A64": "ge_1000",
        "A65": "unknown_or_none",
    },
    "employment_since": {
        "A71": "unemployed",
        "A72": "lt_1y",
        "A73": "1_to_4y",
        "A74": "4_to_7y",
        "A75": "ge_7y",
    },
    "personal_status_sex": {
        "A91": "male_divorced_separated",
        "A92": "female_divorced_separated_married",
        "A93": "male_single",
        "A94": "male_married_widowed",
        "A95": "female_single",
    },
    "other_debtors": {"A101": "none", "A102": "co_applicant", "A103": "guarantor"},
    "property": {
        "A121": "real_estate",
        "A122": "savings_agreement_or_life_insurance",
        "A123": "car_or_other",
        "A124": "unknown_or_none",
    },
    "other_installment_plans": {"A141": "bank", "A142": "stores", "A143": "none"},
    "housing": {"A151": "rent", "A152": "own", "A153": "for_free"},
    "job": {
        "A171": "unemployed_unskilled_non_resident",
        "A172": "unskilled_resident",
        "A173": "skilled_employee",
        "A174": "management_or_self_employed",
    },
    "telephone": {"A191": "none", "A192": "yes"},
    "foreign_worker": {"A201": "yes", "A202": "no"},
}


def decode_row(fields: list[str], line_no: int) -> list[str]:
    if len(fields) != len(COLUMNS):
        raise ValueError(f"line {line_no}: expected {len(COLUMNS)} fields, got {len(fields)}")
    out: list[str] = []
    for name, raw in zip(COLUMNS, fields, strict=True):
        if name in CODES:
            try:
                out.append(CODES[name][raw])
            except KeyError as exc:
                raise ValueError(f"line {line_no}: unknown code {raw!r} for {name}") from exc
        elif name == "target_raw":
            if raw not in {"1", "2"}:
                raise ValueError(f"line {line_no}: bad target {raw!r}")
            out.append("1" if raw == "2" else "0")
        else:
            int(raw)  # numeric attributes are integers in this dataset
            out.append(raw)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--raw", type=Path, required=True, help="path to german.data")
    ap.add_argument("--out", type=Path, required=True, help="output CSV path")
    args = ap.parse_args(argv)

    header = [*COLUMNS[:-1], "default"]
    lines = ["application_id," + ",".join(header)]
    for i, line in enumerate(args.raw.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        lines.append(f"{i:04d}," + ",".join(decode_row(line.split(), i)))
    body = "\n".join(lines) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(body.encode("utf-8"))
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    print(f"wrote {args.out} rows={len(lines) - 1} sha256={digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
