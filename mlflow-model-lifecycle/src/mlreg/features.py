"""Schema-driven preprocessing: one-hot categoricals (unknown → all-zero) + scaled numerics."""

from __future__ import annotations

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import DataSchema


def build_preprocessor(schema: DataSchema) -> ColumnTransformer:
    """Categories come from the contract, so the design matrix is identical across refits."""
    categories = [schema.column(c).categories or [] for c in schema.categorical_features]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), schema.numeric_features),
            (
                "cat",
                OneHotEncoder(categories=categories, handle_unknown="ignore", sparse_output=False),
                schema.categorical_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def feature_names(preprocessor: Any) -> list[str]:
    """Readable names: `num__age_years` → `age_years`, `cat__purpose_x` → `purpose=x`."""
    out: list[str] = []
    for raw in preprocessor.get_feature_names_out():
        name = str(raw)
        if name.startswith("num__"):
            out.append(name[len("num__") :])
        elif name.startswith("cat__"):
            rest = name[len("cat__") :]
            col, _, value = rest.partition("_")
            # categorical column names may themselves contain underscores: match the longest prefix
            for cat_col in sorted(_known_categoricals(preprocessor), key=len, reverse=True):
                if rest.startswith(cat_col + "_"):
                    col, value = cat_col, rest[len(cat_col) + 1 :]
                    break
            out.append(f"{col}={value}")
        else:
            out.append(name)
    return out


def _known_categoricals(preprocessor: Any) -> list[str]:
    for name, _, cols in preprocessor.transformers_:
        if name == "cat":
            return [str(c) for c in cols]
    return []
