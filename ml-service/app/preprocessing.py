"""Preprocessing pipeline construction.

Everything here lives inside a scikit-learn Pipeline, so imputation, scaling and
encoding are fitted only on training data (and only on training folds during
cross-validation). Nothing is ever fitted on the full dataset before splitting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def to_float_frame(X):
    """Coerce numeric inputs to float, turning anything unparsable into NaN."""
    frame = pd.DataFrame(X)
    return frame.apply(pd.to_numeric, errors="coerce").astype(float)


def to_text_frame(X):
    """Coerce categorical inputs to plain Python strings with np.nan for missing.

    This keeps behaviour identical across pandas versions (object vs. the newer
    string dtype) and makes values like 1, 1.0 and "1" encode consistently.
    """
    frame = pd.DataFrame(X)

    def conv(v):
        if v is None:
            return np.nan
        try:
            if pd.isna(v):
                return np.nan
        except (TypeError, ValueError):
            pass
        if isinstance(v, (float, np.floating)) and float(v).is_integer():
            return str(int(v))
        return str(v).strip()

    return frame.apply(lambda col: col.map(conv)).astype(object)


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    transformers = []
    if numeric_features:
        numeric_pipeline = Pipeline(
            steps=[
                ("to_float", FunctionTransformer(to_float_frame, feature_names_out="one-to-one")),
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]
        )
        transformers.append(("num", numeric_pipeline, list(numeric_features)))
    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                ("to_text", FunctionTransformer(to_text_frame, feature_names_out="one-to-one")),
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("cat", categorical_pipeline, list(categorical_features)))
    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=True)


def preprocessing_summary(numeric_features: list[str], categorical_features: list[str], dropped: list[dict]) -> dict:
    return {
        "numeric_features": list(numeric_features),
        "categorical_features": list(categorical_features),
        "dropped_features": list(dropped),
        "numeric_steps": ["Convert to number", "Fill missing with training median", "Standardise (z-score)"]
        if numeric_features
        else [],
        "categorical_steps": [
            "Convert to text",
            "Fill missing with most frequent training value",
            "One-hot encode (unknown categories ignored)",
        ]
        if categorical_features
        else [],
        "fitted_on": "Training data only (inside each cross-validation fold during tuning)",
    }
