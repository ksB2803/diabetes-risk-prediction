"""CSV loading, profiling and training-data validation.

Nothing in this module logs or echoes raw data rows. The only place row values
leave the service is the small preview returned to the user who uploaded the file.
"""
from __future__ import annotations

import csv
import io
import math
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from .config import Settings
from .errors import MLError
from .schemas import TrainConfig

PREVIEW_ROWS = 8
CLASS_COUNT_MAX_UNIQUE = 20
HIGH_MISSING_FRACTION = 0.4
IMBALANCE_FLAG_RATIO = 0.35  # minority share below this is flagged

_ID_NAME = re.compile(r"(^id$|_id$|^id_|\bid\b|identifier|patient.?id|record.?id|^unnamed)", re.IGNORECASE)
_POSITIVE_TOKENS = ["1", "true", "yes", "y", "positive", "pos", "diabetic", "diabetes"]
_TARGET_NAMES = ["diabetes", "outcome", "diabetic", "target", "label", "class"]

WARNING_TEXT = {
    "ALL_EMPTY": "This column is empty in every row.",
    "CONSTANT": "Every row has the same value, so it gives the model nothing to learn from.",
    "HIGH_MISSING": "More than 40% of values are missing.",
    "ID_LIKE": "Looks like an identifier (unique per row). Identifiers rarely help and can mislead the model.",
    "HIGH_CARDINALITY": "Has too many distinct text values to encode safely; it cannot be used as a feature.",
    "DUPLICATE_COLUMN": "Contains exactly the same values as another column.",
    "MIXED_NUMERIC_TEXT": "Mostly numbers, but some entries are text; it will be treated as categories.",
}


# --------------------------------------------------------------------------- loading
def load_csv(data: bytes, settings: Settings) -> pd.DataFrame:
    """Parse uploaded bytes as a CSV table, enforcing size and shape limits."""
    if data is None or len(data) == 0:
        raise MLError("EMPTY_FILE", "The uploaded file is empty.")
    if len(data) > settings.max_upload_bytes:
        mb = settings.max_upload_bytes / (1024 * 1024)
        raise MLError("FILE_TOO_LARGE", f"The file is larger than the {mb:.0f} MB limit.", status_code=413)
    if b"\x00" in data[:65536]:
        raise MLError("NOT_A_CSV", "The file looks like binary data, not a CSV text file.")

    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = data.decode("latin-1")
        except UnicodeDecodeError as exc:  # pragma: no cover - latin-1 decodes any byte
            raise MLError("BAD_ENCODING", "The file could not be read as text. Save it as UTF-8 CSV.") from exc

    if text.strip() == "":
        raise MLError("EMPTY_FILE", "The uploaded file contains no data.")

    delimiter = ","
    first_line = text.lstrip().splitlines()[0]
    if first_line.count(",") == 0 and first_line.count(";") > 0:
        delimiter = ";"

    header = next(csv.reader(io.StringIO(first_line), delimiter=delimiter))
    header = [h.strip() for h in header]
    if any(h == "" for h in header):
        raise MLError("BAD_HEADER", "One or more columns have an empty name. Every column needs a header.")
    seen: set[str] = set()
    dupes = sorted({h for h in header if h in seen or seen.add(h)})
    if dupes:
        raise MLError("DUPLICATE_HEADER", "Column names must be unique.", details=[f"Repeated: {d}" for d in dupes][:10])
    if len(header) > settings.max_columns:
        raise MLError("TOO_MANY_COLUMNS", f"The file has {len(header)} columns; the limit is {settings.max_columns}.")
    if len(header) < 2:
        raise MLError("TOO_FEW_COLUMNS", "A labelled dataset needs at least two columns: one feature and one target.")

    try:
        df = pd.read_csv(
            io.StringIO(text),
            sep=delimiter,
            skipinitialspace=True,
            on_bad_lines="error",
            nrows=settings.max_rows + 1,
            low_memory=False,
        )
    except pd.errors.EmptyDataError as exc:
        raise MLError("EMPTY_FILE", "The file has no rows to read.") from exc
    except (pd.errors.ParserError, csv.Error, ValueError) as exc:
        raise MLError(
            "MALFORMED_CSV",
            "The CSV could not be parsed. Check that every row has the same number of columns as the header.",
        ) from exc

    df.columns = [str(c).strip() for c in df.columns]
    if len(df) == 0:
        raise MLError("NO_ROWS", "The file has a header but no data rows.")
    if len(df) > settings.max_rows:
        raise MLError("TOO_MANY_ROWS", f"The file has more than {settings.max_rows:,} rows, which is the limit.")
    return df


# --------------------------------------------------------------------------- helpers
def normalize_label(value: Any) -> str:
    """Turn a target value into a stable string label (1.0 -> "1", True -> "True")."""
    if isinstance(value, (bool, np.bool_)):
        return "True" if bool(value) else "False"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        f = float(value)
        if math.isfinite(f) and f.is_integer():
            return str(int(f))
        return repr(f)
    return str(value).strip()


def normalize_labels(series: pd.Series) -> pd.Series:
    out = series.map(lambda v: None if _is_missing(v) else normalize_label(v))
    return out.astype(object)


def _is_missing(v: Any) -> bool:
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def suggest_positive_class(labels: list[str]) -> str | None:
    lowered = {lab.lower(): lab for lab in labels}
    for token in _POSITIVE_TOKENS:
        if token in lowered:
            return lowered[token]
    return None


def is_numeric_column(series: pd.Series) -> bool:
    return ptypes.is_numeric_dtype(series) and not ptypes.is_bool_dtype(series)


def _json_value(v: Any) -> Any:
    if _is_missing(v):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return round(f, 6) if math.isfinite(f) else None
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    s = str(v)
    return s if len(s) <= 60 else s[:57] + "..."


def _target_status(series: pd.Series) -> str:
    non_null = series.dropna()
    if len(non_null) == 0:
        return "empty"
    if is_numeric_column(series):
        values = non_null.astype(float)
        if not np.all(np.isclose(values, np.round(values))):
            return "continuous"
    n_unique = normalize_labels(non_null).nunique()
    if n_unique == 1:
        return "one_class"
    if n_unique == 2:
        return "binary"
    return "multiclass"


# --------------------------------------------------------------------------- profiling
def column_profile(df: pd.DataFrame, settings: Settings) -> list[dict]:
    n_rows = len(df)
    profiles: list[dict] = []
    seen_hashes: dict[int, str] = {}
    for name in df.columns:
        s = df[name]
        missing = int(s.isna().sum())
        non_null = s.dropna()
        unique = int(non_null.nunique())
        warnings: list[str] = []

        if is_numeric_column(s):
            inferred = "numeric"
        elif ptypes.is_bool_dtype(s):
            inferred = "boolean"
        else:
            inferred = "categorical"
            if len(non_null) > 0:
                coerced = pd.to_numeric(non_null.astype(str), errors="coerce")
                ok = coerced.notna().mean()
                if 0.8 <= ok < 1.0:
                    warnings.append("MIXED_NUMERIC_TEXT")

        if missing == n_rows:
            warnings.append("ALL_EMPTY")
        elif unique == 1:
            warnings.append("CONSTANT")
        if 0 < missing < n_rows and missing / n_rows > HIGH_MISSING_FRACTION:
            warnings.append("HIGH_MISSING")

        id_like = bool(_ID_NAME.search(name))
        if not id_like and n_rows >= 20 and unique == n_rows:
            if inferred == "categorical":
                id_like = True
            elif inferred == "numeric":
                vals = non_null.astype(float)
                if np.all(np.isclose(vals, np.round(vals))) and (vals.is_monotonic_increasing or vals.is_monotonic_decreasing):
                    id_like = True
        if id_like:
            warnings.append("ID_LIKE")
        if inferred == "categorical" and unique > settings.max_categories:
            warnings.append("HIGH_CARDINALITY")

        if missing < n_rows:
            h = int(pd.util.hash_pandas_object(s.astype(str), index=False).sum())
            if h in seen_hashes and df[seen_hashes[h]].astype(str).equals(s.astype(str)):
                warnings.append("DUPLICATE_COLUMN")
            else:
                seen_hashes.setdefault(h, name)

        class_counts = None
        if 0 < unique <= CLASS_COUNT_MAX_UNIQUE:
            counts = normalize_labels(non_null).value_counts()
            class_counts = [{"label": str(k), "count": int(v)} for k, v in counts.items()]

        status = _target_status(s)
        suggested_positive = None
        if status == "binary" and class_counts:
            suggested_positive = suggest_positive_class([c["label"] for c in class_counts])

        summary: dict[str, Any] = {}
        if inferred == "numeric" and len(non_null) > 0:
            summary = {
                "min": _json_value(non_null.min()),
                "max": _json_value(non_null.max()),
                "mean": _json_value(non_null.mean()),
            }

        excluded = {"ALL_EMPTY", "CONSTANT", "ID_LIKE", "HIGH_CARDINALITY", "DUPLICATE_COLUMN"}
        profiles.append(
            {
                "name": name,
                "inferred_type": inferred,
                "missing_count": missing,
                "missing_pct": round(100.0 * missing / n_rows, 2),
                "unique_count": unique,
                "warnings": warnings,
                "warning_messages": [WARNING_TEXT[w] for w in warnings],
                "suggested_feature": not any(w in excluded for w in warnings),
                "target_status": status,
                "class_counts": class_counts,
                "suggested_positive_class": suggested_positive,
                "summary": summary,
            }
        )
    return profiles


def profile_dataset(df: pd.DataFrame, filename: str, size_bytes: int, settings: Settings) -> dict:
    columns = column_profile(df, settings)
    suggested_target = None
    lowered = {c["name"].lower(): c for c in columns}
    for candidate in _TARGET_NAMES:
        col = lowered.get(candidate)
        if col is not None and col["target_status"] == "binary":
            suggested_target = col["name"]
            break

    preview_df = df.head(PREVIEW_ROWS)
    preview_rows = [[_json_value(v) for v in row] for row in preview_df.itertuples(index=False, name=None)]
    duplicates = int(df.duplicated().sum())

    warnings: list[str] = []
    if duplicates:
        warnings.append(f"{duplicates} duplicate row(s) found. They will be removed before training.")
    if len(df) < settings.min_rows:
        warnings.append(f"Only {len(df)} rows. At least {settings.min_rows} are needed to train.")

    return {
        "filename": filename,
        "size_bytes": size_bytes,
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "duplicate_rows": duplicates,
        "preview": {"columns": list(df.columns), "rows": preview_rows},
        "columns": columns,
        "suggested_target": suggested_target,
        "warnings": warnings,
        "limits": {
            "max_rows": settings.max_rows,
            "max_columns": settings.max_columns,
            "min_rows": settings.min_rows,
            "min_rows_per_class": settings.min_rows_per_class,
        },
    }


# --------------------------------------------------------------------------- training data
@dataclass
class PreparedData:
    X: pd.DataFrame
    y: pd.Series  # 0 = negative class, 1 = positive class
    positive_class: str
    negative_class: str
    numeric_features: list[str]
    categorical_features: list[str]
    dropped_features: list[dict] = field(default_factory=list)
    n_rows_raw: int = 0
    duplicate_rows_removed: int = 0
    missing_target_rows_removed: int = 0
    warnings: list[str] = field(default_factory=list)
    class_counts: dict[str, int] = field(default_factory=dict)


def prepare_training_data(df: pd.DataFrame, config: TrainConfig, settings: Settings) -> PreparedData:
    target = config.target_column
    if target not in df.columns:
        raise MLError("TARGET_NOT_FOUND", f'The target column "{target}" is not in the file.')
    missing_features = [f for f in config.feature_columns if f not in df.columns]
    if missing_features:
        raise MLError(
            "FEATURES_NOT_FOUND",
            "Some selected feature columns are not in the file.",
            details=["Not found: " + ", ".join(missing_features[:20])],
        )
    if target in config.feature_columns:
        raise MLError("TARGET_IN_FEATURES", "The target column cannot also be used as a feature.")

    warnings: list[str] = []
    n_raw = len(df)
    work = df[list(config.feature_columns) + [target]].copy()

    dup_mask = work.duplicated()
    n_dupes = int(dup_mask.sum())
    if n_dupes:
        work = work.loc[~dup_mask]
        warnings.append(f"Removed {n_dupes} duplicate row(s).")

    status = _target_status(work[target])
    if status == "empty":
        raise MLError("TARGET_EMPTY", f'The target column "{target}" has no values.')
    if status == "continuous":
        raise MLError(
            "UNSUPPORTED_TARGET",
            f'The target column "{target}" contains continuous numbers. This app supports binary (two-class) targets only.',
        )

    missing_target = work[target].isna()
    n_missing_target = int(missing_target.sum())
    if n_missing_target:
        work = work.loc[~missing_target]
        warnings.append(f"Removed {n_missing_target} row(s) with no target label (labels are never guessed).")

    labels = normalize_labels(work[target])
    classes = sorted(labels.unique().tolist())
    if len(classes) < 2:
        raise MLError(
            "ONE_CLASS_TARGET",
            f'The target column "{target}" has only one class ({classes[0] if classes else "none"}). '
            "A classifier needs examples of both outcomes.",
        )
    if len(classes) > 2:
        raise MLError(
            "MULTICLASS_UNSUPPORTED",
            f'The target column "{target}" has {len(classes)} classes. This version supports exactly two.',
            details=[f"Classes found: {', '.join(classes[:10])}"],
        )

    positive = config.positive_class
    if positive is not None:
        positive = positive.strip()
        if positive not in classes:
            raise MLError(
                "BAD_POSITIVE_CLASS",
                f'"{positive}" is not one of the target classes.',
                details=[f"Classes found: {', '.join(classes)}"],
            )
    else:
        positive = suggest_positive_class(classes)
        if positive is None:
            raise MLError(
                "POSITIVE_CLASS_REQUIRED",
                "Choose which target class is the positive outcome (for example the class meaning 'has diabetes').",
                details=[f"Classes found: {', '.join(classes)}"],
            )
    negative = next(c for c in classes if c != positive)

    if len(work) < settings.min_rows:
        raise MLError("TOO_FEW_ROWS", f"Only {len(work)} usable rows. At least {settings.min_rows} are needed.")
    counts = labels.value_counts()
    small = [f'"{lab}" has {int(n)}' for lab, n in counts.items() if n < settings.min_rows_per_class]
    if small:
        raise MLError(
            "TOO_FEW_PER_CLASS",
            f"Each class needs at least {settings.min_rows_per_class} rows to split and cross-validate.",
            details=small,
        )

    minority_share = counts.min() / counts.sum()
    if minority_share < IMBALANCE_FLAG_RATIO:
        warnings.append(
            f"Classes are imbalanced (smaller class is {minority_share:.0%} of rows). "
            "Accuracy alone can be misleading; check recall, precision, F1 and ROC-AUC."
        )

    numeric, categorical, dropped = [], [], []
    for name in config.feature_columns:
        s = work[name]
        non_null = s.dropna()
        if len(non_null) == 0:
            dropped.append({"name": name, "reason": "All values are empty."})
            continue
        if non_null.nunique() == 1:
            dropped.append({"name": name, "reason": "Same value in every row."})
            continue
        if is_numeric_column(s):
            numeric.append(name)
        else:
            n_cat = non_null.astype(str).nunique()
            if n_cat > settings.max_categories:
                raise MLError(
                    "HIGH_CARDINALITY_FEATURE",
                    f'Feature "{name}" has {n_cat} distinct text values (limit {settings.max_categories}). '
                    "Deselect it or group its values.",
                )
            categorical.append(name)
    if dropped:
        warnings.append(f"Dropped {len(dropped)} unusable feature(s): " + ", ".join(d["name"] for d in dropped) + ".")
    if not numeric and not categorical:
        raise MLError("NO_USABLE_FEATURES", "None of the selected features can be used. Choose different columns.")

    y = (labels == positive).astype(int)
    X = work[numeric + categorical]
    return PreparedData(
        X=X.reset_index(drop=True),
        y=y.reset_index(drop=True),
        positive_class=positive,
        negative_class=negative,
        numeric_features=numeric,
        categorical_features=categorical,
        dropped_features=dropped,
        n_rows_raw=n_raw,
        duplicate_rows_removed=n_dupes,
        missing_target_rows_removed=n_missing_target,
        warnings=warnings,
        class_counts={str(k): int(v) for k, v in counts.items()},
    )
