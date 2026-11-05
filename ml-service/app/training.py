"""Model training, bounded hyper-parameter tuning and model selection.

Rules enforced here:
* The train/test split happens first. The test set is used once per candidate,
  after tuning, and never influences tuning or model selection.
* Tuning uses StratifiedKFold cross-validation on the training split only.
* Model selection uses the mean cross-validation score for the chosen metric.
* Parameter grids are fixed and small; the user cannot supply arbitrary grids.
"""
from __future__ import annotations

import time
import warnings as py_warnings
from datetime import datetime, timezone

import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .artifact_store import ArtifactStore
from .config import Settings
from .data_validation import PreparedData, prepare_training_data
from .errors import MLError
from .evaluation import evaluate, feature_importance, scorer_for
from .preprocessing import build_preprocessor, preprocessing_summary
from .schemas import ALGORITHM_NAMES, TrainConfig

PIPELINE_VERSION = "1"

FOLDS = {"quick": 3, "balanced": 5, "thorough": 5}

GRIDS: dict[str, dict[str, dict]] = {
    "logistic_regression": {
        "quick": {"model__C": [0.1, 1.0, 10.0]},
        "balanced": {"model__C": [0.01, 0.1, 1.0, 10.0], "model__class_weight": [None, "balanced"]},
        "thorough": {
            "model__C": [0.001, 0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0],
            "model__class_weight": [None, "balanced"],
        },
    },
    "random_forest": {
        "quick": {"model__n_estimators": [100], "model__max_depth": [None, 8]},
        "balanced": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 6, 12],
            "model__min_samples_leaf": [1, 3],
        },
        "thorough": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [None, 4, 6, 8, 12, 16],
            "model__min_samples_leaf": [1, 2, 4, 8],
            "model__max_features": ["sqrt", "log2", 0.5],
            "model__class_weight": [None, "balanced"],
        },
    },
    "decision_tree": {
        "quick": {"model__max_depth": [3, 5, None]},
        "balanced": {"model__max_depth": [3, 5, 8, None], "model__min_samples_leaf": [1, 5, 10]},
        "thorough": {
            "model__criterion": ["gini", "entropy"],
            "model__max_depth": [2, 3, 4, 5, 6, 8, 10, None],
            "model__min_samples_leaf": [1, 2, 5, 10, 20],
            "model__class_weight": [None, "balanced"],
        },
    },
}

METRIC_NAMES = {"roc_auc": "ROC-AUC", "f1": "F1-score", "recall": "recall", "precision": "precision", "accuracy": "accuracy"}

# Large grids are sampled instead of exhaustively searched.
RANDOMIZED = {("random_forest", "thorough"): 20}


def _estimator(algorithm: str, seed: int, n_jobs: int):
    if algorithm == "logistic_regression":
        return LogisticRegression(max_iter=2000, random_state=seed)
    if algorithm == "random_forest":
        return RandomForestClassifier(random_state=seed, n_jobs=n_jobs)
    if algorithm == "decision_tree":
        return DecisionTreeClassifier(random_state=seed)
    raise MLError("UNKNOWN_ALGORITHM", f"Unknown algorithm: {algorithm}")  # pragma: no cover - schema guards this


def _grid_size(grid: dict) -> int:
    size = 1
    for values in grid.values():
        size *= len(values)
    return size


def _schema(prep: PreparedData, X_train: pd.DataFrame, target: str) -> dict:
    features = []
    for name in prep.numeric_features:
        col = pd.to_numeric(X_train[name], errors="coerce").dropna()
        features.append(
            {
                "name": name,
                "type": "numeric",
                "min": round(float(col.min()), 6) if len(col) else None,
                "max": round(float(col.max()), 6) if len(col) else None,
                "median": round(float(col.median()), 6) if len(col) else None,
            }
        )
    for name in prep.categorical_features:
        cats = sorted({str(v).strip() for v in X_train[name].dropna().tolist()})
        features.append({"name": name, "type": "categorical", "categories": cats})
    return {
        "target": target,
        "positive_class": prep.positive_class,
        "negative_class": prep.negative_class,
        "features": features,
    }


def _split(prep: PreparedData, config: TrainConfig):
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            prep.X,
            prep.y,
            train_size=config.train_fraction,
            random_state=config.random_seed,
            stratify=prep.y,
            shuffle=True,
        )
    except ValueError as exc:
        raise MLError(
            "SPLIT_FAILED",
            "The data could not be split with this ratio. Use a larger dataset or a different split.",
        ) from exc
    for part, ys in (("training", y_train), ("test", y_test)):
        if ys.nunique() < 2:
            raise MLError("SPLIT_MISSING_CLASS", f"The {part} set ended up with only one class. Use more data.")
    if y_test.value_counts().min() < 2:
        raise MLError(
            "SPLIT_TOO_SMALL",
            "The test set has fewer than 2 rows of one class. Use more data or a larger test share.",
        )
    return X_train, X_test, y_train, y_test


def run_training(df: pd.DataFrame, config: TrainConfig, settings: Settings, store: ArtifactStore) -> dict:
    started = time.perf_counter()
    prep = prepare_training_data(df, config, settings)
    warnings: list[str] = list(prep.warnings)

    X_train, X_test, y_train, y_test = _split(prep, config)

    min_train_class = int(y_train.value_counts().min())
    n_splits = min(FOLDS[config.tuning_level], min_train_class)
    if n_splits < 2:
        raise MLError("TOO_FEW_FOR_CV", "Not enough rows of each class in the training set for cross-validation.")
    if n_splits < FOLDS[config.tuning_level]:
        warnings.append(
            f"Used {n_splits}-fold cross-validation instead of {FOLDS[config.tuning_level]} because one class is small."
        )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.random_seed)
    scoring = scorer_for(config.scoring_metric)

    schema = _schema(prep, X_train, config.target_column)
    models: list[dict] = []
    for algorithm in config.algorithms:
        grid = GRIDS[algorithm][config.tuning_level]
        pipeline = Pipeline(
            steps=[
                ("preprocess", build_preprocessor(prep.numeric_features, prep.categorical_features)),
                ("model", _estimator(algorithm, config.random_seed, settings.n_jobs)),
            ]
        )
        n_iter = RANDOMIZED.get((algorithm, config.tuning_level))
        if n_iter:
            search = RandomizedSearchCV(
                pipeline,
                grid,
                n_iter=n_iter,
                scoring=scoring,
                cv=cv,
                refit=True,
                n_jobs=settings.n_jobs,
                random_state=config.random_seed,
                error_score="raise",
            )
            n_candidates = n_iter
            search_name = "RandomizedSearchCV"
        else:
            search = GridSearchCV(
                pipeline, grid, scoring=scoring, cv=cv, refit=True, n_jobs=settings.n_jobs, error_score="raise"
            )
            n_candidates = _grid_size(grid)
            search_name = "GridSearchCV"

        t0 = time.perf_counter()
        with py_warnings.catch_warnings(record=True) as caught:
            py_warnings.simplefilter("always", ConvergenceWarning)
            try:
                search.fit(X_train, y_train)
            except MLError:
                raise
            except Exception as exc:
                raise MLError(
                    "TRAINING_FAILED",
                    f"{ALGORITHM_NAMES[algorithm]} could not be trained on this data. Check the selected features.",
                    status_code=422,
                ) from exc
        if any(issubclass(w.category, ConvergenceWarning) for w in caught):
            warnings.append(f"{ALGORITHM_NAMES[algorithm]}: the optimiser did not fully converge for some settings.")
        fit_seconds = time.perf_counter() - t0

        best_index = int(search.best_index_)
        cv_mean = float(search.cv_results_["mean_test_score"][best_index])
        cv_std = float(search.cv_results_["std_test_score"][best_index])
        best_model = search.best_estimator_

        evaluation, notes = evaluate(best_model, X_test, y_test, prep.positive_class, prep.negative_class)
        warnings.extend(f"{ALGORITHM_NAMES[algorithm]}: {n}" for n in notes)

        best_params = {k.replace("model__", ""): (v if v is None or isinstance(v, (int, float, str)) else str(v))
                       for k, v in search.best_params_.items()}
        artifact = store.save_model(
            best_model,
            {
                "job_id": config.job_id,
                "algorithm": algorithm,
                "pipeline_version": PIPELINE_VERSION,
                "sklearn_version": sklearn.__version__,
                "schema": schema,
                "scoring_metric": config.scoring_metric,
                "cv_score": cv_mean,
                "test_metrics": evaluation["metrics"],
                "best_params": best_params,
            },
        )
        models.append(
            {
                "model_id": artifact["model_id"],
                "algorithm": algorithm,
                "display_name": ALGORITHM_NAMES[algorithm],
                "best_params": best_params,
                "search": search_name,
                "n_candidates": n_candidates,
                "cv_score": round(cv_mean, 6),
                "cv_std": round(cv_std, 6),
                "test_metrics": evaluation["metrics"],
                "confusion_matrix": evaluation["confusion_matrix"],
                "classification_report": evaluation["classification_report"],
                "roc_curve": evaluation["roc_curve"],
                "feature_importance": feature_importance(best_model, algorithm),
                "fit_seconds": round(fit_seconds, 3),
                "artifact": {
                    "file": artifact["artifact_file"],
                    "created_at": artifact["created_at"],
                    "pipeline_version": PIPELINE_VERSION,
                    "sklearn_version": sklearn.__version__,
                },
            }
        )

    # Selection by cross-validation score only (first listed wins ties).
    selected = max(models, key=lambda m: m["cv_score"])
    train_counts = y_train.value_counts()
    test_counts = y_test.value_counts()
    label_of = {1: prep.positive_class, 0: prep.negative_class}
    counts = prep.class_counts
    minority_share = min(counts.values()) / sum(counts.values())

    return {
        "job_id": config.job_id,
        "task_type": config.task_type,
        "target_column": config.target_column,
        "scoring_metric": config.scoring_metric,
        "tuning_level": config.tuning_level,
        "random_seed": config.random_seed,
        "dataset_summary": {
            "n_rows_raw": prep.n_rows_raw,
            "n_rows_used": int(len(prep.X)),
            "duplicate_rows_removed": prep.duplicate_rows_removed,
            "missing_target_rows_removed": prep.missing_target_rows_removed,
            "n_features_used": len(prep.numeric_features) + len(prep.categorical_features),
        },
        "preprocessing_summary": preprocessing_summary(
            prep.numeric_features, prep.categorical_features, prep.dropped_features
        ),
        "class_mapping": {
            "positive_class": prep.positive_class,
            "negative_class": prep.negative_class,
            "note": "The positive class is encoded as 1 and the other class as 0 inside the model.",
        },
        "class_distribution": [{"label": k, "count": v} for k, v in sorted(counts.items())],
        "imbalance": {"minority_share": round(minority_share, 4), "flagged": minority_share < 0.35},
        "split": {
            "train_fraction": config.train_fraction,
            "test_fraction": round(1 - config.train_fraction, 4),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "stratified": True,
            "random_seed": config.random_seed,
            "train_class_counts": {label_of[int(k)]: int(v) for k, v in train_counts.items()},
            "test_class_counts": {label_of[int(k)]: int(v) for k, v in test_counts.items()},
        },
        "cross_validation": {"strategy": "StratifiedKFold", "n_splits": n_splits, "shuffle": True},
        "schema": schema,
        "models": models,
        "selected_model_id": selected["model_id"],
        "selection_reason": (
            f"{selected['display_name']} had the highest mean cross-validation {METRIC_NAMES[config.scoring_metric]} "
            "on the training data. Test-set results were not used to choose it."
        ),
        "warnings": warnings,
        "duration_seconds": round(time.perf_counter() - started, 3),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
