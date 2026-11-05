"""Unit tests for splitting, metrics, preprocessing isolation and artifact reload."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from app.artifact_store import ArtifactStore
from app.data_validation import normalize_label, prepare_training_data, suggest_positive_class
from app.errors import MLError
from app.evaluation import evaluate
from app.schemas import TrainConfig
from app.training import _split, run_training

from .conftest import make_config


def test_normalize_label():
    assert normalize_label(1.0) == "1"
    assert normalize_label(np.int64(0)) == "0"
    assert normalize_label(True) == "True"
    assert normalize_label(" yes ") == "yes"


def test_suggest_positive_class():
    assert suggest_positive_class(["0", "1"]) == "1"
    assert suggest_positive_class(["No", "Yes"]) == "Yes"
    assert suggest_positive_class(["groupA", "groupB"]) is None


def test_split_is_reproducible(demo_df, settings):
    cfg = TrainConfig(**make_config())
    prep = prepare_training_data(demo_df, cfg, settings)
    a = _split(prep, cfg)
    b = _split(prep, cfg)
    assert list(a[1].index) == list(b[1].index)
    c = _split(prep, TrainConfig(**make_config(random_seed=7)))
    assert list(a[1].index) != list(c[1].index)


def test_split_is_stratified(demo_df, settings):
    cfg = TrainConfig(**make_config())
    prep = prepare_training_data(demo_df, cfg, settings)
    _, _, y_train, y_test = _split(prep, cfg)
    assert abs(y_train.mean() - y_test.mean()) < 0.05


def test_training_is_reproducible(demo_df, settings):
    store = ArtifactStore(settings.model_dir)
    cfg = TrainConfig(**make_config(algorithms=["random_forest"]))
    r1 = run_training(demo_df, cfg, settings, store)
    r2 = run_training(demo_df, cfg, settings, store)
    assert r1["models"][0]["cv_score"] == r2["models"][0]["cv_score"]
    assert r1["models"][0]["test_metrics"] == r2["models"][0]["test_metrics"]


class _FixedModel:
    """Stand-in classifier with known predictions for metric checks."""

    def __init__(self, preds, proba):
        self._preds = np.asarray(preds)
        self._proba = np.asarray(proba)

    def predict(self, X):
        return self._preds

    def predict_proba(self, X):
        return np.column_stack([1 - self._proba, self._proba])


def test_metrics_match_sklearn():
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 0, 1, 0, 1, 0])
    proba = np.array([0.1, 0.6, 0.8, 0.4, 0.9, 0.2, 0.7, 0.3])
    result, _ = evaluate(_FixedModel(y_pred, proba), np.zeros((8, 1)), y_true, "1", "0")
    m = result["metrics"]
    assert m["accuracy"] == pytest.approx(accuracy_score(y_true, y_pred))
    assert m["precision"] == pytest.approx(precision_score(y_true, y_pred))
    assert m["recall"] == pytest.approx(recall_score(y_true, y_pred))
    assert m["f1"] == pytest.approx(f1_score(y_true, y_pred))
    assert m["roc_auc"] == pytest.approx(roc_auc_score(y_true, proba))
    # rows = actual, columns = predicted: [[TN, FP], [FN, TP]]
    assert result["confusion_matrix"]["matrix"] == [[3, 1], [1, 3]]


def test_undefined_roc_auc_is_reported_not_faked():
    y_true = np.array([1, 1, 1])
    result, notes = evaluate(_FixedModel([1, 1, 0], [0.9, 0.8, 0.2]), np.zeros((3, 1)), y_true, "1", "0")
    assert result["metrics"]["roc_auc"] is None
    assert result["roc_curve"] is None
    assert any("undefined" in n for n in notes)


def test_preprocessing_fitted_on_training_split_only(demo_df, settings):
    """The saved imputer median must equal the median of the training rows, not all rows."""
    store = ArtifactStore(settings.model_dir)
    cfg = TrainConfig(**make_config(algorithms=["logistic_regression"]))
    result = run_training(demo_df, cfg, settings, store)
    pipeline, _ = store.load_model(result["models"][0]["model_id"])

    prep = prepare_training_data(demo_df, cfg, settings)
    X_train, X_test, _, _ = _split(prep, cfg)
    num_pipe = pipeline.named_steps["preprocess"].named_transformers_["num"]
    idx = prep.numeric_features.index("Glucose")
    fitted_median = num_pipe.named_steps["impute"].statistics_[idx]
    assert fitted_median == pytest.approx(X_train["Glucose"].median())
    scaler_mean = num_pipe.named_steps["scale"].mean_[idx]
    train_filled = X_train["Glucose"].fillna(X_train["Glucose"].median())
    assert scaler_mean == pytest.approx(train_filled.mean())


def test_saved_pipeline_reload_matches(demo_df, settings):
    store = ArtifactStore(settings.model_dir)
    cfg = TrainConfig(**make_config(algorithms=["random_forest"]))
    result = run_training(demo_df, cfg, settings, store)
    model_id = result["models"][0]["model_id"]

    fresh_store = ArtifactStore(settings.model_dir)  # no cache
    pipeline, meta = fresh_store.load_model(model_id)
    assert meta["algorithm"] == "random_forest"
    assert meta["schema"]["positive_class"] == "1"
    sample = demo_df[[f["name"] for f in meta["schema"]["features"]]].head(20)
    preds = pipeline.predict(sample)
    assert set(np.unique(preds)) <= {0, 1}
    assert len(preds) == 20


def test_constant_feature_is_dropped_and_reported(demo_df, settings):
    df = demo_df.copy()
    df["Constant"] = 5
    cfg = TrainConfig(**make_config(feature_columns=["Glucose", "BMI", "Constant"]))
    prep = prepare_training_data(df, cfg, settings)
    assert "Constant" not in prep.X.columns
    assert prep.dropped_features[0]["name"] == "Constant"


def test_high_cardinality_feature_rejected(demo_df, settings):
    df = demo_df.copy()
    df["FreeText"] = [f"note-{i}" for i in range(len(df))]
    cfg = TrainConfig(**make_config(feature_columns=["Glucose", "FreeText"]))
    with pytest.raises(MLError) as err:
        prepare_training_data(df, cfg, settings)
    assert err.value.code == "HIGH_CARDINALITY_FEATURE"


def test_duplicates_removed(demo_df, settings):
    df = pd.concat([demo_df, demo_df.head(5)], ignore_index=True)
    cfg = TrainConfig(**make_config())
    prep = prepare_training_data(df, cfg, settings)
    assert prep.duplicate_rows_removed == 5
