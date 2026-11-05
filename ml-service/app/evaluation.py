"""Held-out evaluation and model-explanation helpers."""
from __future__ import annotations

import math
import warnings as py_warnings

import numpy as np
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

MAX_ROC_POINTS = 200
MAX_IMPORTANCE_ITEMS = 25


def scorer_for(metric: str):
    """Scorers for the binary-encoded target (1 = positive class)."""
    if metric == "roc_auc":
        return "roc_auc"
    if metric == "accuracy":
        return "accuracy"
    fn = {"f1": f1_score, "recall": recall_score, "precision": precision_score}[metric]
    return make_scorer(fn, pos_label=1, zero_division=0)


def _clean(value) -> float | None:
    if value is None:
        return None
    f = float(value)
    return round(f, 6) if math.isfinite(f) else None


def evaluate(model, X_test, y_test, positive_label: str, negative_label: str) -> tuple[dict, list[str]]:
    """Return held-out metrics plus warnings about any undefined metric."""
    notes: list[str] = []
    y_pred = model.predict(X_test)
    y_true = np.asarray(y_test)

    with py_warnings.catch_warnings():
        py_warnings.simplefilter("ignore", UndefinedMetricWarning)
        metrics = {
            "accuracy": _clean(accuracy_score(y_true, y_pred)),
            "precision": _clean(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
            "recall": _clean(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
            "f1": _clean(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
            "roc_auc": None,
        }
    if int(np.sum(y_pred == 1)) == 0:
        notes.append("The model predicted no positive cases on the test set, so precision is reported as 0.")

    roc = None
    if hasattr(model, "predict_proba") and len(np.unique(y_true)) == 2:
        proba = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = _clean(roc_auc_score(y_true, proba))
        fpr, tpr, _ = roc_curve(y_true, proba)
        if len(fpr) > MAX_ROC_POINTS:
            idx = np.unique(np.linspace(0, len(fpr) - 1, MAX_ROC_POINTS).astype(int))
            fpr, tpr = fpr[idx], tpr[idx]
        roc = {"fpr": [round(float(v), 4) for v in fpr], "tpr": [round(float(v), 4) for v in tpr]}
    else:
        notes.append("ROC-AUC is undefined here (needs both classes in the test set and probability scores).")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    with py_warnings.catch_warnings():
        py_warnings.simplefilter("ignore", UndefinedMetricWarning)
        report = classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=[negative_label, positive_label],
            output_dict=True,
            zero_division=0,
        )
    report_clean = {
        str(k): ({kk: _clean(vv) for kk, vv in v.items()} if isinstance(v, dict) else _clean(v))
        for k, v in report.items()
    }
    return (
        {
            "metrics": metrics,
            "confusion_matrix": {
                "labels": [negative_label, positive_label],
                "matrix": cm.astype(int).tolist(),
                "layout": "rows = actual class, columns = predicted class",
            },
            "classification_report": report_clean,
            "roc_curve": roc,
        },
        notes,
    )


def feature_importance(fitted_pipeline, algorithm: str) -> dict | None:
    """Importances (trees) or coefficients (logistic regression) on transformed features."""
    try:
        names = list(fitted_pipeline.named_steps["preprocess"].get_feature_names_out())
    except Exception:  # pragma: no cover - defensive; names are optional
        return None
    clf = fitted_pipeline.named_steps["model"]
    if algorithm in ("random_forest", "decision_tree") and hasattr(clf, "feature_importances_"):
        values = np.asarray(clf.feature_importances_, dtype=float)
        kind = "importance"
        caveat = (
            "Impurity-based importance: how much each input helped this model split the training data. "
            "It is not causation and not clinical significance."
        )
    elif algorithm == "logistic_regression" and hasattr(clf, "coef_"):
        values = np.asarray(clf.coef_[0], dtype=float)
        kind = "coefficient"
        caveat = (
            "Coefficients on standardised/one-hot encoded inputs. Positive values push towards the positive class. "
            "Magnitudes depend on scaling and encoding, and they are not causation or clinical significance."
        )
    else:
        return None
    pretty = [n.split("__", 1)[1] if "__" in n else n for n in names]
    order = np.argsort(-np.abs(values))[:MAX_IMPORTANCE_ITEMS]
    return {
        "kind": kind,
        "caveat": caveat,
        "items": [{"feature": pretty[i], "value": round(float(values[i]), 6)} for i in order],
    }
