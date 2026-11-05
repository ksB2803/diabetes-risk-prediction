"""Single-record prediction against a saved pipeline."""
from __future__ import annotations

import math
from typing import Any

import pandas as pd

from .artifact_store import ArtifactStore
from .errors import MLError
from .schemas import PredictRequest, PredictResponse

DISCLAIMER = (
    "Educational project only. This model is not a medical device and cannot diagnose diabetes. "
    "Results depend on the uploaded dataset and model quality. For health concerns, consult a "
    "qualified healthcare professional."
)


def _coerce_numeric(name: str, value: Any) -> float:
    if isinstance(value, bool) or value is None:
        raise MLError("INVALID_VALUE", f'"{name}" must be a number.')
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str) and value.strip() != "":
        try:
            number = float(value.strip())
        except ValueError as exc:
            raise MLError("INVALID_VALUE", f'"{name}" must be a number.') from exc
    else:
        raise MLError("INVALID_VALUE", f'"{name}" must be a number.')
    if not math.isfinite(number):
        raise MLError("INVALID_VALUE", f'"{name}" must be a finite number.')
    return number


def predict(store: ArtifactStore, request: PredictRequest) -> PredictResponse:
    pipeline, meta = store.load_model(request.model_id)
    schema = meta["schema"]
    features = schema["features"]
    expected = [f["name"] for f in features]

    missing = [n for n in expected if n not in request.values]
    unexpected = [n for n in request.values if n not in expected]
    if missing or unexpected:
        details = []
        if missing:
            details.append("Missing: " + ", ".join(missing[:20]))
        if unexpected:
            details.append("Not used by this model: " + ", ".join(unexpected[:20]))
        raise MLError("INCOMPATIBLE_FEATURES", "The input does not match the features this model was trained on.", details=details)

    row: dict[str, Any] = {}
    warnings: list[str] = []
    for feature in features:
        name = feature["name"]
        raw = request.values[name]
        if feature["type"] == "numeric":
            value = _coerce_numeric(name, raw)
            lo, hi = feature.get("min"), feature.get("max")
            if lo is not None and hi is not None and (value < lo or value > hi):
                warnings.append(
                    f'"{name}" = {value:g} is outside the training range ({lo:g} to {hi:g}); '
                    "the model has not seen values like this."
                )
            row[name] = value
        else:
            if raw is None or isinstance(raw, (dict, list)) or str(raw).strip() == "":
                raise MLError("INVALID_VALUE", f'"{name}" needs a value.')
            text = str(raw).strip()
            if len(text) > 200:
                raise MLError("INVALID_VALUE", f'"{name}" is too long.')
            if text not in feature.get("categories", []):
                warnings.append(f'"{name}" = "{text}" was not seen during training; it is ignored by the encoder.')
            row[name] = text

    frame = pd.DataFrame([row], columns=expected)
    positive, negative = schema["positive_class"], schema["negative_class"]
    pred = int(pipeline.predict(frame)[0])
    prob_pos = None
    class_probs = None
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(frame)[0]
        classes = list(pipeline.classes_)
        prob_pos = round(float(proba[classes.index(1)]), 6)
        class_probs = {positive: prob_pos, negative: round(1.0 - prob_pos, 6)}

    return PredictResponse(
        model_id=request.model_id,
        predicted_class=positive if pred == 1 else negative,
        is_positive_class=pred == 1,
        probability_positive=prob_pos,
        class_probabilities=class_probs,
        positive_class=positive,
        negative_class=negative,
        warnings=warnings,
        disclaimer=DISCLAIMER,
    )
