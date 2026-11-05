"""HTTP-level tests for the internal ML API."""
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

from .conftest import post_inspect, post_train


def test_health(client):
    res = client.get("/ml/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["model_store_writable"] is True


# ----------------------------------------------------------------- inspect
def test_inspect_valid_csv(client, demo_csv):
    res = post_inspect(client, demo_csv)
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["n_rows"] == 240
    assert body["n_columns"] == 10
    assert body["suggested_target"] == "Diabetes"
    assert len(body["preview"]["rows"]) == 8
    cols = {c["name"]: c for c in body["columns"]}
    assert "ID_LIKE" in cols["PatientID"]["warnings"]
    assert cols["PatientID"]["suggested_feature"] is False
    assert cols["Glucose"]["inferred_type"] == "numeric"
    assert cols["PhysicalActivity"]["inferred_type"] == "categorical"
    assert cols["Diabetes"]["target_status"] == "binary"
    assert cols["Diabetes"]["suggested_positive_class"] == "1"
    assert sum(c["count"] for c in cols["Diabetes"]["class_counts"]) == 240


def test_inspect_empty_file(client):
    res = post_inspect(client, b"")
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "EMPTY_FILE"


def test_inspect_header_only(client):
    res = post_inspect(client, b"a,b,c\n")
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "NO_ROWS"


def test_inspect_malformed_rows(client):
    res = post_inspect(client, b"a,b\n1,2\n3,4,5,6\n")
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "MALFORMED_CSV"


def test_inspect_binary_content_rejected(client):
    res = post_inspect(client, b"\x89PNG\r\n\x1a\n\x00\x00\x00binary")
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "NOT_A_CSV"


def test_inspect_wrong_extension(client, demo_csv):
    res = post_inspect(client, demo_csv, filename="data.xlsx")
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "NOT_A_CSV"


def test_inspect_duplicate_headers(client):
    res = post_inspect(client, b"a,a,b\n1,2,3\n")
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "DUPLICATE_HEADER"


def test_inspect_too_large(tmp_path, demo_csv):
    small = TestClient(create_app(Settings(model_dir=tmp_path / "m", max_upload_bytes=100)))
    res = post_inspect(small, demo_csv)
    assert res.status_code == 413


# ----------------------------------------------------------------- train
def test_train_valid_csv(client, settings, demo_csv):
    res = post_train(client, demo_csv)
    assert res.status_code == 200, res.text
    body = res.json()
    assert [m["algorithm"] for m in body["models"]] == ["logistic_regression", "random_forest"]
    assert body["split"]["train_rows"] + body["split"]["test_rows"] == body["dataset_summary"]["n_rows_used"]
    assert body["split"]["test_rows"] == round(0.2 * body["dataset_summary"]["n_rows_used"])
    assert body["class_mapping"]["positive_class"] == "1"
    best = max(body["models"], key=lambda m: m["cv_score"])
    assert body["selected_model_id"] == best["model_id"]
    for model in body["models"]:
        for name in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            assert 0.0 <= model["test_metrics"][name] <= 1.0
        assert model["roc_curve"]["fpr"][0] == 0.0
        assert sum(map(sum, model["confusion_matrix"]["matrix"])) == body["split"]["test_rows"]
        assert model["feature_importance"]["items"]
        assert (settings.model_dir / model["artifact"]["file"]).is_file()
    # The ID column was not selected, so it must not appear in the schema.
    assert "PatientID" not in [f["name"] for f in body["schema"]["features"]]


def test_train_decision_tree_and_other_metric(client, demo_csv):
    res = post_train(client, demo_csv, algorithms=["decision_tree"], scoring_metric="f1")
    assert res.status_code == 200, res.text
    assert res.json()["models"][0]["display_name"] == "Decision Tree"


def test_train_missing_target(client, demo_csv):
    res = post_train(client, demo_csv, target_column="Outcome")
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "TARGET_NOT_FOUND"


def test_train_missing_feature(client, demo_csv):
    res = post_train(client, demo_csv, feature_columns=["Glucose", "Insulin"])
    assert res.status_code == 400
    body = res.json()["error"]
    assert body["code"] == "FEATURES_NOT_FOUND"
    assert any("Insulin" in d for d in body["details"])


def test_train_target_as_feature_rejected(client, demo_csv):
    res = post_train(client, demo_csv, feature_columns=["Glucose", "Diabetes"])
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "TARGET_IN_FEATURES"


def test_train_single_class_rejected(client, demo_df):
    df = demo_df.copy()
    df["Diabetes"] = 0
    res = post_train(client, df.to_csv(index=False).encode())
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "ONE_CLASS_TARGET"


def test_train_multiclass_rejected(client, demo_df):
    df = demo_df.copy()
    df["Diabetes"] = [i % 3 for i in range(len(df))]
    res = post_train(client, df.to_csv(index=False).encode())
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "MULTICLASS_UNSUPPORTED"


def test_train_continuous_target_rejected(client, demo_df):
    df = demo_df.copy()
    df["Diabetes"] = df["BMI"].fillna(30.5) + 0.25
    res = post_train(client, df.to_csv(index=False).encode())
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "UNSUPPORTED_TARGET"


def test_train_text_labels_need_positive_class(client, demo_df):
    df = demo_df.copy()
    df["Diabetes"] = df["Diabetes"].map({0: "groupA", 1: "groupB"})
    csv = df.to_csv(index=False).encode()
    res = post_train(client, csv)
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "POSITIVE_CLASS_REQUIRED"
    res = post_train(client, csv, positive_class="groupB")
    assert res.status_code == 200, res.text
    assert res.json()["class_mapping"] == {
        "positive_class": "groupB",
        "negative_class": "groupA",
        "note": "The positive class is encoded as 1 and the other class as 0 inside the model.",
    }


def test_train_missing_target_labels_are_dropped_not_imputed(client, demo_df):
    df = demo_df.copy().astype({"Diabetes": "float"})
    df.loc[:9, "Diabetes"] = float("nan")
    res = post_train(client, df.to_csv(index=False).encode(), algorithms=["logistic_regression"])
    assert res.status_code == 200, res.text
    assert res.json()["dataset_summary"]["missing_target_rows_removed"] == 10


def test_train_too_small(client, demo_df):
    res = post_train(client, demo_df.head(12).to_csv(index=False).encode())
    assert res.status_code == 400
    assert res.json()["error"]["code"] in {"TOO_FEW_ROWS", "TOO_FEW_PER_CLASS"}


def test_train_invalid_config(client, demo_csv):
    res = post_train(client, demo_csv, train_fraction=0.97)
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "INVALID_CONFIG"
    res = post_train(client, demo_csv, algorithms=["svm"])
    assert res.status_code == 400


# ----------------------------------------------------------------- predict
@pytest.fixture()
def trained(client, demo_csv):
    res = post_train(client, demo_csv, algorithms=["logistic_regression"])
    assert res.status_code == 200, res.text
    return res.json()


def _example_values():
    return {
        "Pregnancies": 2,
        "Glucose": 150,
        "BloodPressure": 70,
        "BMI": 33.1,
        "DiabetesPedigreeFunction": 0.5,
        "Age": 45,
        "PhysicalActivity": "Low",
        "FamilyHistory": "Yes",
    }


def test_predict_valid(client, trained):
    model_id = trained["models"][0]["model_id"]
    res = client.post("/ml/predict", json={"model_id": model_id, "values": _example_values()})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["predicted_class"] in {"0", "1"}
    assert 0.0 <= body["probability_positive"] <= 1.0
    assert abs(sum(body["class_probabilities"].values()) - 1.0) < 1e-6
    assert "not a medical device" in body["disclaimer"]


def test_predict_missing_and_extra_features_rejected(client, trained):
    model_id = trained["models"][0]["model_id"]
    values = _example_values()
    values.pop("Glucose")
    values["Insulin"] = 80
    res = client.post("/ml/predict", json={"model_id": model_id, "values": values})
    assert res.status_code == 400
    err = res.json()["error"]
    assert err["code"] == "INCOMPATIBLE_FEATURES"
    assert any("Glucose" in d for d in err["details"])
    assert any("Insulin" in d for d in err["details"])


def test_predict_bad_type_rejected(client, trained):
    model_id = trained["models"][0]["model_id"]
    values = _example_values() | {"Glucose": "high"}
    res = client.post("/ml/predict", json={"model_id": model_id, "values": values})
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "INVALID_VALUE"


def test_predict_out_of_range_and_unknown_category_warn(client, trained):
    model_id = trained["models"][0]["model_id"]
    values = _example_values() | {"Glucose": 900, "PhysicalActivity": "Extreme"}
    res = client.post("/ml/predict", json={"model_id": model_id, "values": values})
    assert res.status_code == 200
    warnings = " ".join(res.json()["warnings"])
    assert "outside the training range" in warnings
    assert "not seen during training" in warnings


def test_predict_unknown_model(client):
    res = client.post("/ml/predict", json={"model_id": "0" * 32, "values": {"a": 1}})
    assert res.status_code == 404
    assert res.json()["error"]["code"] == "MODEL_NOT_FOUND"


def test_predict_rejects_path_like_model_id(client):
    res = client.post("/ml/predict", json={"model_id": "../../etc/passwd", "values": {"a": 1}})
    assert res.status_code == 400


def test_internal_api_key_enforced(tmp_path, demo_csv):
    secured = TestClient(create_app(Settings(model_dir=tmp_path / "m", internal_api_key="s3cret")))
    assert secured.get("/ml/health").status_code == 200
    assert post_inspect(secured, demo_csv).status_code == 401
    ok = secured.post(
        "/ml/inspect",
        files={"file": ("d.csv", demo_csv, "text/csv")},
        headers={"X-Internal-Api-Key": "s3cret"},
    )
    assert ok.status_code == 200
