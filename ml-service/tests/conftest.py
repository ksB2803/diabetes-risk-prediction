import io
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from generate_synthetic_data import generate  # noqa: E402

from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402


@pytest.fixture()
def settings(tmp_path) -> Settings:
    return Settings(model_dir=tmp_path / "models")


@pytest.fixture()
def client(settings) -> TestClient:
    return TestClient(create_app(settings))


@pytest.fixture(scope="session")
def demo_df():
    return generate(n=240, seed=11)


@pytest.fixture()
def demo_csv(demo_df) -> bytes:
    return demo_df.to_csv(index=False).encode()


FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "BMI", "DiabetesPedigreeFunction", "Age", "PhysicalActivity", "FamilyHistory"]


def make_config(**overrides) -> dict:
    cfg = {
        "job_id": "test-1",
        "target_column": "Diabetes",
        "feature_columns": FEATURES,
        "train_fraction": 0.8,
        "algorithms": ["logistic_regression", "random_forest"],
        "scoring_metric": "roc_auc",
        "tuning_level": "quick",
        "random_seed": 42,
        "task_type": "binary_classification",
    }
    cfg.update(overrides)
    return cfg


def post_train(client, csv_bytes: bytes, filename="data.csv", **overrides):
    return client.post(
        "/ml/train",
        files={"file": (filename, io.BytesIO(csv_bytes), "text/csv")},
        data={"config": json.dumps(make_config(**overrides))},
    )


def post_inspect(client, csv_bytes: bytes, filename="data.csv"):
    return client.post("/ml/inspect", files={"file": (filename, io.BytesIO(csv_bytes), "text/csv")})
