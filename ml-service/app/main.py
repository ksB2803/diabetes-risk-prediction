"""Internal FastAPI ML service.

This service is called only by the Spring Boot backend, never by the browser.
Endpoints:
    GET  /ml/health   liveness + model store check
    POST /ml/inspect  profile an uploaded CSV (multipart: file)
    POST /ml/train    train and evaluate models (multipart: file, config JSON)
    POST /ml/predict  predict one record with a saved pipeline (JSON)
"""
from __future__ import annotations

import logging
import threading

import sklearn
from fastapi import Depends, FastAPI, File, Form, Header, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .artifact_store import ArtifactStore
from .config import Settings, get_settings
from .data_validation import load_csv, profile_dataset
from .errors import MLError
from .prediction import predict as run_predict
from .schemas import HealthResponse, PredictRequest, PredictResponse, TrainConfig
from .training import run_training

logger = logging.getLogger("ml-service")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    store = ArtifactStore(settings.model_dir)
    store.ensure_dir()
    training_slots = threading.BoundedSemaphore(max(1, settings.max_concurrent_training))

    app = FastAPI(title="Diabetes Risk Prediction ML service", version="1.0.0", docs_url="/ml/docs", openapi_url="/ml/openapi.json")
    app.state.settings = settings
    app.state.store = store

    def require_key(x_internal_api_key: str | None = Header(default=None)) -> None:
        if settings.internal_api_key and x_internal_api_key != settings.internal_api_key:
            raise MLError("UNAUTHORIZED", "Missing or invalid internal API key.", status_code=401)

    @app.exception_handler(MLError)
    async def _ml_error(_: Request, exc: MLError):
        return JSONResponse(status_code=exc.status_code, content=exc.to_dict())

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError):
        details = [f"{'.'.join(str(p) for p in e.get('loc', []))}: {e.get('msg', 'invalid')}" for e in exc.errors()]
        return JSONResponse(
            status_code=400,
            content={"error": {"code": "VALIDATION_ERROR", "message": "The request is not valid.", "details": details[:20]}},
        )

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, exc: Exception):
        logger.error("Unexpected error: %s", type(exc).__name__)
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "INTERNAL_ERROR", "message": "The ML service hit an unexpected error.", "details": []}},
        )

    def _read_upload(file: UploadFile) -> bytes:
        name = (file.filename or "").lower()
        if not name.endswith(".csv"):
            raise MLError("NOT_A_CSV", "Only .csv files are accepted.")
        data = file.file.read(settings.max_upload_bytes + 1)
        if len(data) > settings.max_upload_bytes:
            raise MLError("FILE_TOO_LARGE", "The file is larger than the upload limit.", status_code=413)
        return data

    @app.get("/ml/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", sklearn_version=sklearn.__version__, model_store_writable=store.is_writable())

    @app.post("/ml/inspect", dependencies=[Depends(require_key)])
    def inspect(file: UploadFile = File(...)) -> dict:
        data = _read_upload(file)
        df = load_csv(data, settings)
        logger.info("Inspected CSV: %d rows x %d columns", df.shape[0], df.shape[1])
        safe_name = (file.filename or "upload.csv").replace("\\", "/").split("/")[-1][:200]
        return profile_dataset(df, safe_name, len(data), settings)

    @app.post("/ml/train", dependencies=[Depends(require_key)])
    def train(file: UploadFile = File(...), config: str = Form(...)) -> dict:
        try:
            cfg = TrainConfig.model_validate_json(config)
        except ValidationError as exc:
            details = [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()]
            raise MLError("INVALID_CONFIG", "The training configuration is not valid.", details=details[:20]) from exc
        data = _read_upload(file)
        if not training_slots.acquire(blocking=False):
            raise MLError("BUSY", "The ML service is already training the maximum number of jobs. Try again shortly.", status_code=429)
        try:
            df = load_csv(data, settings)
            logger.info("Training job %s: %d rows, %d features, algorithms=%s, tuning=%s",
                        cfg.job_id, df.shape[0], len(cfg.feature_columns), cfg.algorithms, cfg.tuning_level)
            result = run_training(df, cfg, settings, store)
            logger.info("Training job %s finished in %.1fs", cfg.job_id, result["duration_seconds"])
            return result
        finally:
            training_slots.release()

    @app.post("/ml/predict", response_model=PredictResponse, dependencies=[Depends(require_key)])
    def predict(request: PredictRequest) -> PredictResponse:
        return run_predict(store, request)

    return app


app = create_app()
