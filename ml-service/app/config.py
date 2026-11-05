"""Runtime configuration for the ML service, read from environment variables.

Every limit has a safe default so the service runs without any configuration.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:  # pragma: no cover - configuration error
        raise RuntimeError(f"Environment variable {name} must be an integer") from exc


@dataclass(frozen=True)
class Settings:
    # Where trained pipelines and their metadata are written.
    model_dir: Path = field(
        default_factory=lambda: Path(os.getenv("ML_MODEL_DIR", "./storage/models")).resolve()
    )
    max_upload_bytes: int = field(default_factory=lambda: _int_env("ML_MAX_UPLOAD_BYTES", 20 * 1024 * 1024))
    max_rows: int = field(default_factory=lambda: _int_env("ML_MAX_ROWS", 100_000))
    max_columns: int = field(default_factory=lambda: _int_env("ML_MAX_COLUMNS", 100))
    min_rows: int = field(default_factory=lambda: _int_env("ML_MIN_ROWS", 20))
    min_rows_per_class: int = field(default_factory=lambda: _int_env("ML_MIN_ROWS_PER_CLASS", 5))
    max_categories: int = field(default_factory=lambda: _int_env("ML_MAX_CATEGORIES", 100))
    # Bounds on simultaneous training requests and parallel workers per search.
    max_concurrent_training: int = field(default_factory=lambda: _int_env("ML_MAX_CONCURRENT_TRAINING", 2))
    n_jobs: int = field(default_factory=lambda: _int_env("ML_N_JOBS", 1))
    # Optional shared secret. When set, every endpoint except /ml/health requires
    # the X-Internal-Api-Key header to match.
    internal_api_key: str = field(default_factory=lambda: os.getenv("ML_INTERNAL_API_KEY", ""))


def get_settings() -> Settings:
    return Settings()
