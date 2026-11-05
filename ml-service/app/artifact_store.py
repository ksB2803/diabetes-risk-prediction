"""Storage for fitted pipelines produced by this service.

Only files written by `save_model` are ever loaded. Model IDs are random 32-char
hex strings and are validated before touching the filesystem, so a caller cannot
point the loader at an arbitrary path. There is no endpoint for uploading models.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import joblib

from .errors import MLError

_MODEL_ID = re.compile(r"^[a-f0-9]{32}$")
_CACHE_SIZE = 8


class ArtifactStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self._cache: "OrderedDict[str, tuple[object, dict]]" = OrderedDict()
        self._lock = threading.Lock()

    def ensure_dir(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def is_writable(self) -> bool:
        try:
            self.ensure_dir()
            with tempfile.NamedTemporaryFile(dir=self.root, prefix=".probe-", delete=True):
                pass
            return True
        except OSError:
            return False

    def _paths(self, model_id: str) -> tuple[Path, Path]:
        if not _MODEL_ID.match(model_id):
            raise MLError("BAD_MODEL_ID", "Model ID is not valid.", status_code=400)
        return self.root / f"{model_id}.joblib", self.root / f"{model_id}.json"

    def _atomic_write(self, target: Path, writer) -> None:
        fd, tmp = tempfile.mkstemp(dir=self.root, prefix=".tmp-", suffix=target.suffix)
        os.close(fd)
        try:
            writer(tmp)
            os.replace(tmp, target)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def save_model(self, pipeline, metadata: dict) -> dict:
        self.ensure_dir()
        model_id = uuid.uuid4().hex
        model_path, meta_path = self._paths(model_id)
        created_at = datetime.now(timezone.utc).isoformat()
        meta = dict(metadata, model_id=model_id, created_at=created_at, artifact_file=model_path.name)

        self._atomic_write(model_path, lambda p: joblib.dump(pipeline, p, compress=3))

        def write_meta(p):
            with open(p, "w", encoding="utf-8") as fh:
                json.dump(meta, fh, indent=2)

        self._atomic_write(meta_path, write_meta)
        return {"model_id": model_id, "created_at": created_at, "artifact_file": model_path.name}

    def load_model(self, model_id: str) -> tuple[object, dict]:
        with self._lock:
            if model_id in self._cache:
                self._cache.move_to_end(model_id)
                return self._cache[model_id]
        model_path, meta_path = self._paths(model_id)
        if not model_path.is_file() or not meta_path.is_file():
            raise MLError("MODEL_NOT_FOUND", "The requested model is not available.", status_code=404)
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        pipeline = joblib.load(model_path)
        with self._lock:
            self._cache[model_id] = (pipeline, meta)
            while len(self._cache) > _CACHE_SIZE:
                self._cache.popitem(last=False)
        return pipeline, meta
