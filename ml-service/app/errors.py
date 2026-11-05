"""A single error type for predictable, user-safe failures.

Messages must never contain filesystem paths, stack traces or raw data rows.
"""
from __future__ import annotations


class MLError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 400, details: list[str] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or []

    def to_dict(self) -> dict:
        return {"error": {"code": self.code, "message": self.message, "details": self.details}}
