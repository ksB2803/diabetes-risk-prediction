#!/usr/bin/env bash
# Start the internal Python ML service on http://127.0.0.1:8000 (Linux/macOS).
# First run creates a virtual environment and installs dependencies.
set -euo pipefail
cd "$(dirname "$0")/../ml-service"

PYTHON="${PYTHON:-python3}"
if [ ! -d .venv ]; then
  "$PYTHON" -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install -r requirements.txt
fi

if [ -f ../.env ]; then
  set -a; . ../.env; set +a
fi

exec .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port "${ML_PORT:-8000}"
