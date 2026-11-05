@echo off
REM Start the internal Python ML service on http://127.0.0.1:8000 (Windows).
REM First run creates a virtual environment and installs dependencies.
cd /d "%~dp0..\ml-service"

if not exist .venv (
  py -3 -m venv .venv || python -m venv .venv
  .venv\Scripts\python -m pip install --upgrade pip
  .venv\Scripts\pip install -r requirements.txt
)

if "%ML_PORT%"=="" set ML_PORT=8000
.venv\Scripts\uvicorn app.main:app --host 127.0.0.1 --port %ML_PORT%
