@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
start "" http://localhost:8000
uvicorn djprep.api.main:app --app-dir backend
