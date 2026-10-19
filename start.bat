@echo off
echo Starting AeroDamage AI server and launching browser...
start "" "http://127.0.0.1:8000"
.\.venv\Scripts\python.exe -m uvicorn aircraft_ai.api:app --host 127.0.0.1 --port 8000 --app-dir src
