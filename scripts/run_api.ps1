# Run AeroDamage AI FastAPI service with Uvicorn
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$workspaceRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $workspaceRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Error "Virtual environment not found at $venvPython. Please create and install dependencies first."
}

Write-Host "Starting AeroDamage AI FastAPI Service on http://127.0.0.1:8000 ..." -ForegroundColor Cyan
& $venvPython -m uvicorn aircraft_ai.api:app --host 127.0.0.1 --port 8000 --reload --app-dir (Join-Path $workspaceRoot "src")
