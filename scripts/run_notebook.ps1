# Execute completed notebook with real outputs preserved
$workspaceRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $workspaceRoot ".venv\Scripts\python.exe"
$execScript = Join-Path $workspaceRoot "scripts\execute_notebook.py"

Write-Host "Executing completed notebook end-to-end..." -ForegroundColor Cyan
& $venvPython $execScript
