# Starts the API (port 7860) and the UI (port 8501) in two new PowerShell windows.
# Run from the project root:  .\scripts\start.ps1
$root = Split-Path $PSScriptRoot -Parent
$py = Join-Path $root "venv\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Host "venv not found - run .\scripts\setup.ps1 first." -ForegroundColor Red; exit 1 }

Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$root'; & '$py' -m uvicorn app:app --app-dir backend --host 127.0.0.1 --port 7860"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$root'; & '$py' -m streamlit run frontend/app.py --server.port 8501"

Write-Host "API  -> http://127.0.0.1:7860/docs" -ForegroundColor Cyan
Write-Host "UI   -> http://localhost:8501  (wait ~30 s on the very first start)" -ForegroundColor Green
