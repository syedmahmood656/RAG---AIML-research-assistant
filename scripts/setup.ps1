# One-time setup: creates venv, installs dependencies, creates .env
# Run from the project root:  .\scripts\setup.ps1
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$launcher = if (Get-Command py -ErrorAction SilentlyContinue) { "py" } else { "python" }
if (-not (Test-Path ".\venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Cyan
    & $launcher -m venv venv
}
Write-Host "Installing dependencies (first run downloads PyTorch - be patient)..." -ForegroundColor Cyan
& .\venv\Scripts\python.exe -m pip install --upgrade pip
& .\venv\Scripts\python.exe -m pip install -r requirements.txt

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env - open it and paste your Hugging Face token (optional)." -ForegroundColor Yellow
}
Write-Host "Setup complete. Start the app with:  .\scripts\start.ps1" -ForegroundColor Green
