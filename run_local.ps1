$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv-api")) {
    python -m venv .venv-api
}
& ".\.venv-api\Scripts\python.exe" -m pip install -r backend\requirements.txt

$api = Start-Process `
    -FilePath ".\.venv-api\Scripts\python.exe" `
    -ArgumentList "-m", "uvicorn", "app.main:app", "--app-dir", "backend", "--host", "0.0.0.0", "--port", "8000" `
    -PassThru

if (-not (Test-Path ".venv-ui")) {
    python -m venv .venv-ui
}
& ".\.venv-ui\Scripts\python.exe" -m pip install -r frontend\requirements.txt
$env:API_URL = "http://localhost:8000"
& ".\.venv-ui\Scripts\python.exe" -m streamlit run frontend\streamlit_app.py `
    --server.address=0.0.0.0 --server.port=8501

Stop-Process -Id $api.Id
