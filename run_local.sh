#!/usr/bin/env bash
set -euo pipefail

python -m venv .venv-api
source .venv-api/bin/activate
pip install -r backend/requirements.txt
uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000 &
API_PID=$!

deactivate
python -m venv .venv-ui
source .venv-ui/bin/activate
pip install -r frontend/requirements.txt
API_URL=http://localhost:8000 streamlit run frontend/streamlit_app.py \
  --server.address=0.0.0.0 --server.port=8501

kill "$API_PID"
