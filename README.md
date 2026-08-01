# TAMVER Scenario Lab AI Assistant

Рабочий прототип **Strategic Decision Intelligence Platform**:

- AIR propagation engine;
- нелинейное распространение влияния;
- Monte Carlo stress testing;
- network centrality;
- decision nodes;
- автоматический поиск решений;
- объяснение причинных цепочек;
- живой AI-ассистент внутри Streamlit Scenario Lab;
- FastAPI backend;
- Docker Compose.

## Быстрый запуск через Docker

Из корня проекта:

```bash
docker compose up --build
```

После запуска:

- Scenario Lab: http://localhost:8501
- FastAPI Swagger: http://localhost:8000/docs
- Health check: http://localhost:8000/health

## Локальный запуск без Docker

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Frontend

Во втором терминале:

```bash
cd frontend
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Команды AI-ассистенту

Примеры:

- `Снизь системный риск при бюджете 0.4`
- `Найди наиболее устойчивую стратегию`
- `Покажи критические узлы`
- `Проведи стресс-тест`
- `Снизь риск и минимизируй затраты`

## Структура

```text
backend/
  app/
    engine.py      # AIR, Monte Carlo, centrality
    agent.py       # поиск решений и объяснение
    schemas.py     # Pydantic API models
    main.py        # FastAPI endpoints
  tests/
frontend/
  streamlit_app.py
data/
  sample_graph.json
docker-compose.yml
```

## API

### POST `/simulate`

Ручная симуляция выбранного решения.

### POST `/agent/recommend`

Автоматический поиск и ранжирование вариантов решения.

### POST `/assistant/respond`

Живой ассистент. Определяет намерение пользователя:

- рекомендация;
- stress test;
- critical nodes.

## Математическая логика

Матрица связей строится по правилу:

```text
W[target, source] = edge_weight × confidence
```

Распространение:

```text
x(t+1) = tanh(α W x(t))
```

Полный эффект:

```text
Impact = Σ x(t)
```

Если `α × spectral_radius(W) ≥ 0.98`, затухание автоматически уменьшается, чтобы ограничить неустойчивое распространение.

## Что усилить перед production

1. PostgreSQL и versioned model registry.
2. Tenant isolation, JWT, RBAC и audit log.
3. Celery/RQ worker для крупных Monte Carlo запусков.
4. Реальный LLM-провайдер только как слой объяснения и генерации гипотез.
5. Approval workflow: AI предлагает, человек утверждает.
6. Calibration registry для коэффициентов AIR.
7. Decision Audit Report в DOCX/PDF.
8. Тесты на исторических кейсах и backtesting.
