# Customer Churn Prediction — MLOps Pipeline

A small ML model wrapped in a full production stack: experiment tracking,
a served API, containerization, automated tests, and CI/CD. The model
itself (a random forest churn classifier) is intentionally simple —
the point of this project is everything _around_ the model.

## Why this project

Building a model is one skill. Shipping it reliably is a different one —
experiment tracking, serving, testing, containerization, and automated
pipelines. This project demonstrates that second skill set end-to-end,
on a problem simple enough that the deployment work stays the focus.

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌───────────────┐
│  data.py      │────▶│  train.py     │────▶│  MLflow        │
│  (synthetic   │     │  (RandomForest│     │  (params,      │
│  churn data)  │     │   + pipeline) │     │   metrics,     │
└──────────────┘     └──────┬───────┘     │   artifacts)   │
                             │              └───────────────┘
                             ▼
                      models/churn_model.joblib
                             │
                             ▼
                     ┌───────────────┐
                     │  api.py        │  FastAPI: /health, /predict, /metrics
                     │  (FastAPI)     │  logs every prediction to logs/predictions.jsonl
                     └──────┬───────┘
                             │
                             ▼
                     ┌───────────────┐
                     │  Dockerfile    │  self-contained image: trains model at
                     │                │  build time, serves via uvicorn
                     └───────────────┘
                             │
                             ▼
                  .github/workflows/ci.yml
                  runs tests + builds Docker image on every push
```

## Model performance

Trained on 5,000 synthetic customers (28% churn rate, matching typical
real-world churn rates):

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 0.78  |
| Precision | 0.60  |
| Recall    | 0.71  |
| F1        | 0.65  |
| ROC AUC   | 0.84  |

(Exact numbers vary slightly by random seed / retraining, but should stay
in this range.)

## Setup

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Running the pipeline locally

**1. Generate data and train the model** (also logs to MLflow):

```powershell
python -m src.data
python -m src.train
```

**2. View experiment tracking:**

```powershell
mlflow ui
```

Open http://localhost:5000 — you'll see the run's parameters, metrics,
confusion matrix image, and the saved model artifact.

**3. Serve the API:**

```powershell
uvicorn src.api:app --reload
```

Open http://localhost:8000/docs for interactive Swagger UI — you can
try `/predict` right from the browser.

**4. Run tests:**

```powershell
pytest tests/ -v
```

## Running with Docker

```powershell
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

Or with Docker Compose:

```powershell
docker-compose up --build
```

The image trains the model at build time, so it's fully self-contained —
no separate setup step needed after `docker run`.

## API reference

**`POST /predict`**

```json
{
  "tenure_months": 5,
  "monthly_charges": 89.5,
  "total_charges": 447.5,
  "num_support_tickets": 4,
  "contract_type": "month-to-month",
  "is_senior_citizen": 0,
  "has_tech_support": 0,
  "payment_delay_days": 6.5
}
```

Returns:

```json
{
  "churn_probability": 0.96,
  "will_churn": true,
  "risk_level": "high"
}
```

**`GET /health`** → liveness check, reports whether the model loaded successfully.

**`GET /metrics`** → request count, average latency, and prediction breakdown — a
minimal monitoring signal without needing a full observability stack.

## CI/CD

`.github/workflows/ci.yml` runs on every push/PR to `main`:

1. Installs dependencies
2. Regenerates data and retrains the model
3. Runs the full test suite
4. Builds the Docker image (doesn't push anywhere by default — see below to extend)

## Project structure

```
churn-mlops/
├── src/
│   ├── data.py       # synthetic churn dataset generator
│   ├── train.py       # training + MLflow logging
│   ├── schemas.py       # Pydantic request/response models
│   └── api.py             # FastAPI service
├── tests/
│   ├── test_model.py         # data + training pipeline tests
│   └── test_api.py             # API endpoint tests
├── .github/workflows/ci.yml
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Honest notes on scope (things worth knowing if asked about this project)

- **Model retrains at Docker build time**, not loaded from a registry.
  For a real production setup, you'd instead pull a specific model
  version from MLflow's Model Registry or S3 — this project doesn't
  implement that to keep within a 1-2 week scope, but it's a natural
  next step (see below).
- **Monitoring is intentionally minimal** — `/metrics` gives request
  count, latency, and prediction breakdown, logged to a JSONL file.
  A production system would use Prometheus + Grafana for this instead.
- **No Kubernetes** in this version — the Dockerfile and docker-compose
  cover single-container deployment; a Helm chart / k8s manifests would
  be the natural extension if you want to demonstrate that too.
- **CI builds but doesn't push** the Docker image anywhere (no registry
  configured) — wiring it to push to Docker Hub or GHCR is a small
  addition if you want the pipeline to be fully deployable.

## Ideas to extend further

- Push the CI-built Docker image to GitHub Container Registry (GHCR) automatically
- Add a `/predict/batch` endpoint for scoring many customers at once
- Add data drift detection: compare incoming request feature distributions
  against training data periodically
- Add a simple Kubernetes deployment manifest + service for a "k8s" talking point
- Swap the JSONL prediction log for a Prometheus client + Grafana dashboard
