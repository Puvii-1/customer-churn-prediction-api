"""
FastAPI service for the churn model.

Endpoints:
  GET  /health   — liveness/readiness check
  POST /predict  — churn prediction for one customer
  GET  /metrics  — basic request count, latency, and prediction stats

Run with: uvicorn src.api:app --reload
"""
import time
import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from src.schemas import ChurnRequest, ChurnResponse, HealthResponse

MODEL_PATH = os.getenv("MODEL_PATH", "models/churn_model.joblib")
LOG_PATH = os.getenv("PREDICTION_LOG_PATH", "logs/predictions.jsonl")

model = None
metrics_state = {
    "request_count": 0,
    "total_latency_ms": 0.0,
    "churn_predictions": 0,
    "no_churn_predictions": 0,
    "errors": 0,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    else:
        model = None  # /health will report this; /predict will 503
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    yield


app = FastAPI(title="Churn Prediction API", version="1.0.0", lifespan=lifespan)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", include_in_schema=False)
def root():
    return FileResponse("static/index.html")


def _log_prediction(request_data: dict, probability: float, latency_ms: float):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input": request_data,
        "churn_probability": probability,
        "latency_ms": round(latency_ms, 2),
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _risk_level(probability: float) -> str:
    if probability >= 0.7:
        return "high"
    if probability >= 0.4:
        return "medium"
    return "low"


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", model_loaded=model is not None)


@app.post("/predict", response_model=ChurnResponse)
def predict(request: ChurnRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train a model first (python -m src.train).")

    start = time.perf_counter()
    input_df = pd.DataFrame([request.model_dump()])

    try:
        probability = float(model.predict_proba(input_df)[0, 1])
    except Exception as e:
        metrics_state["errors"] += 1
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    latency_ms = (time.perf_counter() - start) * 1000

    metrics_state["request_count"] += 1
    metrics_state["total_latency_ms"] += latency_ms
    will_churn = probability >= 0.5
    if will_churn:
        metrics_state["churn_predictions"] += 1
    else:
        metrics_state["no_churn_predictions"] += 1

    _log_prediction(request.model_dump(), probability, latency_ms)

    return ChurnResponse(
        churn_probability=round(probability, 4),
        will_churn=will_churn,
        risk_level=_risk_level(probability),
    )


@app.get("/metrics")
def metrics():
    count = metrics_state["request_count"]
    avg_latency = (metrics_state["total_latency_ms"] / count) if count else 0.0
    return {
        "total_requests": count,
        "average_latency_ms": round(avg_latency, 2),
        "churn_predictions": metrics_state["churn_predictions"],
        "no_churn_predictions": metrics_state["no_churn_predictions"],
        "errors": metrics_state["errors"],
    }