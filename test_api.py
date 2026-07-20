"""
Tests for the FastAPI service. Uses FastAPI's TestClient, which doesn't
need a running server — it calls the app directly in-process.
"""
import os
import pytest
from fastapi.testclient import TestClient

# Ensure a model exists before the app tries to load it at import time
if not os.path.exists("models/churn_model.joblib"):
    from src.train import train
    train()

from src.api import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_predict_high_risk_customer(client):
    payload = {
        "tenure_months": 2,
        "monthly_charges": 95.0,
        "total_charges": 190.0,
        "num_support_tickets": 5,
        "contract_type": "month-to-month",
        "is_senior_citizen": 0,
        "has_tech_support": 0,
        "payment_delay_days": 8.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["will_churn"] is True
    assert data["risk_level"] == "high"


def test_predict_low_risk_customer(client):
    payload = {
        "tenure_months": 60,
        "monthly_charges": 55.0,
        "total_charges": 3300.0,
        "num_support_tickets": 0,
        "contract_type": "two_year",
        "is_senior_citizen": 0,
        "has_tech_support": 1,
        "payment_delay_days": 0.2,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["will_churn"] is False
    assert data["risk_level"] == "low"


def test_predict_rejects_invalid_contract_type(client):
    payload = {
        "tenure_months": 5, "monthly_charges": 89.5, "total_charges": 447.5,
        "num_support_tickets": 4, "contract_type": "monthly",  # invalid
        "is_senior_citizen": 0, "has_tech_support": 0, "payment_delay_days": 6.5,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Pydantic validation error


def test_predict_rejects_negative_tenure(client):
    payload = {
        "tenure_months": -1, "monthly_charges": 89.5, "total_charges": 447.5,
        "num_support_tickets": 4, "contract_type": "month-to-month",
        "is_senior_citizen": 0, "has_tech_support": 0, "payment_delay_days": 6.5,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_metrics_reflects_requests(client):
    client.get("/metrics")  # baseline call
    before = client.get("/metrics").json()["total_requests"]

    payload = {
        "tenure_months": 10, "monthly_charges": 60.0, "total_charges": 600.0,
        "num_support_tickets": 1, "contract_type": "one_year",
        "is_senior_citizen": 0, "has_tech_support": 1, "payment_delay_days": 1.0,
    }
    client.post("/predict", json=payload)

    after = client.get("/metrics").json()["total_requests"]
    assert after == before + 1