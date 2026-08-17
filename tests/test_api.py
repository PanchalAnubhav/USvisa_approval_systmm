import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_health():
    """Test the health check endpoint returns status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_home_page():
    """Test the main web UI loads successfully with 200 OK and HTML content."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "US Visa Approval" in response.text


def test_prediction_approved_case():
    """Test prediction endpoint with known approved feature combination."""
    payload = {
        "continent": "North America",
        "education_of_employee": "Master's",
        "has_job_experience": "Y",
        "requires_job_training": "N",
        "no_of_employees": 5000,
        "region_of_employment": "Northeast",
        "prevailing_wage": 95000.0,
        "unit_of_wage": "Year",
        "full_time_position": "Y",
        "company_age": 20,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] is True
    assert data["prediction"] == "Visa-Approved"
    assert data["prediction_value"] == 1


def test_prediction_denied_case():
    """Test prediction endpoint with known denied feature combination."""
    payload = {
        "continent": "Asia",
        "education_of_employee": "Doctorate",
        "has_job_experience": "N",
        "requires_job_training": "Y",
        "no_of_employees": 50,
        "region_of_employment": "South",
        "prevailing_wage": 20000.0,
        "unit_of_wage": "Year",
        "full_time_position": "N",
        "company_age": 2,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] is True
    assert data["prediction"] == "Visa Not-Approved"
    assert data["prediction_value"] == 0


def test_invalid_prediction_request_validation():
    """Test prediction endpoint returns 422 Unprocessable Entity on missing/invalid schema."""
    payload = {
        "continent": "Asia",
        "no_of_employees": -10,  # Negative number should fail Field(..., ge=0)
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_retrain_status_endpoint():
    """Test the /retrain/status endpoint returns valid status structure."""
    response = client.get("/retrain/status")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] in ["idle", "running", "success", "failed"]
    assert "message" in data