from fastapi.testclient import TestClient

from app import app


client = TestClient(app)


def test_health():

    response = client.get("/health")

    assert response.status_code == 200

    assert response.json()["status"] == "ok"


def test_home_page():

    response = client.get("/")

    assert response.status_code == 200


def test_prediction_validation():

    payload = {
        "continent": "Asia",
        "education_of_employee": "Master's",
        "has_job_experience": "Y",
        "requires_job_training": "N",
        "no_of_employees": 1000,
        "region_of_employment": "West",
        "prevailing_wage": 75000,
        "unit_of_wage": "Year",
        "full_time_position": "Y",
        "company_age": 20,
    }

    # The endpoint may require S3/model access,
    # so this test only verifies that the request
    # passes FastAPI validation.
    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code in [200, 500]


def test_invalid_prediction_request():

    payload = {
        "continent": "Asia",
        "no_of_employees": -10,
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422