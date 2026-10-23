
import pytest
from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True

    with app.test_client() as test_client:
        yield test_client


def test_home_page_loads(client):
    response = client.get("/")
    assert response.status_code == 200


def test_dashboard_loads(client):
    response = client.get("/dashboard")
    assert response.status_code == 200


def test_predict_route_rejects_missing_fields(client):
    response = client.post("/predict", data={})

    # The app should handle incomplete input without crashing.
    assert response.status_code in (200, 400, 422)


def test_unknown_page_returns_404(client):
    response = client.get("/this-page-does-not-exist")
    assert response.status_code == 404
    
def test_predict_route_with_valid_customer_data(client):
    customer_data = {
        "gender": "Female",
        "SeniorCitizen": "0",
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": "12",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": "70.35",
        "TotalCharges": "844.20",
    }

    response = client.post("/predict", data=customer_data)

    assert response.status_code == 200

