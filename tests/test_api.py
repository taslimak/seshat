from fastapi.testclient import TestClient

from seshat.api import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_prediction_contract() -> None:
    response = client.post(
        "/v1/predict",
        json={
            "prompt": "Explain linear regression in simple terms.",
            "model": "example-model",
            "effort": "medium",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["predicted_output_tokens"] >= 0
    assert body["lower_bound"] <= body["predicted_output_tokens"]
    assert body["upper_bound"] >= body["predicted_output_tokens"]
    assert 0.0 <= body["confidence"] <= 1.0
