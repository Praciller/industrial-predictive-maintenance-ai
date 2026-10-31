import pytest


def _api_client():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from industrial_pm import api

    return api, TestClient(api.app)


def test_predict_is_unavailable_without_artifact(monkeypatch, tmp_path):
    api, client = _api_client()
    monkeypatch.setattr(api, "MODEL_PATH", tmp_path / "missing.joblib")
    response = client.post(
        "/predict",
        json={
            "machine_id": "machine_01",
            "timestamp": "2025-01-01T00:00:00Z",
            "vibration": 2.0,
            "temperature": 68.0,
            "current": 14.0,
            "rpm": 1500.0,
        },
    )
    assert response.status_code == 200
    assert response.json()["status"] == "unavailable"


def test_predict_rejects_extra_and_naive_timestamp(monkeypatch, tmp_path):
    api, client = _api_client()
    monkeypatch.setattr(api, "MODEL_PATH", tmp_path / "missing.joblib")
    body = {
        "machine_id": "machine_01",
        "timestamp": "2025-01-01T00:00:00",
        "vibration": 2.0,
        "temperature": 68.0,
        "current": 14.0,
        "rpm": 1500.0,
    }
    assert client.post("/predict", json={**body, "extra": 1}).status_code == 422
    assert client.post("/predict", json=body).status_code == 422
