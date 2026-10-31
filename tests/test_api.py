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

def test_predict_with_explicit_trusted_local_model(monkeypatch, tmp_path):
    import joblib

    from industrial_pm.data import generate_synthetic_data
    from industrial_pm.model import train_model

    api, client = _api_client()
    model, _, _ = train_model(generate_synthetic_data(hours=24 * 45))
    path = tmp_path / "trusted_synthetic_model.joblib"
    joblib.dump(model, path)
    monkeypatch.setattr(api, "MODEL_PATH", path)
    response = client.post(
        "/predict",
        json={
            "machine_id": "machine_01",
            "timestamp": "2025-01-01T00:00:00Z",
            "vibration": 2.5,
            "temperature": 69.0,
            "current": 14.2,
            "rpm": 1498.0,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["machine_id"] == "machine_01"
    assert 0.0 <= body["failure_next_24h_score"] <= 1.0
