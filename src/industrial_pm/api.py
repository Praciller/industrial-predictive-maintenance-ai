"""Optional FastAPI service. Import with the api extra installed."""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, field_validator

from industrial_pm.features import FEATURE_COLUMNS

app = FastAPI(title="Industrial PM synthetic demo", version="0.1.0")
MODEL_PATH = Path(os.environ.get("INDUSTRIAL_PM_MODEL", "artifacts/model.joblib"))


class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    machine_id: str = Field(min_length=1, max_length=64)
    timestamp: str
    vibration: float = Field(ge=0, le=1000)
    temperature: float = Field(ge=-50, le=500)
    current: float = Field(ge=0, le=10000)
    rpm: float = Field(ge=0, le=100000)

    @field_validator("timestamp")
    @classmethod
    def require_utc_timestamp(cls, value: str) -> str:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None or timestamp.utcoffset().total_seconds() != 0:
            raise ValueError("timestamp must include UTC timezone")
        return value


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_available": MODEL_PATH.is_file()}


@app.post("/predict")
def predict(payload: PredictionInput) -> dict:
    if not MODEL_PATH.is_file():
        return {"status": "unavailable", "reason": "model artifact is not available"}
    timestamp = pd.Timestamp(payload.timestamp)
    bundle = joblib.load(MODEL_PATH)
    row = {name: getattr(payload, name) for name in ("vibration", "temperature", "current", "rpm")}
    feature_values = {**row, **{f"{name}_mean_6h": row[name] for name in row}}
    features = pd.DataFrame([feature_values], columns=FEATURE_COLUMNS)
    return {
        "status": "ok",
        "machine_id": payload.machine_id,
        "timestamp": timestamp.isoformat(),
        "failure_next_24h_score": float(bundle.predict_proba(features)[0]),
    }
