"""Causal feature transforms using only present and earlier observations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from industrial_pm.data import LABEL, SENSORS

FEATURE_COLUMNS = [*SENSORS, *(f"{name}_mean_6h" for name in SENSORS)]


def make_features(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return feature matrix and labels, preserving rows and using backward windows only."""
    required = {"timestamp", "machine_id", LABEL, *SENSORS}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    frame = data.copy()
    if not pd.api.types.is_datetime64_any_dtype(frame["timestamp"]):
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    if frame["timestamp"].dt.tz is None:
        raise ValueError("timestamp must be timezone-aware UTC")
    frame = frame.sort_values(["machine_id", "timestamp"], kind="stable")
    features = frame.loc[:, SENSORS].astype(float).copy()
    for sensor in SENSORS:
        features[f"{sensor}_mean_6h"] = frame.groupby("machine_id", sort=False)[sensor].transform(
            lambda values: values.rolling(window=6, min_periods=1).mean()
        )
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("sensor features must be finite")
    labels = frame[LABEL].astype(int)
    return features.reset_index(drop=True), labels.reset_index(drop=True)
