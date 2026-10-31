"""Deterministic synthetic sensor streams and forward-looking labels."""

from __future__ import annotations

import numpy as np
import pandas as pd

SENSORS = ("vibration", "temperature", "current", "rpm")
LABEL = "failure_next_24h"
ASSETS = ("machine_01", "machine_02", "machine_03")


def generate_synthetic_data(
    *, hours: int = 24 * 90, seed: int = 42, start: str = "2025-01-01T00:00:00Z"
) -> pd.DataFrame:
    """Generate hourly UTC sensor readings with deterministic simulated fault windows.

    Failure episodes are seeded independently per asset. The label at time *t* is one
    when the simulated failure instant is in (t, t + 24h], so current readings are
    available at prediction time and the label is strictly forward-looking.
    """
    if not isinstance(hours, int) or hours < 48:
        raise ValueError("hours must be an integer of at least 48")
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    base = pd.Timestamp(start)
    if base.tzinfo is None:
        base = base.tz_localize("UTC")
    else:
        base = base.tz_convert("UTC")
    frames: list[pd.DataFrame] = []
    for asset_index, asset in enumerate(ASSETS):
        # Non-overlapping synthetic date blocks enable a true asset holdout that is
        # also strictly later in time. This is a split convenience, not an operating scenario.
        timestamps = pd.date_range(
            base + pd.Timedelta(hours=asset_index * hours), periods=hours, freq="h"
        )
        rng = np.random.default_rng(seed + asset_index * 1009)
        t = np.arange(hours, dtype=float)
        phase = asset_index * 0.7
        earliest_failure = max(1, hours // 3)
        candidate_failures = np.arange(earliest_failure, hours - 24)
        n_failures = min(max(2, hours // 360), len(candidate_failures))
        failures = np.sort(rng.choice(candidate_failures, size=n_failures, replace=False))
        hours_to_failure = np.full(hours, np.inf)
        for failure in failures:
            mask = (t < failure) & (t >= failure - 36)
            hours_to_failure[mask] = np.minimum(hours_to_failure[mask], failure - t[mask])
        degradation = np.where(hours_to_failure <= 36, (37 - hours_to_failure) / 37, 0.0)
        vibration = (
            2.2 + 0.22 * np.sin(t / 30 + phase) + rng.normal(0, 0.10, hours) + 1.5 * degradation
        )
        temperature = (
            68 + 2.5 * np.sin(t / 80 + phase) + rng.normal(0, 1.0, hours) + 13 * degradation
        )
        current = 14 + 0.7 * np.sin(t / 18 + phase) + rng.normal(0, 0.3, hours) + 2.4 * degradation
        rpm = 1500 + 12 * np.sin(t / 12 + phase) + rng.normal(0, 5, hours) - 45 * degradation
        label = np.zeros(hours, dtype=np.int8)
        for failure in failures:
            label[(t < failure) & (t >= failure - 24)] = 1
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "machine_id": asset,
                    "vibration": vibration,
                    "temperature": temperature,
                    "current": current,
                    "rpm": rpm,
                    LABEL: label,
                }
            )
        )
    result = pd.concat(frames, ignore_index=True)
    return result.sort_values(["timestamp", "machine_id"], kind="stable").reset_index(drop=True)
