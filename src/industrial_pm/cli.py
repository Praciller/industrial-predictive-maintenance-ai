"""Command line demonstration entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from industrial_pm.data import LABEL, generate_synthetic_data
from industrial_pm.model import asset_chronological_split, evaluate, heuristic_scores, train_model


def run_demo(output: Path, hours: int = 24 * 90, seed: int = 42) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    data = generate_synthetic_data(hours=hours, seed=seed)
    data.to_csv(output / "synthetic_sensor_data.csv", index=False)
    train, test = asset_chronological_split(data)
    model, x_test, test = train_model(data)
    trained_scores = model.predict_proba(x_test)
    # Recompute baseline on the same held-out test rows.
    metrics = {
        "scope": "deterministic synthetic data only; no real-world performance claim",
        "split": {
            "train_assets": sorted(train.machine_id.unique()),
            "test_assets": sorted(test.machine_id.unique()),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_end": str(train.timestamp.max()),
            "test_start": str(test.timestamp.min()),
        },
        "model": evaluate(test[LABEL], trained_scores),
        "heuristic_baseline": evaluate(test[LABEL], heuristic_scores(test)),
    }
    joblib.dump(model, output / "model.joblib")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the deterministic synthetic PM demo")
    parser.add_argument("demo", nargs="?", default="demo", choices=["demo"])
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--hours", type=int, default=24 * 90)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(run_demo(args.output, args.hours, args.seed), indent=2))


if __name__ == "__main__":
    main()
