"""Build deterministic, score-only evidence for the static recruiter demo."""

from __future__ import annotations

import json
from pathlib import Path

from industrial_pm.data import LABEL, generate_synthetic_data
from industrial_pm.model import asset_chronological_split, evaluate, heuristic_scores, train_model

DEFAULT_HOURS = 24 * 90
DEFAULT_SEED = 42


def build_evidence(*, hours: int = DEFAULT_HOURS, seed: int = DEFAULT_SEED) -> dict:
    """Train on the reproducible fixture and return score arrays plus provenance."""
    data = generate_synthetic_data(hours=hours, seed=seed)
    train, _ = asset_chronological_split(data)
    model, x_test, test = train_model(data)
    model_scores = model.predict_proba(x_test)
    heuristic = heuristic_scores(test)
    truth = test[LABEL].to_numpy(dtype=int)
    return {
        "scope": "Synthetic fixture only. Not real-world evidence or a production estimate.",
        "generation": {"hours_per_asset": hours, "seed": seed, "assets": 3},
        "split": {
            "train_assets": sorted(train.machine_id.unique().tolist()),
            "test_assets": sorted(test.machine_id.unique().tolist()),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_end": str(train.timestamp.max()),
            "test_start": str(test.timestamp.min()),
            "positive": int(truth.sum()),
            "negative": int(len(truth) - truth.sum()),
        },
        "sources": {
            "model": {
                "name": "HistGradientBoosting classifier",
                "kind": "trained_model_score",
                "threshold": 0.5,
                "metrics": evaluate(truth, model_scores),
                "scores": model_scores.astype(float).tolist(),
            },
            "heuristic": {
                "name": "Fixed sensor threshold heuristic",
                "kind": "fixed_rule_score",
                "threshold": 0.5,
                "metrics": evaluate(truth, heuristic),
                "scores": heuristic.astype(float).tolist(),
            },
        },
        "actual": truth.tolist(),
    }


def write_evidence(
    destination: Path, *, hours: int = DEFAULT_HOURS, seed: int = DEFAULT_SEED
) -> None:
    """Write stable, compact JSON suitable for static hosting."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(build_evidence(hours=hours, seed=seed), separators=(",", ":"), allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build static synthetic recruiter demo evidence")
    parser.add_argument("--output", type=Path, default=Path("static/data.json"))
    parser.add_argument("--hours", type=int, default=DEFAULT_HOURS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    write_evidence(args.output, hours=args.hours, seed=args.seed)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
