# Industrial Predictive Maintenance AI

A small, reproducible AI engineering portfolio project. It generates deterministic **synthetic** hourly readings for three machines (vibration, temperature, current, RPM), creates an explicit `failure_next_24h` target, compares a classical classifier with a fixed-threshold heuristic, and reports held-out metrics.

**This is a software demonstration, not validated predictive maintenance.** The signals, fault episodes, labels, and resulting scores are synthetic and should not be interpreted as real-world efficacy, safety guidance, or an operating threshold.

## Recruiter quickstart

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```powershell
uv sync --extra dev --extra api
uv run pytest
uv run ruff check .
uv run industrial-pm demo
```

The CLI writes `artifacts/synthetic_sensor_data.csv`, `artifacts/metrics.json`, and `artifacts/model.joblib`. These generated files are ignored by git. Run `uv run industrial-pm demo --hours 2160 --seed 42` to reproduce the default 90-day-per-machine dataset.

### Optional API

```powershell
uv sync --extra api
uv run uvicorn industrial_pm.api:app --app-dir src
```

`GET /health` reports service and artifact availability. `POST /predict` accepts a strict JSON sensor record and requires an offset-aware UTC timestamp. Without `artifacts/model.joblib`, prediction returns `{"status":"unavailable", ...}`. With an artifact, it returns a model score, not a calibrated probability or validated risk estimate. The input-only six-hour features use the current reading as their window mean because the endpoint is stateless.

## What this demonstrates

- A seeded, deterministic synthetic generator with both positive and negative forward labels.
- Backward-only per-machine rolling features.
- Asset-disjoint, strictly chronological train/test split, with no timestamp overlap.
- Scikit-learn classifier and transparent, fixed-threshold heuristic baseline.
- Precision, recall, F1, average precision (PR-AUC summary), confusion counts, and their denominators.
- Optional schema-validated FastAPI endpoint with explicit missing-model behavior.

See [architecture](docs/architecture.md) and [limitations](docs/limitations.md).

## License

MIT. See [LICENSE](LICENSE).

