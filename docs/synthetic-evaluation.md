# Synthetic baseline evidence · 2026-10-31

**Scope:** Offline deterministic fixture, `industrial-pm demo --hours 2160 --seed 42`. This is **not a real factory dataset or an estimate of production performance**. The generator intentionally introduces increasing sensor drift before simulated failure events.

## Holdout methodology

- Training: machines `machine_01` and `machine_02`, 4,320 hourly rows.
- Test: entirely held-out `machine_03`, 2,160 hourly rows, strictly later in UTC.
- Test actual positives: 137 hourly labels; negatives: 2,023.
- Adjacent 24-hour positive labels for a single fault are *dependent observations*, not 137 independent incidents. There are only three synthetic machines.
- No model/threshold tuning was performed on the held-out asset.

| Test metric | HistGradientBoosting | Fixed heuristic |
| --- | ---: | ---: |
| Precision | 0.9353 | 1.0000 |
| Recall | 0.9489 | 0.7810 |
| F1 | 0.9420 | 0.8770 |
| Average precision (PR-AUC summary) | 0.9856 | 0.7949 |
| True positives | 130 | 107 |
| False positives | 9 | 0 |
| False negatives | 7 | 30 |
| True negatives | 2,014 | 2,023 |

**Interpretation:** The classifier recovers the relationships programmed into the simulator. That does not validate reliability, calibration, generalization, fault prediction, or operations outcomes outside the fixture.

## Reproduce

```powershell
uv sync --extra dev --extra api
uv run pytest
uv run ruff check .
uv run industrial-pm demo --hours 2160 --seed 42
Get-Content artifacts/metrics.json
```

The generated `artifacts/` dataset, serialized model and machine-specific outputs are deliberately Git-ignored. When adapting to real sensor data, first validate asset identity, timezone, labeling windows, sensor units, data contracts, site-based splits, lag/leakage boundaries, recall versus false alarms, review workflow, and calibration.

**Security note:** `joblib` model artifacts are Python pickle-based and can execute code when loaded. Only load a locally generated, trusted artifact. Never permit unauthenticated users to upload or choose arbitrary model files.
