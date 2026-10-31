# Limitations and honest interpretation

- All data and labels are synthetic. The generator encodes a designed relationship between pre-event sensor drift and the label, so evaluation measures recovery of this toy construction only.
- There are three assets. The complete-asset holdout is a very small and unusual sample; it cannot establish generalization to other machines, sites, operators, fault modes, sensor calibration, or time periods.
- Assets occupy separate, ordered time blocks to permit a strict chronological, asset-disjoint split. This is not representative of a real fleet where machines operate concurrently.
- One simulated event can contribute up to 24 positive hourly labels, so adjacent rows are dependent. Confusion-matrix denominators count rows, not independent failure incidents.
- Average precision is used as the PR-AUC summary. No threshold tuning, uncertainty interval, calibration, cost model, or operational decision analysis is included.
- The fixed heuristic thresholds are illustrative and are not sourced from engineering limits. The endpoint's score is not a calibrated probability.
- The stateless API substitutes the current reading for each six-hour rolling mean because historical readings are absent. This does not reproduce training feature distributions; do not interpret the API's score as a validated inference result.
- The optional API loads a local joblib/pickle model artifact. Pickle deserialization can execute code: load only trusted, locally generated artifacts. Never expose artifact upload/path selection to an untrusted caller.
- Missingness, drift, maintenance actions, competing risks, downtime, delayed labels, and data quality failures are not modeled.

Do not use this artifact to make maintenance, safety, warranty, or uptime decisions. Any real deployment would require domain-approved data governance, incident definitions, leakage review, temporal and site-level validation, monitoring, and human engineering review.
