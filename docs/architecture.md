# Architecture

`data.generate_synthetic_data` creates three independent deterministic machine streams. Every machine has hourly UTC observations and four sensor channels. Synthetic failure instants are selected from the seeded random generator; sensor drift increases during the 36 hours preceding each event. The label is one exactly when an event lies in the next 24 hours, excluding the current instant.

`features.make_features` sorts within machine and computes six-observation rolling means with a right-aligned window. The window includes the current observation and past observations only. Labels are kept separate from feature columns.

`model.asset_chronological_split` holds out the last machine in full and verifies that its timestamps occur strictly after training timestamps. To enable this with only three assets, the generator assigns each asset a distinct, non-overlapping synthetic date block. This is a deliberate evaluation convenience, not a model of simultaneous industrial operations. The split is asset-disjoint and has a strict timestamp gap.

The classifier uses histogram gradient boosting with a fixed seed. The baseline is a simple OR across four documented sensor thresholds. The evaluation helper reports threshold metrics and average precision, with raw confusion counts and denominators so class support and errors are visible.

The CLI persists the dataset, fitted bundle, and JSON metrics beneath `artifacts/`. The optional FastAPI module loads a bundle only for `/predict`; without it, the service returns an explicit unavailable status. The stateless endpoint cannot compute a genuine six-hour history window and uses the present values for those rolling feature slots. Use batch evaluation for the meaningful trained feature pipeline.

