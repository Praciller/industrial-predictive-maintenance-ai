"""Verify the published static evidence against a freshly computed holdout."""

import json
from pathlib import Path

import numpy as np

from industrial_pm.data import LABEL, generate_synthetic_data
from industrial_pm.model import asset_chronological_split, evaluate, heuristic_scores, train_model
from industrial_pm.static_demo import build_evidence, write_evidence


def test_published_evidence_matches_computed_holdout(tmp_path):
    evidence = build_evidence()
    data = generate_synthetic_data(hours=2160, seed=42)
    train, test = asset_chronological_split(data)
    model, x_test, model_test = train_model(data)
    assert model_test.equals(test)
    truth = test[LABEL].to_numpy(dtype=int)
    model_scores = model.predict_proba(x_test)
    baseline_scores = heuristic_scores(test)

    assert evidence["actual"] == truth.tolist()
    assert len(evidence["sources"]["model"]["scores"]) == len(test)
    assert np.array_equal(evidence["sources"]["model"]["scores"], model_scores)
    assert np.array_equal(evidence["sources"]["heuristic"]["scores"], baseline_scores)
    assert evidence["split"]["train_rows"] == len(train)
    assert evidence["split"]["test_rows"] == len(test)
    assert evidence["split"]["positive"] == int(truth.sum())
    assert evidence["split"]["negative"] == int(len(truth) - truth.sum())
    assert evidence["sources"]["model"]["metrics"] == evaluate(truth, model_scores)
    assert evidence["sources"]["heuristic"]["metrics"] == evaluate(truth, baseline_scores)

    output = tmp_path / "data.json"
    write_evidence(output)
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved == evidence
    published = json.loads(
        (Path(__file__).resolve().parents[1] / "static" / "data.json").read_text(
            encoding="utf-8"
        )
    )
    assert published == evidence
