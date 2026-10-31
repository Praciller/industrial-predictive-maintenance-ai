import numpy as np
import pandas as pd
import pytest

from industrial_pm.data import LABEL, generate_synthetic_data
from industrial_pm.features import make_features
from industrial_pm.model import asset_chronological_split, evaluate, train_model


def test_generation_is_deterministic_and_has_both_classes():
    a = generate_synthetic_data(hours=24 * 60)
    b = generate_synthetic_data(hours=24 * 60)
    pd.testing.assert_frame_equal(a, b)
    assert a.timestamp.dt.tz is not None
    assert a.machine_id.nunique() == 3
    assert set(a[LABEL].unique()) == {0, 1}


def test_features_are_causal_and_invalid_values_rejected():
    data = generate_synthetic_data(hours=72)
    features, labels = make_features(data)
    row = data[(data.machine_id == "machine_01")].iloc[5]
    ix = data.index.get_loc(row.name)
    # Six sample rolling mean includes current and preceding observations only.
    expected = data[(data.machine_id == "machine_01")].iloc[:6].vibration.mean()
    assert features.iloc[ix].vibration_mean_6h == pytest.approx(expected)
    assert len(labels) == len(data)
    invalid = data.copy()
    invalid.loc[0, "vibration"] = np.inf
    with pytest.raises(ValueError, match="finite"):
        make_features(invalid)


def test_asset_split_is_strict_and_has_no_machine_overlap():
    train, test = asset_chronological_split(generate_synthetic_data(hours=24 * 60))
    assert train.timestamp.max() < test.timestamp.min()
    assert set(train.machine_id).isdisjoint(test.machine_id)


def test_metrics_denominators_and_training_reproducibility():
    result = evaluate(np.array([0, 0, 1, 1]), np.array([0.1, 0.8, 0.7, 0.9]))
    assert result["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 0, "tp": 2}
    assert result["confusion_denominators"]["actual_positive"] == 2
    data = generate_synthetic_data(hours=24 * 45)
    model_a, xa, ta = train_model(data)
    model_b, xb, tb = train_model(data)
    np.testing.assert_array_equal(model_a.predict_proba(xa), model_b.predict_proba(xb))
    pd.testing.assert_frame_equal(ta, tb)
