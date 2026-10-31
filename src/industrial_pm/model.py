"""Model training, interpretable baseline, chronological asset holdout, and metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from industrial_pm.features import FEATURE_COLUMNS, make_features


@dataclass
class ModelBundle:
    classifier: HistGradientBoostingClassifier
    feature_columns: list[str]

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return self.classifier.predict_proba(features[self.feature_columns])[:, 1]


def asset_chronological_split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out the latest-starting asset; the test period begins after all training data."""
    assets = sorted(data["machine_id"].unique())
    if len(assets) < 2:
        raise ValueError("at least two distinct assets are required")
    test_asset = assets[-1]
    train = data.loc[data["machine_id"] != test_asset].copy()
    test = data.loc[data["machine_id"] == test_asset].copy()
    # Keep every training timestamp strictly earlier than the first test timestamp.
    test_start = test["timestamp"].min()
    train = train.loc[train["timestamp"] < test_start]
    test = test.loc[test["timestamp"] > train["timestamp"].max()]
    if train.empty or test.empty or set(train.machine_id) & set(test.machine_id):
        raise ValueError("unable to form non-overlapping chronological asset holdout")
    if train.timestamp.max() >= test.timestamp.min():
        raise ValueError("train and test timestamps overlap")
    return train.reset_index(drop=True), test.reset_index(drop=True)


def train_model(data: pd.DataFrame) -> tuple[ModelBundle, pd.DataFrame, pd.DataFrame]:
    train, test = asset_chronological_split(data)
    x_train, y_train = make_features(train)
    x_test, _ = make_features(test)
    if y_train.nunique() < 2:
        raise ValueError("training set must contain both label classes")
    model = HistGradientBoostingClassifier(
        max_iter=100, max_leaf_nodes=15, l2_regularization=1.0, random_state=42
    )
    model.fit(x_train[FEATURE_COLUMNS], y_train)
    return ModelBundle(model, FEATURE_COLUMNS), x_test, test.reset_index(drop=True)


def heuristic_scores(data: pd.DataFrame) -> np.ndarray:
    """Transparent fixed-threshold warning score; thresholds are illustrative."""
    return (
        (data["vibration"].to_numpy() >= 3.0)
        | (data["temperature"].to_numpy() >= 78.0)
        | (data["current"].to_numpy() >= 16.0)
        | (data["rpm"].to_numpy() <= 1460.0)
    ).astype(float)


def evaluate(y_true: pd.Series | np.ndarray, scores: np.ndarray, *, threshold: float = 0.5) -> dict:
    """Report threshold metrics and average precision with explicit denominators."""
    truth = np.asarray(y_true, dtype=int)
    prediction = (np.asarray(scores) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(truth, prediction, labels=[0, 1]).ravel()
    return {
        "n": int(len(truth)),
        "positive": int(truth.sum()),
        "negative": int(len(truth) - truth.sum()),
        "threshold": float(threshold),
        "precision": float(precision_score(truth, prediction, zero_division=0)),
        "recall": float(recall_score(truth, prediction, zero_division=0)),
        "f1": float(f1_score(truth, prediction, zero_division=0)),
        "pr_auc_average_precision": float(average_precision_score(truth, scores))
        if len(np.unique(truth)) > 1
        else None,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "confusion_denominators": {
            "actual_negative": int(tn + fp),
            "actual_positive": int(fn + tp),
            "predicted_negative": int(tn + fn),
            "predicted_positive": int(fp + tp),
        },
    }
