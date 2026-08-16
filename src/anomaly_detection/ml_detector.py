from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest


def fit_isolation_forest(features: np.ndarray, contamination: float, random_state: int):
    model = IsolationForest(contamination=contamination, random_state=random_state)
    model.fit(np.nan_to_num(np.asarray(features, dtype=float), nan=0.0))
    return model


def predict_anomalies(model, features: np.ndarray):
    features = np.nan_to_num(np.asarray(features, dtype=float), nan=0.0)
    return model.predict(features) == -1


def anomaly_scores(model, features: np.ndarray):
    """Higher score = more anomalous (sklearn's score_samples is inverted)."""
    features = np.nan_to_num(np.asarray(features, dtype=float), nan=0.0)
    return -model.score_samples(features)
