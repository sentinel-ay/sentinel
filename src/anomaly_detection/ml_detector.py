from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest


def fit_isolation_forest(features: np.ndarray, contamination: float = 0.05):
    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(features)
    return model


def predict_anomalies(model, features: np.ndarray):
    pred = model.predict(features)
    return pred == -1