import numpy as np


def zscore_anomaly_scores(values: np.ndarray, threshold: float = 3.0):
    values = np.asarray(values, dtype=float)
    mean = np.nanmean(values)
    std = np.nanstd(values)
    if std == 0:
        return np.zeros_like(values, dtype=float), np.zeros_like(values, dtype=bool)
    z = np.abs((values - mean) / std)
    return z, z > threshold