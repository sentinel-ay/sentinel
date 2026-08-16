import numpy as np


def _magnitude(values):
    values = np.asarray(values, dtype=float)
    return np.linalg.norm(np.nan_to_num(values, nan=0.0), axis=1) if values.ndim > 1 else values


def zscore_anomaly_scores(values: np.ndarray, threshold: float, reference=None):
    """Z-score of |values|. Feed KF residuals, not raw measurements.

    `reference` calibrates mean/std on a known-normal slice instead of the data itself.
    """
    values = _magnitude(values)
    ref = values if reference is None else _magnitude(reference)
    mean = np.nanmean(ref)
    std = np.nanstd(ref)
    if std == 0:
        return np.zeros_like(values, dtype=float), np.zeros_like(values, dtype=bool)
    z = np.abs((values - mean) / std)
    return z, z > threshold
