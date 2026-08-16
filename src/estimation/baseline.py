import numpy as np


def simple_baseline(states: np.ndarray) -> np.ndarray:
    """Baseline predictor using the previous state as estimate."""
    est = np.asarray(states, dtype=float).copy()
    est[1:] = states[:-1]
    return est