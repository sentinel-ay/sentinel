from __future__ import annotations

import numpy as np


class BaseSensor:
    def __init__(self, noise_std, bias, drift_rate, dropout_prob, outlier_prob, rng=None):
        self.noise_std = noise_std
        self.bias = bias
        self.drift_rate = drift_rate
        self.dropout_prob = dropout_prob
        self.outlier_prob = outlier_prob
        self.rng = np.random.default_rng() if rng is None else rng

    def sample(self, value, t):
        """Measure `value` at elapsed time `t`. Drift accumulates with time, not with magnitude."""
        if self.rng.random() < self.dropout_prob:
            return np.nan
        noise = self.rng.normal(0.0, self.noise_std)
        outlier = self.rng.normal(0.0, 5.0 * self.noise_std) if self.rng.random() < self.outlier_prob else 0.0
        drift = self.bias + self.drift_rate * t
        return value + drift + noise + outlier

    def measure(self, state, t):
        raise NotImplementedError
