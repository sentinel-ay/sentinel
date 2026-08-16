from __future__ import annotations

import numpy as np

from src.sensors.base_sensor import BaseSensor


class GPSSensor(BaseSensor):
    def __init__(self, noise_std=0.3, bias=0.05, drift_rate=0.001, dropout_prob=0.02, outlier_prob=0.01, rng=None):
        super().__init__(noise_std, bias, drift_rate, dropout_prob, outlier_prob, rng)

    def measure(self, state):
        state = np.asarray(state, dtype=float)
        return np.array([self.sample(state[0]), self.sample(state[1])])