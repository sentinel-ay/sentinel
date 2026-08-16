from __future__ import annotations

import numpy as np

from src.sensors.base_sensor import BaseSensor


class IMUSensor(BaseSensor):
    def __init__(self, noise_std=0.2, bias=0.03, drift_rate=0.0005, dropout_prob=0.01, outlier_prob=0.008, rng=None):
        super().__init__(noise_std, bias, drift_rate, dropout_prob, outlier_prob, rng)

    def measure(self, state):
        state = np.asarray(state, dtype=float)
        return np.array([self.sample(state[2]), self.sample(state[3])])