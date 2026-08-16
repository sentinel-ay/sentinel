from __future__ import annotations

import numpy as np

from src.sensors.base_sensor import BaseSensor


class DistanceSensor(BaseSensor):
    def __init__(self, noise_std=0.25, bias=0.04, drift_rate=0.0008, dropout_prob=0.015, outlier_prob=0.008, rng=None):
        super().__init__(noise_std, bias, drift_rate, dropout_prob, outlier_prob, rng)

    def measure(self, state):
        state = np.asarray(state, dtype=float)
        distance = np.hypot(state[0], state[1])
        return self.sample(distance)