from __future__ import annotations

import numpy as np

from src.sensors.base_sensor import BaseSensor


class IMUSensor(BaseSensor):
    """Accelerometer: measures acceleration (ax, ay), not velocity."""

    def measure(self, state, t):
        state = np.asarray(state, dtype=float)
        return np.array([self.sample(state[4], t), self.sample(state[5], t)])
