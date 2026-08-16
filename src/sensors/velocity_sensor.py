from __future__ import annotations

import numpy as np

from src.sensors.base_sensor import BaseSensor


class VelocitySensor(BaseSensor):
    """Measures velocity (vx, vy)."""

    def measure(self, state, t):
        state = np.asarray(state, dtype=float)
        return np.array([self.sample(state[2], t), self.sample(state[3], t)])
