from __future__ import annotations

import numpy as np

from src.sensors.base_sensor import BaseSensor


class DistanceSensor(BaseSensor):
    """Measures range from the origin."""

    def measure(self, state, t):
        state = np.asarray(state, dtype=float)
        return self.sample(np.hypot(state[0], state[1]), t)
