from __future__ import annotations

import numpy as np

from src.sensors.base_sensor import BaseSensor


class GPSSensor(BaseSensor):
    """Measures position (px, py)."""

    def measure(self, state, t):
        state = np.asarray(state, dtype=float)
        return np.array([self.sample(state[0], t), self.sample(state[1], t)])
