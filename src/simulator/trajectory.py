from __future__ import annotations

import numpy as np


def generate_trajectory(steps: int = 200, dt: float = 0.1, x0=(0.0, 0.0), vx: float = 1.0, vy: float = 0.5, accel_std: float = 0.05) -> np.ndarray:
    """Generate a constant-velocity trajectory with noise in acceleration."""
    x0 = np.asarray(x0, dtype=float)
    traj = np.zeros((steps, 4), dtype=float)
    traj[0] = [x0[0], x0[1], vx, vy]

    for i in range(1, steps):
        ax = np.random.normal(0.0, accel_std)
        ay = np.random.normal(0.0, accel_std)
        vx = traj[i - 1, 2] + ax * dt
        vy = traj[i - 1, 3] + ay * dt
        traj[i] = [traj[i - 1, 0] + vx * dt, traj[i - 1, 1] + vy * dt, vx, vy]

    return traj