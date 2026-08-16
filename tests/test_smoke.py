import numpy as np

from src.estimation.baseline import simple_baseline
from src.simulator.trajectory import generate_trajectory


def test_generate_trajectory_and_baseline():
    traj = generate_trajectory(steps=50, dt=0.1, x0=(0.0, 0.0), vx=1.0, vy=0.5)
    assert traj.shape == (50, 4)
    assert np.all(np.isfinite(traj))

    baseline = simple_baseline(traj)
    assert baseline.shape == (50, 4)
    assert np.all(np.isfinite(baseline))