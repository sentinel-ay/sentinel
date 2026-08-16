import numpy as np

from src.simulator.trajectory import generate_trajectory


if __name__ == "__main__":
    np.random.seed(42)
    traj = generate_trajectory(steps=200, dt=0.1)
    print(traj[:5])
    print(traj.shape)