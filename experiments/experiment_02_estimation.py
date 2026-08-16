import numpy as np
import matplotlib.pyplot as plt

from src.simulator.trajectory import generate_trajectory
from src.sensors.gps import GPSSensor
from src.sensors.velocity_sensor import VelocitySensor
from src.estimation.baseline import simple_baseline
from src.estimation.kalman_filter import ConstantVelocityKalmanFilter
from src.evaluation.metrics import mae_rmse


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    traj = generate_trajectory(steps=200, dt=0.1)

    gps = GPSSensor(rng=rng)
    velocity = VelocitySensor(rng=rng)
    obs = np.array([np.concatenate([gps.measure(s), velocity.measure(s)]) for s in traj])
    obs = np.nan_to_num(obs, nan=0.0)  # dropout: hold previous isn't wired yet, zero-fill for this check

    baseline = simple_baseline(obs)
    kf = ConstantVelocityKalmanFilter(dt=0.1, q=0.1, r=0.5)
    kf_est = kf.filter(obs)

    print("raw obs   MAE/RMSE:", mae_rmse(obs, traj))
    print("baseline  MAE/RMSE:", mae_rmse(baseline, traj))
    print("kalman    MAE/RMSE:", mae_rmse(kf_est, traj))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(traj[:, 0], traj[:, 1], color="black", linewidth=2, label="ground truth")
    ax.scatter(obs[:, 0], obs[:, 1], color="lightgray", s=10, label="noisy obs (gps)")
    ax.plot(kf_est[:, 0], kf_est[:, 1], color="tab:blue", linewidth=1.5, label="kalman filter")
    ax.set_xlabel("px")
    ax.set_ylabel("py")
    ax.set_title("KF estimate vs ground truth vs noisy observations")
    ax.legend()
    ax.set_aspect("equal")
    plt.show()
