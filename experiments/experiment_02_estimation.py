import matplotlib.pyplot as plt
import numpy as np

from src.config import load_config
from src.estimation.baseline import simple_baseline
from src.estimation.kalman_filter import build_kalman_filter
from src.evaluation.metrics import mae_rmse
from src.preprocessing.pipeline import forward_fill
from src.sensors.factory import build_sensors
from src.simulator.trajectory import generate_from_config

if __name__ == "__main__":
    cfg = load_config()
    rng = np.random.default_rng(cfg["project"]["random_seed"])

    traj = generate_from_config(cfg, rng=rng)
    truth = traj[:, :4]                                   # KF estimates [px, py, vx, vy]
    sensors = build_sensors(cfg, rng=rng)
    dt = cfg["simulator"]["dt"]

    obs = np.array([np.concatenate([sensors["gps"].measure(s, i * dt),
                                    sensors["velocity"].measure(s, i * dt)])
                    for i, s in enumerate(traj)])

    filled = forward_fill(obs)                            # dropout -> last valid reading
    baseline = simple_baseline(filled)
    kf_est = build_kalman_filter(cfg).filter(obs)         # KF skips the update on NaN directly

    rows = [("raw obs (ffill)", *mae_rmse(filled, truth)),
            ("baseline", *mae_rmse(baseline, truth)),
            ("kalman", *mae_rmse(kf_est, truth))]

    print(f"dropout timesteps: {int(np.isnan(obs).any(axis=1).sum())} / {len(obs)}")
    print(f"{'method':<18}{'MAE':>10}{'RMSE':>10}")
    for name, mae, rmse in rows:
        print(f"{name:<18}{mae:>10.4f}{rmse:>10.4f}")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(truth[:, 0], truth[:, 1], color="black", linewidth=2, label="ground truth")
    ax.scatter(filled[:, 0], filled[:, 1], color="lightgray", s=10, label="noisy obs (gps)")
    ax.plot(kf_est[:, 0], kf_est[:, 1], color="tab:blue", linewidth=1.5, label="kalman filter")
    ax.set_xlabel("px")
    ax.set_ylabel("py")
    ax.set_title(f"KF vs ground truth ({cfg['simulator']['traj_type']})")
    ax.legend()
    ax.set_aspect("equal")
    plt.show()
