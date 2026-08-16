"""Visual check that the five trajectory profiles are actually distinguishable.

Numbered 04 because 03 is reserved for the anomaly-detection comparison.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.config import load_config
from src.simulator.trajectory import TRAJECTORY_TYPES, generate_from_config

OUT = Path(__file__).resolve().parents[1] / "data" / "generated" / "trajectory_types.png"

if __name__ == "__main__":
    cfg = load_config()
    seed = cfg["project"]["random_seed"]
    dt = cfg["simulator"]["dt"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    print(f"{'type':<12}{'end speed':>11}{'path len':>10}{'net heading':>13}")

    for ax, traj_type in zip(axes.ravel(), TRAJECTORY_TYPES):
        traj = generate_from_config(cfg, rng=np.random.default_rng(seed), traj_type=traj_type)
        speed = np.hypot(traj[:, 2], traj[:, 3])
        steps = np.hypot(np.diff(traj[:, 0]), np.diff(traj[:, 1])).sum()
        heading = np.degrees(np.arctan2(traj[-1, 3], traj[-1, 2]) - np.arctan2(traj[0, 3], traj[0, 2]))
        print(f"{traj_type:<12}{speed[-1]:>11.2f}{steps:>10.2f}{heading:>12.1f}°")

        sc = ax.scatter(traj[:, 0], traj[:, 1], c=np.arange(len(traj)) * dt, cmap="viridis", s=12)
        ax.plot(traj[:, 0], traj[:, 1], color="lightgray", linewidth=0.8, zorder=0)
        ax.set_title(f"{traj_type}  (v: {speed[0]:.2f} → {speed[-1]:.2f})")
        ax.set_xlabel("px")
        ax.set_ylabel("py")
        ax.set_aspect("equal")
        fig.colorbar(sc, ax=ax, label="t [s]")

    axes.ravel()[-1].axis("off")
    fig.suptitle("SENTINEL trajectory profiles (same seed, same initial state)")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=110)
    print(f"\nsaved: {OUT}")
