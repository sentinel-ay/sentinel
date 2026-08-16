from __future__ import annotations

import numpy as np

TRAJECTORY_TYPES = ("straight", "curve", "accelerate", "decelerate", "turn")


def _commanded_accel(traj_type, vx, vy, dt, lon_accel, lat_accel, turn_rate):
    """Commanded acceleration for the given profile, in the body frame of the current velocity."""
    speed = np.hypot(vx, vy)
    ux, uy = (vx / speed, vy / speed) if speed > 0 else (1.0, 0.0)
    if traj_type == "straight":
        return 0.0, 0.0
    if traj_type == "accelerate":
        return lon_accel * ux, lon_accel * uy
    if traj_type == "decelerate":
        a = min(lon_accel, speed / dt)      # coast to a stop, never reverse direction
        return -a * ux, -a * uy
    if traj_type == "curve":
        return -lat_accel * uy, lat_accel * ux
    if traj_type == "turn":
        return -turn_rate * speed * uy, turn_rate * speed * ux
    raise ValueError(f"unknown traj_type: {traj_type!r} (expected one of {TRAJECTORY_TYPES})")


def generate_trajectory(steps, dt, x0, vx, vy, accel_std, traj_type, lon_accel, lat_accel, turn_rate, rng=None):
    """Ground-truth trajectory as `[px, py, vx, vy, ax, ay]` per step.

    `ax, ay` is the acceleration actually applied during that step (command + process noise).
    """
    rng = np.random.default_rng() if rng is None else rng
    px, py = np.asarray(x0, dtype=float)
    traj = np.zeros((steps, 6), dtype=float)

    for i in range(steps):
        ax, ay = _commanded_accel(traj_type, vx, vy, dt, lon_accel, lat_accel, turn_rate)
        ax += rng.normal(0.0, accel_std)
        ay += rng.normal(0.0, accel_std)
        traj[i] = [px, py, vx, vy, ax, ay]
        vx += ax * dt
        vy += ay * dt
        px += vx * dt
        py += vy * dt

    return traj


def generate_from_config(cfg: dict, rng=None, traj_type=None) -> np.ndarray:
    sim = cfg["simulator"]
    return generate_trajectory(
        steps=sim["steps"],
        dt=sim["dt"],
        x0=sim["x0"],
        vx=sim["vx"],
        vy=sim["vy"],
        accel_std=sim["accel_std"],
        traj_type=sim["traj_type"] if traj_type is None else traj_type,
        lon_accel=sim["lon_accel"],
        lat_accel=sim["lat_accel"],
        turn_rate=sim["turn_rate"],
        rng=rng,
    )
