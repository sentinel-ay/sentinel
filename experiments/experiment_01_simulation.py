import numpy as np

from src.config import load_config
from src.preprocessing.pipeline import build_sensor_dataframe
from src.sensors.factory import build_sensors
from src.simulator.trajectory import generate_from_config

if __name__ == "__main__":
    cfg = load_config()
    rng = np.random.default_rng(cfg["project"]["random_seed"])

    traj = generate_from_config(cfg, rng=rng)
    sensors = build_sensors(cfg, rng=rng)
    dt = cfg["simulator"]["dt"]

    readings = {name: np.array([s.measure(state, i * dt) for i, state in enumerate(traj)])
                for name, s in sensors.items()}

    df = build_sensor_dataframe(traj, readings["gps"], readings["imu"],
                                readings["velocity"], readings["distance"])

    print(f"trajectory: {traj.shape}  type={cfg['simulator']['traj_type']}")
    print(df.head())
    print("\ndropout(NaN) count per column:")
    print(df.isna().sum()[lambda s: s > 0].to_string() or "  none")
