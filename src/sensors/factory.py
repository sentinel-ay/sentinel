from __future__ import annotations

from src.sensors.distance_sensor import DistanceSensor
from src.sensors.gps import GPSSensor
from src.sensors.imu import IMUSensor
from src.sensors.velocity_sensor import VelocitySensor

SENSOR_TYPES = {
    "gps": GPSSensor,
    "imu": IMUSensor,
    "velocity": VelocitySensor,
    "distance": DistanceSensor,
}


def build_sensors(cfg: dict, rng=None) -> dict:
    """Instantiate every sensor from the `sensors` section of the config."""
    return {name: cls(**cfg["sensors"][name], rng=rng) for name, cls in SENSOR_TYPES.items()}
