import numpy as np
import pandas as pd


def build_sensor_dataframe(states: np.ndarray, gps: np.ndarray, imu: np.ndarray, velocity: np.ndarray, distance: np.ndarray):
    df = pd.DataFrame(states, columns=['px', 'py', 'vx', 'vy'])
    df[['gps_x', 'gps_y']] = gps
    df[['imu_vx', 'imu_vy']] = imu
    df[['vel_x', 'vel_y']] = velocity
    df['distance'] = distance
    return df