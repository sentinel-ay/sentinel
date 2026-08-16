import numpy as np
import pandas as pd


def forward_fill(values: np.ndarray) -> np.ndarray:
    """Replace NaN (sensor dropout) with the last valid observation per column.

    Leading NaNs have no predecessor and stay NaN-free by falling back to the first
    valid value in that column; a fully-missing column is left as zeros.
    """
    arr = np.array(values, dtype=float, copy=True)
    if arr.ndim == 1:
        return forward_fill(arr.reshape(-1, 1)).ravel()

    for j in range(arr.shape[1]):
        col = arr[:, j]
        valid = np.flatnonzero(~np.isnan(col))
        if valid.size == 0:
            col[:] = 0.0
            continue
        col[:valid[0]] = col[valid[0]]                     # backfill the leading gap
        idx = np.maximum.accumulate(np.where(np.isnan(col), 0, np.arange(len(col))))
        arr[:, j] = col[idx]
    return arr


def build_sensor_dataframe(states, gps, imu, velocity, distance) -> pd.DataFrame:
    df = pd.DataFrame(np.asarray(states, dtype=float), columns=['px', 'py', 'vx', 'vy', 'ax', 'ay'])
    df[['gps_x', 'gps_y']] = gps
    df[['imu_ax', 'imu_ay']] = imu
    df[['vel_x', 'vel_y']] = velocity
    df['distance'] = distance
    return df
