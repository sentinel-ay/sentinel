import numpy as np


class ConstantVelocityKalmanFilter:
    """Constant-velocity KF over the 4-D state `[px, py, vx, vy]`.

    Observations are `[gps_x, gps_y, vel_x, vel_y]`; any component may be NaN
    (sensor dropout), in which case that row is skipped in the update step.
    """

    def __init__(self, dt: float, q: float, r):
        self.dt = dt
        self.q = q
        self.F = np.array([[1.0, 0.0, dt, 0.0],
                           [0.0, 1.0, 0.0, dt],
                           [0.0, 0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 1.0]], dtype=float)
        self.H = np.eye(4)
        # Discrete white-noise acceleration model: q is the acceleration process-noise std,
        # so Q follows from dt rather than being a hand-tuned scalar on the diagonal.
        p, pv, v = dt ** 4 / 4, dt ** 3 / 2, dt ** 2
        self.Q = (q ** 2) * np.array([[p, 0.0, pv, 0.0],
                                      [0.0, p, 0.0, pv],
                                      [pv, 0.0, v, 0.0],
                                      [0.0, pv, 0.0, v]], dtype=float)
        r = np.atleast_1d(np.asarray(r, dtype=float))
        self.R = np.diag(np.full(4, r[0]) if r.size == 1 else r)
        self.x = None
        self.P = np.eye(4)

    def predict(self):
        if self.x is None:
            raise ValueError("State is not initialized.")
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, z: np.ndarray):
        if self.x is None:
            raise ValueError("State is not initialized.")
        z = np.asarray(z, dtype=float)
        mask = ~np.isnan(z)
        if not mask.any():          # every sensor dropped out: prediction only
            return self.x.copy()

        H = self.H[mask]
        R = self.R[np.ix_(mask, mask)]
        innovation = z[mask] - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        self.P = (np.eye(4) - K @ H) @ self.P
        return self.x.copy()

    def filter(self, observations: np.ndarray, return_residuals: bool = False):
        obs = np.atleast_2d(np.asarray(observations, dtype=float))
        self.x = np.nan_to_num(obs[0], nan=0.0)
        self.P = np.eye(4)

        estimates, residuals = [], []
        for z in obs:
            pred = self.predict()
            residuals.append(z - self.H @ pred)   # innovation; NaN where the sensor dropped out
            estimates.append(self.update(z))

        if return_residuals:
            return np.asarray(estimates), np.asarray(residuals)
        return np.asarray(estimates)


def build_kalman_filter(cfg: dict) -> ConstantVelocityKalmanFilter:
    est = cfg["estimation"]
    return ConstantVelocityKalmanFilter(
        dt=cfg["simulator"]["dt"],
        q=est["accel_process_std"],
        r=[est["r_gps"], est["r_gps"], est["r_velocity"], est["r_velocity"]],
    )
