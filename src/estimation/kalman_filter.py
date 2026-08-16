import numpy as np


class ConstantVelocityKalmanFilter:
    def __init__(self, dt: float, q: float, r: float):
        self.dt = dt
        self.q = q
        self.r = r
        self.F = np.array([[1.0, 0.0, dt, 0.0],
                          [0.0, 1.0, 0.0, dt],
                          [0.0, 0.0, 1.0, 0.0],
                          [0.0, 0.0, 0.0, 1.0]], dtype=float)
        self.H = np.array([[1.0, 0.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0, 0.0],
                          [0.0, 0.0, 0.0, 1.0]], dtype=float)
        self.Q = np.eye(4) * q
        self.R = np.eye(4) * r
        self.x = None
        self.P = np.eye(4) * 1.0

    def predict(self, x: np.ndarray | None = None):
        if x is not None:
            self.x = x.astype(float)
        if self.x is None:
            raise ValueError("State is not initialized.")
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, z: np.ndarray):
        if self.x is None:
            raise ValueError("State is not initialized.")
        z = np.asarray(z, dtype=float)
        innovation = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.x.copy()

    def filter(self, observations: np.ndarray):
        obs = np.asarray(observations, dtype=float)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        estimates = []
        self.x = obs[0].copy()
        self.P = np.eye(4)
        for z in obs:
            self.predict()
            self.update(z)
            estimates.append(self.x.copy())
        return np.asarray(estimates)