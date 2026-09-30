"""Constant-velocity Kalman filter for 2D tracks (state [x, y, vx, vy]).

Kept deliberately small and dependency-free: the map only needs a stable
position/velocity estimate per object, and every parameter that affects results
(position noise, acceleration density) is visible in this file.
"""

from __future__ import annotations

import numpy as np


class KalmanCV:
    """Constant-velocity KF. `predict(dt)` then `update(zx, zy)` per frame."""

    def __init__(self, position_noise_m: float = 0.3, accel_density: float = 1.0,
                 initial_pos_var: float = 1.0, initial_vel_var: float = 4.0):
        self.accel_density = float(accel_density)
        self.R = np.diag([float(position_noise_m) ** 2] * 2)
        self._P0 = np.diag([initial_pos_var, initial_pos_var,
                            initial_vel_var, initial_vel_var])
        self.x = np.zeros(4, dtype=np.float64)
        self.P = self._P0.copy()
        self.prev_v = np.zeros(2, dtype=np.float64)

    def init(self, x: float, y: float) -> None:
        self.x[:] = [float(x), float(y), 0.0, 0.0]
        self.P = self._P0.copy()
        self.prev_v[:] = 0.0

    def predict(self, dt: float) -> None:
        dt = float(dt)
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        G = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
        Q = (G @ G.T) * self.accel_density
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, zx: float, zy: float) -> None:
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        y = np.array([float(zx), float(zy)]) - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.prev_v[:] = self.x[2:4]
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P

    @property
    def pos(self) -> tuple[float, float]:
        return float(self.x[0]), float(self.x[1])

    @property
    def vel(self) -> tuple[float, float]:
        return float(self.x[2]), float(self.x[3])

    @property
    def speed(self) -> float:
        return float(np.hypot(self.x[2], self.x[3]))

    def acceleration(self, dt: float) -> tuple[float, float]:
        """Finite difference of the KF velocity (labelled estimate, not truth)."""
        if dt <= 1e-9:
            return 0.0, 0.0
        return ((self.x[2] - self.prev_v[0]) / dt, (self.x[3] - self.prev_v[1]) / dt)
