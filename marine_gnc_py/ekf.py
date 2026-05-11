#!/usr/bin/env python3
"""Extended Kalman Filter for the 12-state marine vehicle model.

The EKF owns its own tuning parameters: P0, Q, and assumed measurement R per
sensor. It does not use the simulated sensor's true covariance unless the YAML
configuration makes them match.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml

from marine_gnc_py.dynamics_model import DynamicsModel, STATE_NAMES, wrap_state_angles
from marine_gnc_py.sensor_stack import SensorReading
from marine_gnc_py.symbolic_jacobian import SymbolicJacobians

ANGLE_INDEX_TO_DEG_KEY = {
    3: "roll_deg",
    4: "pitch_deg",
    5: "yaw_deg",
    9: "p_deg",
    10: "q_deg",
    11: "r_deg",
}


def _std_vector_from_mapping(data: Dict[str, Any], defaults: Dict[str, float]) -> np.ndarray:
    out = np.zeros(12, dtype=float)
    for i, name in enumerate(STATE_NAMES):
        if i in ANGLE_INDEX_TO_DEG_KEY:
            deg_key = ANGLE_INDEX_TO_DEG_KEY[i]
            if deg_key in data:
                out[i] = np.deg2rad(float(data[deg_key]))
            elif name in data:
                out[i] = float(data[name])
            else:
                default_val = defaults.get(deg_key, defaults.get(name, 0.0))
                out[i] = np.deg2rad(default_val) if deg_key in defaults else default_val
        else:
            out[i] = float(data.get(name, defaults.get(name, 0.0)))
    return out


def _measurement_std_from_dict(data: Dict[str, Any]) -> np.ndarray:
    if "std" in data:
        return np.asarray(data["std"], dtype=float)
    if "std_deg" in data:
        return np.deg2rad(np.asarray(data["std_deg"], dtype=float))
    raise ValueError("measurement noise entry must contain std or std_deg")


@dataclass
class EkfConfig:
    jacobian_mode: str = "sympy-euler"
    initial_state: np.ndarray = field(default_factory=lambda: np.array([0.15, -0.25, 9.85, 0.02, -0.02, 0.04, 0, 0, 0, 0, 0, 0], dtype=float))
    P0: np.ndarray = field(default_factory=lambda: np.diag([0.75, 0.75, 0.25, np.deg2rad(5), np.deg2rad(5), np.deg2rad(10), 0.25, 0.25, 0.15, np.deg2rad(3), np.deg2rad(3), np.deg2rad(3)]) ** 2)
    Q: np.ndarray = field(default_factory=lambda: np.diag([0.002, 0.002, 0.002, np.deg2rad(0.02), np.deg2rad(0.02), np.deg2rad(0.03), 0.010, 0.010, 0.008, np.deg2rad(0.05), np.deg2rad(0.05), np.deg2rad(0.05)]) ** 2)
    measurement_noise: Dict[str, np.ndarray] = field(default_factory=dict)
    measurement_use: Dict[str, bool] = field(default_factory=dict)

    @staticmethod
    def from_yaml(path: str | Path) -> "EkfConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        initial_state = np.asarray(data.get("initial_state", [0.15, -0.25, 9.85, 0.02, -0.02, 0.04, 0, 0, 0, 0, 0, 0]), dtype=float)

        p0_defaults = {
            "x": 0.75, "y": 0.75, "z": 0.25,
            "roll_deg": 5.0, "pitch_deg": 5.0, "yaw_deg": 10.0,
            "u": 0.25, "v": 0.25, "w": 0.15,
            "p_deg": 3.0, "q_deg": 3.0, "r_deg": 3.0,
        }
        q_defaults = {
            "x": 0.002, "y": 0.002, "z": 0.002,
            "roll_deg": 0.02, "pitch_deg": 0.02, "yaw_deg": 0.03,
            "u": 0.010, "v": 0.010, "w": 0.008,
            "p_deg": 0.05, "q_deg": 0.05, "r_deg": 0.05,
        }
        p0_std = _std_vector_from_mapping(data.get("initial_std", {}), p0_defaults)
        q_std = _std_vector_from_mapping(data.get("process_noise_std", {}), q_defaults)

        meas_noise: Dict[str, np.ndarray] = {}
        meas_use: Dict[str, bool] = {}
        for sensor_name, cfg in data.get("measurement_noise", {}).items():
            meas_use[sensor_name] = bool(cfg.get("use", True))
            meas_noise[sensor_name] = _measurement_std_from_dict(cfg)

        return EkfConfig(
            jacobian_mode=str(data.get("jacobian_mode", "sympy-euler")),
            initial_state=initial_state,
            P0=np.diag(p0_std ** 2),
            Q=np.diag(q_std ** 2),
            measurement_noise=meas_noise,
            measurement_use=meas_use,
        )


class ExtendedKalmanFilter:
    def __init__(self, model: DynamicsModel, config: EkfConfig, environment: Optional[Any] = None):
        self.model = model
        self.config = config
        self.environment = environment
        self.x = wrap_state_angles(config.initial_state.copy())
        self.P = config.P0.copy()
        self.Q = config.Q.copy()
        self.last_F = np.eye(12)
        self._sym = None
        if self.config.jacobian_mode.startswith("sympy"):
            self._sym = SymbolicJacobians(model.params)

    @staticmethod
    def numerical_jacobian(func, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        f0 = np.asarray(func(x), dtype=float)
        J = np.zeros((f0.size, x.size), dtype=float)
        for i in range(x.size):
            dx = np.zeros_like(x)
            dx[i] = eps
            fp = np.asarray(func(x + dx), dtype=float)
            fm = np.asarray(func(x - dx), dtype=float)
            J[:, i] = (fp - fm) / (2.0 * eps)
        return J

    def _discrete_process(self, t: float, x: np.ndarray, tau_cmd: np.ndarray, dt: float) -> np.ndarray:
        x_next, _ = self.model.rk4_step(t, x, tau_cmd, dt, self.environment)
        if self.environment is not None:
            x_next = self.environment.enforce_bounds(x_next).state
        return wrap_state_angles(x_next)

    def _compute_F(self, t: float, tau_cmd: np.ndarray, dt: float) -> np.ndarray:
        mode = self.config.jacobian_mode
        if mode == "numerical":
            return self.numerical_jacobian(lambda xx: self._discrete_process(t, xx, tau_cmd, dt), self.x)
        if mode in ("sympy-euler", "sympy"):
            current_body = self.model.current_body(t, self.x, self.environment)
            A = self._sym.continuous_A(self.x, tau_cmd, current_body=current_body)
            return np.eye(12) + dt * A
        # Simple robust fallback.
        return self.numerical_jacobian(lambda xx: self._discrete_process(t, xx, tau_cmd, dt), self.x)

    def predict(self, t: float, tau_cmd: np.ndarray, dt: float) -> None:
        F = self._compute_F(t, tau_cmd, dt)
        self.x = self._discrete_process(t, self.x, tau_cmd, dt)
        self.P = F @ self.P @ F.T + self.Q
        self.P = 0.5 * (self.P + self.P.T)
        self.last_F = F

    def assumed_R(self, reading: SensorReading) -> Optional[np.ndarray]:
        if not self.config.measurement_use.get(reading.name, True):
            return None
        std = self.config.measurement_noise.get(reading.name)
        if std is None:
            # If the EKF config omits a sensor, fall back to the sensor's true R.
            return reading.R_true.copy()
        if std.size != reading.z.size:
            raise ValueError(f"EKF R for sensor {reading.name} has wrong dimension")
        return np.diag(std ** 2)

    def update(self, reading: SensorReading) -> bool:
        if not reading.available:
            return False
        R = self.assumed_R(reading)
        if R is None:
            return False
        idx = reading.state_indices
        H = np.zeros((idx.size, 12), dtype=float)
        for row, col in enumerate(idx):
            H[row, col] = 1.0
        z_pred = self.x[idx]
        innovation = reading.z - z_pred
        # Wrap angle innovations for directly measured angle/rate states as appropriate.
        for j, col in enumerate(idx):
            if col in (3, 4, 5):
                innovation[j] = (innovation[j] + np.pi) % (2.0 * np.pi) - np.pi
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = wrap_state_angles(self.x + K @ innovation)
        I = np.eye(12)
        # Joseph form for better numerical symmetry/PSD behavior.
        KH = K @ H
        self.P = (I - KH) @ self.P @ (I - KH).T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        return True

    def update_many(self, readings: list[SensorReading]) -> int:
        count = 0
        for reading in readings:
            if self.update(reading):
                count += 1
        return count
