#!/usr/bin/env python3
"""Waypoint-following LQR controller for the marine GNC simulation.

Waypoint mission/progression is intentionally separate from this controller and
lives in waypoints.py.  This file owns only the LQR configuration and LQR
wrench computation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
from math import cos, sin

import numpy as np
import yaml
from scipy.linalg import solve_discrete_are

from marine_gnc_py.dynamics_model import DynamicsModel, wrap_angle, wrap_state_angles
from marine_gnc_py.waypoints import WaypointFollowerConfig, WaypointNavigator, ControlDebug


STATE_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "u", "v", "w", "p", "q", "r"]
TAU_NAMES = ["X", "Y", "Z", "K", "M", "N"]

ANGLE_STATE_KEYS = {
    3: "roll_deg",
    4: "pitch_deg",
    5: "yaw_deg",
    9: "p_deg",
    10: "q_deg",
    11: "r_deg",
}


def _array(data: Any, default: list[float], n: int) -> np.ndarray:
    out = np.asarray(data if data is not None else default, dtype=float)
    if out.size != n:
        raise ValueError(f"Expected array length {n}, got {out.size}")
    return out


def rotation_body_to_ned(phi: float, theta: float, psi: float) -> np.ndarray:
    cphi = cos(phi)
    sphi = sin(phi)
    cth = cos(theta)
    sth = sin(theta)
    cpsi = cos(psi)
    spsi = sin(psi)
    return np.array(
        [
            [cpsi * cth, -spsi * cphi + cpsi * sth * sphi, spsi * sphi + cpsi * cphi * sth],
            [spsi * cth, cpsi * cphi + sphi * sth * spsi, -cpsi * sphi + sth * spsi * cphi],
            [-sth, cth * sphi, cth * cphi],
        ],
        dtype=float,
    )


def _weight_vector(data: Dict[str, Any], defaults: Dict[str, float]) -> np.ndarray:
    data = data or {}
    out = np.zeros(12, dtype=float)
    for i, name in enumerate(STATE_NAMES):
        if i in ANGLE_STATE_KEYS:
            deg_key = ANGLE_STATE_KEYS[i]
            if deg_key in data:
                scale_rad = np.deg2rad(float(data[deg_key]))
                out[i] = 1.0 / max(scale_rad, 1e-9) ** 2
            elif name in data:
                out[i] = float(data[name])
            else:
                scale_rad = np.deg2rad(float(defaults[deg_key]))
                out[i] = 1.0 / max(scale_rad, 1e-9) ** 2
        else:
            out[i] = float(data.get(name, defaults[name]))
    return out


@dataclass
class LQRControllerConfig:
    enabled: bool = True
    state_source: str = "ekf"  # ekf or truth
    tau_limits: np.ndarray = field(default_factory=lambda: np.array([30.0, 30.0, 22.0, 4.0, 4.0, 4.0], dtype=float))
    tau_trim: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=float))
    Q: np.ndarray = field(default_factory=lambda: np.diag([1.0, 1.0, 2.0, 10.0, 10.0, 8.0, 0.8, 0.8, 1.0, 0.5, 0.5, 0.5]))
    R: np.ndarray = field(default_factory=lambda: np.diag([0.20, 0.20, 0.25, 0.45, 0.45, 0.30]))
    linearization_dt: float = 0.03
    update_period_s: float = 0.50
    finite_difference_state_eps: float = 1e-5
    finite_difference_tau_eps: float = 1e-4
    regularization: float = 1e-8
    reference_update_threshold: float = 0.50

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "LQRControllerConfig":
        data = data or {}
        controller = data.get("controller", data)
        state_error_weights = controller.get("state_error_weights", {})
        q_defaults = {
            "x": 1.0,
            "y": 1.0,
            "z": 2.0,
            "roll_deg": 10.0,
            "pitch_deg": 10.0,
            "yaw_deg": 15.0,
            "u": 0.8,
            "v": 0.8,
            "w": 1.0,
            "p_deg": 45.0,
            "q_deg": 45.0,
            "r_deg": 45.0,
        }
        q_vec = _weight_vector(state_error_weights, q_defaults)

        control_weights = controller.get("control_weights", {})
        r_vec = np.array(
            [
                float(control_weights.get("X", 0.20)),
                float(control_weights.get("Y", 0.20)),
                float(control_weights.get("Z", 0.25)),
                float(control_weights.get("K", 0.45)),
                float(control_weights.get("M", 0.45)),
                float(control_weights.get("N", 0.30)),
            ],
            dtype=float,
        )

        return LQRControllerConfig(
            enabled=bool(controller.get("enabled", True)),
            state_source=str(controller.get("state_source", "ekf")).lower(),
            tau_limits=_array(controller.get("tau_limits"), [30.0, 30.0, 22.0, 4.0, 4.0, 4.0], 6),
            tau_trim=_array(controller.get("tau_trim"), [0.0] * 6, 6),
            Q=np.diag(q_vec),
            R=np.diag(r_vec),
            linearization_dt=float(controller.get("linearization_dt", 0.03)),
            update_period_s=float(controller.get("update_period_s", 0.50)),
            finite_difference_state_eps=float(controller.get("finite_difference_state_eps", 1e-5)),
            finite_difference_tau_eps=float(controller.get("finite_difference_tau_eps", 1e-4)),
            regularization=float(controller.get("regularization", 1e-8)),
            reference_update_threshold=float(controller.get("reference_update_threshold", 0.50)),
        )

    @staticmethod
    def from_yaml(path: str | Path) -> "LQRControllerConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return LQRControllerConfig.from_dict(data)


class WaypointLQRController:
    def __init__(
        self,
        waypoint_config: WaypointFollowerConfig,
        lqr_config: LQRControllerConfig,
        model: DynamicsModel,
        environment: Optional[Any] = None,
    ):
        self.navigator = WaypointNavigator(waypoint_config)
        self.waypoint_config = waypoint_config
        self.lqr_config = lqr_config
        self.model = model
        self.environment = environment
        self._K = np.zeros((6, 12), dtype=float)
        self._last_linearization_t = -np.inf
        self._last_target_index = -1
        self._last_reference = None
        self._last_lqr_ok = False

    @property
    def enabled(self) -> bool:
        return self.navigator.enabled and self.lqr_config.enabled

    @property
    def state_source(self) -> str:
        return self.lqr_config.state_source

    @property
    def target_index(self) -> int:
        return self.navigator.target_index

    @property
    def mission_complete(self) -> bool:
        return self.navigator.mission_complete

    def current_waypoint(self):
        return self.navigator.current_waypoint()

    def _reference_state(self, state: np.ndarray) -> tuple[np.ndarray, Any]:
        ref = self.navigator.reference(state)
        x_ref = np.zeros(12, dtype=float)
        x_ref[0:3] = ref.desired_position
        x_ref[3:6] = ref.desired_attitude
        return x_ref, ref

    def _linearize_discrete(self, t: float, x_ref: np.ndarray, tau_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dt = self.lqr_config.linearization_dt
        eps_x = self.lqr_config.finite_difference_state_eps
        eps_u = self.lqr_config.finite_difference_tau_eps

        def f_x(x: np.ndarray) -> np.ndarray:
            x_next, _ = self.model.rk4_step(t, x, tau_ref, dt, self.environment)
            if self.environment is not None:
                x_next = self.environment.enforce_bounds(x_next).state
            return wrap_state_angles(x_next)

        def f_u(u: np.ndarray) -> np.ndarray:
            x_next, _ = self.model.rk4_step(t, x_ref, u, dt, self.environment)
            if self.environment is not None:
                x_next = self.environment.enforce_bounds(x_next).state
            return wrap_state_angles(x_next)

        A = np.zeros((12, 12), dtype=float)
        for i in range(12):
            dx = np.zeros(12, dtype=float)
            dx[i] = eps_x
            fp = f_x(x_ref + dx)
            fm = f_x(x_ref - dx)
            A[:, i] = (fp - fm) / (2.0 * eps_x)

        B = np.zeros((12, 6), dtype=float)
        for i in range(6):
            du = np.zeros(6, dtype=float)
            du[i] = eps_u
            fp = f_u(tau_ref + du)
            fm = f_u(tau_ref - du)
            B[:, i] = (fp - fm) / (2.0 * eps_u)
        return A, B

    def _update_gain_if_needed(self, t: float, x_ref: np.ndarray) -> None:
        cfg = self.lqr_config
        ref_changed = self._last_reference is None or np.linalg.norm(x_ref - self._last_reference) > cfg.reference_update_threshold
        due = (t - self._last_linearization_t) >= cfg.update_period_s
        target_changed = self.navigator.target_index != self._last_target_index
        if not (due or target_changed or ref_changed):
            return

        A, B = self._linearize_discrete(t, x_ref, cfg.tau_trim)
        Q = cfg.Q + cfg.regularization * np.eye(12)
        R = cfg.R + cfg.regularization * np.eye(6)
        try:
            P = solve_discrete_are(A, B, Q, R)
            self._K = np.linalg.solve(B.T @ P @ B + R, B.T @ P @ A)
            self._last_lqr_ok = True
        except Exception:
            # Keep the previous gain if available; otherwise command zero trim.
            self._last_lqr_ok = False
        self._last_linearization_t = t
        self._last_target_index = self.navigator.target_index
        self._last_reference = x_ref.copy()

    def compute(self, t: float, state: np.ndarray, dt: float) -> tuple[np.ndarray, ControlDebug]:
        state = np.asarray(state, dtype=float)
        x_ref, ref = self._reference_state(state)

        if (not self.enabled) or (self.navigator.mission_complete and not self.waypoint_config.hold_last_waypoint):
            zero = np.zeros(6, dtype=float)
            return zero, ControlDebug(
                target_index=self.navigator.target_index,
                target_name=ref.target.name,
                desired_position=ref.desired_position.copy(),
                desired_attitude=state[3:6].copy(),
                position_error_ned=np.zeros(3),
                attitude_error=np.zeros(3),
                tau_cmd=zero.copy(),
                waypoint_reached=False,
                mission_complete=self.navigator.mission_complete,
            )

        R_nb = rotation_body_to_ned(state[3], state[4], state[5])
        v_ned = R_nb @ state[6:9]
        status = self.navigator.advance_if_needed(t, state, dt, ref, speed_ned=v_ned)
        waypoint_reached = status.advanced
        if waypoint_reached:
            self._last_target_index = -1
            self._last_reference = None
            x_ref, ref = self._reference_state(state)

        self._update_gain_if_needed(t, x_ref)

        err = state - x_ref
        err[3:6] = np.array([wrap_angle(state[3 + i] - x_ref[3 + i]) for i in range(3)], dtype=float)
        tau_cmd = self.lqr_config.tau_trim - self._K @ err
        tau_cmd = np.clip(tau_cmd, -self.lqr_config.tau_limits, self.lqr_config.tau_limits)

        return tau_cmd, ControlDebug(
            target_index=self.navigator.target_index,
            target_name=ref.target.name,
            desired_position=ref.desired_position.copy(),
            desired_attitude=ref.desired_attitude.copy(),
            position_error_ned=ref.position_error_ned.copy(),
            attitude_error=ref.attitude_error.copy(),
            tau_cmd=tau_cmd.copy(),
            waypoint_reached=waypoint_reached,
            mission_complete=self.navigator.mission_complete,
            advance_reason=status.reason,
        )
