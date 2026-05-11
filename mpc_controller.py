#!/usr/bin/env python3
"""Waypoint-following nonlinear MPC controller for the marine GNC simulation.

Waypoint mission/progression is shared with PID/LQR through waypoints.py.
This module owns only MPC configuration and receding-horizon wrench selection.

The MPC optimizes a piecewise-constant wrench sequence over a short horizon
using the nominal nonlinear dynamics model.  It is intentionally simple and
slow-but-readable for simulation/experimentation rather than real-time control.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
from math import cos, sin

import numpy as np
import yaml
from scipy.optimize import minimize

from dynamics_model import DynamicsModel, wrap_angle, wrap_state_angles
from waypoints import WaypointFollowerConfig, WaypointNavigator, ControlDebug


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
class MPCControllerConfig:
    enabled: bool = True
    state_source: str = "ekf"  # ekf or truth
    tau_limits: np.ndarray = field(default_factory=lambda: np.array([30.0, 30.0, 22.0, 4.0, 4.0, 4.0], dtype=float))
    tau_trim: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0], dtype=float))
    Q: np.ndarray = field(default_factory=lambda: np.diag([1.0, 1.0, 2.0, 8.0, 8.0, 6.0, 0.3, 0.3, 0.5, 0.1, 0.1, 0.1]))
    Q_terminal: np.ndarray = field(default_factory=lambda: np.diag([2.5, 2.5, 4.0, 10.0, 10.0, 8.0, 0.5, 0.5, 0.8, 0.1, 0.1, 0.1]))
    R: np.ndarray = field(default_factory=lambda: np.diag([0.03, 0.03, 0.05, 0.15, 0.15, 0.12]))
    R_delta: np.ndarray = field(default_factory=lambda: np.diag([0.02, 0.02, 0.03, 0.06, 0.06, 0.05]))
    horizon_steps: int = 8
    prediction_dt: float = 0.09
    update_period_s: float = 0.18
    max_iterations: int = 40
    optimizer_ftol: float = 1e-3
    warm_start: bool = True
    fallback_to_trim_on_failure: bool = False
    reference_speed_mps: float = 0.0  # reserved for future moving-reference MPC

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "MPCControllerConfig":
        data = data or {}
        controller = data.get("controller", data)

        state_error_weights = controller.get("state_error_weights", {})
        q_defaults = {
            "x": 1.0,
            "y": 1.0,
            "z": 2.0,
            "roll_deg": 12.0,
            "pitch_deg": 12.0,
            "yaw_deg": 18.0,
            "u": 0.3,
            "v": 0.3,
            "w": 0.5,
            "p_deg": 60.0,
            "q_deg": 60.0,
            "r_deg": 60.0,
        }
        q_vec = _weight_vector(state_error_weights, q_defaults)

        terminal_weights = controller.get("terminal_state_error_weights", {})
        qf_defaults = {
            "x": 2.5,
            "y": 2.5,
            "z": 4.0,
            "roll_deg": 10.0,
            "pitch_deg": 10.0,
            "yaw_deg": 15.0,
            "u": 0.5,
            "v": 0.5,
            "w": 0.8,
            "p_deg": 60.0,
            "q_deg": 60.0,
            "r_deg": 60.0,
        }
        qf_vec = _weight_vector(terminal_weights, qf_defaults)

        control_weights = controller.get("control_weights", {})
        r_vec = np.array(
            [
                float(control_weights.get("X", 0.03)),
                float(control_weights.get("Y", 0.03)),
                float(control_weights.get("Z", 0.05)),
                float(control_weights.get("K", 0.15)),
                float(control_weights.get("M", 0.15)),
                float(control_weights.get("N", 0.12)),
            ],
            dtype=float,
        )

        control_rate_weights = controller.get("control_rate_weights", {})
        rd_vec = np.array(
            [
                float(control_rate_weights.get("X", 0.02)),
                float(control_rate_weights.get("Y", 0.02)),
                float(control_rate_weights.get("Z", 0.03)),
                float(control_rate_weights.get("K", 0.06)),
                float(control_rate_weights.get("M", 0.06)),
                float(control_rate_weights.get("N", 0.05)),
            ],
            dtype=float,
        )

        return MPCControllerConfig(
            enabled=bool(controller.get("enabled", True)),
            state_source=str(controller.get("state_source", "ekf")).lower(),
            tau_limits=_array(controller.get("tau_limits"), [30.0, 30.0, 22.0, 4.0, 4.0, 4.0], 6),
            tau_trim=_array(controller.get("tau_trim"), [0.0, 0.0, 2.0, 0.0, 0.0, 0.0], 6),
            Q=np.diag(q_vec),
            Q_terminal=np.diag(qf_vec),
            R=np.diag(r_vec),
            R_delta=np.diag(rd_vec),
            horizon_steps=max(1, int(controller.get("horizon_steps", 8))),
            prediction_dt=float(controller.get("prediction_dt", 0.09)),
            update_period_s=float(controller.get("update_period_s", 0.18)),
            max_iterations=max(1, int(controller.get("max_iterations", 40))),
            optimizer_ftol=float(controller.get("optimizer_ftol", 1e-3)),
            warm_start=bool(controller.get("warm_start", True)),
            fallback_to_trim_on_failure=bool(controller.get("fallback_to_trim_on_failure", False)),
            reference_speed_mps=float(controller.get("reference_speed_mps", 0.0)),
        )

    @staticmethod
    def from_yaml(path: str | Path) -> "MPCControllerConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return MPCControllerConfig.from_dict(data)


class WaypointMPCController:
    def __init__(
        self,
        waypoint_config: WaypointFollowerConfig,
        mpc_config: MPCControllerConfig,
        model: DynamicsModel,
        environment: Optional[Any] = None,
    ):
        self.navigator = WaypointNavigator(waypoint_config)
        self.waypoint_config = waypoint_config
        self.mpc_config = mpc_config
        self.model = model
        self.environment = environment
        self._last_tau = self.mpc_config.tau_trim.copy()
        self._last_sequence: Optional[np.ndarray] = None
        self._last_update_t = -np.inf
        self._last_target_index = -1
        self._last_solver_success = False
        self._last_solver_message = "not_run"

    @property
    def enabled(self) -> bool:
        return self.navigator.enabled and self.mpc_config.enabled

    @property
    def state_source(self) -> str:
        return self.mpc_config.state_source

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
        # Zero velocity/rate reference for this simple waypoint-regulation MPC.
        return x_ref, ref

    def _state_error(self, x: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
        err = np.asarray(x, dtype=float) - np.asarray(x_ref, dtype=float)
        err[3:6] = np.array([wrap_angle(x[3 + i] - x_ref[3 + i]) for i in range(3)], dtype=float)
        return err

    def _roll_warm_start(self, horizon: int) -> np.ndarray:
        if self._last_sequence is None or self._last_sequence.shape != (horizon, 6):
            return np.tile(self.mpc_config.tau_trim, (horizon, 1))
        shifted = np.vstack([self._last_sequence[1:, :], self._last_sequence[-1:, :]])
        return shifted.copy()

    def _simulate_candidate_cost(self, t: float, x0: np.ndarray, u_flat: np.ndarray) -> float:
        cfg = self.mpc_config
        horizon = cfg.horizon_steps
        U = np.asarray(u_flat, dtype=float).reshape(horizon, 6)
        x = np.asarray(x0, dtype=float).copy()
        cost = 0.0
        u_prev = self._last_tau.copy()

        for i in range(horizon):
            # Recompute heading reference along the predicted trajectory.  This
            # matters for point_heading mode because desired yaw depends on the
            # current predicted position relative to the active waypoint.
            x_ref, _ = self._reference_state(x)
            err = self._state_error(x, x_ref)
            du = U[i] - u_prev
            u_err = U[i] - cfg.tau_trim
            cost += float(err.T @ cfg.Q @ err)
            cost += float(u_err.T @ cfg.R @ u_err)
            cost += float(du.T @ cfg.R_delta @ du)

            try:
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    x, _ = self.model.rk4_step(t + i * cfg.prediction_dt, x, U[i], cfg.prediction_dt, self.environment)
            except Exception:
                return 1e30
            if (not np.all(np.isfinite(x))) or np.linalg.norm(x) > 1e4:
                return 1e30
            if self.environment is not None:
                x = self.environment.enforce_bounds(x).state
            x = wrap_state_angles(x)
            u_prev = U[i]

        x_ref, _ = self._reference_state(x)
        terminal_err = self._state_error(x, x_ref)
        cost += float(terminal_err.T @ cfg.Q_terminal @ terminal_err)
        if not np.isfinite(cost):
            return 1e30
        return cost

    def _solve_mpc(self, t: float, state: np.ndarray) -> np.ndarray:
        cfg = self.mpc_config
        horizon = cfg.horizon_steps
        if cfg.warm_start:
            u0 = self._roll_warm_start(horizon)
        else:
            u0 = np.tile(cfg.tau_trim, (horizon, 1))
        bounds = [(-float(lim), float(lim)) for _ in range(horizon) for lim in cfg.tau_limits]

        result = minimize(
            lambda u: self._simulate_candidate_cost(t, state, u),
            u0.reshape(-1),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": cfg.max_iterations, "ftol": cfg.optimizer_ftol, "maxls": 20},
        )
        self._last_solver_success = bool(result.success)
        self._last_solver_message = str(result.message)
        if result.success or np.all(np.isfinite(result.x)):
            sequence = np.asarray(result.x, dtype=float).reshape(horizon, 6)
            self._last_sequence = sequence.copy()
            return np.clip(sequence[0], -cfg.tau_limits, cfg.tau_limits)
        if cfg.fallback_to_trim_on_failure:
            return cfg.tau_trim.copy()
        return self._last_tau.copy()

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
            self._last_sequence = None
            self._last_update_t = -np.inf
            self._last_target_index = -1
            x_ref, ref = self._reference_state(state)

        due = (t - self._last_update_t) >= self.mpc_config.update_period_s
        target_changed = self.navigator.target_index != self._last_target_index
        if due or target_changed or self._last_sequence is None:
            tau_cmd = self._solve_mpc(t, state)
            self._last_tau = tau_cmd.copy()
            self._last_update_t = t
            self._last_target_index = self.navigator.target_index
        else:
            tau_cmd = self._last_tau.copy()

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
