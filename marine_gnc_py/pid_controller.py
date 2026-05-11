#!/usr/bin/env python3
"""PID waypoint controller for the marine GNC simulation.

The PID controller is intentionally separate from waypoint mission parsing and
progression.  Waypoint handling lives in waypoints.py so PID, LQR, and future
controllers can share the same waypoint semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict
from math import cos, sin

import numpy as np
import yaml

from marine_gnc_py.waypoints import WaypointFollowerConfig, WaypointNavigator, ControlDebug


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


@dataclass
class PIDAxisConfig:
    kp: float = 0.0
    ki: float = 0.0
    kd: float = 0.0
    integral_limit: float = 0.0
    output_limit: float = 0.0

    @staticmethod
    def from_dict(data: Dict[str, Any], default: "PIDAxisConfig") -> "PIDAxisConfig":
        data = data or {}
        return PIDAxisConfig(
            kp=float(data.get("kp", default.kp)),
            ki=float(data.get("ki", default.ki)),
            kd=float(data.get("kd", default.kd)),
            integral_limit=float(data.get("integral_limit", default.integral_limit)),
            output_limit=float(data.get("output_limit", default.output_limit)),
        )


@dataclass
class PIDControllerConfig:
    enabled: bool = True
    state_source: str = "ekf"  # ekf or truth
    tau_limits: np.ndarray = field(default_factory=lambda: np.array([35.0, 35.0, 25.0, 5.0, 5.0, 5.0], dtype=float))
    axes: Dict[str, PIDAxisConfig] = field(default_factory=dict)

    @staticmethod
    def defaults() -> Dict[str, PIDAxisConfig]:
        return {
            "x": PIDAxisConfig(kp=8.0, ki=0.0, kd=10.0, integral_limit=5.0, output_limit=25.0),
            "y": PIDAxisConfig(kp=8.0, ki=0.0, kd=10.0, integral_limit=5.0, output_limit=25.0),
            "z": PIDAxisConfig(kp=12.0, ki=0.0, kd=12.0, integral_limit=4.0, output_limit=18.0),
            "roll": PIDAxisConfig(kp=4.0, ki=0.0, kd=2.0, integral_limit=0.5, output_limit=4.0),
            "pitch": PIDAxisConfig(kp=4.0, ki=0.0, kd=2.0, integral_limit=0.5, output_limit=4.0),
            "yaw": PIDAxisConfig(kp=5.0, ki=0.0, kd=3.0, integral_limit=1.0, output_limit=4.0),
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "PIDControllerConfig":
        data = data or {}
        controller = data.get("controller", data)
        defaults = PIDControllerConfig.defaults()
        axis_data = controller.get("axes", {})
        axes = {name: PIDAxisConfig.from_dict(axis_data.get(name, {}), defaults[name]) for name in defaults}
        return PIDControllerConfig(
            enabled=bool(controller.get("enabled", True)),
            state_source=str(controller.get("state_source", "ekf")).lower(),
            tau_limits=_array(controller.get("tau_limits"), [35.0, 35.0, 25.0, 5.0, 5.0, 5.0], 6),
            axes=axes,
        )

    @staticmethod
    def from_yaml(path: str | Path) -> "PIDControllerConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return PIDControllerConfig.from_dict(data)


class AxisPID:
    def __init__(self, cfg: PIDAxisConfig):
        self.cfg = cfg
        self.integral = 0.0

    def reset(self) -> None:
        self.integral = 0.0

    def update(self, error: float, rate: float, dt: float) -> float:
        # Derivative on measurement: desired rate is zero, so d(error)/dt ~= -rate.
        if dt > 0.0:
            self.integral += error * dt
        if self.cfg.integral_limit > 0.0:
            self.integral = float(np.clip(self.integral, -self.cfg.integral_limit, self.cfg.integral_limit))
        out = self.cfg.kp * error + self.cfg.ki * self.integral - self.cfg.kd * rate
        if self.cfg.output_limit > 0.0:
            out = float(np.clip(out, -self.cfg.output_limit, self.cfg.output_limit))
        return out


class WaypointPIDController:
    def __init__(self, waypoint_config: WaypointFollowerConfig, pid_config: PIDControllerConfig):
        self.navigator = WaypointNavigator(waypoint_config)
        self.waypoint_config = waypoint_config
        self.pid_config = pid_config
        self.pid = {name: AxisPID(cfg) for name, cfg in pid_config.axes.items()}

    @staticmethod
    def from_yaml(waypoints_yaml: str | Path, pid_yaml: str | Path) -> "WaypointPIDController":
        return WaypointPIDController(WaypointFollowerConfig.from_yaml(waypoints_yaml), PIDControllerConfig.from_yaml(pid_yaml))

    def reset_integrators(self) -> None:
        for pid in self.pid.values():
            pid.reset()

    @property
    def enabled(self) -> bool:
        return self.navigator.enabled and self.pid_config.enabled

    @property
    def state_source(self) -> str:
        return self.pid_config.state_source

    @property
    def target_index(self) -> int:
        return self.navigator.target_index

    @property
    def mission_complete(self) -> bool:
        return self.navigator.mission_complete

    def current_waypoint(self):
        return self.navigator.current_waypoint()

    def compute(self, t: float, state: np.ndarray, dt: float) -> tuple[np.ndarray, ControlDebug]:
        state = np.asarray(state, dtype=float)
        ref = self.navigator.reference(state)
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
        omega_body = state[9:12]

        status = self.navigator.advance_if_needed(t, state, dt, ref, speed_ned=v_ned)
        waypoint_reached = status.advanced
        if waypoint_reached:
            self.reset_integrators()
            ref = self.navigator.reference(state)

        force_ned = np.array(
            [
                self.pid["x"].update(float(ref.position_error_ned[0]), float(v_ned[0]), dt),
                self.pid["y"].update(float(ref.position_error_ned[1]), float(v_ned[1]), dt),
                self.pid["z"].update(float(ref.position_error_ned[2]), float(v_ned[2]), dt),
            ],
            dtype=float,
        )
        force_body = R_nb.T @ force_ned
        moment_body = np.array(
            [
                self.pid["roll"].update(float(ref.attitude_error[0]), float(omega_body[0]), dt),
                self.pid["pitch"].update(float(ref.attitude_error[1]), float(omega_body[1]), dt),
                self.pid["yaw"].update(float(ref.attitude_error[2]), float(omega_body[2]), dt),
            ],
            dtype=float,
        )
        tau_cmd = np.concatenate([force_body, moment_body])
        tau_cmd = np.clip(tau_cmd, -self.pid_config.tau_limits, self.pid_config.tau_limits)

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
