#!/usr/bin/env python3
"""6-DOF marine vehicle dynamics model with optional physical disturbances.

State convention:
    state = [x, y, z, roll, pitch, yaw, u, v, w, p, q, r]

Pose is NED/inertial-style with z positive down. Linear/angular velocity is
body-frame. Hydrodynamic damping is computed from velocity relative to the
water current supplied by the Environment object.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional
from math import sin, cos
import copy

import numpy as np
import yaml

try:
    from environment import Environment
except ImportError:  # pragma: no cover
    Environment = None


STATE_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "u", "v", "w", "p", "q", "r"]
TAU_NAMES = ["X", "Y", "Z", "K", "M", "N"]


def wrap_angle(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def wrap_state_angles(state: np.ndarray) -> np.ndarray:
    s = np.asarray(state, dtype=float).copy()
    s[3:6] = [wrap_angle(a) for a in s[3:6]]
    return s


def _arr3(value: Any, default: list[float]) -> np.ndarray:
    return np.asarray(value if value is not None else default, dtype=float)


@dataclass
class VehicleParams:
    g: float = 9.81
    m: float = 11.5
    B: float = 114.8
    r_g: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.2], dtype=float))
    r_b: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=float))

    I_x: float = 0.16
    I_y: float = 0.16
    I_z: float = 0.16
    I_xy: float = 0.0
    I_xz: float = 0.0
    I_yz: float = 0.0

    X_u: float = 4.03
    Y_v: float = 6.22
    Z_w: float = 5.18
    K_p: float = 3.07
    M_q: float = 3.07
    N_r: float = 4.64

    X_uu: float = 18.18
    Y_vv: float = 21.66
    Z_ww: float = 36.99
    K_pp: float = 0.45
    M_qq: float = 0.45
    N_rr: float = 0.43

    X_u_dot: float = 5.5
    Y_v_dot: float = 12.7
    Z_w_dot: float = 14.57
    K_p_dot: float = 0.12
    M_q_dot: float = 0.2
    N_r_dot: float = 0.24

    @property
    def W(self) -> float:
        return self.m * self.g

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "VehicleParams":
        data = data or {}
        return VehicleParams(
            g=float(data.get("g", 9.81)),
            m=float(data.get("m", 11.5)),
            B=float(data.get("B", data.get("buoyancy", 114.8))),
            r_g=_arr3(data.get("r_g"), [0.0, 0.0, 0.2]),
            r_b=_arr3(data.get("r_b"), [0.0, 0.0, 0.0]),
            I_x=float(data.get("I_x", 0.16)),
            I_y=float(data.get("I_y", 0.16)),
            I_z=float(data.get("I_z", 0.16)),
            I_xy=float(data.get("I_xy", 0.0)),
            I_xz=float(data.get("I_xz", 0.0)),
            I_yz=float(data.get("I_yz", 0.0)),
            X_u=float(data.get("X_u", 4.03)),
            Y_v=float(data.get("Y_v", 6.22)),
            Z_w=float(data.get("Z_w", 5.18)),
            K_p=float(data.get("K_p", 3.07)),
            M_q=float(data.get("M_q", 3.07)),
            N_r=float(data.get("N_r", 4.64)),
            X_uu=float(data.get("X_uu", 18.18)),
            Y_vv=float(data.get("Y_vv", 21.66)),
            Z_ww=float(data.get("Z_ww", 36.99)),
            K_pp=float(data.get("K_pp", 0.45)),
            M_qq=float(data.get("M_qq", 0.45)),
            N_rr=float(data.get("N_rr", 0.43)),
            X_u_dot=float(data.get("X_u_dot", 5.5)),
            Y_v_dot=float(data.get("Y_v_dot", 12.7)),
            Z_w_dot=float(data.get("Z_w_dot", 14.57)),
            K_p_dot=float(data.get("K_p_dot", 0.12)),
            M_q_dot=float(data.get("M_q_dot", 0.2)),
            N_r_dot=float(data.get("N_r_dot", 0.24)),
        )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["r_g"] = self.r_g.tolist()
        d["r_b"] = self.r_b.tolist()
        d["W"] = self.W
        return d


def load_vehicle_params_yaml(path: str | Path) -> tuple[VehicleParams, VehicleParams]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if "truth" not in data or "nominal" not in data:
        raise ValueError("vehicle params YAML must contain top-level 'truth' and 'nominal' sections")
    return VehicleParams.from_dict(data["truth"]), VehicleParams.from_dict(data["nominal"])


@dataclass
class NoiseConfig:
    enabled: bool = False
    tau_bias_enabled: bool = True
    tau_random_enabled: bool = True
    process_noise_enabled: bool = False
    tau_bias_std: np.ndarray = field(default_factory=lambda: np.array([0.15, 0.15, 0.10, 0.02, 0.02, 0.02], dtype=float))
    tau_random_std: np.ndarray = field(default_factory=lambda: np.array([0.20, 0.20, 0.12, 0.015, 0.015, 0.015], dtype=float))
    state_process_noise_std: np.ndarray = field(default_factory=lambda: np.zeros(12, dtype=float))

    @staticmethod
    def from_dict(data: Dict[str, Any], enabled_override: Optional[bool] = None) -> "NoiseConfig":
        data = data or {}
        enabled = bool(data.get("enabled", False)) if enabled_override is None else bool(enabled_override)
        return NoiseConfig(
            enabled=enabled,
            tau_bias_enabled=bool(data.get("tau_bias_enabled", True)),
            tau_random_enabled=bool(data.get("tau_random_enabled", True)),
            process_noise_enabled=bool(data.get("process_noise_enabled", False)),
            tau_bias_std=np.asarray(data.get("tau_bias_std", [0.15, 0.15, 0.10, 0.02, 0.02, 0.02]), dtype=float),
            tau_random_std=np.asarray(data.get("tau_random_std", [0.20, 0.20, 0.12, 0.015, 0.015, 0.015]), dtype=float),
            state_process_noise_std=np.asarray(data.get("state_process_noise_std", [0.0] * 12), dtype=float),
        )


@dataclass
class StepResult:
    state_next: np.ndarray
    state_dot: np.ndarray
    tau_cmd: np.ndarray
    tau_applied: np.ndarray
    tau_bias: np.ndarray
    tau_random: np.ndarray
    process_noise: np.ndarray
    current_ned: np.ndarray
    current_body: np.ndarray
    hit_surface: bool = False
    hit_bottom: bool = False


class DynamicsModel:
    def __init__(
        self,
        params: Optional[VehicleParams] = None,
        noise: Optional[NoiseConfig] = None,
        environment: Optional[Any] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        self.params = copy.deepcopy(params) if params is not None else VehicleParams()
        self.noise = noise if noise is not None else NoiseConfig(enabled=False)
        self.environment = environment
        self.rng = rng if rng is not None else np.random.default_rng()
        self._tau_bias = np.zeros(6, dtype=float)
        if self.noise.enabled and self.noise.tau_bias_enabled:
            self._tau_bias = self.rng.normal(0.0, self.noise.tau_bias_std, size=6)

    @property
    def tau_bias(self) -> np.ndarray:
        return self._tau_bias.copy()

    def M_RB(self) -> np.ndarray:
        p = self.params
        m = p.m
        xg, yg, zg = p.r_g
        return np.array(
            [
                [m, 0, 0, 0, m * zg, -m * yg],
                [0, m, 0, -m * zg, 0, m * xg],
                [0, 0, m, m * yg, -m * xg, 0],
                [0, -m * zg, m * yg, p.I_x, -p.I_xy, -p.I_xz],
                [m * zg, 0, -m * xg, -p.I_xy, p.I_y, -p.I_yz],
                [-m * yg, m * xg, 0, -p.I_xz, -p.I_yz, p.I_z],
            ],
            dtype=float,
        )

    def M_A(self) -> np.ndarray:
        p = self.params
        return np.diag([p.X_u_dot, p.Y_v_dot, p.Z_w_dot, p.K_p_dot, p.M_q_dot, p.N_r_dot]).astype(float)

    def M(self) -> np.ndarray:
        return self.M_RB() + self.M_A()

    def C_RB(self, nu: np.ndarray) -> np.ndarray:
        p = self.params
        u, v, w, pp, q, r = nu
        m = p.m
        return np.array(
            [
                [0, 0, 0, 0, m * w, -m * v],
                [0, 0, 0, -m * w, 0, m * u],
                [0, 0, 0, m * v, -m * u, 0],
                [0, m * w, -m * v, 0, p.I_z * r, -p.I_y * q],
                [-m * w, 0, m * u, -p.I_z * r, 0, p.I_x * pp],
                [m * v, -m * u, 0, p.I_y * q, -p.I_x * pp, 0],
            ],
            dtype=float,
        )

    def C_A(self, nu_rel: np.ndarray) -> np.ndarray:
        p = self.params
        u, v, w, pp, q, r = nu_rel
        return np.array(
            [
                [0, 0, 0, 0, -p.Z_w_dot * w, p.Y_v_dot * v],
                [0, 0, 0, p.Z_w_dot * w, 0, -p.X_u_dot * u],
                [0, 0, 0, -p.Y_v_dot * v, p.X_u_dot * u, 0],
                [0, -p.Z_w_dot * w, p.Y_v_dot * v, 0, -p.N_r_dot * r, p.M_q_dot * q],
                [p.Z_w_dot * w, 0, -p.X_u_dot * u, p.N_r_dot * r, 0, -p.K_p_dot * pp],
                [-p.Y_v_dot * v, p.X_u_dot * u, 0, -p.M_q_dot * q, p.K_p_dot * pp, 0],
            ],
            dtype=float,
        )

    def D_L(self) -> np.ndarray:
        p = self.params
        return np.diag([p.X_u, p.Y_v, p.Z_w, p.K_p, p.M_q, p.N_r]).astype(float)

    def D_NL(self, nu_rel: np.ndarray) -> np.ndarray:
        p = self.params
        u, v, w, pp, q, r = nu_rel
        return np.diag(
            [
                p.X_uu * abs(u),
                p.Y_vv * abs(v),
                p.Z_ww * abs(w),
                p.K_pp * abs(pp),
                p.M_qq * abs(q),
                p.N_rr * abs(r),
            ]
        ).astype(float)

    def g_eta(self, phi: float, theta: float) -> np.ndarray:
        p = self.params
        W = p.W
        B = p.B
        xg, yg, zg = p.r_g
        xb, yb, zb = p.r_b
        sth = sin(theta)
        cth = cos(theta)
        sphi = sin(phi)
        cphi = cos(phi)
        return np.array(
            [
                (W - B) * sth,
                -(W - B) * cth * sphi,
                -(W - B) * cth * cphi,
                -(yg * W - yb * B) * cth * cphi + (zg * W - zb * B) * cth * sphi,
                (zg * W - zb * B) * sth + (xg * W - xb * B) * cth * cphi,
                -(xg * W - xb * B) * cth * sphi - (yg * W - yb * B) * sth,
            ],
            dtype=float,
        )

    def J_eta(self, phi: float, theta: float, psi: float) -> np.ndarray:
        cphi = cos(phi)
        sphi = sin(phi)
        cth = cos(theta)
        sth = sin(theta)
        cpsi = cos(psi)
        spsi = sin(psi)
        if abs(cth) < 1e-6:
            raise ValueError("Euler angle transform is singular near pitch = +/-90 deg")
        R = np.array(
            [
                [cpsi * cth, -spsi * cphi + cpsi * sth * sphi, spsi * sphi + cpsi * cphi * sth],
                [spsi * cth, cpsi * cphi + sphi * sth * spsi, -cpsi * sphi + sth * spsi * cphi],
                [-sth, cth * sphi, cth * cphi],
            ],
            dtype=float,
        )
        T = np.array(
            [
                [1, sphi * sth / cth, cphi * sth / cth],
                [0, cphi, -sphi],
                [0, sphi / cth, cphi / cth],
            ],
            dtype=float,
        )
        return np.block([[R, np.zeros((3, 3))], [np.zeros((3, 3)), T]])

    def current_body(self, t: float, state: np.ndarray, environment: Optional[Any] = None) -> np.ndarray:
        env = environment if environment is not None else self.environment
        if env is None:
            return np.zeros(3, dtype=float)
        return env.current_body_linear(t, state)

    def current_ned(self, t: float, state: np.ndarray, environment: Optional[Any] = None) -> np.ndarray:
        env = environment if environment is not None else self.environment
        if env is None:
            return np.zeros(3, dtype=float)
        return env.current_ned(t, float(state[2]))

    def state_derivative(self, t: float, state: np.ndarray, tau: np.ndarray, environment: Optional[Any] = None) -> np.ndarray:
        state = np.asarray(state, dtype=float)
        tau = np.asarray(tau, dtype=float)
        eta = state[:6]
        nu = state[6:12]
        phi, theta, psi = eta[3:6]

        eta_dot = self.J_eta(phi, theta, psi) @ nu

        current_body = self.current_body(t, state, environment)
        nu_rel = nu.copy()
        nu_rel[:3] -= current_body

        M = self.M()
        C_rb = self.C_RB(nu)
        C_a = self.C_A(nu_rel)
        D = self.D_L() + self.D_NL(nu_rel)
        restoring = self.g_eta(phi, theta)

        # Rigid-body Coriolis acts on actual velocity. Hydrodynamic terms act on
        # water-relative velocity.
        rhs = tau - C_rb @ nu - C_a @ nu_rel - D @ nu_rel - restoring
        nu_dot = np.linalg.solve(M, rhs)
        return np.concatenate([eta_dot, nu_dot])

    def rk4_step(self, t: float, state: np.ndarray, tau: np.ndarray, dt: float, environment: Optional[Any] = None) -> tuple[np.ndarray, np.ndarray]:
        k1 = self.state_derivative(t, state, tau, environment)
        k2 = self.state_derivative(t + 0.5 * dt, state + 0.5 * dt * k1, tau, environment)
        k3 = self.state_derivative(t + 0.5 * dt, state + 0.5 * dt * k2, tau, environment)
        k4 = self.state_derivative(t + dt, state + dt * k3, tau, environment)
        x_next = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return wrap_state_angles(x_next), k1

    def step(
        self,
        t: float,
        state: np.ndarray,
        tau_cmd: np.ndarray,
        dt: float,
        environment: Optional[Any] = None,
    ) -> StepResult:
        state = np.asarray(state, dtype=float)
        tau_cmd = np.asarray(tau_cmd, dtype=float)
        tau_random = np.zeros(6, dtype=float)
        if self.noise.enabled and self.noise.tau_random_enabled:
            tau_random = self.rng.normal(0.0, self.noise.tau_random_std, size=6)
        tau_applied = tau_cmd + (self._tau_bias if self.noise.enabled else 0.0) + tau_random

        state_next, state_dot = self.rk4_step(t, state, tau_applied, dt, environment)

        process_noise = np.zeros(12, dtype=float)
        if self.noise.enabled and self.noise.process_noise_enabled:
            process_noise = self.rng.normal(0.0, self.noise.state_process_noise_std, size=12)
            state_next = wrap_state_angles(state_next + process_noise)

        env = environment if environment is not None else self.environment
        hit_surface = False
        hit_bottom = False
        if env is not None:
            boundary = env.enforce_bounds(state_next)
            state_next = wrap_state_angles(boundary.state)
            hit_surface = boundary.hit_surface
            hit_bottom = boundary.hit_bottom

        return StepResult(
            state_next=state_next,
            state_dot=state_dot,
            tau_cmd=tau_cmd.copy(),
            tau_applied=tau_applied.copy(),
            tau_bias=self._tau_bias.copy(),
            tau_random=tau_random.copy(),
            process_noise=process_noise.copy(),
            current_ned=self.current_ned(t, state, environment),
            current_body=self.current_body(t, state, environment),
            hit_surface=hit_surface,
            hit_bottom=hit_bottom,
        )


def set_axes_equal(ax) -> None:
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)
    plot_radius = 0.5 * max([x_range, y_range, z_range, 1e-9])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
