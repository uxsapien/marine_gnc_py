#!/usr/bin/env python3
"""Environment model for the marine GNC simulation.

Owns water-column bounds and a time/depth-varying current profile.  The
coordinate convention is NED-like: x north/forward inertial, y east/right
inertial, z positive downward.  The water surface is normally z=0 and the
bottom is z=water_depth_m.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml
from math import sin, cos, pi


@dataclass
class BoundaryResult:
    state: np.ndarray
    hit_surface: bool = False
    hit_bottom: bool = False


@dataclass
class CurrentProfileConfig:
    enabled: bool = True
    depth_points_m: np.ndarray = field(default_factory=lambda: np.array([0.0, 5.0, 15.0, 30.0], dtype=float))
    current_ned_profile_mps: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [0.18, 0.04, 0.0],
                [0.14, 0.03, 0.0],
                [0.07, -0.01, 0.0],
                [0.02, -0.03, 0.0],
            ],
            dtype=float,
        )
    )
    sinusoid_amplitude_ned_mps: np.ndarray = field(default_factory=lambda: np.array([0.03, 0.02, 0.0], dtype=float))
    sinusoid_period_s: float = 45.0
    sinusoid_phase_rad: float = 0.0

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "CurrentProfileConfig":
        return CurrentProfileConfig(
            enabled=bool(data.get("enabled", True)),
            depth_points_m=np.asarray(data.get("depth_points_m", [0.0, 5.0, 15.0, 30.0]), dtype=float),
            current_ned_profile_mps=np.asarray(
                data.get(
                    "current_ned_profile_mps",
                    [[0.18, 0.04, 0.0], [0.14, 0.03, 0.0], [0.07, -0.01, 0.0], [0.02, -0.03, 0.0]],
                ),
                dtype=float,
            ),
            sinusoid_amplitude_ned_mps=np.asarray(data.get("sinusoid_amplitude_ned_mps", [0.03, 0.02, 0.0]), dtype=float),
            sinusoid_period_s=float(data.get("sinusoid_period_s", 45.0)),
            sinusoid_phase_rad=float(data.get("sinusoid_phase_rad", 0.0)),
        )

    def validate(self) -> None:
        if self.depth_points_m.ndim != 1:
            raise ValueError("depth_points_m must be a 1-D array")
        if self.current_ned_profile_mps.shape != (self.depth_points_m.size, 3):
            raise ValueError("current_ned_profile_mps must have shape (len(depth_points_m), 3)")
        if np.any(np.diff(self.depth_points_m) <= 0.0):
            raise ValueError("depth_points_m must be strictly increasing")
        if self.sinusoid_amplitude_ned_mps.shape != (3,):
            raise ValueError("sinusoid_amplitude_ned_mps must have length 3")
        if self.sinusoid_period_s <= 0.0:
            raise ValueError("sinusoid_period_s must be positive")


@dataclass
class EnvironmentConfig:
    enabled: bool = True
    surface_z_m: float = 0.0
    water_depth_m: float = 30.0
    current_profile: CurrentProfileConfig = field(default_factory=CurrentProfileConfig)

    @property
    def bottom_z_m(self) -> float:
        return self.surface_z_m + self.water_depth_m

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "EnvironmentConfig":
        bounds = data.get("water_bounds", {})
        current = data.get("current_profile", {})
        return EnvironmentConfig(
            enabled=bool(data.get("enabled", True)),
            surface_z_m=float(bounds.get("surface_z_m", data.get("surface_z_m", 0.0))),
            water_depth_m=float(bounds.get("water_depth_m", data.get("water_depth_m", 30.0))),
            current_profile=CurrentProfileConfig.from_dict(current),
        )

    @staticmethod
    def from_yaml(path: str | Path) -> "EnvironmentConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        cfg = EnvironmentConfig.from_dict(data)
        cfg.validate()
        return cfg

    @staticmethod
    def default() -> "EnvironmentConfig":
        cfg = EnvironmentConfig()
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.water_depth_m <= 0.0:
            raise ValueError("water_depth_m must be positive")
        self.current_profile.validate()


class Environment:
    """Runtime environment object used by dynamics and plotting."""

    def __init__(self, config: Optional[EnvironmentConfig] = None):
        self.config = config if config is not None else EnvironmentConfig.default()

    @staticmethod
    def from_yaml(path: str | Path) -> "Environment":
        return Environment(EnvironmentConfig.from_yaml(path))

    @staticmethod
    def default() -> "Environment":
        return Environment(EnvironmentConfig.default())

    @property
    def surface_z_m(self) -> float:
        return self.config.surface_z_m

    @property
    def bottom_z_m(self) -> float:
        return self.config.bottom_z_m

    def rotation_body_to_ned(self, phi: float, theta: float, psi: float) -> np.ndarray:
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

    def current_ned(self, t: float, depth_m: float) -> np.ndarray:
        """Return water current velocity in NED/inertial axes, m/s."""
        cfg = self.config.current_profile
        if (not self.config.enabled) or (not cfg.enabled):
            return np.zeros(3, dtype=float)

        z = float(np.clip(depth_m, self.surface_z_m, self.bottom_z_m))
        base = np.array(
            [
                np.interp(z, cfg.depth_points_m, cfg.current_ned_profile_mps[:, i])
                for i in range(3)
            ],
            dtype=float,
        )
        omega = 2.0 * pi / cfg.sinusoid_period_s
        return base + cfg.sinusoid_amplitude_ned_mps * sin(omega * float(t) + cfg.sinusoid_phase_rad)

    def current_body_linear(self, t: float, state: np.ndarray) -> np.ndarray:
        """Return water current linear velocity expressed in body axes."""
        state = np.asarray(state, dtype=float)
        phi, theta, psi = state[3:6]
        current_ned = self.current_ned(t, state[2])
        R_nb = self.rotation_body_to_ned(phi, theta, psi)
        return R_nb.T @ current_ned

    def enforce_bounds(self, state: np.ndarray) -> BoundaryResult:
        """Clamp z to the water column and remove velocity into the boundary.

        The body-frame linear velocity is converted to NED, the vertical NED
        component is zeroed if it points out of bounds, and then it is converted
        back to body axes. This avoids continuing to drive through the surface or
        bottom after clamping the position.
        """
        s = np.asarray(state, dtype=float).copy()
        if not self.config.enabled:
            return BoundaryResult(s)

        hit_surface = False
        hit_bottom = False
        if s[2] < self.surface_z_m:
            s[2] = self.surface_z_m
            hit_surface = True
        elif s[2] > self.bottom_z_m:
            s[2] = self.bottom_z_m
            hit_bottom = True

        if hit_surface or hit_bottom:
            phi, theta, psi = s[3:6]
            R_nb = self.rotation_body_to_ned(phi, theta, psi)
            v_ned = R_nb @ s[6:9]
            # NED z is positive downward. At the surface, upward means v_z < 0.
            if hit_surface and v_ned[2] < 0.0:
                v_ned[2] = 0.0
            # At the bottom, downward means v_z > 0.
            if hit_bottom and v_ned[2] > 0.0:
                v_ned[2] = 0.0
            s[6:9] = R_nb.T @ v_ned

        return BoundaryResult(s, hit_surface=hit_surface, hit_bottom=hit_bottom)

    def sample_current_profile_for_plot(self, t_values: np.ndarray, depth_values: np.ndarray) -> np.ndarray:
        out = np.zeros((len(t_values), len(depth_values), 3), dtype=float)
        for i, t in enumerate(t_values):
            for j, z in enumerate(depth_values):
                out[i, j, :] = self.current_ned(float(t), float(z))
        return out
