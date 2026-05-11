#!/usr/bin/env python3
"""Waypoint mission and waypoint progression utilities.

This module intentionally does not implement a specific control law.  It owns
only mission/profile parsing, desired waypoint attitude generation, and target
advancement.  PID, LQR, MPC, or any future controller can share the same
WaypointNavigator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional
from math import atan2

import numpy as np
import yaml

from marine_gnc_py.dynamics_model import wrap_angle


def _array(data: Any, default: list[float], n: int) -> np.ndarray:
    out = np.asarray(data if data is not None else default, dtype=float)
    if out.size != n:
        raise ValueError(f"Expected array length {n}, got {out.size}")
    return out


@dataclass
class Waypoint:
    name: str
    position: np.ndarray
    roll: Optional[float] = None
    pitch: Optional[float] = None
    yaw: Optional[float] = None

    @staticmethod
    def from_dict(i: int, data: dict[str, Any]) -> "Waypoint":
        position = _array(data.get("position"), [0.0, 0.0, 0.0], 3)

        def angle_from(prefix: str) -> Optional[float]:
            if prefix in data:
                return float(data[prefix])
            deg_key = f"{prefix}_deg"
            if deg_key in data:
                return float(np.deg2rad(data[deg_key]))
            return None

        roll = angle_from("roll")
        pitch = angle_from("pitch")
        yaw = angle_from("yaw")

        attitude = data.get("attitude", None)
        if attitude is not None:
            att = np.asarray(attitude, dtype=float)
            if att.size != 3:
                raise ValueError("attitude must be [roll, pitch, yaw] in radians")
            roll, pitch, yaw = float(att[0]), float(att[1]), float(att[2])

        attitude_deg = data.get("attitude_deg", None)
        if attitude_deg is not None:
            att = np.deg2rad(np.asarray(attitude_deg, dtype=float))
            if att.size != 3:
                raise ValueError("attitude_deg must be [roll, pitch, yaw] in degrees")
            roll, pitch, yaw = float(att[0]), float(att[1]), float(att[2])

        return Waypoint(
            name=str(data.get("name", f"wp_{i}")),
            position=position,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
        )


@dataclass
class WaypointFollowerConfig:
    enabled: bool = True
    profile: str = "point_heading"  # point_heading or arbitrary_orientation
    loop: bool = False
    hold_last_waypoint: bool = True

    # Strict waypoint acceptance.
    position_tolerance_m: float = 0.75
    depth_tolerance_m: float = 0.40
    yaw_tolerance_deg: float = 15.0
    require_heading_for_advance: bool = False

    # Robust advancement. This prevents regulator-style controllers such as LQR
    # from parking just outside a tight acceptance radius forever.
    relaxed_position_tolerance_m: float = 1.25
    relaxed_depth_tolerance_m: float = 0.65
    relaxed_speed_tolerance_mps: float = 0.20
    relaxed_acceptance_time_s: float = 0.50

    # If the vehicle crosses past the waypoint along the current leg, advance
    # even if it did not enter the exact acceptance sphere.
    advance_when_past_waypoint: bool = True
    past_waypoint_cross_track_tolerance_m: float = 1.25

    waypoints: List[Waypoint] = field(default_factory=list)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "WaypointFollowerConfig":
        data = data or {}
        waypoints = [Waypoint.from_dict(i, wp) for i, wp in enumerate(data.get("waypoints", []))]
        if not waypoints:
            raise ValueError("waypoint config must contain at least one waypoint")

        profile = str(data.get("profile", "point_heading")).lower()
        if profile not in ("point_heading", "arbitrary_orientation"):
            raise ValueError("waypoint profile must be 'point_heading' or 'arbitrary_orientation'")

        return WaypointFollowerConfig(
            enabled=bool(data.get("enabled", True)),
            profile=profile,
            loop=bool(data.get("loop", False)),
            hold_last_waypoint=bool(data.get("hold_last_waypoint", True)),
            position_tolerance_m=float(data.get("position_tolerance_m", 0.75)),
            depth_tolerance_m=float(data.get("depth_tolerance_m", 0.40)),
            yaw_tolerance_deg=float(data.get("yaw_tolerance_deg", 15.0)),
            require_heading_for_advance=bool(data.get("require_heading_for_advance", False)),
            relaxed_position_tolerance_m=float(data.get("relaxed_position_tolerance_m", 1.25)),
            relaxed_depth_tolerance_m=float(data.get("relaxed_depth_tolerance_m", 0.65)),
            relaxed_speed_tolerance_mps=float(data.get("relaxed_speed_tolerance_mps", 0.20)),
            relaxed_acceptance_time_s=float(data.get("relaxed_acceptance_time_s", 0.50)),
            advance_when_past_waypoint=bool(data.get("advance_when_past_waypoint", True)),
            past_waypoint_cross_track_tolerance_m=float(data.get("past_waypoint_cross_track_tolerance_m", 1.25)),
            waypoints=waypoints,
        )

    @staticmethod
    def from_yaml(path: str | Path) -> "WaypointFollowerConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return WaypointFollowerConfig.from_dict(data)

    @property
    def yaw_tolerance_rad(self) -> float:
        return float(np.deg2rad(self.yaw_tolerance_deg))


@dataclass
class WaypointReference:
    target: Waypoint
    target_index: int
    desired_position: np.ndarray
    desired_attitude: np.ndarray
    position_error_ned: np.ndarray
    attitude_error: np.ndarray


@dataclass
class AdvanceStatus:
    advanced: bool
    reason: str = ""
    mission_complete: bool = False


@dataclass
class ControlDebug:
    target_index: int
    target_name: str
    desired_position: np.ndarray
    desired_attitude: np.ndarray
    position_error_ned: np.ndarray
    attitude_error: np.ndarray
    tau_cmd: np.ndarray
    waypoint_reached: bool
    mission_complete: bool
    advance_reason: str = ""


class WaypointNavigator:
    """Stateful waypoint target manager shared by PID/LQR controllers."""

    def __init__(self, config: WaypointFollowerConfig):
        self.config = config
        self.target_index = 0
        self.mission_complete = False
        self._held_arbitrary_attitude: Optional[np.ndarray] = None
        self._relaxed_timer_s = 0.0

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def current_waypoint(self) -> Waypoint:
        return self.config.waypoints[self.target_index]

    def desired_attitude(self, state: np.ndarray, target: Waypoint) -> np.ndarray:
        cfg = self.config
        current_att = np.asarray(state[3:6], dtype=float)
        if cfg.profile == "point_heading":
            dx = target.position[0] - state[0]
            dy = target.position[1] - state[1]
            desired_yaw = current_att[2] if (abs(dx) < 1e-9 and abs(dy) < 1e-9) else atan2(dy, dx)
            # Heading-only profile: keep roll and pitch commanded level.
            return np.array([0.0, 0.0, desired_yaw], dtype=float)

        # arbitrary_orientation: use waypoint attitude if provided. Otherwise
        # hold the attitude from the moment this target became active.
        if self._held_arbitrary_attitude is None:
            self._held_arbitrary_attitude = current_att.copy()
        return np.array(
            [
                target.roll if target.roll is not None else self._held_arbitrary_attitude[0],
                target.pitch if target.pitch is not None else self._held_arbitrary_attitude[1],
                target.yaw if target.yaw is not None else self._held_arbitrary_attitude[2],
            ],
            dtype=float,
        )

    def reference(self, state: np.ndarray) -> WaypointReference:
        target = self.current_waypoint()
        desired_pos = target.position.copy()
        desired_att = self.desired_attitude(state, target)
        pos_err = desired_pos - state[0:3]
        att_err = np.array([wrap_angle(desired_att[i] - state[3 + i]) for i in range(3)], dtype=float)
        return WaypointReference(
            target=target,
            target_index=self.target_index,
            desired_position=desired_pos,
            desired_attitude=desired_att,
            position_error_ned=pos_err,
            attitude_error=att_err,
        )

    def _attitude_requirement_ok(self, ref: WaypointReference) -> bool:
        cfg = self.config
        yaw_ok = abs(float(ref.attitude_error[2])) <= cfg.yaw_tolerance_rad
        if cfg.profile == "point_heading":
            return (not cfg.require_heading_for_advance) or yaw_ok
        wp = self.current_waypoint()
        needs_attitude = any(v is not None for v in (wp.roll, wp.pitch, wp.yaw))
        return (not needs_attitude) or yaw_ok

    def _strict_acceptance(self, ref: WaypointReference) -> bool:
        cfg = self.config
        xy_ok = float(np.linalg.norm(ref.position_error_ned[:2])) <= cfg.position_tolerance_m
        z_ok = abs(float(ref.position_error_ned[2])) <= cfg.depth_tolerance_m
        return xy_ok and z_ok and self._attitude_requirement_ok(ref)

    def _relaxed_acceptance(self, ref: WaypointReference, speed_ned: np.ndarray | None, dt: float) -> bool:
        cfg = self.config
        if cfg.relaxed_acceptance_time_s <= 0.0:
            return False
        xy_ok = float(np.linalg.norm(ref.position_error_ned[:2])) <= cfg.relaxed_position_tolerance_m
        z_ok = abs(float(ref.position_error_ned[2])) <= cfg.relaxed_depth_tolerance_m
        speed_ok = True
        if speed_ned is not None:
            speed_ok = float(np.linalg.norm(speed_ned[:3])) <= cfg.relaxed_speed_tolerance_mps
        att_ok = self._attitude_requirement_ok(ref)
        if xy_ok and z_ok and speed_ok and att_ok:
            self._relaxed_timer_s += max(float(dt), 0.0)
        else:
            self._relaxed_timer_s = 0.0
        return self._relaxed_timer_s >= cfg.relaxed_acceptance_time_s

    def _past_waypoint_acceptance(self, state: np.ndarray, ref: WaypointReference) -> bool:
        cfg = self.config
        if not cfg.advance_when_past_waypoint or self.target_index <= 0:
            return False
        prev = cfg.waypoints[self.target_index - 1].position
        target = ref.desired_position
        leg = target - prev
        leg_xy = leg[:2]
        leg_len2 = float(np.dot(leg_xy, leg_xy))
        if leg_len2 < 1e-12:
            return False
        vehicle_xy = state[:2] - prev[:2]
        along = float(np.dot(vehicle_xy, leg_xy) / leg_len2)
        closest_xy = prev[:2] + np.clip(along, 0.0, 1.0) * leg_xy
        cross_track = float(np.linalg.norm(state[:2] - closest_xy))
        z_ok = abs(float(ref.position_error_ned[2])) <= max(cfg.depth_tolerance_m, cfg.relaxed_depth_tolerance_m)
        return along >= 1.0 and cross_track <= cfg.past_waypoint_cross_track_tolerance_m and z_ok

    def advance_if_needed(
        self,
        t: float,
        state: np.ndarray,
        dt: float,
        ref: WaypointReference,
        speed_ned: np.ndarray | None = None,
    ) -> AdvanceStatus:
        if self.mission_complete:
            return AdvanceStatus(False, "mission_complete", True)

        reason = ""
        if self._strict_acceptance(ref):
            reason = "strict_acceptance"
        elif self._relaxed_acceptance(ref, speed_ned, dt):
            reason = "relaxed_stationary_acceptance"
        elif self._past_waypoint_acceptance(state, ref):
            reason = "passed_waypoint"
        else:
            return AdvanceStatus(False, "", False)

        cfg = self.config
        if self.target_index < len(cfg.waypoints) - 1:
            self.target_index += 1
            self._held_arbitrary_attitude = None
            self._relaxed_timer_s = 0.0
            return AdvanceStatus(True, reason, False)
        if cfg.loop:
            self.target_index = 0
            self._held_arbitrary_attitude = None
            self._relaxed_timer_s = 0.0
            return AdvanceStatus(True, reason, False)

        self.mission_complete = True
        self._relaxed_timer_s = 0.0
        return AdvanceStatus(True, reason, True)
