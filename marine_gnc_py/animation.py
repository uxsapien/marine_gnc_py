#!/usr/bin/env python3
"""Realtime/recorded 3D animation utilities for the marine GNC simulation.

The animation shows:
  - the selected vehicle state history as an oriented ellipsoid,
  - body-fixed x/y/z axes moving with the vehicle,
  - truth and EKF trajectory traces,
  - waypoint markers and waypoint path,
  - optional surface/bottom reference planes.

State convention:
    state = [x, y, z, roll, pitch, yaw, u, v, w, p, q, r]

Pose is NED/inertial style with z positive down. The plot uses an inverted
z-axis so increasing depth appears lower on screen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
from math import sin, cos

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import yaml


def _as_float_array(value: Any, default: list[float]) -> np.ndarray:
    return np.asarray(value if value is not None else default, dtype=float)


@dataclass
class AnimationConfig:
    enabled: bool = False
    state_source: str = "truth"  # truth or ekf
    realtime: bool = True
    speed: float = 1.0
    stride: int = 3
    interval_ms: Optional[float] = None
    save_path: str = ""
    fps: int = 30
    dpi: int = 120

    # Ellipsoid body dimensions [x_length, y_width, z_height] in meters.
    body_dimensions_m: np.ndarray = field(default_factory=lambda: np.array([1.20, 0.55, 0.35], dtype=float))
    body_mesh_resolution: int = 18
    body_alpha: float = 0.85

    # Body axis lengths [x_axis, y_axis, z_axis] in meters.
    body_axis_lengths_m: np.ndarray = field(default_factory=lambda: np.array([1.00, 0.75, 0.60], dtype=float))
    show_axis_labels: bool = True

    trail_length: int = 300  # animation frames; <=0 means full trail
    show_truth_trace: bool = True
    show_ekf_trace: bool = True
    show_waypoints: bool = True
    show_reference_planes: bool = True
    view_elev_deg: float = 25.0
    view_azim_deg: float = -60.0
    margin_m: float = 2.0

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AnimationConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        cfg = cls()
        cfg.enabled = bool(data.get("enabled", cfg.enabled))
        cfg.state_source = str(data.get("state_source", cfg.state_source)).lower()
        cfg.realtime = bool(data.get("realtime", cfg.realtime))
        cfg.speed = float(data.get("speed", cfg.speed))
        cfg.stride = int(data.get("stride", cfg.stride))
        cfg.interval_ms = data.get("interval_ms", cfg.interval_ms)
        cfg.save_path = str(data.get("save_path", cfg.save_path) or "")
        cfg.fps = int(data.get("fps", cfg.fps))
        cfg.dpi = int(data.get("dpi", cfg.dpi))
        cfg.body_dimensions_m = _as_float_array(data.get("body_dimensions_m"), cfg.body_dimensions_m.tolist())
        cfg.body_mesh_resolution = int(data.get("body_mesh_resolution", cfg.body_mesh_resolution))
        cfg.body_alpha = float(data.get("body_alpha", cfg.body_alpha))
        cfg.body_axis_lengths_m = _as_float_array(data.get("body_axis_lengths_m"), cfg.body_axis_lengths_m.tolist())
        cfg.show_axis_labels = bool(data.get("show_axis_labels", cfg.show_axis_labels))
        cfg.trail_length = int(data.get("trail_length", cfg.trail_length))
        cfg.show_truth_trace = bool(data.get("show_truth_trace", cfg.show_truth_trace))
        cfg.show_ekf_trace = bool(data.get("show_ekf_trace", cfg.show_ekf_trace))
        cfg.show_waypoints = bool(data.get("show_waypoints", cfg.show_waypoints))
        cfg.show_reference_planes = bool(data.get("show_reference_planes", cfg.show_reference_planes))
        cfg.view_elev_deg = float(data.get("view_elev_deg", cfg.view_elev_deg))
        cfg.view_azim_deg = float(data.get("view_azim_deg", cfg.view_azim_deg))
        cfg.margin_m = float(data.get("margin_m", cfg.margin_m))
        return cfg


def body_to_ned_rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Return body-to-NED rotation matrix using ZYX Euler angles."""
    cphi, sphi = cos(roll), sin(roll)
    cth, sth = cos(pitch), sin(pitch)
    cpsi, spsi = cos(yaw), sin(yaw)
    return np.array(
        [
            [cpsi * cth, -spsi * cphi + cpsi * sth * sphi, spsi * sphi + cpsi * cphi * sth],
            [spsi * cth, cpsi * cphi + sphi * sth * spsi, -cpsi * sphi + sth * spsi * cphi],
            [-sth, cth * sphi, cth * cphi],
        ],
        dtype=float,
    )


def make_body_ellipsoid_mesh(dimensions_m: np.ndarray, resolution: int = 18) -> np.ndarray:
    """Return body-frame ellipsoid points as shape (3, n_theta, n_phi)."""
    rx, ry, rz = np.asarray(dimensions_m, dtype=float) / 2.0
    u = np.linspace(0.0, 2.0 * np.pi, resolution)
    v = np.linspace(0.0, np.pi, resolution)
    xb = rx * np.outer(np.cos(u), np.sin(v))
    yb = ry * np.outer(np.sin(u), np.sin(v))
    zb = rz * np.outer(np.ones_like(u), np.cos(v))
    return np.stack([xb, yb, zb], axis=0)


def transform_mesh(body_mesh: np.ndarray, position_ned: np.ndarray, rotation_b_to_ned: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat = body_mesh.reshape(3, -1)
    transformed = rotation_b_to_ned @ flat + position_ned.reshape(3, 1)
    reshaped = transformed.reshape(body_mesh.shape)
    return reshaped[0], reshaped[1], reshaped[2]


def _set_equal_3d_limits(ax, points: np.ndarray, margin_m: float = 2.0) -> None:
    if points.size == 0:
        return
    mins = np.nanmin(points, axis=0)
    maxs = np.nanmax(points, axis=0)
    centers = 0.5 * (mins + maxs)
    radius = 0.5 * np.nanmax(maxs - mins) + margin_m
    if not np.isfinite(radius) or radius <= 0:
        radius = margin_m if margin_m > 0 else 1.0
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def _draw_reference_plane(ax, xlim, ylim, z: float, alpha: float = 0.08):
    """Draw an unlabeled 3D reference plane.

    Matplotlib 3D PolyCollection artists can break legend creation on
    some Matplotlib versions, so labels are handled with simple proxy
    artists instead of assigning labels to plot_surface results.
    """
    x = np.linspace(xlim[0], xlim[1], 2)
    y = np.linspace(ylim[0], ylim[1], 2)
    X, Y = np.meshgrid(x, y)
    Z = np.full_like(X, z, dtype=float)
    return ax.plot_surface(X, Y, Z, alpha=alpha, linewidth=0.0)


def animate_trajectory(
    t: np.ndarray,
    truth: np.ndarray,
    ekf: Optional[np.ndarray] = None,
    waypoints: Optional[np.ndarray] = None,
    cfg: Optional[AnimationConfig] = None,
    surface_z_m: Optional[float] = 0.0,
    bottom_z_m: Optional[float] = None,
    show: bool = True,
    save_path: str | Path | None = None,
):
    """Animate the selected marine vehicle trajectory.

    The returned FuncAnimation must remain referenced by the caller while the
    window is open; this function keeps that reference until show/save returns.
    """
    cfg = cfg or AnimationConfig()
    state_source = cfg.state_source.lower()
    if state_source not in {"truth", "ekf"}:
        raise ValueError("Animation state_source must be 'truth' or 'ekf'.")
    if state_source == "ekf" and ekf is None:
        raise ValueError("EKF animation requested, but no EKF history was provided.")

    vehicle_states = truth if state_source == "truth" else ekf
    ekf = ekf if ekf is not None else np.full_like(truth, np.nan)
    waypoints = waypoints if waypoints is not None else np.zeros((0, 3), dtype=float)

    stride = max(1, int(cfg.stride))
    frame_indices = np.arange(0, len(t), stride, dtype=int)
    if frame_indices[-1] != len(t) - 1:
        frame_indices = np.append(frame_indices, len(t) - 1)

    if cfg.interval_ms is None:
        dt_nominal = float(np.nanmedian(np.diff(t))) if len(t) > 1 else 0.03
        interval_ms = max(1.0, 1000.0 * dt_nominal * stride / max(cfg.speed, 1.0e-6))
    else:
        interval_ms = float(cfg.interval_ms)

    body_mesh = make_body_ellipsoid_mesh(cfg.body_dimensions_m, cfg.body_mesh_resolution)

    fig = plt.figure(figsize=(9.0, 7.0))
    ax = fig.add_subplot(111, projection="3d")

    all_points = [truth[:, 0:3], ekf[:, 0:3]]
    if waypoints.size:
        all_points.append(waypoints[:, 0:3])
    all_points_np = np.vstack([p for p in all_points if p.size])
    _set_equal_3d_limits(ax, all_points_np, margin_m=cfg.margin_m)
    ax.invert_zaxis()
    ax.view_init(elev=cfg.view_elev_deg, azim=cfg.view_azim_deg)

    ax.set_xlabel("x / north [m]")
    ax.set_ylabel("y / east [m]")
    ax.set_zlabel("z / depth [m]")
    ax.set_title("marine vehicle trajectory animation")

    if cfg.show_reference_planes:
        xlim = ax.get_xlim3d()
        ylim = ax.get_ylim3d()
        if surface_z_m is not None:
            _draw_reference_plane(ax, xlim, ylim, float(surface_z_m), alpha=0.06)
        if bottom_z_m is not None:
            _draw_reference_plane(ax, xlim, ylim, float(bottom_z_m), alpha=0.05)

    if waypoints.size and cfg.show_waypoints:
        ax.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], marker="x", s=70)
        ax.plot(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], linestyle="--", linewidth=1.0)

    truth_trace_line, = ax.plot([], [], [], color="green", linewidth=2.0)
    ekf_trace_line, = ax.plot([], [], [], linewidth=1.3)
    body_center, = ax.plot([], [], [], marker="o", markersize=4, linestyle="None", color="black")

    # Mutable artists recreated every frame because 3D surface/quiver updates are limited.
    artists: Dict[str, Any] = {
        "surface": None,
        "axis_lines": [],
        "axis_labels": [],
        "time_text": ax.text2D(0.02, 0.95, "", transform=ax.transAxes),
    }

    axis_colors = ["red", "blue", "purple"]
    axis_names = ["xb", "yb", "zb"]

    def clear_vehicle_artists():
        if artists["surface"] is not None:
            artists["surface"].remove()
            artists["surface"] = None
        for line in artists["axis_lines"]:
            line.remove()
        artists["axis_lines"] = []
        for txt in artists["axis_labels"]:
            txt.remove()
        artists["axis_labels"] = []

    def update(frame_no: int):
        k = int(frame_indices[frame_no])
        clear_vehicle_artists()

        state = vehicle_states[k, :]
        pos = state[0:3]
        R = body_to_ned_rotation(state[3], state[4], state[5])
        X, Y, Z = transform_mesh(body_mesh, pos, R)
        artists["surface"] = ax.plot_surface(X, Y, Z, alpha=cfg.body_alpha, linewidth=0.0, shade=True)

        for j in range(3):
            end = pos + R[:, j] * cfg.body_axis_lengths_m[j]
            line, = ax.plot([pos[0], end[0]], [pos[1], end[1]], [pos[2], end[2]], color=axis_colors[j], linewidth=2.5)
            artists["axis_lines"].append(line)
            if cfg.show_axis_labels:
                txt = ax.text(end[0], end[1], end[2], axis_names[j], color=axis_colors[j], fontsize=9)
                artists["axis_labels"].append(txt)

        if cfg.trail_length > 0:
            start_frame = max(0, frame_no - cfg.trail_length)
            trail_idx = frame_indices[start_frame : frame_no + 1]
        else:
            trail_idx = frame_indices[: frame_no + 1]

        if cfg.show_truth_trace:
            truth_trace_line.set_data(truth[trail_idx, 0], truth[trail_idx, 1])
            truth_trace_line.set_3d_properties(truth[trail_idx, 2])
        if cfg.show_ekf_trace:
            ekf_trace_line.set_data(ekf[trail_idx, 0], ekf[trail_idx, 1])
            ekf_trace_line.set_3d_properties(ekf[trail_idx, 2])

        body_center.set_data([pos[0]], [pos[1]])
        body_center.set_3d_properties([pos[2]])
        artists["time_text"].set_text(f"t = {t[k]:.2f} s | frame {frame_no + 1}/{len(frame_indices)}")

        return [truth_trace_line, ekf_trace_line, body_center, artists["time_text"], artists["surface"], *artists["axis_lines"], *artists["axis_labels"]]

    # Use simple 2D proxy artists for the legend. Passing actual 3D
    # PolyCollection/Path3DCollection artists to legend() can fail on some
    # Matplotlib versions with errors like: tuple has no attribute size.
    legend_handles = []
    if cfg.show_truth_trace:
        legend_handles.append(Line2D([0], [0], color="green", linewidth=2.0, label="truth trail"))
    if cfg.show_ekf_trace:
        legend_handles.append(Line2D([0], [0], linewidth=1.3, label="EKF trail"))
    legend_handles.append(Line2D([0], [0], marker="o", color="black", linestyle="None", label=f"animated {state_source}"))
    if waypoints.size and cfg.show_waypoints:
        legend_handles.append(Line2D([0], [0], marker="x", color="black", linestyle="None", label="waypoints"))
        legend_handles.append(Line2D([0], [0], linestyle="--", color="black", linewidth=1.0, label="waypoint path"))
    if cfg.show_reference_planes and surface_z_m is not None:
        legend_handles.append(Patch(alpha=0.15, label="surface"))
    if cfg.show_reference_planes and bottom_z_m is not None:
        legend_handles.append(Patch(alpha=0.15, label="bottom"))
    ax.legend(handles=legend_handles, loc="upper right", fontsize="small")
    anim = FuncAnimation(fig, update, frames=len(frame_indices), interval=interval_ms, blit=False, repeat=False)

    output_path = Path(save_path or cfg.save_path) if (save_path or cfg.save_path) else None
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = output_path.suffix.lower()
        if suffix == ".gif":
            anim.save(output_path, writer=PillowWriter(fps=cfg.fps), dpi=cfg.dpi)
        elif suffix in {".mp4", ".m4v"}:
            anim.save(output_path, writer=FFMpegWriter(fps=cfg.fps), dpi=cfg.dpi)
        else:
            raise ValueError("Animation save path must end with .gif or .mp4")

    if show:
        plt.show()
    else:
        plt.close(fig)
    return anim
