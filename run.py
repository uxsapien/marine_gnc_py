#!/usr/bin/env python3
"""Run marine GNC truth, sensor, EKF, and optional waypoint-controller simulation.

The runner owns iteration and logging. The model returns one step at a time,
the sensor stack generates measurements from truth, the EKF estimates state,
and the waypoint controller optionally produces tau_cmd from EKF/truth state.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from dynamics_model import (
    DynamicsModel,
    NoiseConfig,
    STATE_NAMES,
    TAU_NAMES,
    load_vehicle_params_yaml,
    set_axes_equal,
)
from environment import Environment, EnvironmentConfig
from sensor_stack import SensorStack, SensorStackConfig, readings_to_flat_dict
from ekf import EkfConfig, ExtendedKalmanFilter
from waypoints import WaypointFollowerConfig
from pid_controller import PIDControllerConfig, WaypointPIDController
from lqr_controller import LQRControllerConfig, WaypointLQRController
from mpc_controller import MPCControllerConfig, WaypointMPCController
from animation import AnimationConfig, animate_trajectory


DEFAULT_CONFIG_DIR = Path("config")
DEFAULT_VEHICLE_PARAMS = DEFAULT_CONFIG_DIR / "vehicle_params.yaml"
DEFAULT_ENVIRONMENT_CONFIG = DEFAULT_CONFIG_DIR / "environment_config.yaml"
DEFAULT_SENSOR_CONFIG = DEFAULT_CONFIG_DIR / "sensor_config.yaml"
DEFAULT_EKF_CONFIG = DEFAULT_CONFIG_DIR / "ekf_config.yaml"
DEFAULT_WAYPOINTS_CONFIG = DEFAULT_CONFIG_DIR / "waypoints.yaml"
DEFAULT_PID_CONFIG = DEFAULT_CONFIG_DIR / "pid_config.yaml"
DEFAULT_LQR_CONFIG = DEFAULT_CONFIG_DIR / "lqr_config.yaml"
DEFAULT_MPC_CONFIG = DEFAULT_CONFIG_DIR / "mpc_config.yaml"
DEFAULT_ANIMATION_CONFIG = DEFAULT_CONFIG_DIR / "animation_config.yaml"


def load_truth_noise_config(path: Path, enabled_override: bool | None = None, state_process_noise_override: bool | None = None) -> NoiseConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    disturbance = data.get("truth_disturbance", {})
    cfg = NoiseConfig.from_dict(disturbance, enabled_override=enabled_override)
    if state_process_noise_override is not None:
        cfg.process_noise_enabled = bool(state_process_noise_override)
    return cfg


def ensure_output_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def initialize_histories(n_steps: int, sensor_stack: SensorStack) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[int, List[str]]]:
    truth = np.zeros((n_steps, 12), dtype=float)
    ekf = np.zeros((n_steps, 12), dtype=float)
    tau_cmd = np.zeros((n_steps, 6), dtype=float)
    tau_applied = np.zeros((n_steps, 6), dtype=float)

    sensor_series: Dict[str, np.ndarray] = {}
    sensor_labels_by_state: Dict[int, List[str]] = {i: [] for i in range(12)}
    for sname, cfg in sensor_stack.sensors.items():
        if not cfg.enabled:
            continue
        for local_j, state_idx in enumerate(cfg.state_indices):
            meas_name = cfg.measurement_names[local_j]
            key = f"{sname}:{meas_name}"
            sensor_series[key] = np.full(n_steps, np.nan, dtype=float)
            sensor_labels_by_state[int(state_idx)].append(key)
    return truth, ekf, tau_cmd, tau_applied, sensor_series, sensor_labels_by_state


def update_sensor_series(sensor_series: Dict[str, np.ndarray], k: int, readings) -> None:
    for reading in readings:
        if not reading.available:
            continue
        for j, meas_name in enumerate(reading.measurement_names):
            key = f"{reading.name}:{meas_name}"
            if key in sensor_series:
                sensor_series[key][k] = reading.z[j]


def make_dataframe(
    t: np.ndarray,
    truth: np.ndarray,
    ekf: np.ndarray,
    tau_cmd: np.ndarray,
    tau_applied: np.ndarray,
    current_ned: np.ndarray,
    current_body: np.ndarray,
    sensor_series: Dict[str, np.ndarray],
    waypoint_index: np.ndarray,
    desired_position: np.ndarray,
    desired_attitude: np.ndarray,
    position_error_ned: np.ndarray,
    attitude_error: np.ndarray,
    mission_complete: np.ndarray,
) -> pd.DataFrame:
    data = {"t": t}
    for i, name in enumerate(STATE_NAMES):
        data[f"truth_{name}"] = truth[:, i]
        data[f"ekf_{name}"] = ekf[:, i]
        data[f"error_{name}"] = truth[:, i] - ekf[:, i]
    for i, name in enumerate(TAU_NAMES):
        data[f"tau_cmd_{name}"] = tau_cmd[:, i]
        data[f"tau_applied_{name}"] = tau_applied[:, i]
    for i, name in enumerate(["north", "east", "down"]):
        data[f"current_ned_{name}"] = current_ned[:, i]
    for i, name in enumerate(["u", "v", "w"]):
        data[f"current_body_{name}"] = current_body[:, i]
    for key, values in sensor_series.items():
        safe_key = key.replace(":", "_")
        data[f"sensor_{safe_key}"] = values
    data["waypoint_index"] = waypoint_index
    data["mission_complete"] = mission_complete.astype(int)
    for i, name in enumerate(["x", "y", "z"]):
        data[f"desired_{name}"] = desired_position[:, i]
        data[f"position_error_{name}"] = position_error_ned[:, i]
    for i, name in enumerate(["roll", "pitch", "yaw"]):
        data[f"desired_{name}"] = desired_attitude[:, i]
        data[f"attitude_error_{name}"] = attitude_error[:, i]
    return pd.DataFrame(data)


def plot_state_group(
    t: np.ndarray,
    truth: np.ndarray,
    ekf: np.ndarray,
    sensor_series: Dict[str, np.ndarray],
    sensor_labels_by_state: Dict[int, List[str]],
    indices: List[int],
    titles: List[str],
    ylabel: str,
    output_path: Path,
    angle_deg: bool = False,
    show: bool = False,
) -> None:
    fig, axs = plt.subplots(1, len(indices), figsize=(6.0 * len(indices), 4.5), squeeze=False)
    axs = axs[0]
    for ax, idx, title in zip(axs, indices, titles):
        truth_vals = truth[:, idx]
        ekf_vals = ekf[:, idx]
        if angle_deg:
            truth_vals = np.rad2deg(truth_vals)
            ekf_vals = np.rad2deg(ekf_vals)
        ax.plot(t, truth_vals, label="truth", color="green", linewidth=2.0)
        ax.plot(t, ekf_vals, label="EKF", linewidth=1.8)

        for sensor_key in sensor_labels_by_state.get(idx, []):
            vals = sensor_series[sensor_key]
            valid = ~np.isnan(vals)
            if np.any(valid):
                plot_vals = np.rad2deg(vals[valid]) if angle_deg else vals[valid]
                ax.scatter(t[valid], plot_vals, s=14, alpha=0.65, color="orange", label=f"sensor {sensor_key}")
        ax.set_title(title)
        ax.set_xlabel("t [s]")
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def plot_environment_current(t: np.ndarray, current_ned: np.ndarray, output_path: Path, show: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(t, current_ned[:, 0], label="current north")
    ax.plot(t, current_ned[:, 1], label="current east")
    ax.plot(t, current_ned[:, 2], label="current down")
    ax.set_title("Current at vehicle depth")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("current [m/s]")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def plot_trajectory(
    truth: np.ndarray,
    ekf: np.ndarray,
    waypoints: Optional[np.ndarray],
    output_path: Path,
    show: bool = False,
) -> None:
    fig = plt.figure(figsize=(7.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(truth[:, 0], truth[:, 1], truth[:, 2], label="truth", color="green", linewidth=2.0)
    ax.plot(ekf[:, 0], ekf[:, 1], ekf[:, 2], label="EKF", linewidth=1.6)
    if waypoints is not None and waypoints.size:
        ax.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], marker="x", s=70, label="waypoints")
        ax.plot(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], linestyle="--", linewidth=1.0, label="waypoint path")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("3D trajectory")
    ax.legend()
    set_axes_equal(ax)
    ax.invert_zaxis()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def plot_tau(t: np.ndarray, tau_cmd: np.ndarray, tau_applied: np.ndarray, output_path: Path, show: bool = False) -> None:
    fig, axs = plt.subplots(2, 3, figsize=(15, 7), squeeze=False)
    axs = axs.ravel()
    for i, name in enumerate(TAU_NAMES):
        axs[i].plot(t, tau_cmd[:, i], label=f"cmd {name}")
        axs[i].plot(t, tau_applied[:, i], label=f"applied {name}", alpha=0.75)
        axs[i].set_title(name)
        axs[i].set_xlabel("t [s]")
        axs[i].grid(True)
        axs[i].legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def save_config_snapshot(paths: Dict[str, Path], output_path: Path) -> None:
    snapshot = {}
    for name, path in paths.items():
        with open(path, "r", encoding="utf-8") as f:
            snapshot[name] = yaml.safe_load(f)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)


def run(args: argparse.Namespace) -> Dict[str, Path]:
    vehicle_yaml = Path(args.vehicle_params_yaml)
    environment_yaml = Path(args.environment_config_yaml)
    sensor_yaml = Path(args.sensor_config_yaml)
    ekf_yaml = Path(args.ekf_config_yaml)
    waypoints_yaml = Path(args.waypoints_yaml)
    pid_yaml = Path(args.pid_config_yaml)
    lqr_yaml = Path(args.lqr_config_yaml)
    mpc_yaml = Path(args.mpc_config_yaml)
    animation_yaml = Path(args.animation_config_yaml)
    output_dir = ensure_output_dir(args.output_dir)

    true_params, nominal_params = load_vehicle_params_yaml(vehicle_yaml)
    truth_environment = Environment(EnvironmentConfig.from_yaml(environment_yaml))
    # By default, only the truth plant knows the real environment/current.
    # EKF and model-based controllers are environment-blind unless explicitly opted in.
    ekf_environment = truth_environment if args.ekf_knows_environment else None
    controller_environment = truth_environment if args.controllers_know_environment else None
    truth_noise = load_truth_noise_config(
        environment_yaml,
        enabled_override=args.truth_noise,
        state_process_noise_override=args.truth_state_process_noise,
    )
    sensor_cfg = SensorStackConfig.from_yaml(sensor_yaml)
    ekf_cfg = EkfConfig.from_yaml(ekf_yaml)

    truth_model = DynamicsModel(
        params=true_params,
        noise=truth_noise,
        environment=truth_environment,
        rng=np.random.default_rng(args.seed + 10),
    )
    ekf_model = DynamicsModel(
        params=nominal_params,
        noise=NoiseConfig(enabled=False),
        environment=ekf_environment,
        rng=np.random.default_rng(args.seed + 20),
    )
    controller_model = DynamicsModel(
        params=nominal_params,
        noise=NoiseConfig(enabled=False),
        environment=controller_environment,
        rng=np.random.default_rng(args.seed + 30),
    )
    sensor_stack = SensorStack(
        config=sensor_cfg,
        rng=np.random.default_rng(args.seed + sensor_cfg.seed_offset),
        noise_enabled=args.sensor_noise,
    )
    ekf = ExtendedKalmanFilter(model=ekf_model, config=ekf_cfg, environment=ekf_environment)

    controller = None
    waypoint_positions: Optional[np.ndarray] = None
    controller_type = "disabled"
    if not args.disable_waypoint_controller:
        waypoint_cfg = WaypointFollowerConfig.from_yaml(waypoints_yaml)
        controller_type = args.controller.lower()
        if controller_type == "pid":
            pid_cfg = PIDControllerConfig.from_yaml(pid_yaml)
            if args.controller_state_source is not None:
                pid_cfg.state_source = args.controller_state_source.lower()
            controller = WaypointPIDController(waypoint_cfg, pid_cfg)
        elif controller_type == "lqr":
            lqr_cfg = LQRControllerConfig.from_yaml(args.lqr_config_yaml)
            if args.controller_state_source is not None:
                lqr_cfg.state_source = args.controller_state_source.lower()
            controller = WaypointLQRController(waypoint_cfg, lqr_cfg, model=controller_model, environment=controller_environment)
        elif controller_type == "mpc":
            mpc_cfg = MPCControllerConfig.from_yaml(args.mpc_config_yaml)
            if args.controller_state_source is not None:
                mpc_cfg.state_source = args.controller_state_source.lower()
            controller = WaypointMPCController(waypoint_cfg, mpc_cfg, model=controller_model, environment=controller_environment)
        else:
            raise ValueError(f"Unknown controller type: {args.controller}")
        waypoint_positions = np.vstack([wp.position for wp in waypoint_cfg.waypoints])

    n_steps = int(np.floor(args.duration / args.dt)) + 1
    t = np.arange(n_steps, dtype=float) * args.dt
    fixed_tau = np.asarray(args.tau, dtype=float)

    truth, ekf_hist, tau_cmd_hist, tau_applied, sensor_series, sensor_labels_by_state = initialize_histories(n_steps, sensor_stack)
    current_ned = np.zeros((n_steps, 3), dtype=float)
    current_body = np.zeros((n_steps, 3), dtype=float)
    boundary_hits = np.zeros((n_steps, 2), dtype=bool)
    waypoint_index = np.full(n_steps, -1, dtype=int)
    mission_complete = np.zeros(n_steps, dtype=bool)
    desired_position = np.full((n_steps, 3), np.nan, dtype=float)
    desired_attitude = np.full((n_steps, 3), np.nan, dtype=float)
    position_error_ned = np.full((n_steps, 3), np.nan, dtype=float)
    attitude_error = np.full((n_steps, 3), np.nan, dtype=float)

    truth[0, :] = np.asarray(args.initial_state, dtype=float)
    ekf_hist[0, :] = ekf.x

    flat_sensor_rows = []

    for k in range(n_steps):
        readings = sensor_stack.step(float(t[k]), truth[k, :])
        update_sensor_series(sensor_series, k, readings)
        ekf.update_many(readings)
        ekf_hist[k, :] = ekf.x

        flat_row = {"t": float(t[k])}
        flat_row.update(readings_to_flat_dict(readings))
        flat_sensor_rows.append(flat_row)

        current_ned[k, :] = truth_environment.current_ned(float(t[k]), truth[k, 2])
        current_body[k, :] = truth_environment.current_body_linear(float(t[k]), truth[k, :])

        if controller is not None and controller.enabled:
            if controller.state_source == "truth":
                feedback_state = truth[k, :]
            else:
                feedback_state = ekf.x
            tau_cmd, dbg = controller.compute(float(t[k]), feedback_state, args.dt)
            waypoint_index[k] = dbg.target_index
            mission_complete[k] = dbg.mission_complete
            desired_position[k, :] = dbg.desired_position
            desired_attitude[k, :] = dbg.desired_attitude
            position_error_ned[k, :] = dbg.position_error_ned
            attitude_error[k, :] = dbg.attitude_error
        else:
            tau_cmd = fixed_tau

        tau_cmd_hist[k, :] = tau_cmd

        if k == n_steps - 1:
            tau_applied[k, :] = tau_applied[k - 1, :] if k > 0 else tau_cmd
            break

        result = truth_model.step(float(t[k]), truth[k, :], tau_cmd, args.dt, environment=truth_environment)
        truth[k + 1, :] = result.state_next
        tau_applied[k, :] = result.tau_applied
        boundary_hits[k, :] = [result.hit_surface, result.hit_bottom]
        ekf.predict(float(t[k]), tau_cmd, args.dt)

    # Log final posterior state after updates at final sample.
    ekf_hist[-1, :] = ekf.x

    prefix = args.prefix
    csv_path = output_dir / f"{prefix}_truth_sensors_ekf.csv"
    npz_path = output_dir / f"{prefix}_truth_sensors_ekf.npz"
    sensor_event_csv_path = output_dir / f"{prefix}_sensor_events.csv"
    config_snapshot_path = output_dir / f"{prefix}_config_snapshot.json"

    df = make_dataframe(
        t, truth, ekf_hist, tau_cmd_hist, tau_applied, current_ned, current_body, sensor_series,
        waypoint_index, desired_position, desired_attitude, position_error_ned, attitude_error, mission_complete,
    )
    df.to_csv(csv_path, index=False)
    pd.DataFrame(flat_sensor_rows).to_csv(sensor_event_csv_path, index=False)

    np.savez(
        npz_path,
        t=t,
        truth=truth,
        ekf=ekf_hist,
        tau_cmd=tau_cmd_hist,
        tau_applied=tau_applied,
        current_ned=current_ned,
        current_body=current_body,
        boundary_hits=boundary_hits,
        waypoint_index=waypoint_index,
        mission_complete=mission_complete,
        desired_position=desired_position,
        desired_attitude=desired_attitude,
        position_error_ned=position_error_ned,
        attitude_error=attitude_error,
        waypoint_positions=waypoint_positions if waypoint_positions is not None else np.zeros((0, 3)),
        state_names=np.asarray(STATE_NAMES),
        tau_names=np.asarray(TAU_NAMES),
        sensor_keys=np.asarray(list(sensor_series.keys())),
        sensor_values=np.vstack([sensor_series[k] for k in sensor_series]).T if sensor_series else np.zeros((n_steps, 0)),
    )

    snapshot_paths = {
        "vehicle_params": vehicle_yaml,
        "environment_config": environment_yaml,
        "sensor_config": sensor_yaml,
        "ekf_config": ekf_yaml,
    }
    if waypoints_yaml.exists():
        snapshot_paths["waypoints"] = waypoints_yaml
    if pid_yaml.exists():
        snapshot_paths["pid_config"] = pid_yaml
    if lqr_yaml.exists():
        snapshot_paths["lqr_config"] = lqr_yaml
    if mpc_yaml.exists():
        snapshot_paths["mpc_config"] = mpc_yaml
    if animation_yaml.exists():
        snapshot_paths["animation_config"] = animation_yaml
    save_config_snapshot(snapshot_paths, config_snapshot_path)

    show = not args.no_show
    plot_state_group(
        t, truth, ekf_hist, sensor_series, sensor_labels_by_state,
        [0, 1, 2], ["x", "y", "z/depth"], "position [m]",
        output_dir / f"{prefix}_position_truth_ekf_sensors.png", show=show,
    )
    plot_state_group(
        t, truth, ekf_hist, sensor_series, sensor_labels_by_state,
        [3, 4, 5], ["roll", "pitch", "yaw"], "attitude [deg]",
        output_dir / f"{prefix}_attitude_truth_ekf_sensors.png", angle_deg=True, show=show,
    )
    plot_state_group(
        t, truth, ekf_hist, sensor_series, sensor_labels_by_state,
        [6, 7, 8], ["u", "v", "w"], "body velocity [m/s]",
        output_dir / f"{prefix}_linear_velocity_truth_ekf_sensors.png", show=show,
    )
    plot_state_group(
        t, truth, ekf_hist, sensor_series, sensor_labels_by_state,
        [9, 10, 11], ["p", "q", "r"], "angular rate [deg/s]",
        output_dir / f"{prefix}_angular_rate_truth_ekf_sensors.png", angle_deg=True, show=show,
    )
    plot_environment_current(t, current_ned, output_dir / f"{prefix}_environment_current.png", show=show)
    plot_trajectory(truth, ekf_hist, waypoint_positions, output_dir / f"{prefix}_trajectory.png", show=show)
    plot_tau(t, tau_cmd_hist, tau_applied, output_dir / f"{prefix}_tau_cmd_applied.png", show=show)

    animation_output_path = None
    animation_cfg = None
    if args.animate or args.animation_save:
        animation_cfg = AnimationConfig.from_yaml(animation_yaml) if animation_yaml.exists() else AnimationConfig()
        animation_cfg.enabled = True
        if args.animation_state_source is not None:
            animation_cfg.state_source = args.animation_state_source.lower()
        if args.animation_speed is not None:
            animation_cfg.speed = float(args.animation_speed)
        if args.animation_stride is not None:
            animation_cfg.stride = int(args.animation_stride)
        if args.animation_save:
            animation_cfg.save_path = str(args.animation_save)
        animation_output_path = Path(animation_cfg.save_path) if animation_cfg.save_path else None
        if animation_output_path is not None and not animation_output_path.is_absolute():
            animation_output_path = output_dir / animation_output_path
        animate_trajectory(
            t=t,
            truth=truth,
            ekf=ekf_hist,
            waypoints=waypoint_positions,
            cfg=animation_cfg,
            surface_z_m=truth_environment.surface_z_m,
            bottom_z_m=truth_environment.bottom_z_m,
            show=(not args.no_show and not args.animation_save),
            save_path=animation_output_path,
        )

    outputs = {
        "csv": csv_path,
        "npz": npz_path,
        "sensor_events_csv": sensor_event_csv_path,
        "config_snapshot": config_snapshot_path,
        "position_plot": output_dir / f"{prefix}_position_truth_ekf_sensors.png",
        "attitude_plot": output_dir / f"{prefix}_attitude_truth_ekf_sensors.png",
        "linear_velocity_plot": output_dir / f"{prefix}_linear_velocity_truth_ekf_sensors.png",
        "angular_rate_plot": output_dir / f"{prefix}_angular_rate_truth_ekf_sensors.png",
        "environment_current_plot": output_dir / f"{prefix}_environment_current.png",
        "trajectory_plot": output_dir / f"{prefix}_trajectory.png",
        "tau_plot": output_dir / f"{prefix}_tau_cmd_applied.png",
    }
    if animation_output_path is not None:
        outputs["animation"] = animation_output_path

    print("Simulation completed.")
    print(f"Final truth state: {truth[-1, :]}")
    print(f"Final EKF state:   {ekf_hist[-1, :]}")
    print(f"Final error:       {truth[-1, :] - ekf_hist[-1, :]}")
    if controller is not None:
        print(f"Controller type: {controller_type}")
        print(f"Final waypoint index: {waypoint_index[-1]}")
        print(f"Mission complete: {bool(mission_complete[-1])}")
    print("Outputs:")
    for key, path in outputs.items():
        print(f"  {key}: {path}")
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run configured marine vehicle truth/sensor/EKF/waypoint-controller simulation")
    parser.add_argument("--vehicle-params-yaml", default=str(DEFAULT_VEHICLE_PARAMS))
    parser.add_argument("--environment-config-yaml", default=str(DEFAULT_ENVIRONMENT_CONFIG))
    parser.add_argument("--sensor-config-yaml", default=str(DEFAULT_SENSOR_CONFIG))
    parser.add_argument("--ekf-config-yaml", default=str(DEFAULT_EKF_CONFIG))
    parser.add_argument("--waypoints-yaml", default=str(DEFAULT_WAYPOINTS_CONFIG))
    parser.add_argument("--pid-config-yaml", default=str(DEFAULT_PID_CONFIG))
    parser.add_argument("--lqr-config-yaml", default=str(DEFAULT_LQR_CONFIG))
    parser.add_argument("--mpc-config-yaml", default=str(DEFAULT_MPC_CONFIG))
    parser.add_argument("--animation-config-yaml", default=str(DEFAULT_ANIMATION_CONFIG))
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--dt", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--truth-noise", action="store_true", help="Enable physical truth disturbances from environment_config.yaml")
    parser.add_argument("--truth-state-process-noise", action="store_true", help="Allow direct state process noise in truth model; off by default")
    parser.add_argument("--sensor-noise", action="store_true", help="Enable simulated sensor white noise, biases, and dropouts")
    parser.add_argument("--disable-waypoint-controller", action="store_true", help="Disable waypoint controller and use fixed --tau instead")
    parser.add_argument("--controller", choices=["pid", "lqr", "mpc"], default="pid", help="Waypoint controller type; default is pid")
    parser.add_argument("--controller-state-source", choices=["ekf", "truth"], default=None, help="Override controller feedback source")
    parser.add_argument("--ekf-knows-environment", action="store_true", help="Opt-in debug mode: let EKF prediction use the true environment/current")
    parser.add_argument("--controllers-know-environment", action="store_true", help="Opt-in debug mode: let LQR/MPC prediction use the true environment/current")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--animate", action="store_true", help="Show realtime 3D animation after the simulation finishes")
    parser.add_argument("--animation-save", default="", help="Save animation to a GIF or MP4 path. Relative paths are placed in --output-dir")
    parser.add_argument("--animation-speed", type=float, default=None, help="Animation playback speed multiplier; overrides config")
    parser.add_argument("--animation-stride", type=int, default=None, help="Animate every N simulation samples; overrides config")
    parser.add_argument("--animation-state-source", choices=["truth", "ekf"], default=None, help="Animate truth or EKF pose; overrides config")
    parser.add_argument("--output-dir", default="sim_outputs")
    parser.add_argument("--prefix", default="configured_ekf")
    parser.add_argument("--initial-state", nargs=12, type=float, default=[0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    parser.add_argument("--tau", nargs=6, type=float, default=[7, 9, 2, 0, 0, -1.9], help="Fixed tau when waypoint controller is disabled")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
