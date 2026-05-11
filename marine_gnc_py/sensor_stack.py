#!/usr/bin/env python3
"""Sensor simulation layer for the marine GNC simulation.

The sensor stack consumes the ground-truth propagated state and produces raw
simulated sensor measurements.  These are the *actual* sensor characteristics;
the EKF has its own separate assumed measurement-noise configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

STATE_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "u", "v", "w", "p", "q", "r"]
ANGLE_STATE_INDICES = {3, 4, 5, 9, 10, 11}


def _as_array(data: Any, default: list[float]) -> np.ndarray:
    return np.asarray(data if data is not None else default, dtype=float)


def _maybe_deg_to_rad(data: Dict[str, Any], key_rad: str, key_deg: str, default: list[float]) -> np.ndarray:
    if key_rad in data:
        return np.asarray(data[key_rad], dtype=float)
    if key_deg in data:
        return np.deg2rad(np.asarray(data[key_deg], dtype=float))
    return np.asarray(default, dtype=float)


@dataclass
class SensorConfig:
    name: str
    enabled: bool
    state_indices: np.ndarray
    measurement_names: List[str]
    std: np.ndarray
    bias_std: np.ndarray
    rate_hz: float
    dropout_prob: float = 0.0

    @staticmethod
    def from_dict(name: str, data: Dict[str, Any]) -> "SensorConfig":
        state_indices = np.asarray(data["state_indices"], dtype=int)
        measurement_names = list(data.get("measurement_names", [STATE_NAMES[i] for i in state_indices]))
        std = _maybe_deg_to_rad(data, "std", "std_deg", [0.0] * len(state_indices))
        bias_std = _maybe_deg_to_rad(data, "bias_std", "bias_std_deg", [0.0] * len(state_indices))
        if std.shape != state_indices.shape:
            raise ValueError(f"sensor {name}: std length must match state_indices")
        if bias_std.shape != state_indices.shape:
            raise ValueError(f"sensor {name}: bias_std length must match state_indices")
        return SensorConfig(
            name=name,
            enabled=bool(data.get("enabled", True)),
            state_indices=state_indices,
            measurement_names=measurement_names,
            std=std,
            bias_std=bias_std,
            rate_hz=float(data.get("rate_hz", 1.0)),
            dropout_prob=float(data.get("dropout_prob", 0.0)),
        )

    @property
    def period_s(self) -> float:
        if self.rate_hz <= 0.0:
            return np.inf
        return 1.0 / self.rate_hz


@dataclass
class SensorReading:
    name: str
    t: float
    z: np.ndarray
    z_true: np.ndarray
    noise: np.ndarray
    bias: np.ndarray
    R_true: np.ndarray
    state_indices: np.ndarray
    measurement_names: List[str]
    available: bool


@dataclass
class SensorStackConfig:
    seed_offset: int = 1000
    sensors: Dict[str, SensorConfig] = field(default_factory=dict)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "SensorStackConfig":
        sensor_data = data.get("sensors", {})
        sensors = {name: SensorConfig.from_dict(name, cfg) for name, cfg in sensor_data.items()}
        return SensorStackConfig(seed_offset=int(data.get("seed_offset", 1000)), sensors=sensors)

    @staticmethod
    def from_yaml(path: str | Path) -> "SensorStackConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return SensorStackConfig.from_dict(data)


class SensorStack:
    def __init__(
        self,
        config: SensorStackConfig,
        rng: Optional[np.random.Generator] = None,
        noise_enabled: bool = True,
    ):
        self.config = config
        self.rng = rng if rng is not None else np.random.default_rng()
        self.noise_enabled = bool(noise_enabled)
        self._next_sample_time = {name: 0.0 for name in config.sensors}
        self._biases = {}
        for name, cfg in config.sensors.items():
            if self.noise_enabled:
                self._biases[name] = self.rng.normal(0.0, cfg.bias_std, size=cfg.state_indices.size)
            else:
                self._biases[name] = np.zeros(cfg.state_indices.size, dtype=float)

    @staticmethod
    def from_yaml(path: str | Path, seed: int = 0, noise_enabled: bool = True) -> "SensorStack":
        cfg = SensorStackConfig.from_yaml(path)
        return SensorStack(cfg, rng=np.random.default_rng(seed + cfg.seed_offset), noise_enabled=noise_enabled)

    @property
    def sensors(self) -> Dict[str, SensorConfig]:
        return self.config.sensors

    def step(self, t: float, truth_state: np.ndarray) -> List[SensorReading]:
        truth_state = np.asarray(truth_state, dtype=float)
        readings: List[SensorReading] = []
        eps = 1e-10
        for name, cfg in self.config.sensors.items():
            if not cfg.enabled:
                continue
            if t + eps < self._next_sample_time[name]:
                continue
            self._next_sample_time[name] += cfg.period_s
            available = True
            if self.noise_enabled and cfg.dropout_prob > 0.0:
                available = bool(self.rng.random() >= cfg.dropout_prob)
            if not available:
                readings.append(
                    SensorReading(
                        name=name,
                        t=float(t),
                        z=np.full(cfg.state_indices.size, np.nan),
                        z_true=truth_state[cfg.state_indices].copy(),
                        noise=np.full(cfg.state_indices.size, np.nan),
                        bias=self._biases[name].copy(),
                        R_true=np.diag(cfg.std ** 2),
                        state_indices=cfg.state_indices.copy(),
                        measurement_names=list(cfg.measurement_names),
                        available=False,
                    )
                )
                continue
            noise = self.rng.normal(0.0, cfg.std, size=cfg.state_indices.size) if self.noise_enabled else np.zeros(cfg.state_indices.size)
            z_true = truth_state[cfg.state_indices].copy()
            z = z_true + self._biases[name] + noise
            readings.append(
                SensorReading(
                    name=name,
                    t=float(t),
                    z=z,
                    z_true=z_true,
                    noise=noise,
                    bias=self._biases[name].copy(),
                    R_true=np.diag(cfg.std ** 2),
                    state_indices=cfg.state_indices.copy(),
                    measurement_names=list(cfg.measurement_names),
                    available=True,
                )
            )
        return readings


def readings_to_flat_dict(readings: List[SensorReading]) -> Dict[str, float]:
    row: Dict[str, float] = {}
    for r in readings:
        for j, mname in enumerate(r.measurement_names):
            row[f"sensor_{r.name}_{mname}"] = float(r.z[j]) if r.available else np.nan
            row[f"sensor_{r.name}_{mname}_true"] = float(r.z_true[j])
            row[f"sensor_{r.name}_{mname}_bias"] = float(r.bias[j])
            row[f"sensor_{r.name}_{mname}_noise"] = float(r.noise[j]) if r.available else np.nan
    return row
