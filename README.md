# Marine Guidance, Navigation, and Control Simulator

This is a self-contained Python simulation for a 12-state marine vehicle model:

```text
state = [x, y, z, roll, pitch, yaw, u, v, w, p, q, r]
```

`z` is positive downward. Pose is inertial/NED-style and velocities are body-frame.

## Configuration files

All user-tunable YAML files live in `config/`:

```text
config/vehicle_params.yaml      # truth and nominal vehicle parameters
config/environment_config.yaml  # water depth, current profile, truth tau disturbances
config/sensor_config.yaml       # true simulated sensor noise/bias/rate/dropout
config/ekf_config.yaml          # EKF P0, Q, assumed R, sensor usage, Jacobian mode
config/waypoints.yaml           # waypoint list, waypoint profile, and advancement rules
config/pid_config.yaml          # simple waypoint PID gains and limits
config/lqr_config.yaml          # waypoint LQR weights, limits, trim, and linearization settings
config/mpc_config.yaml          # waypoint MPC horizon, weights, limits, and optimizer settings
```


## Controller modules

Waypoint logic is separated from control-law logic:

```text
waypoints.py          # waypoint YAML parsing, desired attitudes, target advancement
pid_controller.py     # PID gains and PID wrench command
lqr_controller.py     # LQR weights/linearization and LQR wrench command
mpc_controller.py     # nonlinear receding-horizon MPC wrench command
```

This keeps the waypoint mission file independent from PID/LQR/MPC-specific tuning.
The waypoint navigator also has relaxed/pass-through advancement rules so a
regulator-style controller such as LQR does not park just outside an intermediate
waypoint forever.

## Waypoint follower

The waypoint follower can use PID, local LQR, or simple nonlinear MPC. PID is the default.

The waypoint follower can run in two profiles:

```yaml
profile: point_heading
```

The controller commands yaw toward the active waypoint while keeping roll and pitch near zero.

```yaml
profile: arbitrary_orientation
```

The controller intercepts waypoints with the waypoint's specified attitude if one is provided. If a waypoint does not specify attitude, it holds the attitude it had when that waypoint became active.

Waypoints use:

```yaml
waypoints:
  - name: wp_1
    position: [8.0, 0.0, 10.0]

  - name: wp_2
    position: [8.0, 5.0, 10.0]
    attitude_deg: [0.0, 0.0, 90.0]
```

## Install

```bash
python3 -m pip install -r requirements.txt
```

## Smoke test

```bash
./run_test.sh
./run_lqr_test.sh
./run_mpc_test.sh
```

## Normal run

```bash
python3 run.py \
  --truth-noise \
  --sensor-noise \
  --no-show
```

## Controller selection

PID is the default:

```bash
python3 run.py \
  --controller pid \
  --truth-noise \
  --sensor-noise \
  --no-show
```

Run the local waypoint LQR controller. The default LQR config includes a small positive `Z` trim to cancel the example vehicle's slight positive buoyancy:

```bash
python3 run.py \
  --controller lqr \
  --truth-noise \
  --sensor-noise \
  --no-show
```

Run the nonlinear waypoint MPC controller. The default MPC config uses a short receding horizon and SciPy L-BFGS-B optimization, so it is intended for simulation/testing rather than hard realtime control:

```bash
python3 run.py \
  --controller mpc \
  --truth-noise \
  --sensor-noise \
  --no-show
```

## Useful options

Use truth instead of EKF state for controller feedback:

```bash
python3 run.py \
  --truth-noise \
  --sensor-noise \
  --controller-state-source truth \
  --no-show
```

Disable waypoint controller and use a fixed wrench:

```bash
python3 run.py \
  --disable-waypoint-controller \
  --tau 7 9 2 0 0 -1.9 \
  --truth-noise \
  --sensor-noise \
  --no-show
```

Override YAML paths:

```bash
python3 run.py \
  --vehicle-params-yaml config/vehicle_params.yaml \
  --environment-config-yaml config/environment_config.yaml \
  --sensor-config-yaml config/sensor_config.yaml \
  --ekf-config-yaml config/ekf_config.yaml \
  --waypoints-yaml config/waypoints.yaml \
  --pid-config-yaml config/pid_config.yaml \
  --lqr-config-yaml config/lqr_config.yaml \
  --mpc-config-yaml config/mpc_config.yaml \
  --no-show
```


## Environment knowledge policy

By default, only the truth plant receives the true environment/current model from `config/environment_config.yaml`. The EKF and model-based controllers are intentionally environment-blind so currents act like an unknown disturbance, which is closer to a real vehicle unless you explicitly provide a current estimate.

For debug/perfect-model comparisons only, opt in with:

```bash
python3 run.py --ekf-knows-environment --controllers-know-environment --truth-noise --sensor-noise --no-show
```

## Outputs

The runner writes CSV/NPZ logs and plots to the selected output directory. Truth traces are plotted in green. Sensor outputs are plotted in orange and overlaid on truth-vs-EKF plots wherever a state is directly measured.


## Realtime / recorded 3D animation

The project includes `animation.py`, which animates the selected vehicle pose as an ovoid/ellipsoid with body-fixed coordinate axes:

- red: body x-axis (`xb`)
- blue: body y-axis (`yb`)
- purple: body z-axis (`zb`)
- green trail: ground-truth trajectory
- default EKF trail: EKF estimated trajectory
- waypoint markers/path are shown when the waypoint controller is enabled

Animation settings live in:

```text
config/animation_config.yaml
```

Show a realtime animation window after the simulation finishes:

```bash
python3 run.py   --truth-noise   --sensor-noise   --animate
```

Save a GIF instead of opening a window:

```bash
python3 run.py   --truth-noise   --sensor-noise   --no-show   --animation-save animation.gif
```

Animate the EKF estimate instead of truth:

```bash
python3 run.py   --truth-noise   --sensor-noise   --animate   --animation-state-source ekf
```

Control playback speed and frame stride:

```bash
python3 run.py   --truth-noise   --sensor-noise   --animate   --animation-speed 2.0   --animation-stride 2
```

A short headless animation save test is available:

```bash
./run_animation_test.sh
```
