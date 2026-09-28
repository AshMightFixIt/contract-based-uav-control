# contract_uav_control — ROS 2 / PX4 bridge for DEXI

Runs the contract-based three-tier adaptive controller (PID → MPC → H-inf, with
CBF safety filter and Pacti horizon planner) as a **PX4 offboard** ROS 2 node.
Targets the **DEXI / PX4** platform over **micro-XRCE-DDS** (`px4_msgs`).

## How it fits together

```
PX4 estimator topics ─┐
  vehicle_local_pos   │   build_raw_sensors()      AdaptiveDroneController
  vehicle_attitude    ├──► raw_sensors dict ──────► .control_step()  (PID/MPC/H-inf + CBF)
  angular_velocity    │                                   │
  gps / battery       ┘                                   ▼
                                          VehicleAttitudeSetpoint (q_d + thrust_body)
                                          OffboardControlMode (attitude heartbeat) ──► PX4
```

- Both the controller and PX4 `VehicleLocalPosition` use the **NED** frame, so
  position/velocity pass through with no axis remap (negative z = altitude).
- The node commands **attitude + collective thrust**; PX4's onboard rate
  controller and mixer run underneath. The controller's outer position/attitude
  loops and the CBF filter stay in charge of the trajectory.

## Monitoring & tuning

The node publishes full per-cycle telemetry on **`/contract_uav/state`**
(`contract_uav_msgs/ContractState`): estimated state, setpoint, `position_error_norm`,
attitude error, commands, active controller / flight mode, CBF interventions, planner
margin, and per-component contract status. Every numeric field is plottable.

```bash
ros2 topic echo /contract_uav/state           # quick text view
ros2 run plotjuggler plotjuggler              # live graphs (subscribe to the topic)
ros2 bag record /contract_uav/state -o flight # record a run for offline analysis
```

See [`docs/TESTING_MANUAL.md`](../docs/TESTING_MANUAL.md) for the full
step-by-step testing & tuning guide written for the whole team.

## Prerequisites

- ROS 2 Humble
- [`px4_msgs`](https://github.com/PX4/px4_msgs) built in your workspace
- [`Micro-XRCE-DDS-Agent`](https://docs.px4.io/main/en/middleware/uxrce_dds.html)
- PX4 (SITL for testing, or DEXI flight controller for hardware)
- The core controller package, `contract_uav_core/` in this repo (next to this
  package). It is plain Python — numpy only.

## Build

```bash
# The repo root is also the colcon workspace; clone px4_msgs into it first
# (git clone https://github.com/PX4/px4_msgs.git; it is gitignored there).
cd /path/to/contract-based-uav-control
colcon build --packages-select px4_msgs contract_uav_msgs contract_uav_core contract_uav_control
source install/setup.bash
```

> The node imports the core controller from the `contract_uav_core` package,
> which colcon builds and installs next to this one (`exec_depend` in
> `package.xml`), so no environment variable is needed. Outside colcon,
> `pip install -e ./contract_uav_core` makes it importable too. Make sure
> `numpy` (and optionally `pacti`) are on the same Python the node runs with.

## Run (PX4 SITL first — always validate in sim before flight)

```bash
# Terminal 1 — DDS agent (UDP for SITL)
MicroXRCEAgent udp4 -p 8888

# Terminal 2 — PX4 SITL
make px4_sitl gz_x500

# Terminal 3 — the controller
ros2 launch contract_uav_control contract_control.launch.py
```

On DEXI hardware the agent runs over serial to the PX4 FMU
(`MicroXRCEAgent serial --dev /dev/ttyAMA0 -b 921600` or the DEXI-documented port).

## Parameters (`config/params.yaml`)

| param | default | meaning |
|-------|---------|---------|
| `control_frequency` | `50.0` | loop rate (Hz); controller `dt = 1/freq` |
| `use_horizon_planner` | `true` | enable Pacti horizon planner |
| `takeoff_altitude` | `5.0` | metres above start (NED z = -alt) |
| `auto_arm` | `false` | **keep false on hardware**; arm via QGroundControl |
| `waypoints` | `[0,0,-5, ...]` | flat NED list `[x,y,z, ...]` |

## Safety notes / current limitations

- **`auto_arm` defaults to false.** On hardware, arm and switch to OFFBOARD
  manually from QGroundControl with a safety pilot ready. Auto-arm is for SITL.
- IMU temperature/calibration are stubbed (PX4 doesn't expose them on these
  topics); contract inputs use nominal values. Wire real telemetry before relying
  on the sensor-degradation contracts.
- **Body rates are optional.** PX4 comments out `vehicle_angular_velocity` in the
  default `dds_topics.yaml`, so the node falls back to zero rates if that topic is
  absent (the IMU stays valid on attitude alone). For full rate feedback, uncomment
  that topic in PX4's `dds_topics.yaml` and rebuild PX4.
- No watchdog/failsafe is implemented in the node yet — rely on PX4 failsafes
  (RC loss, offboard-loss, geofence, battery) as the safety net.
- Controller gains were tuned in a simplified simulation. **Re-tune against PX4
  SITL** (and HITL) before any flight.

## Next steps

1. Build against `px4_msgs` and confirm topic names match your PX4 version
   (some builds expose `/fmu/out/vehicle_gps_position` vs `/fmu/out/sensor_gps`).
2. Fly the waypoint mission in `gz_x500` SITL; compare against the sim demos.
3. Add an offboard-loss/RC-override watchdog and a LAND fallback in the node.
4. HITL on the DEXI FMU, then tethered flight with a safety pilot.
