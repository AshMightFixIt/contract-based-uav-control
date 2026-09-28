# Contract-Based Adaptive UAV Control System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Pacti](https://img.shields.io/badge/Pacti-Enabled-green.svg)](https://github.com/pacti-org/pacti)

A hierarchical contract-based architecture integrating Assume-Guarantee (A/G) contracts with Control Barrier Function (CBF) safety filtering for provably safe multi-mode UAV control. Features **horizon-based predictive planning** using Pacti for formal contract composition and a **three-tier controller cascade** (PID → MPC → H-infinity) with live visualization.

**Research Project** | University of Michigan
**Courses:** AE552 (Aerospace Information Systems) & ECE599 (Formal Methods)
**Author:** Aswatth Sunil | **Advisor:** Prof. Iñigo Incer (EECS)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Innovation](#key-innovation)
- [Horizon-Based Planning](#horizon-based-planning)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Controller Modes & Contracts](#controller-modes--contracts)
- [Experimental Results](#experimental-results)
- [Development Roadmap](#development-roadmap)
- [Technical Details](#technical-details)
- [References](#references)

---

## 🎯 Overview

This system demonstrates formal methods applied to autonomous UAV control through hierarchical contract composition. Instead of ad-hoc threshold-based switching, the system uses mathematically provable Assume-Guarantee contracts to ensure safe controller transitions, with **predictive horizon-based planning** using Pacti.

### Core Components

- **Horizon-Based Pacti Planner** - Predictive N-step contract cascade with safety margin optimization
- **Hierarchical Contract Framework** - Compositional verification through A/G contracts
- **Flight Mode Supervisor** - Mission-level mode management (TRACK, HOVER, LAND, EMERGENCY)
- **Three-Tier Controller Cascade** - PID (wind ≤ 3 m/s) → MPC (wind ≤ 8 m/s) → H-infinity (wind ≤ 25 m/s)
- **Contract-Aware EKF** - Adaptive sensor fusion based on contract satisfaction
- **CBF Safety Filter** - Minimally-invasive safety enforcement with 4 barrier functions
- **Sensor Degradation Testing** - 10 fault types with time-scheduled injection validating graceful degradation
- **Live Visualizations** - Pygame 3D flight visualizer and matplotlib live 2×2 racing speed panels

**Implementation:** 3,500+ lines of production Python code

### System Performance

- **Real-time monitoring:** 50 Hz (20 ms period)
- **Computation time:** 2.5 ms average, 4.1 ms peak  
- **CPU utilization:** 12.5%
- **Contract checks:** 750+ per 15-second flight
- **Switching:** Formal violation-based (zero false positives)

---

## 💡 Key Innovation

### Traditional Approach: Reactive Heuristic Thresholds
```python
if wind_speed > 3.0:  # Ad-hoc threshold
    switch_to_robust_controller()  # Reactive - already in trouble!
```

**Problems:** No safety guarantees, reactive (not predictive), prone to chattering

### Our Approach: Predictive Horizon-Based Contract Planning
```python
# Cascade contracts over N-step planning horizon using Pacti
for step in range(horizon):
    step_contract = controller.compose(dynamics)
    safety_margin = safety_limit - tracking_error_bound

    if safety_margin < threshold:
        replan()  # Predictive - see trouble coming!

# Pre-flight verification of entire pipeline
pipeline = sensor >> ekf >> controller >> actuator
if not pipeline.satisfies(mission):
    abort_mission()  # Provably unsafe
```

**Benefits:** ✅ Formal verification ✅ Predictive planning ✅ Safety margins ✅ Automatic re-planning

---

## 🔮 Horizon-Based Planning

The system uses [Pacti](https://github.com/pacti-org/pacti) for formal contract composition over a planning horizon.

### Architecture
```
Sensors -> EKF -> [HORIZON PLANNER] -> Supervisor -> [PID|H-inf] -> CBF -> Actuators
                        │
                 Pacti Contract Cascade
                        │
                 Safety Margin Monitor
                        │
              Replan if margin < threshold
```

### How It Works

1. **Contract Cascade**: Compose contracts over N steps (default: 10)
2. **Safety Optimization**: Use Pacti's `get_variable_bounds()` to find achievable tracking error
3. **Margin Monitoring**: `margin = safety_limit - tracking_error_bound`
4. **Automatic Re-planning** when margin falls below threshold:
   - Try shorter horizon (10 → 3 steps)
   - Switch controller (PID → H-inf)
   - Enter emergency mode if all strategies fail

### Example Output
```
Time  | Phase    | Wind | Controller | Horizon | Margin | Status
------|----------|------|------------|---------|--------|--------
0.02  | NOMINAL  | 0.5  | PID        |      10 |   4.65 | feasible
3.02  | WIND_UP  | 4.0  | PID        |      10 |   4.31 | feasible
8.02  | EXTREME  | 10.0 | Hinf       |      10 |   2.00 | feasible
10.52 | RECOVERY | 2.0  | Hinf       |      10 |   0.98 | marginal
```

### Pacti Contract Library

Seven formal contracts defined:
- **GPS/IMU**: Sensor measurement quality
- **EKF**: State estimation accuracy
- **PID/H-inf**: Controller tracking performance
- **Actuator**: Thrust/torque delivery
- **Dynamics**: State evolution bounds

---

## 📦 Installation

```bash
git clone https://github.com/AshMightFixIt/contract-based-uav-control.git
cd contract-based-uav-control
pip install -e ./contract_uav_core    # the core package (numpy, PyYAML)
pip install -r requirements.txt       # matplotlib and pygame for the demos and benchmarks
```

**Requirements:** Python 3.8+, NumPy, Matplotlib

---

## 🚀 Quick Start

### Run Complete Simulation
```bash
python simulation_demo.py
```

**Demonstrates:**
1. Pre-flight contract verification
2. 15-second flight with wind disturbance
3. Contract-based switching (PID → H-infinity)
4. Runtime safety monitoring
5. Results visualization

### Run Multi-Waypoint Mission
```bash
python demo_waypoint_mission.py
```

**Demonstrates:**
1. 4-waypoint square mission with altitude variation
2. Wind disturbance phases (nominal → moderate → extreme → recovery)
3. Contract-based controller switching during flight
4. Waypoint detection with confirmation counter

### Run Sensor Degradation Test
```bash
python demo_sensor_degradation.py
```

**Demonstrates:**
1. 7-phase degradation: nominal → GPS loss → recovery → IMU degradation → combined failure → intermittent GPS → full recovery
2. EKF fusion mode transitions (full → IMU-only → dead reckoning)
3. Contract-driven controller switching and flight mode changes
4. Mission completion despite sensor faults (4/4 waypoints)
5. 8-panel visualization with sensor health, fusion modes, and covariance

### Run Live Racing Visualization
```bash
python demo_racing_live.py
```

**Demonstrates:**
1. All 4 controller configurations (Adaptive, PID, MPC, H-inf) racing simultaneously
2. Live 2×2 speed-heatmap panels — plasma colormap trail (dark = slow → bright = fast)
3. Per-drone speedometer bar gauge updating in real time
4. Waypoint progress and elapsed time per panel
5. Shared colorbar for cross-drone speed comparison

### Run Racing Benchmark
```bash
python benchmark_racing.py
```

**Produces `benchmark_racing.png` with 8 panels:**
1. 3D trajectory comparison
2. XY racing trajectory with numbered waypoints
3. Altitude over time
4. Speed over time
5. Tracking error over time
6. Per-waypoint completion time (grouped bars)
7. Performance scores — all metrics on a unified 0–100 scale
8. Performance radar — normalized polygon chart (larger = better)

### Run Horizon Planner Demo
```bash
python demo_horizon_planner.py
```

**Demonstrates:**
1. Pacti contract cascade over N-step horizon
2. Safety margin monitoring and optimization
3. Automatic re-planning when margins are tight
4. Horizon reduction under extreme conditions
5. Predictive controller switching

### Test Components
```bash
python -m contract_uav_core.contracts.monitor            # Contract framework
python -m contract_uav_core.estimation.ekf               # Extended Kalman Filter
python -m contract_uav_core.control.switcher             # Controllers
python -m contract_uav_core.safety.cbf                   # CBF safety filter
python -m contract_uav_core.planning.integrated_planner  # Horizon planner
python -m contract_uav_core.core                         # Complete system
```

---

## 🏗️ System Architecture

```
┌───────────────────────────────────────────────────────┐
│        PRE-FLIGHT CONTRACT VERIFICATION               │
│   Hierarchical Composition: G₁ ⇒ A₂ ⇒ ... ⇒ Safety  │
└───────────────────────────────────────────────────────┘
                        ↓
┌───────────────────────────────────────────────────────┐
│              RUNTIME SUPERVISOR                        │
│      Contract Monitoring (50 Hz) + CBF Safety        │
├───────────────────────────────────────────────────────┤
│  Sensors → EKF → Controller → CBF → Actuators        │
│  [Contract] [Contract] [Contract] [Contract]          │
└───────────────────────────────────────────────────────┘
```

**Contract Composition Example:**
- Sensor guarantees position accuracy < 3m
- Estimator assumes position accuracy < 3m → Guarantees state error < 2m
- Controller assumes state error < 2m → Guarantees tracking < 1.5m

If G₁ ⇒ A₂, the pipeline is formally verified!

---

## 📁 Project Structure

```
contract-based-uav-control/
│
├── contract_uav_core/                  # Core package: pip install -e ./contract_uav_core (also ament_python)
│   ├── setup.py, setup.cfg, package.xml
│   └── contract_uav_core/
│       ├── core.py                     # Integration: AdaptiveDroneController
│       ├── conditions.py               # System-condition estimation
│       ├── switching_policy.py         # Contract-based controller recommendation
│       ├── telemetry.py                # Monitor updates and flight log
│       ├── interfaces.py               # AccelCommand and other shared types (NED frame)
│       ├── frames.py                   # Acceleration -> attitude/thrust/torque mappings
│       ├── config/
│       │   ├── airframe.py             # Airframe file schema and loader
│       │   └── airframes/              # x500_sitl.yaml, dexi.yaml (PX4 v1.16.2 values)
│       ├── contracts/
│       │   ├── spec.py                 # A/G contract types (SimpleContract, LinearConstraint)
│       │   ├── library.py              # Component contract definitions
│       │   ├── preflight.py            # Pipeline composition and pre-flight check
│       │   └── monitor.py              # HierarchicalContractMonitor (runtime)
│       ├── estimation/
│       │   └── ekf.py                  # Contract-aware EKF
│       ├── control/
│       │   ├── base.py                 # Controller base class
│       │   ├── pid.py, mpc.py, hinf.py # PID, MPC & H-infinity
│       │   ├── switcher.py             # Controller switching
│       │   └── supervisor.py           # Mission mode management
│       ├── planning/                   # Horizon-based planning
│       │   ├── pacti_contracts.py      # Pacti contract library
│       │   ├── horizon_planner.py      # N-step contract cascade
│       │   └── integrated_planner.py   # Control system integration
│       ├── safety/
│       │   ├── cbf.py                  # CBF filter
│       │   └── runtime_monitor.py      # Runtime condition monitors
│       ├── sim/
│       │   ├── sensor_faults.py        # Fault injection module
│       │   ├── battery_model.py        # Battery model
│       │   └── quadrotor.py            # Reference plant (not used by the scripts yet)
│       └── viz/
│           └── live_visualizer.py      # Pygame 3D live visualizer
│
├── contract_uav_control/               # ROS 2 / PX4 offboard bridge (ament_python)
├── contract_uav_msgs/                  # ROS 2 telemetry message (ament_cmake)
├── docs/                               # ROS 2 testing manual and hardware procedure
├── tests/                              # make test: smoke, golden, unit and ROS-node tests
├── tools/contracts_offline/            # Offline Pacti contract reconciliation
│
├── simulation_demo.py                  # Main demonstration
├── demo_waypoint_mission.py           # Three-tier wind demo (PID→MPC→H-inf)
├── demo_sensor_degradation.py         # Sensor degradation testing demo
├── demo_horizon_planner.py            # Horizon planner demonstration
├── demo_racing_live.py                # Live 2x2 racing speed panels (NEW)
├── benchmark_racing.py                # Racing course benchmark (4 controllers)
├── benchmark_controllers.py           # Controller performance benchmark
├── benchmark_sensor_degradation.py    # Sensor degradation benchmark
├── requirements.txt                    # Dependencies
├── Makefile                            # make test
├── README.md                           # This file
└── DEVELOPMENT.md                      # Testing & development guide
```

**Time and interfaces.** `AdaptiveDroneController.control_step(raw_sensors, t=None)` has two
time modes: without `t` (all the scripts) it steps its own clock by the nominal `dt`, exactly as
before; with `t` in seconds (the ROS node passes the PX4 timestamp) the EKF predicts with the
measured step, clamped to 0.5–2× `dt`, and the telemetry's `timing` entry counts overruns.
`interfaces.py` (the NED frame convention, `AccelCommand`), `frames.py` (the tiers'
acceleration-to-attitude mapping, unchanged) and each tier's `compute_accel_command` are the
interfaces the next refactors build on. The reference plant `sim/quadrotor.py` and the airframe
files `config/airframes/` are not used by the scripts yet.

---

## 🎮 Controller Modes & Contracts

The system uses three feedback controllers in a graduated cascade, managed by a Flight Mode Supervisor that selects the appropriate tier based on contract satisfaction.

### Controllers

#### 1. Nominal PID Controller
**Use:** Efficient control in calm conditions (wind ≤ 3 m/s)

**Contract:**
- Assumptions: Wind < 3 m/s, GPS available, Position error < 3m, Disturbance < 3
- Guarantees: Position error < 1.5m, Velocity error < 1.5 m/s

#### 2. Model Predictive Controller (MPC)
**Use:** Optimal control in moderate wind (wind ≤ 8 m/s)

**Contract:**
- Assumptions: Wind < 8 m/s, GPS available, Computation time < 50 ms
- Guarantees: Tracking error < 2m (tighter than PID due to horizon optimization)
- Implementation: Condensed QP with N=15 horizon, ~0.26 ms solve time

#### 3. Wind-Robust H-infinity Controller
**Use:** Robust stabilization under strong disturbances (wind ≤ 25 m/s)

**Contract:**
- Assumptions: Wind < 25 m/s, GPS available
- Guarantees: Tracking error < 3m, integral wind rejection via anti-windup

**Key:** Graduated degradation — PID handles calm flight efficiently; MPC optimizes moderate conditions; H-inf is always available as the safety net.

### Flight Mode Supervisor

The supervisor manages mission-level modes (not control laws):

| Mode | Setpoint Strategy | Controller Selection |
|------|-------------------|---------------------|
| **TRACK** | Follow waypoint sequence | PID (nominal) or H-inf (disturbance) |
| **HOVER** | Hold fixed position | PID (nominal) or H-inf (GPS-denied) |
| **LAND** | Descending profile at 0.5 m/s | Always H-inf |
| **EMERGENCY** | Hold position, safe altitude | Always H-inf |

**Automatic Transitions:**
- Wind > 8 m/s → EMERGENCY
- GPS lost during TRACK → HOVER
- All waypoints reached → HOVER
- EMERGENCY recovery (wind < 5, rates < 0.5) → HOVER

---

## 📊 Experimental Results

### Demonstration Scenario (15 seconds)

**Phase 1 (0-5s):** Nominal flight, PID active, wind 0.5 m/s  
**Phase 2 (5-10s):** Wind disturbance 5.4 m/s → Switch to H-infinity  
**Phase 3 (10-15s):** Wind subsides → System stabilizes

### Performance Metrics

**Contract Monitoring:**
- Total checks: 750+ (50 Hz × 15s)
- Violations detected: 1 (velocity error at t=2.02s)
- False positives: 0
- Switching latency: < 100ms

**Safety Validation:**
- Altitude: [2m, 50m] ✓
- Velocity: Peak 6.2 m/s (limit 15 m/s) ✓
- Tilt: Max 12° (limit 30°) ✓
- CBF interventions: 0

**Computation:**
- Frequency: 50 Hz
- Average: 2.5 ms/cycle
- Peak: 4.1 ms/cycle
- CPU: 12.5%

---

## 🗓️ Development Roadmap

### Current Status (February 2026)

**✅ Completed:**
- Hierarchical contract framework with theory-grounded constants
- Three-tier controller cascade: PID → MPC → H-infinity with formal contracts
  - MPC: condensed QP, N=15 horizon, ~0.26 ms solve time, warm-start
- Flight Mode Supervisor (TRACK, HOVER, LAND, EMERGENCY)
- Contract-aware EKF with 3 fusion modes
- CBF safety filter with 4 barrier functions
- Horizon-based Pacti planner (N-step cascade, safety margins, re-planning)
- Multi-waypoint mission demo with three-tier wind switching
- Sensor degradation testing
  - 10 fault types: GPS satellite loss, HDOP increase, position drift, complete loss, intermittent; IMU noise, temperature drift, bias accumulation, calibration loss, spikes
  - 7-phase degradation schedule validating contract-driven graceful degradation
  - Mission completion (4/4 waypoints) despite sensor faults
- Pygame 3D live flight visualizer (orbital camera, controller-coloured trail)
- **Live 2×2 racing speed panels** (NEW) — plasma heatmap trail + speedometer gauge
- **Racing benchmark** (NEW) — 10-waypoint figure-8 course, 4 controller comparison
  - 8-panel plot: 3D/XY trajectory, altitude, speed, tracking error, per-waypoint times, performance scores (unified 0–100 scale), performance radar
- Runtime monitoring at 50 Hz
- Bidirectional controller switching (contract-violation triggered)
- Python simulation (3,500+ lines)

### Upcoming

- SITL/Gazebo integration
- Pacti contract refinement
- Multi-agent coordination
- C++ implementation / ROS2 integration

---

## 🔬 Technical Details

### Assume-Guarantee Contracts

Contract C = (A, G):
- **A (Assumptions):** Required conditions
- **G (Guarantees):** Promised behavior when A holds

**Composition:** C₁ and C₂ compose if G₁ ⇒ A₂

### Control Barrier Functions

Safe set: C = {x : h(x) ≥ 0}  
Safety constraint: ḣ(x,u) ≥ -α(h(x))

**Four barriers:**
```python
h₁(x) = z - 2              # Altitude lower (z > 2m)
h₂(x) = 50 - z             # Altitude upper (z < 50m)
h₃(x) = 225 - ||v||²       # Velocity (||v|| < 15 m/s)
h₄(x) = 0.5 - ||θ||²       # Tilt (||θ|| < 30°)
```

**Safety filter:**
```
u* = argmin ||u - u_d||²
s.t. ḣᵢ(x,u) ≥ -αᵢhᵢ(x) for all i
```

### Contract-Aware EKF

**Three fusion modes:**
1. Full GPS: Position error < 0.5m
2. IMU-dominated: Position drift < 0.2 m/s
3. Dead-reckoning: Best-effort with warning

---

## 📚 References

1. Benveniste, A., et al. "Contracts for System Design," *FnT EDA*, 2018
2. Ames, A., et al. "Control Barrier Functions," *ECC*, 2019
3. Nuzzo, P., et al. "Contract-Based Aircraft Design," *IEEE Access*, 2014
4. Incer, I., et al. "Pacti: Assume-Guarantee Reasoning," arXiv:2303.17751, 2023
5. Beard, R. & McLain, T. *Small Unmanned Aircraft*, Princeton, 2012

**Tools:** [PX4](https://px4.io/) | [ROS2](https://docs.ros.org/) | [Gazebo](https://gazebosim.org/) | [Pacti](https://github.com/pacti-org/pacti)

---

## 📄 License

MIT License - See LICENSE file

---

## 📧 Contact

**Aswatth Sunil**  
University of Michigan | EECS  
aswatth@umich.edu  
[@AshMightFixIt](https://github.com/AshMightFixIt)

**Advisor:** Prof. Iñigo Incer

---

**Status:** Active Development | **Updated:** February 2026 | **Version:** 1.3.0
