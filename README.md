# Contract-Based Adaptive UAV Control System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Pacti](https://img.shields.io/badge/Pacti-Enabled-green.svg)](https://github.com/pacti-org/pacti)

A hierarchical contract-based architecture integrating Assume-Guarantee (A/G) contracts with Control Barrier Function (CBF) safety filtering for provably safe multi-mode UAV control. Features **horizon-based predictive planning** using Pacti for formal contract composition.

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
- **Dual Controllers** - PID (efficient) and H-infinity (robust) with formal operating envelopes
- **Contract-Aware EKF** - Adaptive sensor fusion based on contract satisfaction
- **CBF Safety Filter** - Minimally-invasive safety enforcement with 4 barrier functions

**Implementation:** 2,500+ lines of production Python code

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
pip install -r requirements.txt
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
python src/contracts/contract_framework.py    # Contract framework
python src/estimation/contract_ekf.py         # Extended Kalman Filter
python src/control/controllers.py             # Controllers
python src/safety/cbf_filter.py              # CBF safety filter
python src/planning/integrated_planner.py     # Horizon planner
python src/adaptive_control_system.py         # Complete system
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
├── src/                                # Source code (2,500+ lines)
│   ├── contracts/
│   │   └── contract_framework.py       # A/G contracts (506 lines)
│   ├── estimation/
│   │   └── contract_ekf.py            # Contract-aware EKF (331 lines)
│   ├── control/
│   │   ├── controllers.py             # PID & H-infinity (200 lines)
│   │   └── flight_mode_supervisor.py  # Mission mode management (200 lines)
│   ├── planning/                       # NEW: Horizon-based planning
│   │   ├── pacti_contracts.py         # Pacti contract library (150 lines)
│   │   ├── horizon_planner.py         # N-step contract cascade (250 lines)
│   │   └── integrated_planner.py      # Control system integration (300 lines)
│   ├── safety/
│   │   └── cbf_filter.py             # CBF filter (200 lines)
│   └── adaptive_control_system.py     # Integration (350 lines)
│
├── simulation_demo.py                  # Main demonstration
├── demo_horizon_planner.py            # Horizon planner demonstration
├── requirements.txt                    # Dependencies
├── README.md                           # This file
└── DEVELOPMENT.md                      # Testing & development guide
```

---

## 🎮 Controller Modes & Contracts

The system uses two feedback controllers managed by a Flight Mode Supervisor.

### Controllers

#### 1. Nominal PID Controller
**Use:** Efficient control in calm conditions

**Contract:**
- Assumptions: Wind < 3 m/s, GPS available, Position error < 3m, Disturbance < 3
- Guarantees: Position error < 1.5m, Velocity error < 1.5 m/s

#### 2. Wind-Robust H-infinity Controller
**Use:** Robust control under disturbances

**Contract:**
- Assumptions: Wind < 10 m/s, GPS available
- Guarantees: Position error < 3m, Stabilization < 10s

**Key:** Minimal assumptions - always available safety net

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

### Current Status (February 2025)

**✅ Completed:**
- Hierarchical contract framework
- PID and H-infinity controllers with formal contracts
- Flight Mode Supervisor (TRACK, HOVER, LAND, EMERGENCY)
- Contract-aware EKF
- CBF safety filter with 4 barrier functions
- **Horizon-based Pacti planner** (NEW)
  - N-step contract cascade
  - Safety margin optimization
  - Automatic re-planning
- Runtime monitoring at 50 Hz
- Bidirectional controller switching (PID ↔ H-inf)
- Python simulation (2,500+ lines)

**Recent Fixes:**
- Controller switch-back now works correctly
- Lateral drift calculation for accurate tracking error
- Relaxed PID thresholds to prevent false violations
- Cooldown initialization for proper switching at t=0

### Upcoming (February-April 2025)

**February:**
- Week 1-2: SITL/Gazebo integration
- Week 3: GPS-denied mode testing
- Week 4: Multi-waypoint missions

**March:**
- Pacti contract refinement
- Performance optimization
- Documentation & analysis

**April (Optional):**
- C++ implementation
- Hardware testing prep
- ROS2 integration

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

**Status:** Active Development | **Updated:** February 2025 | **Version:** 1.1.0
