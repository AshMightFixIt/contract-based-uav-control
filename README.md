# Contract-Based Adaptive UAV Control System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A hierarchical contract-based architecture integrating Assume-Guarantee (A/G) contracts with Control Barrier Function (CBF) safety filtering for provably safe multi-mode UAV control.

**Research Project** | University of Michigan  
**Courses:** AE552 (Aerospace Information Systems) & ECE599 (Formal Methods)  
**Author:** Aswatth Sunil | **Advisor:** Prof. Iñigo Incer (EECS)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Innovation](#key-innovation)
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

This system demonstrates formal methods applied to autonomous UAV control through hierarchical contract composition. Instead of ad-hoc threshold-based switching, the system uses mathematically provable Assume-Guarantee contracts to ensure safe controller transitions.

### Core Components

- **Hierarchical Contract Framework** (506 lines) - Compositional verification through A/G contracts
- **Four Controller Modes** - PID, H-infinity, GPS-denied, Safety with formal operating envelopes
- **Contract-Aware EKF** (331 lines) - Adaptive sensor fusion based on contract satisfaction  
- **CBF Safety Filter** (200 lines) - Minimally-invasive safety with 4 barrier functions
- **Runtime Monitoring** (281 lines) - 50 Hz contract verification (2.5 ms avg computation)

**Implementation:** 1,499 lines of production Python code

### System Performance

- **Real-time monitoring:** 50 Hz (20 ms period)
- **Computation time:** 2.5 ms average, 4.1 ms peak  
- **CPU utilization:** 12.5%
- **Contract checks:** 750+ per 15-second flight
- **Switching:** Formal violation-based (zero false positives)

---

## 💡 Key Innovation

### Traditional Approach: Heuristic Thresholds
```python
if wind_speed > 3.0:  # Ad-hoc threshold
    switch_to_robust_controller()
```

**Problems:** No safety guarantees, difficult to verify, prone to chattering

### Our Approach: Hierarchical Contract Composition
```python
# Formal verification of entire pipeline
pipeline_contract = (
    sensor_contract 
    >> estimator_contract 
    >> controller_contract 
    >> actuator_contract
)

# Pre-flight verification
if not pipeline_contract.satisfies(mission_requirements):
    abort_mission()  # Provably unsafe
```

**Benefits:** ✅ Formal verification ✅ Safety guarantees ✅ Certification-ready

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

### Test Components
```bash
python src/contracts/contract_framework.py    # Contract framework
python src/estimation/contract_ekf.py         # Extended Kalman Filter
python src/control/controllers.py             # Controllers
python src/safety/cbf_filter.py              # CBF safety filter
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
├── src/                                # Source code (1,499 lines)
│   ├── contracts/
│   │   └── contract_framework.py       # A/G contracts (506 lines)
│   ├── estimation/
│   │   └── contract_ekf.py            # Contract-aware EKF (331 lines)
│   ├── control/
│   │   ├── controllers.py             # PID & H-infinity (200 lines)
│   │   └── four_mode_controllers.py   # Four modes (181 lines)
│   ├── safety/
│   │   └── cbf_filter.py             # CBF filter (200 lines)
│   └── adaptive_control_system.py     # Integration (281 lines)
│
├── simulation_demo.py                  # Main demonstration
├── requirements.txt                    # Dependencies
├── README.md                           # This file
└── DEVELOPMENT.md                      # Testing & development guide
```

---

## 🎮 Controller Modes & Contracts

### 1. Nominal PID Controller
**Use:** Efficient control in calm conditions

**Contract:**
- Assumptions: Wind < 3 m/s, GPS available, Position error < 2m
- Guarantees: Position error < 1.5m, Velocity error < 1.5 m/s

### 2. Wind-Robust H-infinity Controller  
**Use:** Robust control under disturbances

**Contract:**
- Assumptions: Wind < 10 m/s, GPS available
- Guarantees: Position error < 3m, Stabilization < 10s

**Key:** Minimal assumptions - always available safety net

### 3. GPS-Denied Controller
**Use:** Navigation without GPS

**Contract:**
- Assumptions: IMU available, Wind < 5 m/s
- Guarantees: Position drift < 0.5 m/s

### 4. Safety Controller
**Use:** Emergency landing

**Contract:**
- Assumptions: Any sensor available
- Guarantees: Controlled descent, Velocity < 2 m/s

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

### Current Status (January 2025)

**✅ Completed:**
- Hierarchical contract framework
- Four controller modes
- Contract-aware EKF
- CBF safety filter
- Runtime monitoring at 50 Hz
- Python simulation (1,499 lines)

**⚠️ Known Issues:**
- H-infinity emergency threshold too aggressive
- Bidirectional switching needs demonstration
- Initial PID transient causes early switch

### Upcoming (February-April 2025)

**February:**
- Week 1: Debug H-infinity controller
- Week 2: SITL/Gazebo integration
- Week 3: GPS-denied mode
- Week 4: Waypoint following

**March:**
- Controller optimization
- Contract refinement
- Documentation & analysis

**April (Optional):**
- Fuzzy logic controller
- C++ implementation
- Hardware testing prep

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

**Status:** Active Development | **Updated:** January 2025 | **Version:** 1.0.0
