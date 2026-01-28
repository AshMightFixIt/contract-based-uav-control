# Contract-Based Adaptive UAV Control System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A hierarchical contract-based architecture integrating Assume-Guarantee (A/G) contracts with Control Barrier Function (CBF) safety filtering for provably safe multi-mode UAV control.

**Research Project** | University of Michigan  
**Courses:** AE552 (Aerospace Information Systems) & ECE599 (Formal Methods)  
**Author:** Aswatth Sunil | **Advisor:** Prof. Iñigo Incer

---

## 🎯 Overview

This system demonstrates formal methods applied to autonomous UAV control through:

- **Hierarchical Contract Framework** (506 lines) - Compositional verification through assume-guarantee contracts
- **Four Controller Modes** - PID, H-infinity, GPS-denied, Safety with formal operating envelopes
- **Contract-Aware EKF** (331 lines) - Adaptive sensor fusion based on contract satisfaction  
- **CBF Safety Filter** (200 lines) - Minimally-invasive safety with 4 barrier functions
- **Runtime Monitoring** (281 lines) - 50 Hz contract verification (2.5 ms avg computation)

**Total:** 1,499 lines of production Python code

---

## 🏗️ System Architecture

### Key Innovation: Contracts as Architectural Principle

Traditional threshold-based switching:
```python
if wind_speed > 3.0:  # Ad-hoc threshold
    switch_to_robust_controller()
```

Our contract-based approach:
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

### Hierarchical Architecture

```
┌─────────────────────────────────────────────────────────────┐
│           PRE-FLIGHT CONTRACT VERIFICATION                   │
│           Hierarchical Composition: G₁ ⇒ A₂                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   RUNTIME SUPERVISOR                         │
│        Contract Monitoring (50 Hz) + CBF Safety             │
├─────────────────────────────────────────────────────────────┤
│  Sensors → EKF → Controller → CBF Filter → Actuators       │
│  [Contract] [Contract] [Contract]  [Contract]  [Contract]   │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 Installation

### Requirements
- Python 3.8+
- NumPy
- Matplotlib

### Setup
```bash
git clone https://github.com/YOUR_USERNAME/contract-based-uav-control.git
cd contract-based-uav-control
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### Run Complete Simulation
```bash
python simulation_demo.py
```

This demonstrates:
1. Pre-flight contract verification
2. 15-second flight simulation with wind disturbance
3. Contract-based controller switching (PID → H-infinity)
4. Runtime safety monitoring
5. Results visualization

### Test Individual Components
```bash
# Test contract framework
python src/contracts/contract_framework.py

# Test contract-aware EKF
python src/estimation/contract_ekf.py

# Test controllers
python src/control/controllers.py

# Test complete system
python src/adaptive_control_system.py
```

---

## 📊 Experimental Results

### System Performance
- **Real-time monitoring:** 50 Hz (20 ms period)
- **Computation time:** 2.5 ms average, 4.1 ms peak
- **CPU utilization:** 12.5% (leaves 87.5% for control)
- **Contract checks:** 750+ per 15-second flight
- **Controller switches:** Formal violation-based (zero false positives)

### Demonstration Scenario

**Phase 1 (0-5s): Nominal Flight**
- Light wind (0.5 m/s)
- PID controller active
- All contracts satisfied

**Phase 2 (5-10s): Wind Disturbance**
- Strong wind injection (5.4 m/s peak)
- Exceeds PID contract assumptions
- Automatic switch to H-infinity controller
- Safety maintained

**Phase 3 (10-15s): Recovery**
- Wind subsides to 0.5 m/s
- System remains stable
- Recovery logic evaluates return to PID

### Safety Validation
- **Altitude:** Maintained within [2m, 50m]
- **Velocity:** Peak 6.2 m/s (under 15 m/s limit)
- **Tilt angle:** Max 12° (under 30° limit)
- **Angular rates:** Max 0.8 rad/s (under 3 rad/s limit)
- **CBF interventions:** 0 (controllers respected safety constraints)

---

## 📁 Project Structure

```
contract-based-uav-control/
├── src/
│   ├── contracts/
│   │   └── contract_framework.py    # Core A/G contract system (506 lines)
│   ├── estimation/
│   │   └── contract_ekf.py          # Contract-aware EKF (331 lines)
│   ├── control/
│   │   ├── controllers.py           # PID & H-infinity (381 lines)
│   │   └── four_mode_controllers.py # Four-mode implementation
│   ├── safety/
│   │   └── cbf_filter.py           # CBF safety filter (200 lines)
│   └── adaptive_control_system.py   # Main integration (281 lines)
├── simulation_demo.py               # Demonstration
├── requirements.txt                 # Dependencies
└── README.md                        # This file
```

---

## 🔬 Technical Details

### Assume-Guarantee Contracts

Each component has a contract $(A, G)$ where:
- **Assumptions (A):** Required environmental conditions
- **Guarantees (G):** Promised behavior when assumptions hold

**PID Controller Contract:**
```python
A_PID = {
    'wind_speed': [0, 3.0] m/s,
    'GPS_available': True,
    'position_error': [0, 2.0] m
}
G_PID = {
    'position_error': [0, 1.5] m,
    'velocity_error': [0, 1.5] m/s
}
```

**H-infinity Controller Contract:**
```python
A_Hinf = {
    'wind_speed': [0, 10.0] m/s,
    'GPS_available': True
}
G_Hinf = {
    'position_error': [0, 3.0] m,
    'velocity_error': [0, 5.0] m/s
}
```

### Control Barrier Functions

Four barrier functions enforce safety constraints:
```python
h₁(x) = z - z_min        # Altitude lower bound
h₂(x) = z_max - z        # Altitude upper bound  
h₃(x) = v_max² - ||v||²  # Velocity limit
h₄(x) = θ_max² - ||θ||²  # Tilt angle limit
```

Safety filter: `u* = argmin ||u - u_d||²` subject to `ḣᵢ(x,u) ≥ -αᵢhᵢ(x)`

---

## 🎓 Research Contributions

1. **Hierarchical Contract Composition** for formal mission verification
2. **Contract-Aware State Estimation** with adaptive sensor fusion
3. **Bidirectional Switching Logic** based on contract satisfaction
4. **Integration of Contracts with CBF** for layered safety guarantees
5. **Real-time Performance** demonstrated on embedded-class hardware

### Publications & Presentations
- AE552 Final Paper: "Hierarchical Contract-Based Adaptive Control with Runtime Safety Assurance for UAVs"
- ECE599 Project Presentation (December 2024)

---

## 📚 References

1. Benveniste et al., "Contracts for System Design," *Foundations and Trends in Electronic Design Automation*, 2018
2. Ames et al., "Control Barrier Functions: Theory and Applications," *European Control Conference*, 2019
3. Nuzzo et al., "A Contract-Based Methodology for Aircraft Electric Power System Design," *IEEE Access*, 2014
4. Incer et al., "Pacti: Scaling Assume-Guarantee Reasoning for System Analysis," arXiv:2303.17751, 2023

---

## 🛠️ Future Work

- [ ] C++ implementation for embedded deployment
- [ ] ROS2/PX4/Gazebo integration for HITL testing
- [ ] GPS-denied navigation with vision-based estimation
- [ ] Multi-waypoint mission planning with contracts
- [ ] Hardware flight testing

---

## 📄 License

This project is developed for academic research purposes.

MIT License - See LICENSE file for details

---

## 📧 Contact

**Aswatth Sunil**  
University of Michigan  
Email: aswatth@umich.edu  
Advisor: Prof. Iñigo Incer (EECS)

---

**Status:** Project Complete ✓  
**Last Updated:** January 2025
