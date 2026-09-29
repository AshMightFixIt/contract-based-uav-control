# Development & Testing Guide

Complete guide for testing, developing, and extending the contract-based UAV control system.

---

## 📋 Table of Contents

- [Development Setup](#development-setup)
- [Testing](#testing)
- [Code Structure](#code-structure)
- [Adding New Controllers](#adding-new-controllers)
- [Debugging](#debugging)
- [Performance Optimization](#performance-optimization)
- [Known Issues](#known-issues)
- [Contributing](#contributing)

---

## 🛠️ Development Setup

### Prerequisites

```bash
# Python 3.8 or higher
python --version  # Should be >= 3.8

# Install the core package (editable) and the demo dependencies
pip install -e ./contract_uav_core
pip install -r requirements.txt

# For the tests (make test), and optionally the pacti planner
pip install -r requirements-test.txt
pip install -r requirements-legacy.txt   # optional: pacti, for the horizon planner and make test-pacti

# Verify installation
python -c "import numpy; import matplotlib; print('✓ Dependencies OK')"
```

### Development Environment

**Recommended IDE:** VSCode with Python extension

**Python path:** no PYTHONPATH setup is needed. `pip install -e ./contract_uav_core`
makes `contract_uav_core` importable from any directory, and edits to the package
take effect without reinstalling. The same commands work in PowerShell on Windows.

**Package layout: keep `contract_uav_core/contract_uav_core/__init__.py` docstring-only.**
The package sits one folder down (`contract_uav_core/contract_uav_core/`, the ROS 2
convention). With the default editable install, a Python started in the repository
root imports the outer project folder `contract_uav_core/` as a namespace package:
the inner `__init__.py` is not executed there, although every submodule
(`contract_uav_core.core`, ...) still resolves to the real package. Code in that
`__init__.py` would therefore run or not depending on the working directory, so the
file holds only a docstring, and `tests/test_package_layout.py` fails otherwise.
Installing with `pip install -e ./contract_uav_core --config-settings editable_mode=compat`
(a plain path entry instead of an import hook) removes the effect.

---

## 🧪 Testing

### Quick Test - All Components

```bash
# Test everything: module self-tests, golden benchmark tables, node import check
make test

# Or run single module self-tests:
python -m contract_uav_core.contracts.monitor
python -m contract_uav_core.estimation.ekf
python -m contract_uav_core.control.switcher
python -m contract_uav_core.safety.cbf
python -m contract_uav_core.core
```

`make help` lists the other targets (`make golden`, `make test-pacti`, ...).

### Component-Level Testing

#### 1. Contract Framework Test

```bash
python -m contract_uav_core.contracts.monitor
```

**Expected Output:**
```
=== Testing Contract Framework ===
✓ Basic contract creation
✓ Contract composition
✓ Hierarchical pipeline verification
✓ Contract violation detection
All tests passed!
```

**What it tests:**
- Contract creation and initialization
- Assumption/guarantee checking
- Hierarchical composition (G₁ ⇒ A₂)
- Violation detection logic

#### 2. Extended Kalman Filter Test

```bash
python -m contract_uav_core.estimation.ekf
```

**Expected Output:**
```
=== Testing Contract-Aware EKF ===
✓ State initialization
✓ GPS fusion mode
✓ IMU-dominated mode
✓ Contract satisfaction checking
Position error: 0.34m (< 0.5m limit)
All tests passed!
```

**What it tests:**
- 12-state EKF initialization
- GPS measurement fusion
- IMU integration
- Fusion mode switching
- Contract guarantee validation

#### 3. Controller Test

```bash
python -m contract_uav_core.control.switcher
```

**Expected Output:**
```
=== Testing Controllers ===
✓ PID controller - Nominal conditions
✓ H-infinity controller - High wind
✓ Contract checking
✓ Control output bounds
All tests passed!
```

**What it tests:**
- PID control law implementation
- H-infinity robust control
- Contract assumption checking
- Control output saturation

#### 4. CBF Safety Filter Test

```bash
python -m contract_uav_core.safety.cbf
```

**Expected Output:**
```
=== Testing CBF Safety Filter ===
✓ Barrier function initialization
✓ Safety constraint checking
✓ Minimal intervention filter
✓ All barriers positive
All tests passed!
```

**What it tests:**
- Four barrier functions (altitude, velocity, tilt, rates)
- Safety constraint ḣ(x,u) ≥ -αh(x)
- Minimally-invasive filtering
- Constraint satisfaction

#### 5. Complete System Integration Test

```bash
python -m contract_uav_core.core
```

**Expected Output:**
```
=== Testing Adaptive Control System ===
✓ Pre-flight contract verification
✓ Controller switching logic
✓ Runtime monitoring
✓ Safety filter integration
Simulation: 10 seconds
Switches: 1 (PID → H-infinity at t=2.5s)
All tests passed!
```

**What it tests:**
- End-to-end pipeline
- Contract-based switching
- Runtime monitoring at 50 Hz
- CBF integration

### Full Demonstration

```bash
python simulation_demo.py
```

**Runs 15-second flight simulation:**
- Pre-flight verification
- Wind disturbance injection
- Contract-based switching
- Results visualization
- Performance logging

---

## 📚 Code Structure

### Core Modules

The core is the package `contract_uav_core` (numpy, and PyYAML for the airframe files). Paths below are
relative to `contract_uav_core/contract_uav_core/`. Mixins let one class span several files: each
file owns part of the class, and the class name and constructor stay the same.

#### `contracts/` — assume-guarantee contracts

- `spec.py`: `ContractStatus` (enum), `ContractMetrics` (dataclass),
  `LinearConstraint` (`output <= sum(c*x) + k`, or `>=`) and `SimpleContract`
  (range assumptions + linear guarantees).
- `library.py`: `define_contracts()`, the component contracts (sensors, EKF,
  PID, MPC, H-inf, actuators) and their named constants.
- `preflight.py`: `compose_pipeline(controller_name)` and
  `verify_mission_feasibility(controller_name, conditions)` for the monitor, and
  `AdaptiveDroneController.pre_flight_check(initial_conditions, mission)`.
- `monitor.py`: `HierarchicalContractMonitor`, with `monitor_runtime`,
  `get_contract_status`, `should_switch_controller` and `export_metrics`.

**Main methods:**
```python
SimpleContract.check_assumptions(values) -> (bool, margins)
SimpleContract.check_guarantees(values) -> (bool, margins)
SimpleContract.predict_guarantees(values) -> {"<output>_max": bound, ...}
SimpleContract.compose(other) -> SimpleContract      # self feeds into other
HierarchicalContractMonitor.verify_mission_feasibility(controller_name, conditions) -> (bool, message)
```

#### `estimation/ekf.py`

**Key class:** `ContractAwareEKF`, a 12-state EKF. It picks its update from the
sensor contracts: `update_full` (GPS + IMU), `update_imu_only`, or
`propagate_only` (dead reckoning).

**State vector (12 states):**
```python
x = [px, py, pz,        # Position (NED frame)
     vx, vy, vz,        # Velocity
     φ, θ, ψ,           # Euler angles (roll, pitch, yaw)
     p, q, r]           # Angular rates
```

**Main methods:**
```python
predict(dt=None)
update(measurements, sensors, timestamp) -> (state, estimation_ok)
check_sensor_contracts(timestamp) -> (gps_ok, imu_ok)
get_state() -> dict
get_covariance() -> np.ndarray
```

#### `control/`

- `base.py`: `BaseController` (abstract: `compute_accel_command(state, setpoint)`, `reset()`;
  `compute_control` returns the command's legacy dict).
- `pid.py`, `mpc.py`, `hinf.py`: `PIDController`, `MPCController`,
  `HInfinityController`. Each has `compute_accel_command`, `reset` and
  `get_integral_state` / `set_integral_state` (bumpless transfer).
- `switcher.py`: `ControllerSwitcher`, which holds the three controllers under
  the names `'PID'`, `'MPC'` and `'Hinf'` (`switch_to(name, current_time, reason,
  priority)`, `compute_control`, `compute_accel_command`, `get_active_controller`).
- `supervisor.py`: `FlightMode` and `FlightModeSupervisor` (setpoints and the
  controller choice per flight mode).

#### `safety/`

- `cbf.py`: `CBFSafetyFilter`, with `barrier_altitude`, `barrier_velocity`,
  `barrier_tilt`, `barrier_rates`, `evaluate_safety(state)` and
  `filter_control(state, nominal_control, timestamp) -> (control, intervened)`.
- `runtime_monitor.py`: `RuntimeMonitor` (wind, GPS quality, compute delay).

#### Integration: `core.py` and its mixins

- `core.py`: `AdaptiveDroneController` (`__init__`, `update_sensors`,
  `control_step(raw_sensors, t=None) -> (control, telemetry)`, mission commands).
- `conditions.py`: `estimate_system_conditions`.
- `switching_policy.py`: the multi-factor controller recommendation of
  `control_step` (signals A–E, hysteresis, planner escalation).
- `telemetry.py`: runtime and contract monitor updates, the flight log and
  `save_flight_log`.

#### Other packages

- `planning/`: the Pacti horizon planner (needs pacti; the core runs without it).
- `sim/`: `sensor_faults.py` and `battery_model.py` for the demos and benchmarks;
  `quadrotor.py`, the reference plant.
- `config/`: `airframe.py` (schema and loader) and `airframes/*.yaml`.
- `viz/`: `live_visualizer.py` (needs pygame).

#### Interfaces and time

`control_step(raw_sensors, t=None)`: without `t` (the scripts) the step runs at the
accumulated `self.time` and the EKF predicts with the nominal `dt`, bit for bit as before;
with `t` in seconds (the ROS node passes the PX4 `VehicleLocalPosition` timestamp) the EKF
predicts with the measured step clamped to 0.5–2× `dt`, a step above 1.5× `dt` counts as an
overrun and a repeated or older `t` as non-increasing (telemetry `timing`). Only the EKF sees
the measured step; the controllers, supervisor and switcher keep the nominal `dt`.
`interfaces.py` holds the shared types and documents the frame convention (NED, z down;
ZYX Euler angles): `AccelCommand` (acceleration demand + yaw, with the legacy thrust,
torques and attitude/rate setpoints passed through; `to_legacy_dict()`),
`AttitudeThrustCommand` and `SwitchPriority` (recorded only until P1.5). `frames.py` holds
the tiers' acceleration -> attitude/thrust/torque mappings, moved unchanged (still
yaw-blind). `sim/quadrotor.py` (the reference plant, driven by `AttitudeThrustCommand`) and
`config/airframes/` (PX4 v1.16.2 values with provenance, `load_airframe('x500_sitl')`) are not
used by the legacy scripts yet. Their tests are in `tests/core`, `tests/sim` and
`tests/config`.

---

## ➕ Adding New Controllers

### Step 1: Create Controller Class

```python
# In contract_uav_core/contract_uav_core/control/ (a new module next to pid.py)

class MyNewController:
    def __init__(self):
        self.gains = {...}
        
    def compute_control(self, state, setpoint):
        """Compute control output"""
        # Your control law here
        u = your_algorithm(state, setpoint)
        return self.saturate(u)
        
    def saturate(self, u):
        """Ensure control limits"""
        return np.clip(u, -self.u_max, self.u_max)
```

### Step 2: Define Contract

```python
# Contracts are defined in contract_uav_core/contract_uav_core/contracts/library.py
# (define_contracts)

my_controller_contract = {
    'assumptions': {
        'condition1': (min_val, max_val),
        'condition2': (min_val, max_val),
    },
    'guarantees': {
        'performance1': (min_val, max_val),
        'performance2': (min_val, max_val),
    }
}
```

### Step 3: Add to Controller Dictionary

```python
self.controllers = {
    'PID': PIDController(),
    'Hinf': HInfinityController(),
    'MyNew': MyNewController(),  # Add your controller
    'Safety': SafetyController()
}

self.controller_contracts = {
    'PID': pid_contract,
    'Hinf': hinf_contract,
    'MyNew': my_controller_contract,  # Add contract
    'Safety': safety_contract
}
```

### Step 4: Test

```python
# Test your controller
controller = MyNewController()
u = controller.compute_control(test_state, test_setpoint)
assert contract_satisfied(u, test_state)
```

---

## 🐛 Debugging

### Enable Debug Logging

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Common Issues

#### Issue 1: Contract Violations Not Detected

**Symptom:** Controller doesn't switch when expected

**Debug:**
```python
# In contract_uav_core/contract_uav_core/switching_policy.py, add:
print(f"State: {state}")
print(f"Assumptions: {assumptions_met}")
print(f"Violations: {violated_assumptions}")
```

**Common causes:**
- Incorrect contract bounds
- State not updated properly
- Sensor data missing

#### Issue 2: Excessive Switching (Chattering)

**Symptom:** Controller switches too frequently

**Debug:**
```python
# Check switching frequency
switches_per_second = total_switches / simulation_time
print(f"Switch rate: {switches_per_second} Hz")
```

**Solutions:**
- Add hysteresis to contract bounds
- Implement cooldown timer
- Adjust contract margins

#### Issue 3: CBF Always Intervening

**Symptom:** Safety filter modifies every control command

**Debug:**
```python
# In contract_uav_core/contract_uav_core/safety/cbf.py:
for i, h in enumerate(self.barriers):
    print(f"Barrier {i}: h={h.value(state):.3f}")
```

**Solutions:**
- Check barrier function definitions
- Adjust safety margins (α parameter)
- Verify controller tuning

#### Issue 4: Simulation Divergence

**Symptom:** States go to infinity, NaN, or unstable

**Debug:**
```python
# Check for NaN
assert not np.isnan(state).any(), "NaN in state!"

# Check bounds
assert abs(state['z']) < 100, "Altitude unrealistic!"
```

**Solutions:**
- Reduce simulation time step
- Check controller gains
- Add saturation limits
- Verify dynamics model

### Visualization for Debugging

```python
import matplotlib.pyplot as plt

# Plot state history
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(time, position)
plt.ylabel('Position [m]')

plt.subplot(3, 1, 2)
plt.plot(time, velocity)
plt.ylabel('Velocity [m/s]')

plt.subplot(3, 1, 3)
plt.plot(time, controller_mode)
plt.ylabel('Active Controller')
plt.xlabel('Time [s]')

plt.show()
```

---

## ⚡ Performance Optimization

### Profiling

```python
import cProfile
import pstats

# Profile simulation
cProfile.run('simulation_demo.main()', 'output.stats')

# View results
stats = pstats.Stats('output.stats')
stats.sort_stats('cumulative')
stats.print_stats(20)
```

### Common Bottlenecks

**1. Contract Checking:**
- Cache assumption evaluations
- Use numpy vectorization
- Pre-compute constants

**2. EKF Updates:**
- Update only when new measurements available
- Use sparse matrices where possible
- Optimize Kalman gain computation

**3. CBF Optimization:**
- Analytical solution when possible
- Warm-start iterative solver
- Simplify constraint checking

### Optimization Tips

```python
# Bad: Loop over elements
for i in range(len(array)):
    result[i] = array[i]**2

# Good: Vectorized operation
result = array**2

# Bad: Repeated computation
for i in range(1000):
    x = expensive_function()
    
# Good: Compute once
x = expensive_function()
for i in range(1000):
    # use x
```

---

## ⚠️ Known Issues

### 1. H-infinity Emergency Threshold

**Issue:** Emergency landing triggered too early (altitude < -8m)

**Status:** Identified, fix in progress

**Workaround:**
```python
# In contract_uav_core/contract_uav_core/control/hinf.py, change:
if altitude < -40.0:  # Instead of -8.0
    emergency_land()
```

### 2. Bidirectional Switching

**Issue:** System doesn't switch back from H-infinity to PID

**Status:** Recovery logic implemented but blocked by Issue #1

**Workaround:** Fix H-infinity controller first

### 3. Initial PID Transient

**Issue:** Large velocity spike at t=0 causes premature switch

**Status:** Controller tuning needed

**Workaround:**
```python
# Softer startup gains
if time < 2.0:
    kp = kp * 0.6
```

---

## 🤝 Contributing

### Code Style

- Follow PEP 8
- Use type hints
- Add docstrings (Google style)
- Keep functions < 50 lines

### Example Function

```python
def compute_control(
    state: Dict[str, np.ndarray],
    setpoint: Dict[str, float],
    dt: float = 0.02
) -> np.ndarray:
    """
    Compute control output for given state and setpoint.
    
    Args:
        state: Current system state with keys 'position', 'velocity'
        setpoint: Desired state
        dt: Time step in seconds
        
    Returns:
        Control command as numpy array [thrust, roll, pitch, yaw]
        
    Raises:
        ValueError: If state or setpoint missing required keys
    """
    # Implementation
    return control_output
```

### Testing New Features

1. Write unit tests
2. Test with simulation
3. Document changes
4. Submit pull request

---

## 📝 Next Steps

For integration with PX4 SITL/Gazebo, see the ROS 2 bridge in `contract_uav_control/` and `docs/TESTING_MANUAL.md`.

For adding fuzzy logic controller, follow the "Adding New Controllers" section.

For hardware deployment, see C++ implementation guide (coming soon).

---

**Questions?** Open an issue on GitHub or contact aswatth@umich.edu
