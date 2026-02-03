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

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import numpy; import matplotlib; print('✓ Dependencies OK')"
```

### Development Environment

**Recommended IDE:** VSCode with Python extension

**Python Path Setup:**
```bash
# Add to ~/.bashrc or ~/.zshrc
export PYTHONPATH="/path/to/contract-based-uav-control/src:$PYTHONPATH"

# Or set in current session
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
```

**Windows (PowerShell):**
```powershell
$env:PYTHONPATH = "$(Get-Location)\src;$env:PYTHONPATH"
```

---

## 🧪 Testing

### Quick Test - All Components

```bash
# Test everything
./test_all.sh

# Or manually:
python src/contracts/contract_framework.py
python src/estimation/contract_ekf.py
python src/control/controllers.py
python src/safety/cbf_filter.py
python src/adaptive_control_system.py
```

### Component-Level Testing

#### 1. Contract Framework Test

```bash
python src/contracts/contract_framework.py
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
python src/estimation/contract_ekf.py
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
python src/control/controllers.py
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
python src/safety/cbf_filter.py
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
python src/adaptive_control_system.py
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

#### `src/contracts/contract_framework.py` (506 lines)

**Key Classes:**
- `Contract`: Base A/G contract class
- `ComponentContract`: Contract with metadata
- `ContractChecker`: Runtime verification
- `HierarchicalComposer`: Contract composition

**Main Functions:**
```python
def check_assumptions(state) -> (bool, list):
    """Check if contract assumptions are satisfied"""
    
def check_guarantees(state) -> (bool, list):
    """Verify contract guarantees hold"""
    
def compose(contract1, contract2) -> Contract:
    """Compose two contracts: G₁ ⇒ A₂"""
```

#### `src/estimation/contract_ekf.py` (331 lines)

**Key Classes:**
- `ContractAwareEKF`: 12-state EKF with contract checking
- `SensorFusionMode`: Enum for GPS/IMU/Dead-reckoning

**State Vector (12 states):**
```python
x = [px, py, pz,        # Position (NED frame)
     vx, vy, vz,        # Velocity
     φ, θ, ψ,           # Euler angles (roll, pitch, yaw)
     p, q, r]           # Angular rates
```

**Main Methods:**
```python
def predict(u, dt):
    """Prediction step with process model"""
    
def update_gps(z_gps):
    """GPS measurement update"""
    
def update_imu(z_imu):
    """IMU measurement update"""
    
def check_contract():
    """Verify estimation contract"""
```

#### `src/control/controllers.py` (200 lines)

**Key Classes:**
- `PIDController`: Nominal PID control
- `HInfinityController`: Robust H∞ control

**PID Implementation:**
```python
def compute_control(state, setpoint):
    error = setpoint - state
    u_p = self.kp * error
    u_i = self.ki * integral
    u_d = self.kd * derivative
    return saturate(u_p + u_i + u_d)
```

**Contract Checking:**
```python
def check_assumptions(state):
    wind_ok = state['wind'] < 3.0
    gps_ok = state['gps_available']
    error_ok = state['position_error'] < 2.0
    return wind_ok and gps_ok and error_ok
```

#### `src/safety/cbf_filter.py` (200 lines)

**Key Classes:**
- `ControlBarrierFunction`: Single barrier function
- `CBFSafetyFilter`: Multi-barrier safety filter

**Barrier Functions:**
```python
def h_altitude_lower(state):
    return state['z'] - 2.0  # z > 2m

def h_altitude_upper(state):
    return 50.0 - state['z']  # z < 50m
    
def h_velocity(state):
    return 15.0**2 - np.linalg.norm(state['v'])**2
    
def h_tilt(state):
    return 0.5 - (state['phi']**2 + state['theta']**2)
```

**Safety Filter:**
```python
def filter_control(u_desired, state):
    """
    Solve: u* = argmin ||u - u_d||²
           s.t. ḣᵢ(x,u) ≥ -αᵢhᵢ(x)
    """
    if all_safe(state):
        return u_desired  # No intervention needed
    else:
        return project_to_safe_set(u_desired, state)
```

---

## ➕ Adding New Controllers

### Step 1: Create Controller Class

```python
# In src/control/controllers.py

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
# In src/adaptive_control_system.py

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
# In adaptive_control_system.py, add:
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
# In cbf_filter.py:
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
# In controllers.py, change:
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

For integration with SITL/Gazebo, see the ROS2 integration templates in `/ros2_adaptive_controller/`.

For adding fuzzy logic controller, follow the "Adding New Controllers" section.

For hardware deployment, see C++ implementation guide (coming soon).

---

**Questions?** Open an issue on GitHub or contact aswatth@umich.edu
