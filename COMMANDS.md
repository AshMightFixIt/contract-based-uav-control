# Quick Command Reference
## Daily Workflow Commands

**Project Location**: `/home/claude/adaptive_drone_contracts/`

---

## 🔧 Setup (Run Once Per Session)

```bash
# Navigate to project
cd /home/claude/adaptive_drone_contracts

# Set Python path
export PYTHONPATH=/home/claude/adaptive_drone_contracts/src:$PYTHONPATH
```

---

## 🧪 Testing Commands

### Test Individual Components
```bash
# Test contract framework
python3 src/contracts/contract_framework.py

# Test EKF
python3 src/estimation/contract_ekf.py

# Test controllers
python3 src/control/controllers.py

# Test complete system
python3 src/adaptive_control_system.py
```

### Run All Tests
```bash
./test_all.sh
```

---

## 🚀 Run Demonstrations

### Full Flight Simulation
```bash
python3 simulation_demo.py
# Outputs:
#   - simulation_results.png (visualization)
#   - flight_log.json (flight data)
#   - flight_log_contracts.json (contract monitoring data)
```

### View Results
```bash
# View plot
eog simulation_results.png
# or
xdg-open simulation_results.png

# View logs
cat flight_log.json | python3 -m json.tool | less
cat flight_log_contracts.json | python3 -m json.tool | less
```

---

## 📊 Data Analysis

### Extract Statistics
```bash
# Count contract violations
python3 -c "
import json
with open('flight_log_contracts.json') as f:
    data = json.load(f)
    violations = [m for m in data['history'] if not m['assumptions_met']]
    print(f'Total violations: {len(violations)}')
"

# Get switching times
python3 -c "
import json
with open('flight_log.json') as f:
    data = json.load(f)
    controllers = [d['controller'] for d in data]
    switches = [(i, controllers[i-1], controllers[i]) 
                for i in range(1, len(controllers)) 
                if controllers[i] != controllers[i-1]]
    print('Switches:')
    for i, (idx, from_c, to_c) in enumerate(switches, 1):
        print(f'  {i}. t={data[idx][\"time\"]:.2f}s: {from_c} → {to_c}')
"
```

---

## 🔍 Code Exploration

### View Key Files
```bash
# Contract framework
less src/contracts/contract_framework.py

# Contract-aware EKF
less src/estimation/contract_ekf.py

# Controllers
less src/control/controllers.py

# Main system
less src/adaptive_control_system.py
```

### Check Line Counts
```bash
wc -l src/**/*.py simulation_demo.py
```

---

## 📝 Documentation

### View Documentation
```bash
# Main README
less README.md

# Week 1 summary
less WEEK1_SUMMARY.md

# Innovation summary
less INNOVATION_SUMMARY.md

# Complete project summary
less PROJECT_SUMMARY.md
```

---

## 🛠️ Development Workflow

### Add New Scenario (Example: GPS Degradation)

1. **Modify simulation_demo.py**
```python
# Add new phase
phases = [
    ("NOMINAL", 0, 5.0, np.array([0.5, 0.0, 0.0]), 12),  # 12 satellites
    ("GPS_DEGRADE", 5.0, 10.0, np.array([0.5, 0.0, 0.0]), 4),  # 4 satellites
    ("RECOVERY", 10.0, 15.0, np.array([0.5, 0.0, 0.0]), 12),
]

# Update sensor data generation
satellites = phase[4]  # Get from phase
sensors['gps']['satellites'] = satellites
```

2. **Run simulation**
```bash
python3 simulation_demo.py
```

3. **Analyze results**
```bash
eog simulation_results.png
cat flight_log_contracts.json | grep "sensor"
```

### Add New Controller

1. **Edit src/control/controllers.py**
```python
class MPCController(BaseController):
    def __init__(self, dt=0.02):
        super().__init__("MPC", dt)
        # Add MPC implementation
    
    def compute_control(self, state, setpoint):
        # MPC logic
        pass
```

2. **Update switcher**
```python
self.controllers = {
    'PID': PIDController(dt),
    'MPC': MPCController(dt),  # Add new controller
    'Hinf': HInfinityController(dt)
}
```

3. **Define contract**
```python
# In contract_framework.py
self.controller_contracts['MPC'] = SimpleContract(
    name="MPC_Controller",
    assumptions={
        "position_error": (0.0, 5.0),
        "computation_time": (0.0, 0.05),
    },
    guarantees={
        "tracking_error": (0.0, 0.5),
        "constraints_satisfied": (1.0, 1.0),
    }
)
```

---

## 🐛 Debugging

### Enable Verbose Logging
```python
# In any Python file
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Contract Status
```bash
# Add to any script
python3 -c "
from adaptive_control_system import AdaptiveDroneController
controller = AdaptiveDroneController()
controller.contract_monitor.define_contracts()
status = controller.contract_monitor.get_contract_status('sensors')
print(f'Sensor contract status: {status}')
"
```

---

## 📦 Export Results

### Create Results Package
```bash
# Create archive
tar -czf week1_results.tar.gz \
    simulation_results.png \
    flight_log.json \
    flight_log_contracts.json \
    README.md \
    WEEK1_SUMMARY.md

# Extract later
tar -xzf week1_results.tar.gz
```

### Generate Report
```bash
# Create markdown report
python3 -c "
import json

with open('flight_log.json') as f:
    data = json.load(f)

with open('RESULTS_REPORT.md', 'w') as out:
    out.write('# Flight Test Results\n\n')
    out.write(f'**Total steps**: {len(data)}\n')
    out.write(f'**Duration**: {data[-1][\"time\"]:.1f}s\n\n')
    
    # Count controller usage
    controllers = {}
    for d in data:
        ctrl = d['controller']
        controllers[ctrl] = controllers.get(ctrl, 0) + 1
    
    out.write('## Controller Usage\n')
    for ctrl, count in controllers.items():
        pct = 100 * count / len(data)
        out.write(f'- **{ctrl}**: {count} steps ({pct:.1f}%)\n')

print('Report generated: RESULTS_REPORT.md')
"
```

---

## 🔄 Git Workflow (If Using Version Control)

```bash
# Initialize repo
git init
git add .
git commit -m "Initial commit: Week 1 complete"

# Create branch for Week 2
git checkout -b week2-enhanced-testing

# Make changes, commit
git add .
git commit -m "Add GPS degradation scenario"

# Merge back
git checkout main
git merge week2-enhanced-testing
```

---

## 📅 Daily Checklist

### Morning Routine
- [ ] `cd /home/claude/adaptive_drone_contracts`
- [ ] `export PYTHONPATH=/home/claude/adaptive_drone_contracts/src:$PYTHONPATH`
- [ ] `./test_all.sh` (verify system still works)

### Development
- [ ] Make changes to code
- [ ] `python3 src/<modified_file>.py` (test component)
- [ ] `python3 simulation_demo.py` (test integration)
- [ ] `eog simulation_results.png` (check results)

### End of Day
- [ ] Run full test suite: `./test_all.sh`
- [ ] Archive results: `tar -czf day_X_results.tar.gz *.png *.json`
- [ ] Update progress notes

---

## 🎯 Week 2 Goals Quick Reference

### Day 8-10: GPS Degradation
```bash
# Modify simulation_demo.py
# - Add GPS satellite count to phase definition
# - Update sensor generation logic
# Run: python3 simulation_demo.py
```

### Day 11-12: Multiple Waypoints
```bash
# Add to adaptive_control_system.py
# - Waypoint list
# - Waypoint switching logic
# Run: python3 simulation_demo.py
```

### Day 13-14: Data Collection
```bash
# Create batch script
for i in {1..50}; do
    python3 simulation_demo.py > logs/flight_$i.log 2>&1
    mv simulation_results.png results/flight_$i.png
    mv flight_log.json logs/flight_$i.json
done
```

---

## 📞 Quick Help

**Stuck? Check these files:**
1. `README.md` - Complete system documentation
2. `WEEK1_SUMMARY.md` - What we built
3. `INNOVATION_SUMMARY.md` - Why it's novel
4. `PROJECT_SUMMARY.md` - Overall status

**Common Issues:**
- Import errors → Set PYTHONPATH
- Plot not showing → Install matplotlib (`pip install matplotlib --break-system-packages`)
- JSON errors → Check file paths

---

**Last Updated**: November 24, 2025  
**Status**: Week 1 Complete - Ready for Week 2 Development
