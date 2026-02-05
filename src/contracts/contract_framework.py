"""
Hierarchical Contract Framework for Adaptive Drone Control
Uses Pacti library for formal Assume-Guarantee contract composition

Key Concepts:
- Every component has a contract (assumptions → guarantees)
- Contracts compose to prove system-level properties
- Runtime monitoring verifies contracts hold
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

try:
    from pacti.contracts import PolyhedralIoContract
    PACTI_AVAILABLE = True
except ImportError:
    PACTI_AVAILABLE = False
    print("WARNING: Pacti not available. Using simplified contracts.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContractStatus(Enum):
    """Status of contract checking"""
    SATISFIED = "satisfied"
    ASSUMPTIONS_VIOLATED = "assumptions_violated"
    GUARANTEES_VIOLATED = "guarantees_violated"
    BOTH_VIOLATED = "both_violated"
    UNKNOWN = "unknown"


@dataclass
class ContractMetrics:
    """Metrics for evaluating contract satisfaction"""
    timestamp: float
    component_name: str
    assumptions_met: bool
    guarantees_met: bool
    assumption_values: Dict[str, float] = field(default_factory=dict)
    guarantee_values: Dict[str, float] = field(default_factory=dict)
    margin: Optional[float] = None  # How close to violation (0 = at boundary, >0 = safe)


class SimpleContract:
    """
    Simplified contract for when Pacti is not available
    Uses threshold-based checking
    """
    def __init__(self, 
                 name: str,
                 assumptions: Dict[str, Tuple[float, float]],  # {var: (min, max)}
                 guarantees: Dict[str, Tuple[float, float]]):
        self.name = name
        self.assumptions = assumptions
        self.guarantees = guarantees
        
    def check_assumptions(self, values: Dict[str, float]) -> Tuple[bool, Dict[str, float]]:
        """Check if assumptions are satisfied"""
        margins = {}
        all_met = True
        
        for var, (min_val, max_val) in self.assumptions.items():
            if var in values:
                val = values[var]
                # Compute margin (positive = safe, negative = violated)
                margin_min = val - min_val
                margin_max = max_val - val
                margins[var] = min(margin_min, margin_max)
                
                if val < min_val or val > max_val:
                    all_met = False
            else:
                all_met = False
                margins[var] = -float('inf')
                
        return all_met, margins
    
    def check_guarantees(self, values: Dict[str, float]) -> Tuple[bool, Dict[str, float]]:
        """Check if guarantees are satisfied"""
        margins = {}
        all_met = True
        
        for var, (min_val, max_val) in self.guarantees.items():
            if var in values:
                val = values[var]
                margin_min = val - min_val
                margin_max = max_val - val
                margins[var] = min(margin_min, margin_max)
                
                if val < min_val or val > max_val:
                    all_met = False
            else:
                all_met = False
                margins[var] = -float('inf')
                
        return all_met, margins
    
    def compose(self, other: 'SimpleContract') -> 'SimpleContract':
        """
        Simple composition: Connect guarantees of self to assumptions of other
        This is a simplified version of proper contract composition
        """
        # New assumptions = self.assumptions + (other.assumptions not covered by self.guarantees)
        new_assumptions = self.assumptions.copy()
        for var, bounds in other.assumptions.items():
            if var not in self.guarantees:
                new_assumptions[var] = bounds
        
        # New guarantees = other.guarantees
        new_guarantees = other.guarantees.copy()
        
        return SimpleContract(
            name=f"{self.name}>>{other.name}",
            assumptions=new_assumptions,
            guarantees=new_guarantees
        )


class HierarchicalContractMonitor:
    """
    Monitors hierarchical contract composition for the entire system
    
    Pipeline: Sensors → Estimator → Controller → Actuators → Dynamics
    Each component has contracts, and we compose them for system-level guarantees
    """
    
    def __init__(self, use_pacti: bool = False):
        self.use_pacti = use_pacti and PACTI_AVAILABLE
        
        # Component contracts
        self.sensor_contract = None
        self.estimator_contract = None
        self.controller_contracts = {}  # Multiple controllers
        self.actuator_contract = None
        
        # Composed contracts
        self.pipeline_contracts = {}  # For each controller
        self.current_pipeline = None
        
        # Monitoring history
        self.history: List[ContractMetrics] = []
        
        logger.info(f"Contract Monitor initialized (Pacti: {self.use_pacti})")
    
    def define_contracts(self):
        """Define all component contracts"""
        
        # SENSOR CONTRACT
        # Assumes: Hardware functioning
        # Guarantees: Measurement quality bounds
        self.sensor_contract = SimpleContract(
            name="Sensors",
            assumptions={
                "gps_satellites": (6.0, 30.0),
                "imu_temperature_stable": (0.0, 1.0),  # Boolean-like
            },
            guarantees={
                "gps_accuracy": (0.0, 3.0),  # meters
                "imu_noise": (0.0, 0.15),  # rad/s
                "measurement_rate": (50.0, 500.0),  # Hz
            }
        )
        
        # ESTIMATOR CONTRACT (EKF)
        # Assumes: Sensor quality bounds
        # Guarantees: State estimation accuracy
        self.estimator_contract = SimpleContract(
            name="Estimator",
            assumptions={
                "gps_accuracy": (0.0, 3.0),
                "imu_noise": (0.0, 0.15),
                "measurement_rate": (50.0, 500.0),
            },
            guarantees={
                "position_error": (0.0, 2.0),  # meters
                "velocity_error": (0.0, 0.5),  # m/s
                "attitude_error": (0.0, 0.1),  # radians (~5.7 deg)
            }
        )
        
        # CONTROLLER CONTRACTS
        # PID Controller
        self.controller_contracts['PID'] = SimpleContract(
            name="PID_Controller",
            assumptions={
                "position_error": (0.0, 3.0),  # tracking error (lateral drift weighted)
                "velocity_error": (0.0, 2.0),  # m/s lateral drift tolerance
                "wind_speed": (0.0, 3.0),  # m/s - primary switching criterion
                "disturbance": (0.0, 3.0),  # Nm (torque magnitude)
            },
            guarantees={
                "tracking_error": (0.0, 1.0),  # meters
                "settling_time": (0.0, 5.0),  # seconds
                "control_effort": (0.0, 0.8),  # Normalized [0,1]
            }
        )
        
        # H-infinity Controller (Emergency)
        self.controller_contracts['Hinf'] = SimpleContract(
            name="Hinf_Controller",
            assumptions={
                # No assumptions - always available (safety net!)
            },
            guarantees={
                "stabilization_time": (0.0, 10.0),  # seconds
                "tilt_angle": (0.0, 0.35),  # radians (~20 deg)
                "altitude_safe": (3.0, 50.0),  # meters
            }
        )
        
        # MPC Controller (for future)
        self.controller_contracts['MPC'] = SimpleContract(
            name="MPC_Controller",
            assumptions={
                "position_error": (0.0, 5.0),
                "velocity_error": (0.0, 2.0),
                "wind_speed": (0.0, 8.0),
                "computation_time": (0.0, 0.05),  # 50ms max
            },
            guarantees={
                "tracking_error": (0.0, 0.5),
                "settling_time": (0.0, 8.0),
                "constraints_satisfied": (1.0, 1.0),  # Boolean
            }
        )
        
        # ACTUATOR CONTRACT
        # Assumes: Valid control commands
        # Guarantees: Thrust delivery accuracy
        self.actuator_contract = SimpleContract(
            name="Actuators",
            assumptions={
                "control_effort": (0.0, 1.0),
                "battery_voltage": (11.0, 12.6),  # 3S LiPo
                "motor_temperature": (20.0, 80.0),  # Celsius
            },
            guarantees={
                "thrust_accuracy": (0.9, 1.1),  # Ratio (1.0 = perfect)
                "response_time": (0.01, 0.05),  # seconds
            }
        )
        
        logger.info("All component contracts defined")
    
    def compose_pipeline(self, controller_name: str) -> SimpleContract:
        """
        Compose sensor → estimator → controller → actuator pipeline
        Returns composed contract for entire pipeline
        """
        if controller_name not in self.controller_contracts:
            logger.error(f"Controller {controller_name} not defined!")
            return None
        
        # Compose step by step
        pipeline = self.sensor_contract
        pipeline = pipeline.compose(self.estimator_contract)
        pipeline = pipeline.compose(self.controller_contracts[controller_name])
        pipeline = pipeline.compose(self.actuator_contract)
        
        logger.info(f"Pipeline composed for {controller_name}")
        logger.info(f"  Assumptions: {list(pipeline.assumptions.keys())}")
        logger.info(f"  Guarantees: {list(pipeline.guarantees.keys())}")
        
        return pipeline
    
    def verify_mission_feasibility(self, 
                                   controller_name: str,
                                   current_conditions: Dict[str, float]) -> Tuple[bool, str]:
        """
        BEFORE TAKEOFF: Check if mission is feasible given current conditions
        This is the key innovation - formal verification before flight!
        """
        pipeline = self.compose_pipeline(controller_name)
        
        if pipeline is None:
            return False, "Pipeline composition failed"
        
        # Check if current conditions satisfy pipeline assumptions
        assumptions_met, margins = pipeline.check_assumptions(current_conditions)
        
        if assumptions_met:
            min_margin = min(margins.values()) if margins else 0.0
            return True, f"Mission feasible with {controller_name} (margin: {min_margin:.2f})"
        else:
            violated = [k for k, v in margins.items() if v < 0]
            return False, f"Assumptions violated: {violated}"
    
    def monitor_runtime(self, 
                       component: str,
                       values: Dict[str, float],
                       timestamp: float) -> ContractMetrics:
        """
        DURING FLIGHT: Monitor if contracts are satisfied at runtime
        """
        contract = None
        
        if component == "sensors":
            contract = self.sensor_contract
        elif component == "estimator":
            contract = self.estimator_contract
        elif component.startswith("controller_"):
            ctrl_name = component.split("_")[1]
            contract = self.controller_contracts.get(ctrl_name)
        elif component == "actuators":
            contract = self.actuator_contract
        
        if contract is None:
            logger.warning(f"Unknown component: {component}")
            return ContractMetrics(
                timestamp=timestamp,
                component_name=component,
                assumptions_met=False,
                guarantees_met=False
            )
        
        # Check assumptions and guarantees
        assumptions_met, assumption_margins = contract.check_assumptions(values)
        guarantees_met, guarantee_margins = contract.check_guarantees(values)
        
        # Compute overall margin (worst case)
        all_margins = {**assumption_margins, **guarantee_margins}
        min_margin = min(all_margins.values()) if all_margins else 0.0
        
        metrics = ContractMetrics(
            timestamp=timestamp,
            component_name=component,
            assumptions_met=assumptions_met,
            guarantees_met=guarantees_met,
            assumption_values=values.copy(),
            guarantee_values=values.copy(),
            margin=min_margin
        )
        
        self.history.append(metrics)
        
        return metrics
    
    def get_contract_status(self, component: str) -> ContractStatus:
        """Get current contract status for a component"""
        if not self.history:
            return ContractStatus.UNKNOWN
        
        # Get latest metrics for this component
        recent = [m for m in self.history if m.component_name == component]
        if not recent:
            return ContractStatus.UNKNOWN
        
        latest = recent[-1]
        
        if latest.assumptions_met and latest.guarantees_met:
            return ContractStatus.SATISFIED
        elif not latest.assumptions_met and latest.guarantees_met:
            return ContractStatus.ASSUMPTIONS_VIOLATED
        elif latest.assumptions_met and not latest.guarantees_met:
            return ContractStatus.GUARANTEES_VIOLATED
        else:
            return ContractStatus.BOTH_VIOLATED
    
    def should_switch_controller(self, 
                                current_controller: str,
                                system_state: Dict[str, float]) -> Tuple[bool, Optional[str], str]:
        """
        Determine if controller should switch based on contract violations
        Returns: (should_switch, new_controller, reason)
        """
        # Check current controller's contract
        current_contract = self.controller_contracts.get(current_controller)
        if current_contract is None:
            return False, None, "Current controller not found"
        
        assumptions_met, margins = current_contract.check_assumptions(system_state)
        
        if assumptions_met:
            # Current controller OK, but check if more efficient option available
            # Priority: PID (efficient) > MPC (optimal) > Hinf (safe)
            
            if current_controller == 'Hinf':
                # Try to switch back to more efficient controllers
                pid_contract = self.controller_contracts['PID']
                pid_ok, _ = pid_contract.check_assumptions(system_state)
                if pid_ok:
                    return True, 'PID', "Conditions improved, switching back to efficient PID"
                
                mpc_contract = self.controller_contracts.get('MPC')
                if mpc_contract:
                    mpc_ok, _ = mpc_contract.check_assumptions(system_state)
                    if mpc_ok:
                        return True, 'MPC', "Conditions improved, switching to optimal MPC"
            
            elif current_controller == 'MPC':
                # Try to switch to PID if even more efficient
                pid_contract = self.controller_contracts['PID']
                pid_ok, _ = pid_contract.check_assumptions(system_state)
                if pid_ok:
                    return True, 'PID', "Conditions normal, switching to more efficient PID"
            
            # Current controller is the best option
            return False, None, "Current controller assumptions satisfied"
        
        # Current controller assumptions violated - need to switch
        violated_vars = [k for k, v in margins.items() if v < 0]
        
        # Try other controllers in priority order
        # Priority: PID (efficient) > MPC (optimal) > Hinf (safe)
        candidates = ['PID', 'MPC', 'Hinf']
        if current_controller in candidates:
            candidates.remove(current_controller)
        
        for candidate in candidates:
            candidate_contract = self.controller_contracts[candidate]
            candidate_ok, _ = candidate_contract.check_assumptions(system_state)
            
            if candidate_ok:
                reason = f"{current_controller} violated {violated_vars}, switching to {candidate}"
                return True, candidate, reason
        
        # If no controller assumptions satisfied, force Hinf (no assumptions)
        if current_controller != 'Hinf':
            reason = f"Emergency: All controllers violated, forcing Hinf"
            return True, 'Hinf', reason
        
        return False, None, "Already in Hinf, no alternatives"
    
    def export_metrics(self, filename: str):
        """Export contract monitoring history for analysis"""
        import json
        
        data = {
            "history": [
                {
                    "timestamp": m.timestamp,
                    "component": m.component_name,
                    "assumptions_met": m.assumptions_met,
                    "guarantees_met": m.guarantees_met,
                    "margin": m.margin,
                    "values": m.assumption_values
                }
                for m in self.history
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported {len(self.history)} metrics to {filename}")


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("Hierarchical Contract Framework Test")
    print("=" * 60)
    
    # Create monitor
    monitor = HierarchicalContractMonitor()
    monitor.define_contracts()
    
    # Test 1: Pre-flight verification (NOMINAL)
    print("\n[TEST 1] Pre-flight Check - Nominal Conditions")
    nominal_conditions = {
        "gps_satellites": 12.0,
        "imu_temperature_stable": 1.0,
        "battery_voltage": 12.4,
        "motor_temperature": 25.0,
        "wind_speed": 1.5,
        "disturbance": 0.5,
    }
    
    feasible, msg = monitor.verify_mission_feasibility("PID", nominal_conditions)
    print(f"  Result: {msg}")
    print(f"  Feasible: {feasible}")
    
    # Test 2: Pre-flight verification (HIGH WIND)
    print("\n[TEST 2] Pre-flight Check - High Wind")
    windy_conditions = {
        "gps_satellites": 12.0,
        "imu_temperature_stable": 1.0,
        "battery_voltage": 12.4,
        "motor_temperature": 25.0,
        "wind_speed": 5.0,  # Exceeds PID assumption
        "disturbance": 3.0,
    }
    
    feasible, msg = monitor.verify_mission_feasibility("PID", windy_conditions)
    print(f"  Result: {msg}")
    print(f"  Feasible: {feasible}")
    
    # Try with Hinf instead
    feasible, msg = monitor.verify_mission_feasibility("Hinf", windy_conditions)
    print(f"  With Hinf: {msg}")
    
    # Test 3: Runtime monitoring
    print("\n[TEST 3] Runtime Monitoring")
    runtime_state = {
        "position_error": 1.5,
        "velocity_error": 0.3,
        "wind_speed": 2.5,
        "disturbance": 1.5,
    }
    
    metrics = monitor.monitor_runtime("controller_PID", runtime_state, timestamp=1.0)
    print(f"  Component: {metrics.component_name}")
    print(f"  Assumptions met: {metrics.assumptions_met}")
    print(f"  Guarantees met: {metrics.guarantees_met}")
    print(f"  Margin: {metrics.margin:.3f}")
    
    # Test 4: Controller switching logic
    print("\n[TEST 4] Controller Switching")
    degraded_state = {
        "position_error": 3.0,  # Exceeds PID assumption
        "velocity_error": 0.8,
        "wind_speed": 4.0,
        "disturbance": 3.5,
    }
    
    should_switch, new_ctrl, reason = monitor.should_switch_controller("PID", degraded_state)
    print(f"  Should switch: {should_switch}")
    print(f"  New controller: {new_ctrl}")
    print(f"  Reason: {reason}")
    
    print("\n" + "=" * 60)
    print("[OK] Contract Framework Test Complete")
    print("=" * 60)
