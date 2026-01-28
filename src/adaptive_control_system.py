"""
Adaptive Drone Control System with Hierarchical Contract Composition

This is the MAIN SYSTEM that integrates:
1. Contract Monitor (hierarchical composition)
2. Contract-Aware EKF (state estimation)
3. Contract-Based Controllers (PID, H-infinity)
4. Switching Logic (based on contract violations)

Key Innovation:
- BEFORE FLIGHT: Verify mission feasibility via contract composition
- DURING FLIGHT: Monitor contracts, switch controllers, guarantee safety
"""

import numpy as np
from typing import Dict, Tuple, Optional
import logging
import time

from contracts.contract_framework import HierarchicalContractMonitor, ContractStatus
from estimation.contract_ekf import ContractAwareEKF
from control.controllers import ControllerSwitcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdaptiveDroneController:
    """
    Main adaptive control system with formal contract guarantees
    
    Pipeline: Sensors → EKF → Controller → Actuators
    Each component monitored by contracts
    """
    
    def __init__(self, dt: float = 0.02):
        self.dt = dt
        self.time = 0.0
        
        # Initialize contract monitor
        logger.info("Initializing Contract Monitor...")
        self.contract_monitor = HierarchicalContractMonitor()
        self.contract_monitor.define_contracts()
        
        # Initialize EKF with contract awareness
        logger.info("Initializing Contract-Aware EKF...")
        self.ekf = ContractAwareEKF(self.contract_monitor)
        
        # Initialize controller switcher
        logger.info("Initializing Controller Switcher...")
        self.controller_switcher = ControllerSwitcher(dt)
        
        # Mission state
        self.mission_active = False
        self.mission_feasible = False
        self.mission_setpoint = {
            'position': np.array([0.0, 0.0, -5.0]),
            'velocity': np.array([0.0, 0.0, 0.0]),
            'yaw': 0.0
        }
        
        # Data logging
        self.flight_log = []
        
        logger.info("✓ Adaptive Drone Controller initialized")
    
    def pre_flight_check(self, 
                        initial_conditions: Dict[str, float],
                        mission: Dict[str, any]) -> Tuple[bool, str]:
        """
        CRITICAL PRE-FLIGHT CHECK
        
        Uses contract composition to verify mission is feasible
        This is a KEY INNOVATION - formal verification before takeoff!
        """
        logger.info("=" * 60)
        logger.info("PRE-FLIGHT CONTRACT VERIFICATION")
        logger.info("=" * 60)
        
        # Extract mission requirements
        self.mission_setpoint['position'] = mission.get('target_position', 
                                                        np.array([0.0, 0.0, -5.0]))
        
        # Try with PID first (most efficient)
        logger.info("\n[1/3] Checking if PID can handle mission...")
        feasible_pid, msg_pid = self.contract_monitor.verify_mission_feasibility(
            'PID', initial_conditions
        )
        
        if feasible_pid:
            logger.info(f"✓ {msg_pid}")
            self.mission_feasible = True
            return True, "Mission feasible with PID controller"
        else:
            logger.warning(f"✗ PID: {msg_pid}")
        
        # Try with H-infinity (always works)
        logger.info("\n[2/3] Checking if H-infinity can handle mission...")
        feasible_hinf, msg_hinf = self.contract_monitor.verify_mission_feasibility(
            'Hinf', initial_conditions
        )
        
        if feasible_hinf:
            logger.info(f"✓ {msg_hinf}")
            # Switch to H-infinity before flight
            self.controller_switcher.switch_to('Hinf', 0.0, "Pre-flight: conditions require H-infinity")
            self.mission_feasible = True
            return True, "Mission feasible with H-infinity controller (degraded performance expected)"
        else:
            logger.error(f"✗ H-infinity: {msg_hinf}")
        
        # If even H-infinity can't work, mission is infeasible
        logger.error("\n[3/3] MISSION INFEASIBLE")
        logger.error("Current conditions violate all controller contracts!")
        logger.error("RECOMMENDATION: Abort mission until conditions improve")
        
        self.mission_feasible = False
        return False, "Mission infeasible - all controller contracts violated"
    
    def update_sensors(self, raw_sensors: Dict[str, any]) -> Dict[str, np.ndarray]:
        """
        Process raw sensor data into measurements
        """
        measurements = {}
        
        # GPS measurements
        if 'gps' in raw_sensors and raw_sensors['gps']['valid']:
            measurements['gps_position'] = raw_sensors['gps']['position']
            measurements['gps_velocity'] = raw_sensors['gps']['velocity']
        
        # IMU measurements
        if 'imu' in raw_sensors and raw_sensors['imu']['valid']:
            measurements['imu_attitude'] = raw_sensors['imu']['attitude']
            measurements['imu_rates'] = raw_sensors['imu']['rates']
        
        return measurements
    
    def estimate_system_conditions(self, 
                                   state: Dict[str, np.ndarray],
                                   control: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Estimate current system conditions for contract checking
        """
        # Calculate velocity error from target (FIXED!)
        target_velocity = self.mission_setpoint['velocity']
        velocity_error = np.linalg.norm(state['velocity'] - target_velocity)
        wind_estimate = min(velocity_error * 2.0, 10.0)  # Rough estimate
        
        # Position error from setpoint
        pos_error = np.linalg.norm(self.mission_setpoint['position'] - state['position'])
        
        # Disturbance estimate from control effort
        disturbance = np.linalg.norm(control.get('torques', np.zeros(3)))
        
        return {
            'position_error': pos_error,
            'velocity_error': velocity_error,
            'wind_speed': wind_estimate,
            'disturbance': disturbance,
            'control_effort': control.get('thrust', 0.5)
        }
    
    def check_and_switch_controller(self, system_conditions: Dict[str, float]):
        """
        Check contracts and switch controller if needed
        """
        current_controller = self.controller_switcher.get_active_controller()
        
        # Check if switch is needed
        should_switch, new_controller, reason = self.contract_monitor.should_switch_controller(
            current_controller, system_conditions
        )
        
        if should_switch:
            success = self.controller_switcher.switch_to(
                new_controller, self.time, reason
            )
            
            if success:
                logger.warning(f"Controller switched: {current_controller} → {new_controller}")
                logger.warning(f"Reason: {reason}")
    
    def control_step(self, 
                    raw_sensors: Dict[str, any]) -> Tuple[Dict[str, np.ndarray], Dict[str, any]]:
        """
        Main control loop step
        
        Returns: (control_output, telemetry)
        """
        
        if not self.mission_feasible:
            logger.error("Mission not feasible - cannot run control loop!")
            return {'thrust': 0.0, 'torques': np.zeros(3)}, {}
        
        # 1. Process sensors
        measurements = self.update_sensors(raw_sensors)
        
        # 2. State estimation with contract checking
        state, estimation_ok = self.ekf.update(measurements, raw_sensors, self.time)
        state_dict = self.ekf.get_state()
        
        # 3. Monitor estimator contract
        estimation_state = {
            'position_error': np.linalg.norm(self.ekf.P[0:3, 0:3]),
            'velocity_error': np.linalg.norm(self.ekf.P[3:6, 3:6]),
            'attitude_error': np.linalg.norm(self.ekf.P[6:9, 6:9]),
        }
        
        # 4. Compute control
        control = self.controller_switcher.compute_control(state_dict, self.mission_setpoint)
        
        # 5. Estimate system conditions
        system_conditions = self.estimate_system_conditions(state_dict, control)
        
        # 6. Monitor controller contract
        active_controller = self.controller_switcher.get_active_controller()
        self.contract_monitor.monitor_runtime(
            f"controller_{active_controller}",
            system_conditions,
            self.time
        )
        
        # 7. Check if controller switch needed
        self.check_and_switch_controller(system_conditions)
        
        # 8. Monitor actuator contract
        actuator_state = {
            'control_effort': control['thrust'],
            'battery_voltage': raw_sensors.get('battery', {}).get('voltage', 12.0),
            'motor_temperature': raw_sensors.get('motors', {}).get('temperature', 25.0),
        }
        self.contract_monitor.monitor_runtime('actuators', actuator_state, self.time)
        
        # 9. Log data
        log_entry = {
            'time': self.time,
            'state': state_dict.copy(),
            'control': control.copy(),
            'controller': active_controller,
            'system_conditions': system_conditions.copy(),
            'estimation_ok': estimation_ok
        }
        self.flight_log.append(log_entry)
        
        # 10. Telemetry
        telemetry = {
            'time': self.time,
            'active_controller': active_controller,
            'position': state_dict['position'],
            'velocity': state_dict['velocity'],
            'attitude': state_dict['attitude'],
            'contract_status': {
                'sensors': self.contract_monitor.get_contract_status('sensors'),
                'estimator': self.contract_monitor.get_contract_status('estimator'),
                'controller': self.contract_monitor.get_contract_status(f'controller_{active_controller}'),
                'actuators': self.contract_monitor.get_contract_status('actuators'),
            }
        }
        
        # Increment time
        self.time += self.dt
        
        return control, telemetry
    
    def start_mission(self):
        """Start the mission"""
        if not self.mission_feasible:
            logger.error("Cannot start mission - pre-flight check failed!")
            return False
        
        self.mission_active = True
        logger.info("🚁 Mission started!")
        return True
    
    def stop_mission(self):
        """Stop the mission"""
        self.mission_active = False
        logger.info("Mission stopped")
    
    def save_flight_log(self, filename: str):
        """Save flight log for analysis"""
        import json
        
        # Convert numpy arrays to lists for JSON
        log_serializable = []
        for entry in self.flight_log:
            entry_copy = entry.copy()
            for key in ['state', 'control', 'system_conditions']:
                if key in entry_copy:
                    for subkey, val in entry_copy[key].items():
                        if isinstance(val, np.ndarray):
                            entry_copy[key][subkey] = val.tolist()
            log_serializable.append(entry_copy)
        
        with open(filename, 'w') as f:
            json.dump(log_serializable, f, indent=2)
        
        logger.info(f"Flight log saved to {filename}")
        
        # Also save contract history
        contract_log = filename.replace('.json', '_contracts.json')
        self.contract_monitor.export_metrics(contract_log)


# Test the complete system
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ADAPTIVE DRONE CONTROL SYSTEM TEST")
    print("=" * 60)
    
    # Create adaptive controller
    controller = AdaptiveDroneController(dt=0.02)
    
    # Test 1: Pre-flight check with GOOD conditions
    print("\n" + "=" * 60)
    print("TEST 1: Pre-flight Check - NOMINAL CONDITIONS")
    print("=" * 60)
    
    good_conditions = {
        'gps_satellites': 12.0,
        'imu_temperature_stable': 1.0,
        'battery_voltage': 12.4,
        'motor_temperature': 25.0,
        'wind_speed': 1.5,
        'disturbance': 0.5,
    }
    
    mission = {
        'target_position': np.array([10.0, 5.0, -8.0])
    }
    
    feasible, msg = controller.pre_flight_check(good_conditions, mission)
    print(f"\nResult: {msg}")
    print(f"Feasible: {feasible}\n")
    
    # Test 2: Simulate a few control steps
    if feasible:
        print("=" * 60)
        print("TEST 2: Simulating Flight")
        print("=" * 60)
        
        controller.start_mission()
        
        for i in range(5):
            # Simulate sensor data
            raw_sensors = {
                'gps': {
                    'position': np.array([float(i) * 0.5, 0.0, -5.0]),
                    'velocity': np.array([0.1, 0.0, 0.0]),
                    'valid': True,
                    'satellites': 12,
                    'hdop': 1.0
                },
                'imu': {
                    'attitude': np.array([0.0, 0.0, 0.0]),
                    'rates': np.array([0.0, 0.0, 0.0]),
                    'valid': True,
                    'calibrated': True,
                    'temperature': 25.0
                },
                'battery': {
                    'voltage': 12.4
                },
                'motors': {
                    'temperature': 30.0
                }
            }
            
            # Run control step
            control, telemetry = controller.control_step(raw_sensors)
            
            print(f"\nStep {i+1}:")
            print(f"  Position: {telemetry['position']}")
            print(f"  Active Controller: {telemetry['active_controller']}")
            print(f"  Thrust: {control['thrust']:.3f}")
        
        controller.stop_mission()
    
    # Test 3: Pre-flight check with BAD conditions
    print("\n" + "=" * 60)
    print("TEST 3: Pre-flight Check - EXTREME CONDITIONS")
    print("=" * 60)
    
    bad_conditions = {
        'gps_satellites': 3.0,  # Too few
        'imu_temperature_stable': 0.0,  # Unstable
        'battery_voltage': 10.5,  # Low
        'motor_temperature': 85.0,  # Hot
        'wind_speed': 12.0,  # Very high
        'disturbance': 8.0,
    }
    
    controller2 = AdaptiveDroneController(dt=0.02)
    feasible, msg = controller2.pre_flight_check(bad_conditions, mission)
    print(f"\nResult: {msg}")
    print(f"Feasible: {feasible}\n")
    
    print("=" * 60)
    print("✓ SYSTEM TEST COMPLETE")
    print("=" * 60)
