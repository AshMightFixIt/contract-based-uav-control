"""
Adaptive Drone Control System with Hierarchical Contract Composition

Pipeline: Sensors -> EKF -> [Horizon Planner] -> Supervisor -> [PID | MPC | H-inf] -> CBF Filter -> Actuators

Three-tier graduated degradation:
  PID  (wind <= 3 m/s)  —  efficient, nominal conditions
  MPC  (wind <= 8 m/s)  —  optimal, moderate conditions
  H-inf (wind <= 25 m/s) — robust, emergency stabilization

Key Innovation:
- BEFORE FLIGHT: Verify mission feasibility via contract composition
- DURING FLIGHT: Horizon planner predicts safety margins, triggers re-planning
  Supervisor manages flight modes, CBF filter enforces hard safety constraints
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
import logging
import time

from contracts.contract_framework import HierarchicalContractMonitor, ContractStatus
from estimation.contract_ekf import ContractAwareEKF
from control.controllers import ControllerSwitcher
from control.flight_mode_supervisor import FlightModeSupervisor, FlightMode
from safety.cbf_filter import CBFSafetyFilter, RuntimeMonitor

# Optional: Horizon-based Pacti planner
try:
    from planning.integrated_planner import IntegratedPlanner
    from planning.horizon_planner import PlannerConfig
    HORIZON_PLANNER_AVAILABLE = True
except ImportError:
    HORIZON_PLANNER_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdaptiveDroneController:
    """
    Main adaptive control system with formal contract guarantees.

    Architecture:
        Sensors -> EKF -> Supervisor -> Controller -> CBF -> Actuators
                          (setpoint +   (PID or      (safety
                           ctrl pick)    H-inf)       filter)
    """

    def __init__(self, dt: float = 0.02, use_horizon_planner: bool = True):
        self.dt = dt
        self.time = 0.0

        # Contract monitor
        logger.info("Initializing Contract Monitor...")
        self.contract_monitor = HierarchicalContractMonitor()
        self.contract_monitor.define_contracts()

        # State estimation
        logger.info("Initializing Contract-Aware EKF...")
        self.ekf = ContractAwareEKF(self.contract_monitor)

        # Controller switcher (PID + H-inf)
        logger.info("Initializing Controller Switcher...")
        self.controller_switcher = ControllerSwitcher(dt)

        # Flight mode supervisor
        logger.info("Initializing Flight Mode Supervisor...")
        self.supervisor = FlightModeSupervisor(dt)

        # CBF safety filter
        logger.info("Initializing CBF Safety Filter...")
        self.cbf_filter = CBFSafetyFilter()
        self.runtime_monitor = RuntimeMonitor()

        # Horizon-based Pacti planner (optional)
        self.horizon_planner = None
        self.use_horizon_planner = use_horizon_planner and HORIZON_PLANNER_AVAILABLE
        if self.use_horizon_planner:
            logger.info("Initializing Horizon Planner (Pacti)...")
            planner_config = PlannerConfig(
                max_horizon=10,
                min_horizon=3,
                dt=0.1,  # Coarser planning timestep
                safety_limit=5.0,
                margin_threshold=1.0,
                replan_threshold=0.5
            )
            self.horizon_planner = IntegratedPlanner(planner_config)
        else:
            if use_horizon_planner and not HORIZON_PLANNER_AVAILABLE:
                logger.warning("Horizon planner requested but Pacti not available")

        # Mission state
        self.mission_active = False
        self.mission_feasible = False
        self.current_target = np.array([0.0, 0.0, -5.0])

        # Wind estimation with low-pass filter (prevents noise from triggering emergency)
        self.wind_estimate_filtered = 0.0
        self.wind_filter_alpha = 0.15  # Moderate smoothing - responsive but stable

        # Data logging
        self.flight_log = []

        logger.info("Adaptive Drone Controller initialized")

    def pre_flight_check(self,
                        initial_conditions: Dict[str, float],
                        mission: Dict[str, any]) -> Tuple[bool, str]:
        """
        Pre-flight contract verification.
        Uses contract composition to verify mission is feasible.
        Sets up the supervisor with the mission target.
        """
        logger.info("=" * 60)
        logger.info("PRE-FLIGHT CONTRACT VERIFICATION")
        logger.info("=" * 60)

        target = mission.get('target_position', np.array([0.0, 0.0, -5.0]))
        waypoints = mission.get('waypoints', [target])

        # Try controllers in order: PID (efficient) → MPC (optimal) → H-inf (robust)
        controllers = [
            ('PID', "efficient PID"),
            ('MPC', "optimal MPC"),
            ('Hinf', "robust H-infinity"),
        ]

        for i, (name, desc) in enumerate(controllers, 1):
            logger.info(f"\n[{i}/{len(controllers)}] Checking {desc}...")
            # MPC pipeline needs computation_time in conditions
            check_conditions = initial_conditions.copy()
            if name == 'MPC' and 'computation_time' not in check_conditions:
                check_conditions['computation_time'] = 0.01  # typical solve time

            feasible, msg = self.contract_monitor.verify_mission_feasibility(
                name, check_conditions
            )

            if feasible:
                logger.info(f"{name} feasible: {msg}")
                if name != 'PID':
                    self.controller_switcher.switch_to(
                        name, 0.0, f"Pre-flight: conditions require {desc}")
                self.supervisor.set_waypoints(waypoints)
                self.mission_feasible = True
                perf_note = "" if name == 'PID' else " (degraded performance expected)"
                return True, f"Mission feasible with {desc}{perf_note}"
            else:
                logger.warning(f"{name} infeasible: {msg}")

        logger.error("MISSION INFEASIBLE - all controller contracts violated!")
        self.mission_feasible = False
        return False, "Mission infeasible - all controller contracts violated"

    def update_sensors(self, raw_sensors: Dict[str, any]) -> Dict[str, np.ndarray]:
        """Process raw sensor data into measurements."""
        measurements = {}

        if 'gps' in raw_sensors and raw_sensors['gps']['valid']:
            measurements['gps_position'] = raw_sensors['gps']['position']
            measurements['gps_velocity'] = raw_sensors['gps']['velocity']

        if 'imu' in raw_sensors and raw_sensors['imu']['valid']:
            measurements['imu_attitude'] = raw_sensors['imu']['attitude']
            measurements['imu_rates'] = raw_sensors['imu']['rates']

        return measurements

    def estimate_system_conditions(self,
                                   state: Dict[str, np.ndarray],
                                   setpoint: Dict,
                                   control: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Estimate current system conditions for contract checking.

        position_error: Tracking error metric, NOT distance to goal.
            - For a drone in transit, we check if it's moving toward the target.
            - If closing velocity is positive, tracking is fine (error = 0).
            - If closing velocity is negative or lateral drift is high, error grows.

        velocity_error: Lateral drift (velocity perpendicular to target direction).
            - Forward velocity towards target is intentional, not an error.
            - Only sideways drift counts as velocity error.

        wind_speed: Estimated from lateral drift (unexpected motion).

        disturbance: Estimated from control torques magnitude.
        """
        target_position = setpoint.get('position', state['position'])
        position_diff = target_position - state['position']
        distance_to_target = np.linalg.norm(position_diff)

        # Decompose velocity into forward (intentional) and lateral (error) components
        if distance_to_target > 0.1:
            direction_to_target = position_diff / distance_to_target
            closing_velocity = np.dot(state['velocity'], direction_to_target)
            # Lateral drift (velocity perpendicular to target direction)
            lateral_velocity = state['velocity'] - closing_velocity * direction_to_target
            lateral_drift = np.linalg.norm(lateral_velocity)

            # Tracking error: high if we're drifting or moving away
            # Low if we're closing in on target with minimal lateral drift
            if closing_velocity > 0.5:
                # Moving toward target at reasonable speed - good tracking
                tracking_error = lateral_drift * 0.5
            elif closing_velocity > 0:
                # Moving toward target slowly
                tracking_error = lateral_drift + (0.5 - closing_velocity)
            else:
                # Moving away from target - bad tracking
                tracking_error = lateral_drift + abs(closing_velocity) + 1.0

            # Velocity error is lateral drift only (forward motion is intentional)
            velocity_error = lateral_drift
        else:
            # Very close to target - use position error and total velocity
            tracking_error = distance_to_target
            target_velocity = setpoint.get('velocity', np.zeros(3))
            velocity_error = np.linalg.norm(state['velocity'] - target_velocity)
            lateral_drift = velocity_error

        # Wind estimate from lateral drift (unexpected sideways motion)
        # Use low-pass filter to prevent transient high velocities from triggering false alarms
        raw_wind_estimate = min(max(lateral_drift - 0.5, 0.0) * 1.5, 10.0)
        self.wind_estimate_filtered = (
            self.wind_filter_alpha * raw_wind_estimate +
            (1 - self.wind_filter_alpha) * self.wind_estimate_filtered
        )
        wind_estimate = self.wind_estimate_filtered

        # Override with direct wind sensor if available (more accurate)
        # This is accessed in control_step via self.last_wind_sensor
        if hasattr(self, 'last_wind_sensor') and self.last_wind_sensor is not None:
            sensor_wind = self.last_wind_sensor
            # Use max of sensor reading and drift-based estimate
            wind_estimate = max(wind_estimate, sensor_wind)

        # Disturbance from control effort (also filtered for stability)
        raw_disturbance = np.linalg.norm(control.get('torques', np.zeros(3)))
        disturbance = min(raw_disturbance, 3.0)  # Cap disturbance estimate

        return {
            'position_error': tracking_error,
            'velocity_error': velocity_error,
            'wind_speed': wind_estimate,
            'disturbance': disturbance,
            'control_effort': control.get('thrust', 0.5)
        }

    def control_step(self,
                    raw_sensors: Dict[str, any]) -> Tuple[Dict[str, np.ndarray], Dict[str, any]]:
        """
        Main control loop step.

        Pipeline:
        1. Sensors -> measurements
        2. EKF update -> state estimate
        3. [HORIZON PLANNER] -> predictive safety margins, recommended controller
        4. Estimate conditions + check PID contract
        5. Supervisor -> setpoint + controller choice
        6. Controller -> nominal control
        7. CBF filter -> safe control
        8. Contract monitoring + logging

        Returns: (control_output, telemetry)
        """
        if not self.mission_feasible:
            logger.error("Mission not feasible - cannot run control loop!")
            return {'thrust': 0.0, 'torques': np.zeros(3)}, {}

        # 1. Process sensors
        measurements = self.update_sensors(raw_sensors)

        # Capture wind sensor reading if available
        if 'wind' in raw_sensors:
            self.last_wind_sensor = raw_sensors['wind'].get('speed', 0.0)
        else:
            self.last_wind_sensor = None

        # 2. State estimation (predict + update)
        self.ekf.predict(self.dt)  # Propagate state forward
        state, estimation_ok = self.ekf.update(measurements, raw_sensors, self.time)
        state_dict = self.ekf.get_state()

        # 3. GPS availability from sensor contracts
        gps_ok, imu_ok = self.ekf.check_sensor_contracts(self.time)

        # 4. Get current setpoint for condition estimation (before supervisor update)
        current_setpoint = self.supervisor.get_current_setpoint(state_dict, self.time)

        # 5. Estimate system conditions
        # Use a neutral control for initial estimation
        system_conditions = self.estimate_system_conditions(
            state_dict, current_setpoint, {'thrust': 0.5, 'torques': np.zeros(3)}
        )

        # 6. [NEW] Horizon Planner - predictive contract cascade
        planner_info = {}
        planner_emergency = False
        planner_controller = None

        if self.use_horizon_planner and self.horizon_planner:
            # Build environment state for planner
            env = {
                'wind_speed': system_conditions['wind_speed'],
                'gps_satellites': raw_sensors.get('gps', {}).get('satellites', 12),
                'gps_hdop': raw_sensors.get('gps', {}).get('hdop', 1.0),
                'battery_voltage': raw_sensors.get('battery', {}).get('voltage', 12.0),
            }

            # Update target from supervisor
            target = current_setpoint.get('position', self.current_target)

            # Run horizon planner
            planner_controller, planner_emergency, planner_info = self.horizon_planner.update(
                self.time, state_dict, env, target
            )

            # If planner says emergency, force H-inf
            if planner_emergency:
                logger.warning(f"Horizon planner: EMERGENCY MODE")

        # 7. Three-tier reactive contract check: PID > MPC > H-inf
        # Add computation_time for MPC contract check
        check_conditions = system_conditions.copy()
        mpc_ctrl = self.controller_switcher.controllers.get('MPC')
        if mpc_ctrl:
            check_conditions['computation_time'] = getattr(mpc_ctrl, 'last_solve_time', 0.01)

        recommended = 'Hinf'  # default safety net
        pid_contract = self.contract_monitor.controller_contracts.get('PID')
        mpc_contract = self.contract_monitor.controller_contracts.get('MPC')

        if pid_contract:
            pid_ok, _ = pid_contract.check_assumptions(check_conditions)
            if pid_ok:
                recommended = 'PID'
        if recommended != 'PID' and mpc_contract:
            mpc_ok, _ = mpc_contract.check_assumptions(check_conditions)
            if mpc_ok:
                recommended = 'MPC'

        # Combine with planner recommendation (planner can only escalate, not downgrade)
        controller_priority = {'PID': 0, 'MPC': 1, 'Hinf': 2}
        if planner_emergency:
            recommended = 'Hinf'
        elif planner_controller:
            planner_mapped = {'pid': 'PID', 'mpc': 'MPC', 'hinf': 'Hinf'}.get(
                planner_controller, planner_controller)
            if controller_priority.get(planner_mapped, 0) > controller_priority.get(recommended, 0):
                recommended = planner_mapped
                logger.info(f"Horizon planner: preemptive escalation to {recommended}")

        # 8. Supervisor: get setpoint + controller choice
        setpoint, controller_name = self.supervisor.update(
            state_dict, system_conditions, recommended, gps_ok, self.time
        )

        # 8. Switch controller if needed
        current_controller = self.controller_switcher.get_active_controller()
        if controller_name != current_controller:
            switched = self.controller_switcher.switch_to(
                controller_name, self.time,
                f"Supervisor: {self.supervisor.get_mode().value} mode"
            )
            if switched:
                logger.warning(f"Switched from {current_controller} to {controller_name}")

        # 9. Compute nominal control
        nominal_control = self.controller_switcher.compute_control(state_dict, setpoint)

        # 10. CBF safety filter
        control, cbf_intervened = self.cbf_filter.filter_control(
            state_dict, nominal_control, self.time
        )

        # 11. Update runtime monitors
        self.runtime_monitor.update_wind_estimate(
            state_dict['velocity'],
            setpoint.get('velocity', np.zeros(3))
        )
        if 'gps' in raw_sensors:
            gps = raw_sensors['gps']
            self.runtime_monitor.update_gps_quality(
                gps.get('satellites', 0), gps.get('hdop', 100.0)
            )

        # 12. Contract monitoring
        active_controller = self.controller_switcher.get_active_controller()
        # Re-estimate conditions with actual control output
        system_conditions = self.estimate_system_conditions(state_dict, setpoint, control)
        self.contract_monitor.monitor_runtime(
            f"controller_{active_controller}",
            system_conditions,
            self.time
        )

        # 13. Actuator contract monitoring
        actuator_state = {
            'control_effort': control['thrust'],
            'battery_voltage': raw_sensors.get('battery', {}).get('voltage', 12.0),
            'motor_temperature': raw_sensors.get('motors', {}).get('temperature', 25.0),
        }
        self.contract_monitor.monitor_runtime('actuators', actuator_state, self.time)

        # 14. Log data
        flight_mode = self.supervisor.get_mode().value
        log_entry = {
            'time': self.time,
            'state': state_dict.copy(),
            'control': control.copy(),
            'controller': active_controller,
            'flight_mode': flight_mode,
            'cbf_intervened': cbf_intervened,
            'system_conditions': system_conditions.copy(),
            'estimation_ok': estimation_ok,
            'planner_info': planner_info.copy() if planner_info else {}
        }
        self.flight_log.append(log_entry)

        # 15. Telemetry
        telemetry = {
            'time': self.time,
            'active_controller': active_controller,
            'flight_mode': flight_mode,
            'cbf_intervened': cbf_intervened,
            'cbf_stats': self.cbf_filter.get_statistics(),
            'position': state_dict['position'],
            'velocity': state_dict['velocity'],
            'attitude': state_dict['attitude'],
            'contract_status': {
                'sensors': self.contract_monitor.get_contract_status('sensors'),
                'estimator': self.contract_monitor.get_contract_status('estimator'),
                'controller': self.contract_monitor.get_contract_status(f'controller_{active_controller}'),
                'actuators': self.contract_monitor.get_contract_status('actuators'),
            },
            # Horizon planner info (if available)
            'planner': {
                'active': self.use_horizon_planner,
                'emergency': planner_emergency,
                'safety_margin': planner_info.get('safety_margin', -1.0),
                'horizon': planner_info.get('horizon', 0),
                'plan_status': planner_info.get('plan_status', 'none'),
                'replanned': planner_info.get('replanned', False),
            }
        }

        self.time += self.dt
        return control, telemetry

    # --- Mission convenience methods ---

    def set_waypoint_mission(self, waypoints: List[np.ndarray]):
        """Load a waypoint mission into the supervisor."""
        self.supervisor.set_waypoints(waypoints)

    def command_hover(self, position: np.ndarray, yaw: float = 0.0):
        """Command hover at a specific position."""
        self.supervisor.command_hover(position, yaw)

    def command_land(self):
        """Initiate controlled landing."""
        self.supervisor.command_land()

    def command_emergency(self):
        """Trigger emergency mode."""
        self.supervisor.command_emergency()

    def start_mission(self):
        """Start the mission."""
        if not self.mission_feasible:
            logger.error("Cannot start mission - pre-flight check failed!")
            return False
        self.mission_active = True
        logger.info("Mission started")
        return True

    def stop_mission(self):
        """Stop the mission."""
        self.mission_active = False
        logger.info("Mission stopped")

    def save_flight_log(self, filename: str):
        """Save flight log for analysis."""
        import json

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

        contract_log = filename.replace('.json', '_contracts.json')
        self.contract_monitor.export_metrics(contract_log)


# Test the complete system
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ADAPTIVE DRONE CONTROL SYSTEM TEST")
    print("=" * 60)

    controller = AdaptiveDroneController(dt=0.02)

    # Test 1: Pre-flight check with GOOD conditions
    print("\n" + "=" * 60)
    print("TEST 1: Pre-flight Check - NOMINAL CONDITIONS")
    print("=" * 60)

    good_conditions = {
        'gps_satellites': 12.0,
        'gps_hdop': 1.0,
        'imu_temperature': 25.0,
        'imu_calibrated': 1.0,
        'battery_voltage': 12.4,
        'motor_temperature': 25.0,
        'wind_speed': 1.5,
        'disturbance': 0.5,
    }

    mission = {
        'target_position': np.array([10.0, 5.0, -8.0]),
        'waypoints': [np.array([10.0, 5.0, -8.0])]
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
                'battery': {'voltage': 12.4},
                'motors': {'temperature': 30.0}
            }

            control, telemetry = controller.control_step(raw_sensors)

            print(f"\nStep {i+1}:")
            print(f"  Position: {telemetry['position']}")
            print(f"  Controller: {telemetry['active_controller']}")
            print(f"  Flight Mode: {telemetry['flight_mode']}")
            print(f"  CBF Intervened: {telemetry['cbf_intervened']}")
            print(f"  Thrust: {control['thrust']:.3f}")

        controller.stop_mission()

    # Test 3: Pre-flight check with BAD conditions
    print("\n" + "=" * 60)
    print("TEST 3: Pre-flight Check - EXTREME CONDITIONS")
    print("=" * 60)

    bad_conditions = {
        'gps_satellites': 3.0,
        'gps_hdop': 6.0,
        'imu_temperature': 60.0,
        'imu_calibrated': 0.0,
        'battery_voltage': 10.5,
        'motor_temperature': 85.0,
        'wind_speed': 12.0,
        'disturbance': 8.0,
    }

    controller2 = AdaptiveDroneController(dt=0.02)
    feasible, msg = controller2.pre_flight_check(bad_conditions, mission)
    print(f"\nResult: {msg}")
    print(f"Feasible: {feasible}\n")

    print("=" * 60)
    print("SYSTEM TEST COMPLETE")
    print("=" * 60)
