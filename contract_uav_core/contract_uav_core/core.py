"""
Adaptive Drone Control System with Hierarchical Contract Composition

Pipeline: Sensors -> EKF -> [Horizon Planner] -> Supervisor -> [PID | MPC | H-inf] -> CBF Filter -> Actuators

Three-tier graduated degradation:
  PID  (wind <= 3 m/s)  —  efficient, nominal conditions
  MPC  (wind <= 15 m/s) —  optimal, moderate to strong conditions
  H-inf (wind > 15 m/s, dual sensor failure) — robust, emergency only

Key Innovation:
- BEFORE FLIGHT: Verify mission feasibility via contract composition
- DURING FLIGHT: Horizon planner predicts safety margins, triggers re-planning
  Supervisor manages flight modes, CBF filter enforces hard safety constraints
"""

import math
import numpy as np
from typing import Dict, Tuple, Optional, List
import logging
import time

from .contracts.monitor import HierarchicalContractMonitor
from .contracts.spec import ContractStatus
from .contracts.preflight import ControllerPreflightMixin
from .estimation.ekf import ContractAwareEKF
from .control.switcher import ControllerSwitcher
from .control.supervisor import FlightModeSupervisor, FlightMode
from .safety.cbf import CBFSafetyFilter
from .safety.runtime_monitor import RuntimeMonitor
from .conditions import ConditionsMixin
from .switching_policy import SwitchingPolicyMixin
from .telemetry import TelemetryMixin

# Optional: Horizon-based Pacti planner
try:
    from .planning.integrated_planner import IntegratedPlanner
    from .planning.horizon_planner import PlannerConfig
    HORIZON_PLANNER_AVAILABLE = True
except ImportError:
    HORIZON_PLANNER_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdaptiveDroneController(ControllerPreflightMixin, ConditionsMixin,
                              SwitchingPolicyMixin, TelemetryMixin):
    """
    Main adaptive control system with formal contract guarantees.

    Architecture:
        Sensors -> EKF -> Supervisor -> Controller -> CBF -> Actuators
                          (setpoint +   (PID or      (safety
                           ctrl pick)    H-inf)       filter)
    """

    # control_step(raw_sensors, t): bounds on the measured step, as multiples of the
    # nominal dt. The EKF predicts with the raw step clamped to [DT_MIN, DT_MAX] x dt;
    # a raw step above OVERRUN x dt counts as an overrun.
    DT_MIN_FACTOR = 0.5
    DT_MAX_FACTOR = 2.0
    OVERRUN_FACTOR = 1.5

    def __init__(self, dt: float = 0.02, use_horizon_planner: bool = True):
        self.dt = dt
        self.time = 0.0

        # Step timing, reported in the telemetry dict under 'timing' (see control_step)
        self.step_count = 0
        self.overrun_count = 0
        self.nonincreasing_count = 0
        self.last_dt_raw = None
        self.last_dt_clamped = None
        self._last_t = None

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

        # Multi-factor switching state
        self._prev_tracking_error = 0.0   # for error-trend detection

        # Data logging
        self.flight_log = []

        logger.info("Adaptive Drone Controller initialized")

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

    def control_step(self,
                    raw_sensors: Dict[str, any],
                    t: Optional[float] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, any]]:
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

        Time, two modes:
        - t is None (the top-level scripts): the step runs at the accumulated
          self.time and the EKF predicts with the nominal self.dt; self.time += self.dt
          at the end. This is the unchanged legacy path.
        - t given (seconds, from the caller's clock; the ROS node passes the PX4
          timestamp): self.time = t for this step and is not advanced at the end.
          The raw step is t minus the previous t (the nominal dt on the first timed
          call). The EKF predicts with the raw step clamped to
          [DT_MIN_FACTOR, DT_MAX_FACTOR] x dt. A raw step above OVERRUN_FACTOR x dt
          counts in overrun_count. A non-increasing t (raw step <= 0: a repeated or
          older timestamp) counts in nonincreasing_count and is clamped like any
          other step, so the EKF still predicts DT_MIN_FACTOR x dt; self.time still
          follows t, so it can go backwards. A non-finite t raises ValueError.
          Only the EKF sees the measured step: the controllers, supervisor and
          switcher keep their nominal dt. Do not mix the two modes on one controller.
        The telemetry dict reports both modes under 'timing'.

        Returns: (control_output, telemetry)
        """
        if not self.mission_feasible:
            logger.error("Mission not feasible - cannot run control loop!")
            return {'thrust': 0.0, 'torques': np.zeros(3)}, {}

        # 0. Step time (see the docstring)
        if t is None:
            dt_raw = dt_predict = self.dt
        else:
            dt_raw, dt_predict = self._advance_clock(t)
        self.step_count += 1
        self.last_dt_raw = dt_raw
        self.last_dt_clamped = dt_predict

        # 1. Process sensors
        measurements = self.update_sensors(raw_sensors)

        # Capture wind sensor reading if available
        if 'wind' in raw_sensors:
            self.last_wind_sensor = raw_sensors['wind'].get('speed', 0.0)
        else:
            self.last_wind_sensor = None

        # 2. State estimation (predict + update)
        self.ekf.predict(dt_predict)  # Propagate state forward (self.dt when t is None)
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

        # 7. Multi-factor contract-based switching: PID > MPC > H-inf (switching_policy.py)
        recommended = self._recommend_controller(
            system_conditions, gps_ok, imu_ok, planner_emergency, planner_controller
        )

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

        # 11.-14. Runtime monitors, contract monitoring and the flight log (telemetry.py)
        active_controller, flight_mode = self._record_step(
            raw_sensors, state_dict, setpoint, control, cbf_intervened, estimation_ok, planner_info
        )

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
            'setpoint': {
                'position': setpoint.get('position', np.zeros(3)),
                'velocity': setpoint.get('velocity', np.zeros(3)),
            },
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
            },
            # Step timing (control_step's docstring)
            'timing': {
                'time_source': 'accumulated' if t is None else 'caller',
                'step_count': self.step_count,
                'overrun_count': self.overrun_count,
                'nonincreasing_count': self.nonincreasing_count,
                'dt_raw': dt_raw,
                'dt_clamped': dt_predict,
            },
        }

        if t is None:
            self.time += self.dt
        return control, telemetry

    def _advance_clock(self, t: float) -> Tuple[float, float]:
        """control_step with a caller time t: set self.time, count, return (raw, clamped) dt."""
        t = float(t)
        if not math.isfinite(t):
            raise ValueError(f"control_step: t must be a finite time in seconds, got {t!r}")
        dt_raw = self.dt if self._last_t is None else t - self._last_t
        if dt_raw > self.OVERRUN_FACTOR * self.dt:
            self.overrun_count += 1
        if dt_raw <= 0.0:
            self.nonincreasing_count += 1
        dt_clamped = min(max(dt_raw, self.DT_MIN_FACTOR * self.dt), self.DT_MAX_FACTOR * self.dt)
        self._last_t = t
        self.time = t
        return dt_raw, dt_clamped

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
