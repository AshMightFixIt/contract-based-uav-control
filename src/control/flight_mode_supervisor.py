"""
Flight Mode Supervisor
Manages mission-level flight modes and delegates control selection.

Does NOT compute control — generates setpoints and selects controller.
Supports three-tier controller selection: PID > MPC > H-inf.

Flight Modes:
- TRACK: Follow waypoints using position setpoints
- HOVER: Hold position (e.g. after GPS loss or mission complete)
- LAND: Controlled descent profile
- EMERGENCY: Immediate stabilization at safe altitude
"""

import numpy as np
from typing import Dict, Tuple, List
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class FlightMode(Enum):
    TRACK = "track"
    HOVER = "hover"
    LAND = "land"
    EMERGENCY = "emergency"


class FlightModeSupervisor:
    """
    Higher-level layer managing mission logic.

    Generates setpoints and selects PID or H-inf based on
    flight mode, contract status, and sensor availability.
    """

    def __init__(self, dt: float = 0.02):
        self.dt = dt
        self.mode = FlightMode.HOVER
        self.previous_mode = None

        # Waypoint tracking
        self.waypoints: List[np.ndarray] = []
        self.current_waypoint_idx = 0
        self.waypoint_tolerance = 1.5  # meters
        # Require distance to be below tolerance for multiple consecutive checks
        self.waypoint_confirm_count = 0
        self.waypoint_confirm_threshold = 5  # Need 5 consecutive checks within tolerance
        self.last_distance = float('inf')

        # Hover state
        self.hover_position = np.array([0.0, 0.0, -5.0])
        self.hover_yaw = 0.0

        # Landing state
        self.landing_descent_rate = 0.5  # m/s (positive = downward in NED)
        self.landing_start_position = None
        self.landing_start_time = 0.0

        # Emergency state
        self.emergency_altitude = -5.0

        # Mode transition cooldown
        self.last_transition_time = 0.0
        self.transition_cooldown = 1.0  # seconds

        # Sensor availability
        self.gps_available = True

        logger.info("Flight Mode Supervisor initialized")

    def set_waypoints(self, waypoints: List[np.ndarray]):
        """Load a waypoint mission."""
        self.waypoints = [wp.copy() for wp in waypoints]
        self.current_waypoint_idx = 0
        self.mode = FlightMode.TRACK
        logger.info(f"Waypoint mission loaded: {len(waypoints)} waypoints")

    def command_hover(self, position: np.ndarray, yaw: float = 0.0):
        """Command hover at a specific position."""
        self.hover_position = position.copy()
        self.hover_yaw = yaw
        self.mode = FlightMode.HOVER

    def command_land(self):
        """Initiate controlled landing."""
        self.mode = FlightMode.LAND
        self.landing_start_position = None

    def command_emergency(self):
        """Trigger emergency mode."""
        self.mode = FlightMode.EMERGENCY

    def update(self,
               state: Dict[str, np.ndarray],
               system_conditions: Dict[str, float],
               recommended_controller: str,
               gps_available: bool,
               current_time: float) -> Tuple[Dict, str]:
        """
        Main supervisor tick.

        Args:
            state: Current state dict (position, velocity, attitude, rates)
            system_conditions: Wind speed, disturbance, errors
            recommended_controller: Contract-based recommendation ('PID', 'MPC', or 'Hinf')
            gps_available: Whether GPS sensor contract is satisfied
            current_time: Simulation time

        Returns:
            (setpoint, controller_name)
        """
        self.gps_available = gps_available

        # Check automatic mode transitions
        self._check_transitions(state, system_conditions, gps_available, current_time)

        # Generate setpoint for current mode
        setpoint = self._generate_setpoint(state, current_time)

        # Select controller (supervisor may override recommendation)
        controller = self._select_controller(recommended_controller)

        return setpoint, controller

    def _check_transitions(self, state, conditions, gps_available, current_time):
        """Automatic mode transitions based on conditions."""
        if current_time - self.last_transition_time < self.transition_cooldown:
            return

        old_mode = self.mode

        # EMERGENCY takes priority: severe wind or disturbance
        if self.mode != FlightMode.EMERGENCY:
            wind = conditions.get('wind_speed', 0.0)
            disturbance = conditions.get('disturbance', 0.0)
            if wind > 8.0 or disturbance > 5.0:
                self.mode = FlightMode.EMERGENCY
                self.emergency_altitude = min(state['position'][2] - 2.0, -5.0)

        # GPS loss during TRACK -> capture position and HOVER
        if self.mode == FlightMode.TRACK and not gps_available:
            self.hover_position = state['position'].copy()
            self.hover_yaw = state['attitude'][2]
            self.mode = FlightMode.HOVER

        # Recovery from EMERGENCY -> HOVER when conditions improve
        if self.mode == FlightMode.EMERGENCY:
            wind = conditions.get('wind_speed', 0.0)
            disturbance = conditions.get('disturbance', 0.0)
            rates_norm = np.linalg.norm(state['rates'])
            if wind < 5.0 and disturbance < 3.0 and rates_norm < 0.5:
                self.hover_position = state['position'].copy()
                self.hover_yaw = state['attitude'][2]
                self.mode = FlightMode.HOVER

        if self.mode != old_mode:
            self.previous_mode = old_mode
            self.last_transition_time = current_time
            logger.warning(f"FLIGHT MODE: {old_mode.value} -> {self.mode.value}")

    def _generate_setpoint(self, state, current_time) -> Dict:
        """Generate setpoint based on current mode."""
        if self.mode == FlightMode.TRACK:
            return self._setpoint_track(state)
        elif self.mode == FlightMode.HOVER:
            return self._setpoint_hover()
        elif self.mode == FlightMode.LAND:
            return self._setpoint_land(state, current_time)
        elif self.mode == FlightMode.EMERGENCY:
            return self._setpoint_emergency(state)
        return self._setpoint_hover()

    def _setpoint_track(self, state) -> Dict:
        """Track waypoints sequentially with robust detection."""
        if not self.waypoints:
            return self._setpoint_hover()

        target = self.waypoints[self.current_waypoint_idx]
        dist = np.linalg.norm(target - state['position'])

        # Robust waypoint detection: require multiple consecutive checks within tolerance
        # AND distance should not be increasing significantly
        if dist < self.waypoint_tolerance:
            # Check if we're not moving away
            if dist <= self.last_distance + 0.2:  # Allow small increase due to noise
                self.waypoint_confirm_count += 1
            else:
                self.waypoint_confirm_count = max(0, self.waypoint_confirm_count - 1)
        else:
            self.waypoint_confirm_count = 0

        self.last_distance = dist

        # Only advance if consistently within tolerance
        if self.waypoint_confirm_count >= self.waypoint_confirm_threshold:
            self.waypoint_confirm_count = 0  # Reset for next waypoint
            self.last_distance = float('inf')

            if self.current_waypoint_idx < len(self.waypoints) - 1:
                self.current_waypoint_idx += 1
                target = self.waypoints[self.current_waypoint_idx]
                logger.info(f"Waypoint {self.current_waypoint_idx} reached (dist={dist:.2f}m), "
                           f"advancing to WP{self.current_waypoint_idx + 1}")
            else:
                self.hover_position = target.copy()
                self.mode = FlightMode.HOVER
                logger.info("All waypoints reached, switching to HOVER")
                return self._setpoint_hover()

        return {
            'position': target.copy(),
            'velocity': np.zeros(3),
            'yaw': 0.0
        }

    def _setpoint_hover(self) -> Dict:
        """Hold position."""
        return {
            'position': self.hover_position.copy(),
            'velocity': np.zeros(3),
            'yaw': self.hover_yaw
        }

    def _setpoint_land(self, state, current_time) -> Dict:
        """Controlled descent profile.

        NED frame: negative Z = altitude above ground
        - Start: z = -5.0 (5m above ground)
        - Descend: z increases toward 0
        - Stop at: z = -0.5 (0.5m above ground for safety)
        """
        if self.landing_start_position is None:
            self.landing_start_position = state['position'].copy()
            self.landing_start_time = current_time

        elapsed = current_time - self.landing_start_time

        # In NED, to descend we need Z to become less negative (increase toward 0)
        # descent_rate is positive, so we ADD it to make Z increase
        target_z = self.landing_start_position[2] + self.landing_descent_rate * elapsed

        # Clamp to safe landing altitude (0.5m above ground)
        landing_altitude = -0.5  # Stop at 0.5m above ground
        target_z = min(target_z, landing_altitude)

        # If we've reached landing altitude, zero out descent velocity
        descent_vel = self.landing_descent_rate if target_z > landing_altitude else 0.0

        return {
            'position': np.array([
                self.landing_start_position[0],
                self.landing_start_position[1],
                target_z
            ]),
            'velocity': np.array([0.0, 0.0, descent_vel]),
            'yaw': state['attitude'][2]
        }

    def _setpoint_emergency(self, state) -> Dict:
        """Hold safe altitude, level attitude, zero velocity."""
        return {
            'position': np.array([
                state['position'][0],
                state['position'][1],
                self.emergency_altitude
            ]),
            'velocity': np.zeros(3),
            'yaw': state['attitude'][2]
        }

    def _select_controller(self, recommended: str) -> str:
        """
        Select controller based on flight mode and contract recommendation.

        Mode overrides:
        - EMERGENCY / LAND -> always H-inf
        - HOVER without GPS -> H-inf
        Otherwise, use the contract-recommended controller.
        """
        if self.mode == FlightMode.EMERGENCY:
            return 'Hinf'
        if self.mode == FlightMode.LAND:
            return 'Hinf'
        if self.mode == FlightMode.HOVER and not self.gps_available:
            return 'Hinf'
        return recommended

    def get_mode(self) -> FlightMode:
        return self.mode

    def get_current_setpoint(self, state: Dict[str, np.ndarray], current_time: float) -> Dict:
        """Get the current setpoint without running transitions."""
        return self._generate_setpoint(state, current_time)

    def get_status(self) -> Dict:
        return {
            'mode': self.mode.value,
            'waypoint_idx': self.current_waypoint_idx,
            'waypoint_count': len(self.waypoints),
            'gps_available': self.gps_available
        }


if __name__ == "__main__":
    print("=" * 60)
    print("Flight Mode Supervisor Test")
    print("=" * 60)

    supervisor = FlightModeSupervisor()

    state = {
        'position': np.array([0.0, 0.0, -5.0]),
        'velocity': np.array([0.0, 0.0, 0.0]),
        'attitude': np.array([0.0, 0.0, 0.0]),
        'rates': np.array([0.0, 0.0, 0.0])
    }

    # Test 1: Waypoint tracking with PID
    print("\n[TEST 1] Waypoint tracking, nominal conditions")
    supervisor.set_waypoints([
        np.array([5.0, 0.0, -5.0]),
        np.array([5.0, 5.0, -8.0]),
    ])
    conditions = {'wind_speed': 1.0, 'disturbance': 0.5}
    setpoint, ctrl = supervisor.update(state, conditions, recommended_controller='PID', gps_available=True, current_time=0.0)
    print(f"  Mode: {supervisor.get_mode().value}, Controller: {ctrl}")
    print(f"  Setpoint: {setpoint['position']}")

    # Test 2: Wind → MPC recommended
    print("\n[TEST 2] Moderate wind -> MPC recommended")
    conditions = {'wind_speed': 4.0, 'disturbance': 1.0}
    setpoint, ctrl = supervisor.update(state, conditions, recommended_controller='MPC', gps_available=True, current_time=2.0)
    print(f"  Mode: {supervisor.get_mode().value}, Controller: {ctrl}")

    # Test 3: GPS loss -> HOVER (overrides recommendation)
    print("\n[TEST 3] GPS loss during tracking -> H-inf override")
    conditions = {'wind_speed': 1.0, 'disturbance': 0.5}
    setpoint, ctrl = supervisor.update(state, conditions, recommended_controller='PID', gps_available=False, current_time=4.0)
    print(f"  Mode: {supervisor.get_mode().value}, Controller: {ctrl}")
    print(f"  Hover position: {setpoint['position']}")

    # Test 4: Severe wind -> EMERGENCY
    print("\n[TEST 4] Severe wind -> EMERGENCY (H-inf forced)")
    supervisor.mode = FlightMode.TRACK
    conditions = {'wind_speed': 9.0, 'disturbance': 6.0}
    setpoint, ctrl = supervisor.update(state, conditions, recommended_controller='Hinf', gps_available=True, current_time=6.0)
    print(f"  Mode: {supervisor.get_mode().value}, Controller: {ctrl}")

    # Test 5: Recovery from emergency
    print("\n[TEST 5] Recovery from EMERGENCY")
    conditions = {'wind_speed': 2.0, 'disturbance': 1.0}
    state['rates'] = np.array([0.1, 0.1, 0.0])
    setpoint, ctrl = supervisor.update(state, conditions, recommended_controller='PID', gps_available=True, current_time=8.0)
    print(f"  Mode: {supervisor.get_mode().value}, Controller: {ctrl}")

    # Test 6: Landing (H-inf always)
    print("\n[TEST 6] Landing command -> H-inf always")
    supervisor.command_land()
    setpoint, ctrl = supervisor.update(state, conditions, recommended_controller='PID', gps_available=True, current_time=10.0)
    print(f"  Mode: {supervisor.get_mode().value}, Controller: {ctrl}")
    print(f"  Landing setpoint: {setpoint['position']}")

    print("\n" + "=" * 60)
    print("Flight Mode Supervisor Test Complete")
    print("=" * 60)
