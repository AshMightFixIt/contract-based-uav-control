"""
Integrated Horizon-Based Planner

Integrates the horizon planner with the adaptive control system.
Provides predictive planning layer on top of reactive control.

Architecture:
    Sensors -> EKF -> [HORIZON PLANNER] -> Supervisor -> [PID|Hinf] -> CBF -> Actuators
                           |
                    Pacti Contract Cascade
                           |
                    Safety Margin Monitor
                           |
                    Re-planning Trigger
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import logging

from .horizon_planner import (
    HorizonPlanner, AdaptiveRePlanner, Plan, PlanStatus, PlannerConfig
)

logger = logging.getLogger(__name__)


@dataclass
class PlanningState:
    """Current planning state for integration with control system."""
    current_plan: Optional[Plan] = None
    last_plan_time: float = 0.0
    replan_interval: float = 0.5  # Replan every 0.5s
    consecutive_failures: int = 0
    max_failures: int = 3
    emergency_mode: bool = False


class IntegratedPlanner:
    """
    Integrates horizon-based planning with the control system.

    Provides:
    1. Periodic planning updates
    2. Safety margin monitoring
    3. Automatic re-planning triggers
    4. Emergency mode fallback
    """

    def __init__(self, config: Optional[PlannerConfig] = None):
        self.config = config or PlannerConfig(
            max_horizon=10,
            min_horizon=3,
            dt=0.1,
            safety_limit=5.0,
            margin_threshold=1.0,
            replan_threshold=0.5
        )
        self.planner = HorizonPlanner(self.config)
        self.replanner = AdaptiveRePlanner(self.planner)
        self.state = PlanningState()

        logger.info("IntegratedPlanner initialized")

    def update(self,
               current_time: float,
               drone_state: Dict[str, np.ndarray],
               environment: Dict[str, float],
               target: np.ndarray) -> Tuple[str, bool, Dict]:
        """
        Main update called by control system.

        Args:
            current_time: Current simulation time
            drone_state: Current drone state (position, velocity, attitude, rates)
            environment: Environmental conditions (wind, GPS, battery)
            target: Current target position

        Returns:
            (recommended_controller, needs_emergency, planning_info)
        """
        planning_info = {
            'replanned': False,
            'plan_status': 'unknown',
            'horizon': 0,
            'safety_margin': 0.0,
            'replan_reason': None
        }

        # Check if we need to replan
        time_since_plan = current_time - self.state.last_plan_time
        needs_replan = (
            self.state.current_plan is None or
            time_since_plan >= self.state.replan_interval or
            self._safety_critical(drone_state, environment)
        )

        if needs_replan:
            # Convert drone state to planning format
            planning_state = self._convert_state(drone_state, target)

            # Try to find a feasible plan
            plan, modifications = self.replanner.find_feasible_plan(
                planning_state, target, environment
            )

            self.state.current_plan = plan
            self.state.last_plan_time = current_time
            planning_info['replanned'] = True
            planning_info['modifications'] = modifications

            # Track failures
            if not plan.is_safe():
                self.state.consecutive_failures += 1
                if self.state.consecutive_failures >= self.state.max_failures:
                    self.state.emergency_mode = True
                    logger.error(f"Emergency mode triggered after {self.state.max_failures} failures")
            else:
                self.state.consecutive_failures = 0
                self.state.emergency_mode = False

        # Fill planning info
        if self.state.current_plan:
            planning_info['plan_status'] = self.state.current_plan.status.value
            planning_info['horizon'] = self.state.current_plan.horizon_length
            planning_info['safety_margin'] = self.state.current_plan.min_safety_margin
            planning_info['replan_reason'] = self.state.current_plan.replan_reason

        # Determine recommended controller
        if self.state.emergency_mode:
            return 'hinf', True, planning_info
        elif self.state.current_plan and self.state.current_plan.is_safe():
            return self.state.current_plan.recommended_controller, False, planning_info
        else:
            return 'hinf', False, planning_info

    def _convert_state(self,
                       drone_state: Dict[str, np.ndarray],
                       target: np.ndarray) -> Dict[str, float]:
        """Convert drone state to planning state format."""
        position = drone_state.get('position', np.zeros(3))
        velocity = drone_state.get('velocity', np.zeros(3))

        # Compute tracking error considering BOTH lateral drift AND distance growth
        to_target = target - position
        dist_to_target = np.linalg.norm(to_target)

        if dist_to_target > 0.1:
            # Decompose velocity into toward-target and lateral components
            direction = to_target / dist_to_target
            closing_speed = np.dot(velocity, direction)
            lateral_velocity = velocity - closing_speed * direction
            lateral_drift = np.linalg.norm(lateral_velocity)

            # Track if we're making progress toward target
            # Large distance when we should be close indicates overshoot
            expected_progress_rate = 2.0  # m/s reasonable approach speed
            max_reasonable_distance = 15.0  # If further than this, something is wrong

            # Tracking error combines:
            # 1. Lateral drift (perpendicular deviation)
            # 2. Penalty for moving away from target
            # 3. Penalty for excessive distance (indicates overshoot or loss of tracking)
            if closing_speed >= 0.5:
                # Moving toward target at good speed - only lateral drift matters
                tracking_error = lateral_drift
            elif closing_speed >= 0:
                # Slow approach - add small penalty
                tracking_error = lateral_drift + 0.5
            else:
                # Moving AWAY from target - significant penalty
                tracking_error = lateral_drift + abs(closing_speed) * 2.0

            # Add distance-based penalty when far from target
            # This catches cases where we've overshot badly
            if dist_to_target > max_reasonable_distance:
                distance_penalty = (dist_to_target - max_reasonable_distance) * 0.3
                tracking_error += distance_penalty

        else:
            # Very close to target - use position offset
            tracking_error = dist_to_target

        # Estimate position error (from EKF covariance if available)
        position_est_error = drone_state.get('position_uncertainty', 0.5)
        velocity_est_error = drone_state.get('velocity_uncertainty', 0.1)

        return {
            'tracking_error': tracking_error,
            'position_est_error': position_est_error,
            'velocity_est_error': velocity_est_error,
            'distance_to_target': dist_to_target,
        }

    def _safety_critical(self,
                         drone_state: Dict[str, np.ndarray],
                         environment: Dict[str, float]) -> bool:
        """Check if current conditions warrant immediate re-planning."""
        if self.state.current_plan is None:
            return True

        # Check if wind exceeds current controller's envelope
        current_wind = environment.get('wind_speed', 0.0)
        ctrl = self.state.current_plan.recommended_controller
        wind_limits = {'pid': 3.0, 'mpc': 8.0, 'hinf': 15.0}
        wind_limit = wind_limits.get(ctrl, 15.0)
        if current_wind > wind_limit:
            logger.warning(f"Wind ({current_wind:.1f} m/s) exceeds {ctrl} envelope ({wind_limit} m/s)")
            return True

        # Check safety margin
        if self.state.current_plan.min_safety_margin < self.config.replan_threshold:
            return True

        # Check for altitude anomaly (positive Z in NED = below ground)
        position = drone_state.get('position', np.zeros(3))
        if position[2] > -0.5:  # Less than 0.5m above ground
            logger.warning(f"Altitude critical: z={position[2]:.2f}")
            return True

        # Check for high velocity (indicates loss of control)
        velocity = drone_state.get('velocity', np.zeros(3))
        speed = np.linalg.norm(velocity)
        if speed > 8.0:  # Very high speed indicates loss of control
            logger.warning(f"High speed detected: {speed:.2f} m/s")
            return True

        # Check attitude (large tilt indicates instability)
        attitude = drone_state.get('attitude', np.zeros(3))
        tilt = np.sqrt(attitude[0]**2 + attitude[1]**2)
        if tilt > 0.5:  # ~30 degrees
            logger.warning(f"Large tilt detected: {np.degrees(tilt):.1f} deg")
            return True

        return False

    def get_plan_visualization(self) -> List[Dict]:
        """Get plan data for visualization."""
        if not self.state.current_plan or not self.state.current_plan.steps:
            return []

        return [
            {
                'timestep': step.timestep,
                'tracking_error_bound': step.tracking_error_bound,
                'safety_margin': step.safety_margin,
                'controller': step.controller
            }
            for step in self.state.current_plan.steps
        ]

    def get_status(self) -> Dict:
        """Get current planner status."""
        return {
            'emergency_mode': self.state.emergency_mode,
            'consecutive_failures': self.state.consecutive_failures,
            'current_plan_status': (
                self.state.current_plan.status.value
                if self.state.current_plan else 'none'
            ),
            'recommended_controller': (
                self.state.current_plan.recommended_controller
                if self.state.current_plan else 'hinf'
            ),
            'safety_margin': (
                self.state.current_plan.min_safety_margin
                if self.state.current_plan else -1.0
            ),
        }


# Test integration
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("Integrated Planner Test")
    print("=" * 70)

    planner = IntegratedPlanner()

    # Simulate a flight scenario
    print("\n=== Simulated Flight Scenario ===")

    # Phase 1: Nominal
    print("\n--- Phase 1: Nominal Conditions ---")
    state = {
        'position': np.array([0.0, 0.0, -5.0]),
        'velocity': np.array([1.0, 0.0, 0.0]),
        'attitude': np.array([0.0, 0.0, 0.0]),
        'rates': np.array([0.0, 0.0, 0.0]),
    }
    env = {'wind_speed': 1.0, 'gps_satellites': 12, 'gps_hdop': 1.0, 'battery_voltage': 12.4}
    target = np.array([10.0, 0.0, -5.0])

    for t in [0.0, 0.5, 1.0]:
        ctrl, emergency, info = planner.update(t, state, env, target)
        print(f"t={t:.1f}s: controller={ctrl}, emergency={emergency}, "
              f"margin={info['safety_margin']:.2f}, status={info['plan_status']}")

    # Phase 2: Wind increases
    print("\n--- Phase 2: Wind Increases ---")
    env['wind_speed'] = 4.0

    for t in [1.5, 2.0, 2.5]:
        ctrl, emergency, info = planner.update(t, state, env, target)
        print(f"t={t:.1f}s: controller={ctrl}, emergency={emergency}, "
              f"margin={info['safety_margin']:.2f}, status={info['plan_status']}")
        if info.get('replanned'):
            print(f"         REPLANNED! Reason: {info.get('replan_reason')}")

    # Phase 3: Extreme conditions
    print("\n--- Phase 3: Extreme Conditions ---")
    env['wind_speed'] = 12.0
    state['position'] = np.array([2.0, 1.0, -5.0])

    for t in [3.0, 3.5, 4.0, 4.5, 5.0]:
        ctrl, emergency, info = planner.update(t, state, env, target)
        status = planner.get_status()
        print(f"t={t:.1f}s: controller={ctrl}, emergency={emergency}, "
              f"failures={status['consecutive_failures']}, "
              f"margin={info['safety_margin']:.2f}")

    # Phase 4: Recovery
    print("\n--- Phase 4: Recovery ---")
    env['wind_speed'] = 2.0

    for t in [5.5, 6.0, 6.5]:
        ctrl, emergency, info = planner.update(t, state, env, target)
        print(f"t={t:.1f}s: controller={ctrl}, emergency={emergency}, "
              f"margin={info['safety_margin']:.2f}, status={info['plan_status']}")

    print("\n=== Final Status ===")
    print(planner.get_status())

    print("\n" + "=" * 70)
    print("Integrated Planner Test Complete")
    print("=" * 70)
