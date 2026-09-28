"""
System-condition estimation for AdaptiveDroneController (core.py).

estimate_system_conditions, moved unchanged from adaptive_control_system.py. It
reads and updates the controller's wind_estimate_filtered, wind_filter_alpha and
last_wind_sensor, which core.py sets.
"""

import numpy as np
from typing import Dict


class ConditionsMixin:
    """Estimates the conditions the contracts are checked against."""

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
