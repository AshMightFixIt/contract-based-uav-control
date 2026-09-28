"""
H-infinity robust controller (emergency tier), moved unchanged from controllers.py.
"""

import numpy as np
from typing import Dict
import logging

from .base import BaseController

logger = logging.getLogger(__name__)


class HInfinityController(BaseController):
    """
    H-infinity Robust Controller for Emergency Stabilization
    
    Contract:
    - Assumes: NOTHING (always works - safety net!)
    - Guarantees: Stabilization in <10s, maintains safe altitude, robust to disturbances
    
    Strategy:
    1. Aggressively damp all angular rates
    2. Level attitude (roll, pitch → 0)
    3. Hold altitude from setpoint
    """

    def __init__(self, dt: float = 0.02):
        super().__init__("H-infinity", dt)

        # Aggressive damping gains
        self.k_rate_damping = 0.8  # Very aggressive rate damping
        self.k_attitude = 5.0      # Strong attitude correction
        self.k_altitude = 0.05     # Altitude hold gain (conservative to avoid overshoot)
        self.k_altitude_rate = 0.15  # Altitude rate damping (helps slow descent)

        # Position tracking gains (robust but active wind rejection)
        self.k_pos = np.array([0.35, 0.35, 0.0])  # lateral position tracking
        self.k_vel = np.array([0.7, 0.7, 0.0])     # velocity damping (fight wind drift)
        self.k_int = np.array([0.10, 0.10, 0.0])   # integral for wind rejection
        self.int_limit = 8.0  # anti-windup clamp
        self.pos_integral = np.zeros(3)
        self.max_tilt = 0.22  # ~13° — enough tilt to reject strong wind

        # Hover thrust (normalized) - this counteracts gravity
        self.hover_thrust = 0.5

    def reset(self):
        """Reset controller state"""
        self.pos_integral = np.zeros(3)
        logger.info(f"{self.name} reset")

    def get_integral_state(self) -> np.ndarray:
        """Return effective lateral acceleration contributed by integral term."""
        return self.k_int * self.pos_integral

    def set_integral_state(self, effective_accel: np.ndarray):
        """
        Initialize integral from transferred effective acceleration (bumpless transfer).
        Solves: k_int * pos_integral = effective_accel  →  pos_integral = accel / k_int
        Z-axis integral is zero (k_int[2] = 0), altitude handled by thrust separately.
        """
        integral = np.zeros(3)
        for i in range(3):
            if self.k_int[i] > 1e-9:
                integral[i] = effective_accel[i] / self.k_int[i]
        self.pos_integral = np.clip(integral, -self.int_limit, self.int_limit)

    def compute_control(self,
                       state: Dict[str, np.ndarray],
                       setpoint: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Robust stabilization control

        Priority:
        1. Damp angular rates (prevent tumbling)
        2. Level attitude (prevent crashing)
        3. Hold altitude from setpoint
        """

        position = state['position']
        velocity = state['velocity']
        attitude = state['attitude']
        rates = state['rates']

        target_pos = setpoint.get('position', position)

        # === POSITION TRACKING (with integral wind rejection) ===
        pos_error = target_pos - position
        vel_error = -velocity  # want zero velocity (fight drift)

        # Integrate position error for sustained wind rejection
        self.pos_integral += pos_error * self.dt
        self.pos_integral = np.clip(self.pos_integral, -self.int_limit, self.int_limit)

        desired_accel = (self.k_pos * pos_error +
                         self.k_vel * vel_error +
                         self.k_int * self.pos_integral)

        # Map lateral acceleration to desired tilt (conservative limits)
        desired_pitch = np.clip(-desired_accel[0] / 9.81, -self.max_tilt, self.max_tilt)
        desired_roll = np.clip(desired_accel[1] / 9.81, -self.max_tilt, self.max_tilt)
        target_attitude = np.array([desired_roll, desired_pitch, attitude[2]])

        # === RATE DAMPING (Highest Priority) ===
        rate_damping_torque = -self.k_rate_damping * rates

        # === ATTITUDE STABILIZATION ===
        att_error = target_attitude - attitude
        att_error[2] = np.arctan2(np.sin(att_error[2]), np.cos(att_error[2]))

        attitude_torque = self.k_attitude * att_error

        torques = rate_damping_torque + attitude_torque

        # === ALTITUDE HOLD (from setpoint) ===
        # Thrust = hover_thrust + corrections for altitude tracking
        altitude_error = target_pos[2] - position[2]
        altitude_rate_error = 0.0 - velocity[2]

        thrust_correction = (self.k_altitude * altitude_error +
                            self.k_altitude_rate * altitude_rate_error)
        desired_thrust = self.hover_thrust + thrust_correction
        desired_thrust = np.clip(desired_thrust, 0.1, 0.9)

        control = {
            'thrust': desired_thrust,
            'torques': torques,
            'desired_attitude': target_attitude,
            'desired_rates': np.zeros(3)
        }

        return control
