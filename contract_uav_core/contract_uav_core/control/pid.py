"""
Contract-Based Controllers for Adaptive Drone System

Implements three-tier graduated degradation:
1. PID Controller     - Efficient, for nominal conditions (wind <= 3 m/s)
2. MPC Controller     - Optimal, for moderate conditions (wind <= 15 m/s)
3. H-infinity Controller - Robust, for emergency stabilization (wind <= 25 m/s)

Each controller has formal contracts defining when they work.

This module holds PIDController. BaseController is in control/base.py, the
MPC and H-infinity controllers in control/mpc.py and control/hinf.py, and
ControllerSwitcher (with the self-test) in control/switcher.py.
"""

import numpy as np
from typing import Dict
import logging

from .base import BaseController
from ..frames import accel_to_attitude, attitude_to_rates_torques
from ..interfaces import AccelCommand

logger = logging.getLogger(__name__)


class PIDController(BaseController):
    """
    Cascaded PID Controller for position and attitude

    Contract:
    - Assumes: Low wind (<3 m/s), small errors (<2m), nominal conditions
    - Guarantees: Fast response (<5s), low overshoot (<30%), efficient
    """

    def __init__(self, dt: float = 0.02):
        super().__init__("PID", dt)

        # Position PID gains (outer loop) - tuned for stability in simplified simulation
        # MUCH lower gains to prevent overshoot
        self.kp_pos = np.array([0.35, 0.35, 0.8])  # [x, y, z]
        self.ki_pos = np.array([0.005, 0.005, 0.05])  # minimal integral
        self.kd_pos = np.array([0.5, 0.5, 0.6])  # strong velocity damping

        # Attitude PID gains (inner loop)
        self.kp_att = np.array([1.5, 1.5, 1.0])  # [roll, pitch, yaw] - reduced
        self.ki_att = np.array([0.02, 0.02, 0.02])
        self.kd_att = np.array([0.4, 0.4, 0.3])  # increased damping

        # Rate PID gains (innermost loop)
        self.kp_rate = np.array([0.08, 0.08, 0.06])  # [p, q, r]

        # Limits
        self.max_tilt = 0.22  # ~13 degrees max tilt
        self.max_rate = 1.0  # rad/s
        self.max_thrust = 1.0
        self.min_thrust = 0.0

        # Hover thrust (normalized) - this counteracts gravity
        self.hover_thrust = 0.5

        # Anti-windup
        self.integral_limit = 2.0  # reduced to prevent windup
        
    def reset(self):
        """Reset integrators"""
        self.integral_error = np.zeros(3)
        self.last_error = np.zeros(3)
        logger.info(f"{self.name} reset")

    def get_integral_state(self) -> np.ndarray:
        """Return effective lateral acceleration contributed by integral term."""
        return self.ki_pos * self.integral_error

    def set_integral_state(self, effective_accel: np.ndarray):
        """
        Initialize integral from transferred effective acceleration (bumpless transfer).
        Solves: ki_pos * integral_error = effective_accel  →  integral_error = accel / ki
        """
        ratio = np.where(self.ki_pos > 1e-9,
                         effective_accel / np.where(self.ki_pos > 1e-9, self.ki_pos, 1.0),
                         0.0)
        self.integral_error = np.clip(ratio, -self.integral_limit, self.integral_limit)

    def compute_accel_command(self,
                              state: Dict[str, np.ndarray],
                              setpoint: Dict[str, np.ndarray]) -> AccelCommand:
        """
        Cascaded PID control
        
        Outer loop: Position → desired attitude
        Inner loop: Attitude → desired rates
        Innermost loop: Rates → motor commands

        Returns the AccelCommand: accel is the PID acceleration demand (all
        three axes; its z is not used, the thrust comes from the altitude law
        below), yaw is the setpoint's yaw. compute_control (base.py) returns its
        legacy dict.
        """
        
        # Extract state
        position = state['position']  # [x, y, z]
        velocity = state['velocity']  # [vx, vy, vz]
        attitude = state['attitude']  # [roll, pitch, yaw]
        rates = state['rates']  # [p, q, r]
        
        # Extract setpoints
        target_pos = setpoint.get('position', position)
        target_vel = setpoint.get('velocity', np.zeros(3))
        target_yaw = setpoint.get('yaw', attitude[2])
        
        # === OUTER LOOP: Position Control ===
        pos_error = target_pos - position
        vel_error = target_vel - velocity
        
        # PID for desired acceleration
        self.integral_error += pos_error * self.dt
        self.integral_error = np.clip(self.integral_error, 
                                      -self.integral_limit, 
                                      self.integral_limit)
        
        d_error = (pos_error - self.last_error) / self.dt
        self.last_error = pos_error.copy()
        
        desired_accel = (self.kp_pos * pos_error + 
                        self.ki_pos * self.integral_error +
                        self.kd_pos * d_error)
        
        # Convert desired acceleration to desired attitude
        # In NED frame with standard quadrotor convention:
        # - Positive pitch (nose up) -> backward (negative X)
        # - Positive roll (right wing down) -> rightward (positive Y)
        # So to accelerate forward (pos X), need negative pitch
        # And to accelerate right (pos Y), need positive roll
        # (frames.py: accel_to_attitude, yaw-blind; limit tilt angles)
        desired_attitude = accel_to_attitude(desired_accel, target_yaw, self.max_tilt)
        
        # === INNER LOOPS: Attitude -> rates -> torques (frames.py) ===
        desired_rates, torques = attitude_to_rates_torques(
            desired_attitude, attitude, rates,
            self.kp_att, self.kd_att, self.max_rate, self.kp_rate)
        
        # === THRUST CONTROL ===
        # Desired thrust = hover thrust + altitude correction
        # hover_thrust counteracts gravity, corrections adjust for tracking
        altitude_error = target_pos[2] - position[2]
        altitude_rate_error = -velocity[2]  # Want zero vertical velocity

        # PD control on altitude with gravity feedforward
        thrust_correction = (self.kp_pos[2] * altitude_error * 0.1 +
                            self.kd_pos[2] * altitude_rate_error * 0.05)
        desired_thrust = self.hover_thrust + thrust_correction
        desired_thrust = np.clip(desired_thrust, self.min_thrust, self.max_thrust)
        
        return AccelCommand(
            accel=desired_accel,
            yaw=target_yaw,
            thrust=desired_thrust,
            torques=torques,  # [roll_torque, pitch_torque, yaw_torque]
            desired_attitude=desired_attitude,
            desired_rates=desired_rates,
        )
