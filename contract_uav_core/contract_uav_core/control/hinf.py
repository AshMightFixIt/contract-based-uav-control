"""
H-infinity robust controller (emergency tier), moved unchanged from controllers.py.

Batch 3 (W4-05, P1.1/P1.2): the altitude hold is a vertical acceleration demand
turned into thrust by frames.accel_to_thrust (physically signed, normalised by
the airframe's hover thrust), and the tilt follows the heading it holds.
"""

import numpy as np
from typing import Dict
import logging

from .base import BaseController
from ..frames import (GRAVITY, LEGACY_THRUST_TO_ACCEL, accel_to_attitude, accel_to_thrust,
                      hinf_attitude_torques)
from ..interfaces import AccelCommand

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

    def __init__(self, dt: float = 0.02, airframe=None):
        """airframe: optional config.airframe.Airframe; sets hover_thrust, min_thrust and
        max_thrust (base.py). Without one: hover 0.5, thrust limits [0, 1]."""
        super().__init__("H-infinity", dt, airframe)

        # Aggressive damping gains
        self.k_rate_damping = 0.8  # Very aggressive rate damping
        self.k_attitude = 5.0      # Strong attitude correction
        # Altitude gains in legacy thrust units at hover 0.5 (times LEGACY_THRUST_TO_ACCEL
        # = g / 0.5 they are m/s^2 per m and per m/s; compute_accel_command)
        self.k_altitude = 0.05     # Altitude hold gain (conservative to avoid overshoot)
        self.k_altitude_rate = 0.15  # Altitude rate damping (helps slow descent)
        # Vertical demand limit: the legacy thrust clip [0.1, 0.9] was hover -+ 0.4 at
        # hover 0.5, i.e. -+0.8 g; as an acceleration it scales with any hover thrust
        self.max_vertical_accel = 0.8 * GRAVITY

        # Position tracking gains (robust but active wind rejection)
        self.k_pos = np.array([0.35, 0.35, 0.0])  # lateral position tracking
        self.k_vel = np.array([0.7, 0.7, 0.0])     # velocity damping (fight wind drift)
        self.k_int = np.array([0.10, 0.10, 0.0])   # integral for wind rejection
        self.int_limit = 8.0  # anti-windup clamp
        self.pos_integral = np.zeros(3)
        self.max_tilt = 0.22  # ~13° — enough tilt to reject strong wind

        # hover_thrust, min_thrust, max_thrust: base.py (from the airframe; else 0.5, 0, 1)

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

    def compute_accel_command(self,
                              state: Dict[str, np.ndarray],
                              setpoint: Dict[str, np.ndarray]) -> AccelCommand:
        """
        Robust stabilization control

        Priority:
        1. Damp angular rates (prevent tumbling)
        2. Level attitude (prevent crashing)
        3. Hold altitude from setpoint

        Returns the AccelCommand: accel x and y are the lateral acceleration
        demand (they set the tilt), accel z is the altitude hold's vertical
        demand after its limit (it sets the thrust, frames.accel_to_thrust); yaw
        is the current yaw (the H-inf holds heading and ignores the setpoint's
        yaw). The tilt is rotated by that same yaw, the one it commands, so the
        lateral demand acts in the right direction at any heading (P1.2).
        compute_control (base.py) returns its legacy dict.
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

        # Map lateral acceleration to desired tilt (conservative limits; frames.py),
        # rotated by the held yaw attitude[2], which is also the yaw commanded
        target_attitude = accel_to_attitude(desired_accel, attitude[2], self.max_tilt)

        # === RATE DAMPING (Highest Priority) + ATTITUDE STABILIZATION (frames.py) ===
        torques = hinf_attitude_torques(target_attitude, attitude, rates,
                                        self.k_rate_damping, self.k_attitude)

        # === ALTITUDE HOLD (from setpoint; P1.1, P1.2) ===
        # NED vertical acceleration demand -> thrust about hover. A target above
        # (altitude_error < 0) gives accel_z < 0, i.e. thrust above hover. The legacy
        # law was thrust = hover + correction: less thrust to climb (K01).
        altitude_error = target_pos[2] - position[2]
        altitude_rate_error = 0.0 - velocity[2]

        thrust_correction = (self.k_altitude * altitude_error +
                            self.k_altitude_rate * altitude_rate_error)
        accel_z = np.clip(thrust_correction * LEGACY_THRUST_TO_ACCEL,
                          -self.max_vertical_accel, self.max_vertical_accel)
        desired_thrust = accel_to_thrust(accel_z, self.hover_thrust, self.min_thrust, self.max_thrust)

        return AccelCommand(
            accel=np.array([desired_accel[0], desired_accel[1], accel_z]),
            yaw=attitude[2],
            thrust=desired_thrust,
            torques=torques,
            desired_attitude=target_attitude,
            desired_rates=np.zeros(3),
        )
