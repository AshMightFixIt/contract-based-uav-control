"""
Data types shared across the core (the interfaces frozen in W4-04, P0a.9).

Frame convention, as the code uses it:
- World frame NED: x north, y east, z down, in metres. Altitude above the start
  point is -z: the default target (0, 0, -5) in core.py is 5 m up, the
  supervisor lands by increasing z toward 0 and holds EMERGENCY at
  min(z - 2, -5), and the ROS node passes PX4's NED VehicleLocalPosition
  through unchanged (contract_uav_control/px4_conversions.py).
- Attitude: Euler angles [roll, pitch, yaw] in rad, ZYX order, body frame FRD
  (x forward, y right, z down), as in px4_conversions.py. Positive roll puts the
  right side down, positive pitch raises the nose, and positive yaw turns from
  north toward east.
- Thrust: normalised collective thrust in [0, 1]. Physically it pushes along
  body -z, i.e. up; the node sends thrust_body = [0, 0, -thrust] to PX4.

Known legacy behaviour that these types carry unchanged (fixed in batch 3):
- The tiers' altitude laws raise thrust above hover to *increase* z, i.e. to
  descend (K01; P1.1).
- The acceleration-to-tilt mapping ignores yaw (frames.py; A-R2, P1.2).
"""

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


@dataclass(frozen=True, eq=False)
class AccelCommand:
    """One tier's command for one step: an acceleration demand plus yaw.

    accel and yaw are the command itself. The other four fields are the tier's
    legacy outputs, passed through unchanged: to_legacy_dict() rebuilds, with
    the same objects in the same key order, the dict that compute_control
    returned before this type existed. The CBF filter, the telemetry, the flight
    log and the ROS node consume that dict; the Smoothness and Effort metrics of
    the benchmarks read its torques (K16).

    accel: shape (3,), NED, m/s^2. The kinematic acceleration the tier's
        position law asks for, without gravity (zero means hold). It is taken
        before the tilt limit. Which components drive the legacy outputs
        differs per tier (see each compute_accel_command): x and y always set
        the tilt; z sets the thrust only for the MPC, while the PID and H-inf
        thrust come from their own altitude laws (the H-inf z is always 0).
    yaw: desired yaw in rad. The PID and MPC use the setpoint's yaw (current yaw
        if the setpoint has none); the H-inf holds the current yaw.
    thrust: legacy normalised collective thrust, from the tier's altitude law.
    torques: legacy body torques [roll, pitch, yaw], from the tier's inner loop.
    desired_attitude: legacy [roll, pitch, yaw] setpoint in rad; the node turns
        it into the attitude setpoint quaternion it sends to PX4.
    desired_rates: legacy body-rate setpoint in rad/s (zeros for the H-inf).
    """
    accel: np.ndarray
    yaw: float
    thrust: float
    torques: np.ndarray
    desired_attitude: np.ndarray
    desired_rates: np.ndarray

    def to_legacy_dict(self) -> Dict[str, Any]:
        """The legacy control dict: thrust, torques, desired_attitude, desired_rates."""
        return {
            'thrust': self.thrust,
            'torques': self.torques,
            'desired_attitude': self.desired_attitude,
            'desired_rates': self.desired_rates,
        }
