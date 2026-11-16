"""The command the core has just sent, as the NED acceleration it produces (W4-08b, T1).

Without GPS velocity the estimator propagates the velocity with this acceleration
(ContractAwareEKF.predict's accel_cmd, or set_command for the next predict), so the
estimate follows what the vehicle does in response to its own commands. Before
W4-08b it kept the last velocity: right only when nobody steers. In closed loop the
supervisor answers a GPS loss with HOVER, the controller then brakes or adds thrust,
the estimate did not see it, and the vehicle ran away (46.6 m up in an 8 s outage).

Which command, and at which attitude:
- The thrust is the final one, control['thrust'] after the safety filter (CBF) and the
  tier's saturation: what the vehicle receives, not the tier's AccelCommand before
  them.
- Its direction is the estimated attitude (from the IMU), not the commanded one. The
  vehicle reaches a commanded attitude only through its attitude loop's lag (tau 0.1 s
  in sim/quadrotor.py; PX4's attitude controller likewise), and the velocity error of
  that lag does not decay: it stays at tau times the change of tilt acceleration. Used
  with the commanded attitude, an outage while braking from 3 m/s ended with the
  estimate 2.8 m off after 8 s; with the estimated attitude, 1.1 m (W4-08b report).
  Without a usable IMU (imu_ok False) the commanded attitude is used.
- The hover thrust is the one the tier normalised the thrust with (its hover_thrust:
  0.5 without an airframe), so the conversion is the exact inverse of the tier's.

The remaining mismatch (drag, wind, a hover-thrust error) is learned by the estimator
while GPS is there (ContractAwareEKF.command_bias) and covered by its process noise.
"""

from typing import Any, Mapping, Optional

import numpy as np

from ..frames import attitude_thrust_to_accel


def commanded_acceleration(control: Mapping[str, Any], estimated_attitude, imu_ok: bool,
                           hover_thrust: float) -> Optional[np.ndarray]:
    """NED acceleration (m/s^2, gravity included) of the command just sent, or None.

    control: the final control dict (after the CBF filter): its 'thrust', and its
    'desired_attitude' when imu_ok is False. estimated_attitude: the estimator's
    [roll, pitch, yaw]. hover_thrust: the sending tier's. None if the dict has no
    thrust (or no attitude to use); the estimator then keeps the velocity, as before.
    """
    thrust = control.get('thrust')
    attitude = estimated_attitude if imu_ok else control.get('desired_attitude')
    if thrust is None or attitude is None:
        return None
    return attitude_thrust_to_accel(attitude, thrust, hover_thrust)
