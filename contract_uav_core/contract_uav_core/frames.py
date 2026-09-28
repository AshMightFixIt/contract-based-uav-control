"""
Acceleration -> attitude, thrust and torque mappings of the three tiers.

Moved verbatim (W4-04, P0a.9) from control/pid.py, control/mpc.py and
control/hinf.py; before the package split they were controllers.py:157-158
(PID), :293-295 (H-inf) and :526-527 (MPC). The arithmetic and its order are
unchanged, so the tiers' outputs are bit-identical. Frames: see interfaces.py.

Legacy behaviour kept on purpose (batch 3 fixes it, P1.1 and P1.2):
- Yaw-blind: accel_to_attitude maps the NED x/y acceleration straight to
  pitch/roll, as if the vehicle always pointed north; the demand is not rotated
  into the body frame by the yaw (A-R2).
- Sign: pitch = -a_x / 9.81 (nose down to go north at yaw 0) and
  roll = a_y / 9.81.
- The MPC thrust mapping adds a_z / 9.81 to hover: with NED z down this raises
  thrust to go down (K01), and hover is the constant 0.5 of each tier.

How the three copies compared (checked line by line):
- accel -> attitude: the same formula in all three. They differ only in the
  tilt limit (PID and H-inf: max_tilt = 0.22 rad; MPC:
  arctan2(max_lateral_accel, 9.81) = arctan2(3.0, 9.81)) and in the yaw passed
  through (PID and MPC: the setpoint's yaw, else the current yaw; H-inf: the
  current yaw). Both are parameters here.
- attitude -> rates -> torques: the PID and MPC share one structure with
  different constants (rate clip 1.0 vs 2.0 rad/s; torque gain kp_rate
  [0.08, 0.08, 0.06] vs the scalar 0.08). The H-inf law is different: a rate
  damping term plus an attitude term, with no rate setpoint.
- accel -> thrust: only the MPC derives thrust from the acceleration demand.
"""

import numpy as np


def accel_to_attitude(accel: np.ndarray, yaw, max_tilt) -> np.ndarray:
    """Desired [roll, pitch, yaw] from an NED acceleration demand (yaw-blind).

    From pid.py (formerly controllers.py:157-158); mpc.py and hinf.py computed
    the same values.
    """
    desired_pitch = -accel[0] / 9.81  # Negative: forward accel needs nose down
    desired_roll = accel[1] / 9.81    # Positive: right accel needs right roll

    # Limit tilt angles
    desired_roll = np.clip(desired_roll, -max_tilt, max_tilt)
    desired_pitch = np.clip(desired_pitch, -max_tilt, max_tilt)

    return np.array([desired_roll, desired_pitch, yaw])


def attitude_to_rates_torques(desired_attitude: np.ndarray,
                              attitude: np.ndarray,
                              rates: np.ndarray,
                              kp_att, kd_att, max_rate, kp_rate):
    """PID/MPC inner loops: attitude error -> rate setpoint -> torques.

    Returns (desired_rates, torques). From pid.py (MPC: max_rate 2.0, kp_rate 0.08).
    Note: this writes nothing into its inputs; att_error is a new array.
    """
    # === INNER LOOP: Attitude Control ===
    att_error = desired_attitude - attitude

    # Normalize yaw error to [-pi, pi]
    att_error[2] = np.arctan2(np.sin(att_error[2]), np.cos(att_error[2]))

    # PD control for desired rates
    desired_rates = kp_att * att_error - kd_att * rates
    desired_rates = np.clip(desired_rates, -max_rate, max_rate)

    # === INNERMOST LOOP: Rate Control ===
    rate_error = desired_rates - rates

    # P control for torques
    torques = kp_rate * rate_error

    return desired_rates, torques


def hinf_attitude_torques(target_attitude: np.ndarray,
                          attitude: np.ndarray,
                          rates: np.ndarray,
                          k_rate_damping, k_attitude) -> np.ndarray:
    """H-inf torques: rate damping plus attitude correction. From hinf.py."""
    # === RATE DAMPING (Highest Priority) ===
    rate_damping_torque = -k_rate_damping * rates

    # === ATTITUDE STABILIZATION ===
    att_error = target_attitude - attitude
    att_error[2] = np.arctan2(np.sin(att_error[2]), np.cos(att_error[2]))

    attitude_torque = k_attitude * att_error

    torques = rate_damping_torque + attitude_torque
    return torques


def mpc_accel_to_thrust(accel_z, hover_thrust):
    """MPC: normalised thrust from the vertical acceleration demand. From mpc.py.

    Simulation convention: thrust > hover -> z increases (descent in NED).
    MPC accel_cmd[2] < 0 means "move z negative" (climb) -> need less thrust.
    """
    desired_thrust = hover_thrust + accel_z / 9.81
    desired_thrust = np.clip(desired_thrust, 0.0, 1.0)
    return desired_thrust
