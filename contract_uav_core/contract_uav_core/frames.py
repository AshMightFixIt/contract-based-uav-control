"""
Acceleration -> attitude, thrust and torque mappings of the three tiers.

Moved verbatim (W4-04, P0a.9) from control/pid.py, control/mpc.py and
control/hinf.py; before the package split they were controllers.py:157-158
(PID), :293-295 (H-inf) and :526-527 (MPC). Frames: see interfaces.py (NED
world, z down; FRD body; ZYX Euler angles).

Changed on purpose in batch 3 (W4-05):
- Heading-aware tilt (P1.2, A-R2): accel_to_attitude rotates the NED
  horizontal demand by the yaw of the attitude setpoint it returns, into the
  heading frame (forward, right), before mapping it to pitch and roll. It used
  to map a_x/a_y straight to pitch/roll, as if the vehicle always faced north.
- Physically signed, hover-normalised thrust (P1.1, P1.2, K01, A-G6):
  accel_to_thrust turns the NED vertical acceleration demand into normalised
  collective thrust, thrust = hover_thrust * (1 - a_z / g): a demand to climb
  (a_z < 0, z is down) gives thrust above hover. This is the inverse of the
  thrust model of sim/quadrotor.py (specific force (thrust / hover) * g along
  body -z) and of PX4's MPC_THR_HOVER. It used to be hover + a_z / g (MPC only;
  more thrust to descend, and a gain that is right only for a hover of 1.0).
  All three tiers now derive their thrust from it.

How the three copies compared (checked line by line in W4-04):
- accel -> attitude: the same formula in all three. They differ only in the
  tilt limit (PID and H-inf: max_tilt = 0.22 rad; MPC:
  arctan2(max_lateral_accel, 9.81) = arctan2(3.0, 9.81)) and in the yaw passed
  through (PID and MPC: the setpoint's yaw, else the current yaw; H-inf: the
  current yaw). Both are parameters here.
- attitude -> rates -> torques: the PID and MPC share one structure with
  different constants (rate clip 1.0 vs 2.0 rad/s; torque gain kp_rate
  [0.08, 0.08, 0.06] vs the scalar 0.08). The H-inf law is different: a rate
  damping term plus an attitude term, with no rate setpoint.
"""

import numpy as np

GRAVITY = 9.81  # m/s^2; the same constant as sim/quadrotor.py

# The hover thrust every tier assumed before P1.2; a tier built without an
# airframe still uses it (control/base.py).
DEFAULT_HOVER_THRUST = 0.5

# m/s^2 per unit of normalised thrust at the legacy hover 0.5: g / 0.5. The PID
# and H-inf altitude gains were tuned in thrust units at hover 0.5; times this
# they are the same gains in m/s^2 per m and per m/s, so with no airframe the
# vertical loop keeps its legacy strength (control/pid.py, control/hinf.py).
LEGACY_THRUST_TO_ACCEL = GRAVITY / DEFAULT_HOVER_THRUST


def accel_to_attitude(accel: np.ndarray, yaw, max_tilt) -> np.ndarray:
    """Desired [roll, pitch, yaw] from an NED acceleration demand, heading-aware (P1.2).

    accel[0:2] (north, east) is rotated by yaw into the heading frame:
        a_forward =  cos(yaw) a_north + sin(yaw) a_east
        a_right   = -sin(yaw) a_north + cos(yaw) a_east
    then pitch = -a_forward / g (nose down to accelerate forward) and
    roll = a_right / g (right side down to accelerate right), each clipped to
    +-max_tilt. accel[2] is not used here (accel_to_thrust).

    yaw is returned as the attitude setpoint's yaw, so the rotation always uses
    the yaw the setpoint carries: PX4 (and sim/quadrotor.py) apply roll and pitch
    in the frame of that yaw (ZYX Euler). At yaw 0 this is the legacy mapping.
    """
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    a_forward = cos_yaw * accel[0] + sin_yaw * accel[1]
    a_right = -sin_yaw * accel[0] + cos_yaw * accel[1]

    desired_pitch = -a_forward / GRAVITY  # Negative: forward accel needs nose down
    desired_roll = a_right / GRAVITY      # Positive: right accel needs right roll

    # Limit tilt angles (in the heading frame, as before at yaw 0)
    desired_roll = np.clip(desired_roll, -max_tilt, max_tilt)
    desired_pitch = np.clip(desired_pitch, -max_tilt, max_tilt)

    return np.array([desired_roll, desired_pitch, yaw])


def accel_to_thrust(accel_z, hover_thrust, min_thrust=0.0, max_thrust=1.0):
    """Normalised collective thrust for an NED vertical acceleration demand (P1.1, P1.2).

    accel_z in m/s^2, z down: negative asks to accelerate up. The thrust model is
    linear with hover_thrust balancing gravity (sim/quadrotor.py; PX4's
    MPC_THR_HOVER), so the vertical acceleration is g - (thrust / hover) g and
        thrust = hover_thrust * (1 - accel_z / g),
    clipped to [min_thrust, max_thrust]. No tilt compensation (level-flight law).
    """
    desired_thrust = hover_thrust * (1.0 - accel_z / GRAVITY)
    desired_thrust = np.clip(desired_thrust, min_thrust, max_thrust)
    return desired_thrust


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
    """accel_to_thrust with the MPC's legacy clip [0, 1]; kept so no name disappears.

    Since P1.1/P1.2 (W4-05) this is the physically signed, hover-normalised law
    (thrust = hover * (1 - a_z / g)), no longer hover + a_z / g. The MPC itself
    calls accel_to_thrust with its airframe's limits.
    """
    return accel_to_thrust(accel_z, hover_thrust, 0.0, 1.0)
