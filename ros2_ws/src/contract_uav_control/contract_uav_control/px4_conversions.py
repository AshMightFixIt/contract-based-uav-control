"""Conversions between PX4 (px4_msgs) and the contract-based controller's dict format.

Frame conventions — both sides use NED, so positions/velocities pass through directly:
  * PX4 VehicleLocalPosition: x=North, y=East, z=Down (positive down) in metres.
  * The controller's state['position'] / 'velocity' use the same NED convention
    (negative z = altitude above ground), so no axis remapping is needed.

Attitude:
  * PX4 VehicleAttitude.q is [w, x, y, z], rotation from NED earth frame to FRD body.
  * The controller uses Euler [roll, pitch, yaw] (rad), so we convert quat -> Euler in,
    and Euler -> quat out for the attitude setpoint.

Thrust:
  * The controller emits a normalized collective thrust in [0, 1] (0.5 = hover).
  * PX4 VehicleAttitudeSetpoint.thrust_body is body-frame; for a multicopter the
    collective acts along body -Z (up), so thrust_body = [0, 0, -thrust].
"""

import numpy as np


def quaternion_to_euler(q):
    """PX4 quaternion [w, x, y, z] -> Euler [roll, pitch, yaw] (rad), ZYX convention."""
    w, x, y, z = q[0], q[1], q[2], q[3]

    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])


def euler_to_quaternion(roll, pitch, yaw):
    """Euler [roll, pitch, yaw] (rad) -> PX4 quaternion [w, x, y, z], ZYX convention."""
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return [float(w), float(x), float(y), float(z)]


def build_raw_sensors(local_pos, attitude, ang_vel, sensor_gps, battery):
    """Assemble the raw_sensors dict expected by AdaptiveDroneController.control_step().

    Any argument may be None if that message has not been received yet; the
    corresponding sub-dict is marked invalid so the EKF degrades gracefully.
    """
    raw = {}

    if local_pos is not None and local_pos.xy_valid and local_pos.z_valid:
        raw['gps'] = {
            'position': np.array([local_pos.x, local_pos.y, local_pos.z]),
            'velocity': np.array([local_pos.vx, local_pos.vy, local_pos.vz]),
            'valid': True,
            'satellites': int(sensor_gps.satellites_used) if sensor_gps is not None else 12,
            'hdop': float(sensor_gps.hdop) if sensor_gps is not None else 1.0,
        }
    else:
        raw['gps'] = {
            'position': np.zeros(3),
            'velocity': np.zeros(3),
            'valid': False,
            'satellites': int(sensor_gps.satellites_used) if sensor_gps is not None else 0,
            'hdop': float(sensor_gps.hdop) if sensor_gps is not None else 99.0,
        }

    if attitude is not None and ang_vel is not None:
        raw['imu'] = {
            'attitude': quaternion_to_euler(attitude.q),
            'rates': np.array([ang_vel.xyz[0], ang_vel.xyz[1], ang_vel.xyz[2]]),
            'valid': True,
            'calibrated': True,
            'temperature': 25.0,
        }
    else:
        raw['imu'] = {
            'attitude': np.zeros(3),
            'rates': np.zeros(3),
            'valid': False,
            'calibrated': False,
            'temperature': 25.0,
        }

    if battery is not None:
        raw['battery'] = {'voltage': float(battery.voltage_v)}
    else:
        raw['battery'] = {'voltage': 12.0}

    raw['motors'] = {'temperature': 25.0}
    return raw
