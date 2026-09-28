"""
Reference quadrotor plant (W4-04, P0a.10): rigid body, attitude + collective thrust.

Not used by the legacy scripts; batch 3 closes the loop on it (gate 1). It is
deterministic: no global state, no randomness, fixed-step integration.

Frames (interfaces.py): NED world, z down, so altitude is -z; FRD body;
attitude = ZYX Euler angles [roll, pitch, yaw] in rad.

Input: AttitudeThrustCommand (interfaces.py), i.e. what the ROS node sends to
PX4 (VehicleAttitudeSetpoint: attitude quaternion + normalised collective
thrust). The plant stands in for PX4 plus the airframe. It deliberately does not
take an AccelCommand: the acceleration -> attitude/thrust mapping (frames.py) is
part of the controller under test, and the defects batch 3 must fix live there
(yaw blindness A-R2, thrust sign K01, hover normalisation). A legacy control
dict converts with AttitudeThrustCommand.from_legacy(control).

Model, one step of length dt (semi-implicit Euler):
1. Attitude: first-order lag toward the commanded attitude with time constant
   tau: att += (cmd - att) * min(dt / tau, 1), the yaw error wrapped to
   [-pi, pi]. rates is the resulting Euler-angle rate of the step.
2. Thrust: u = clip(thrust, 0, max_thrust); the specific force (u / hover_thrust) * g
   acts along body -z, i.e. UP when level. Linear in u by assumption, so
   u = hover_thrust exactly balances gravity; mass cancels.
3. Gravity: +g along NED z.
4. Drag and wind: F = -k |v_rel| v_rel with v_rel = v - wind (wind in NED, m/s),
   divided by the mass. Wind acts only through this force.
5. v += a dt, then p += v dt (with the new v).
6. Ground at z = ground_z (default 0): the vehicle cannot go below it; on
   contact z is set to ground_z and a downward vertical velocity to 0 (no
   bounce). There is no friction: horizontal motion on the ground continues.
Heading: the thrust direction comes from the full rotation Rz(yaw) Ry(pitch)
Rx(roll), so a nose-down pitch moves the vehicle along its heading (north at
yaw 0, east at yaw 90 deg).

Parameters (QuadrotorParams) come from an airframe file
(config/airframes/*.yaml) via QuadrotorParams.from_airframe.
"""

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from ..interfaces import AttitudeThrustCommand

GRAVITY = 9.81  # m/s^2, the constant the core's mappings use (frames.py)


@dataclass(frozen=True)
class QuadrotorParams:
    """Plant parameters. mass kg; hover_thrust and max_thrust normalised; tau s; drag kg/m."""
    mass: float
    hover_thrust: float
    max_thrust: float
    attitude_time_constant: float
    drag_coefficient: float
    gravity: float = GRAVITY

    def __post_init__(self):
        for name in ('mass', 'hover_thrust', 'max_thrust', 'attitude_time_constant', 'gravity'):
            value = getattr(self, name)
            if not (isinstance(value, (int, float)) and math.isfinite(value) and value > 0):
                raise ValueError(f"QuadrotorParams.{name} must be a finite number > 0, got {value!r}")
        if not (math.isfinite(self.drag_coefficient) and self.drag_coefficient >= 0):
            raise ValueError(f"QuadrotorParams.drag_coefficient must be >= 0, got {self.drag_coefficient!r}")
        if self.hover_thrust > self.max_thrust:
            raise ValueError(f"hover_thrust {self.hover_thrust} exceeds max_thrust {self.max_thrust}: "
                             "the plant could not hover")

    @classmethod
    def from_airframe(cls, airframe, mass: Optional[float] = None) -> 'QuadrotorParams':
        """Parameters from a config.airframe.Airframe; mass overrides the file (needed if unknown)."""
        if mass is None:
            mass = airframe.mass
        if mass is None:
            note = airframe.provenance['mass'].note
            raise ValueError(f"airframe {airframe.name!r}: the mass is unknown ({note}); "
                             "pass mass= explicitly")
        return cls(mass=float(mass), hover_thrust=airframe.hover_thrust,
                   max_thrust=airframe.max_thrust,
                   attitude_time_constant=airframe.attitude_time_constant,
                   drag_coefficient=airframe.drag_coefficient)


@dataclass(frozen=True, eq=False)
class QuadrotorState:
    """A snapshot of the plant (copies). NED m, m/s; attitude rad; rates rad/s (Euler-angle rates)."""
    position: np.ndarray
    velocity: np.ndarray
    attitude: np.ndarray
    rates: np.ndarray
    on_ground: bool


def _wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def _vector(value, name) -> np.ndarray:
    array = np.array(value, dtype=float)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be 3 finite numbers, got {value!r}")
    return array


class QuadrotorPlant:
    """The reference plant; see the module docstring for the model."""

    def __init__(self, params: QuadrotorParams,
                 position: Sequence[float] = (0.0, 0.0, 0.0),
                 velocity: Sequence[float] = (0.0, 0.0, 0.0),
                 attitude: Sequence[float] = (0.0, 0.0, 0.0),
                 ground_z: float = 0.0):
        self.params = params
        self.ground_z = float(ground_z)
        self._position = _vector(position, 'position')
        self._velocity = _vector(velocity, 'velocity')
        self._attitude = _vector(attitude, 'attitude')
        self._rates = np.zeros(3)
        if self._position[2] > self.ground_z:
            raise ValueError(f"initial z {self._position[2]} is below the ground (z = {self.ground_z}, NED)")

    @property
    def state(self) -> QuadrotorState:
        return QuadrotorState(position=self._position.copy(), velocity=self._velocity.copy(),
                              attitude=self._attitude.copy(), rates=self._rates.copy(),
                              on_ground=bool(self._position[2] >= self.ground_z))

    def step(self, cmd: AttitudeThrustCommand, dt: float,
             wind: Optional[Sequence[float]] = None) -> QuadrotorState:
        """Advance by dt seconds under cmd and the wind (NED, m/s); return the new state."""
        if not (isinstance(dt, (int, float)) and math.isfinite(dt) and dt > 0):
            raise ValueError(f"dt must be a finite number > 0, got {dt!r}")
        target = _vector(cmd.attitude, 'cmd.attitude')
        thrust = float(cmd.thrust)
        if not math.isfinite(thrust):
            raise ValueError(f"cmd.thrust must be finite, got {cmd.thrust!r}")
        wind = np.zeros(3) if wind is None else _vector(wind, 'wind')
        p = self.params

        # 1. attitude: first-order lag
        alpha = min(dt / p.attitude_time_constant, 1.0)
        error = target - self._attitude
        error[2] = _wrap(error[2])
        attitude = self._attitude + alpha * error
        attitude[2] = _wrap(attitude[2])
        rates = alpha * error / dt

        # 2.-4. forces at the new attitude, per unit mass
        u = min(max(thrust, 0.0), p.max_thrust)
        specific_thrust = (u / p.hover_thrust) * p.gravity
        roll, pitch, yaw = attitude
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        body_z = np.array([cy * sp * cr + sy * sr,     # third column of Rz(yaw) Ry(pitch) Rx(roll)
                           sy * sp * cr - cy * sr,
                           cp * cr])
        v_rel = self._velocity - wind
        accel = (-specific_thrust * body_z
                 + np.array([0.0, 0.0, p.gravity])
                 - (p.drag_coefficient / p.mass) * np.linalg.norm(v_rel) * v_rel)

        # 5. semi-implicit Euler
        velocity = self._velocity + accel * dt
        position = self._position + velocity * dt

        # 6. ground contact (NED: below ground means z > ground_z)
        if position[2] >= self.ground_z:
            position[2] = self.ground_z
            velocity[2] = min(velocity[2], 0.0)

        self._position, self._velocity = position, velocity
        self._attitude, self._rates = attitude, rates
        return self.state
