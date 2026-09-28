"""The reference plant sim/quadrotor.py: thrust up, ground, heading, wind, determinism."""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

pytestmark = pytest.mark.usefixtures("core_package")

DT = 0.01


@pytest.fixture(scope='module')
def x500():
    from contract_uav_core.config.airframe import load_airframe
    from contract_uav_core.sim.quadrotor import QuadrotorParams
    return QuadrotorParams.from_airframe(load_airframe('x500_sitl'))


def plant(params, **kw):
    from contract_uav_core.sim.quadrotor import QuadrotorPlant
    return QuadrotorPlant(params, **kw)


def cmd(thrust, roll=0.0, pitch=0.0, yaw=0.0):
    from contract_uav_core.interfaces import AttitudeThrustCommand
    return AttitudeThrustCommand(attitude=np.array([roll, pitch, yaw]), thrust=thrust)


def fly(p, command, seconds, wind=None):
    states = []
    for _ in range(int(round(seconds / DT))):
        states.append(p.step(command, DT, wind=wind))
    return states


def test_params_come_from_the_airframe(x500):
    assert (x500.mass, x500.hover_thrust, x500.max_thrust) == (2.0643, 0.60, 1.0)
    assert (x500.attitude_time_constant, x500.drag_coefficient) == (0.15, 0.05)


def test_thrust_above_hover_climbs(x500):
    p = plant(x500, position=(0.0, 0.0, -10.0))
    last = fly(p, cmd(1.2 * x500.hover_thrust), 2.0)[-1]
    assert last.position[2] < -10.0 - 3.0      # NED: climbing is z decreasing
    assert last.velocity[2] < 0.0
    # Without drag the climb rate is exactly (u / hover - 1) g t.
    p = plant(dataclasses.replace(x500, drag_coefficient=0.0), position=(0.0, 0.0, -10.0))
    assert fly(p, cmd(1.2 * x500.hover_thrust), 2.0)[-1].velocity[2] == pytest.approx(-0.2 * 9.81 * 2.0)


def test_thrust_below_hover_descends(x500):
    p = plant(x500, position=(0.0, 0.0, -10.0))
    last = fly(p, cmd(0.8 * x500.hover_thrust), 2.0)[-1]
    assert last.position[2] > -10.0 + 3.0
    assert last.velocity[2] > 0.0


def test_thrust_at_hover_holds_altitude(x500):
    p = plant(x500, position=(1.0, 2.0, -5.0))
    for state in fly(p, cmd(x500.hover_thrust), 10.0):
        assert abs(state.position[2] + 5.0) < 1e-9
        assert np.all(np.abs(state.velocity) < 1e-9)


def test_ground_stops_descent_without_bounce(x500):
    p = plant(x500, position=(0.0, 0.0, -1.0))
    states = fly(p, cmd(0.0), 3.0)
    touchdown = next(i for i, s in enumerate(states) if s.on_ground)
    assert all(s.position[2] <= 0.0 for s in states)                  # never below the ground
    assert all(s.position[2] == 0.0 and s.velocity[2] == 0.0 and s.on_ground
               for s in states[touchdown:])                          # stays down: no bounce
    assert (touchdown + 1) * DT == pytest.approx(math.sqrt(2 * 1.0 / 9.81), abs=0.03)
    # thrust above hover lifts off again
    assert fly(p, cmd(1.5 * x500.hover_thrust), 0.5)[-1].position[2] < -0.1


def test_heading_rotates_the_body_axes(x500):
    """A nose-down pitch moves the vehicle along its heading: north, east, south."""
    pitch = -0.1
    for yaw_deg, direction in [(0, (1.0, 0.0)), (90, (0.0, 1.0)), (180, (-1.0, 0.0))]:
        yaw = math.radians(yaw_deg)
        p = plant(x500, position=(0.0, 0.0, -20.0), attitude=(0.0, 0.0, yaw))
        last = fly(p, cmd(x500.hover_thrust / math.cos(pitch), pitch=pitch, yaw=yaw), 3.0)[-1]
        horizontal = last.position[:2]
        distance = np.linalg.norm(horizontal)
        assert distance > 2.0, yaw_deg
        assert np.dot(horizontal / distance, direction) > 0.999, (yaw_deg, horizontal)
        assert last.attitude[2] == pytest.approx(yaw, abs=1e-9)


def test_wind_pushes_a_hovering_vehicle_downwind(x500):
    for wind, axis, sign in [((4.0, 0.0, 0.0), 0, 1), ((0.0, -4.0, 0.0), 1, -1)]:
        p = plant(x500, position=(0.0, 0.0, -10.0))
        states = fly(p, cmd(x500.hover_thrust), 5.0, wind=wind)
        assert sign * states[-1].position[axis] > 1.0
        assert sign * states[-1].velocity[axis] > 0.0
        # the force acts on the air-relative velocity, so the vehicle never outruns the wind
        assert abs(states[-1].velocity[axis]) < 4.0
        assert abs(states[-1].position[2] + 10.0) < 1e-9     # horizontal wind, level: no climb


def test_same_inputs_give_an_identical_trajectory(x500):
    def run():
        p = plant(x500, position=(0.0, 0.0, -3.0), attitude=(0.0, 0.0, 0.3))
        out = []
        for k in range(800):
            c = cmd(0.55 + 0.1 * math.sin(0.01 * k), roll=0.05 * math.cos(0.02 * k),
                    pitch=-0.08, yaw=0.3 + 0.001 * k)
            s = p.step(c, DT, wind=(2.0 * math.sin(0.005 * k), 1.0, 0.0))
            out.append(np.concatenate([s.position, s.velocity, s.attitude, s.rates]).tobytes())
        return out
    assert run() == run()


def test_max_thrust_clips_the_command():
    from contract_uav_core.config.airframe import load_airframe
    from contract_uav_core.sim.quadrotor import QuadrotorParams

    dexi = load_airframe('dexi')
    with pytest.raises(ValueError, match="mass is unknown"):
        QuadrotorParams.from_airframe(dexi)
    params = QuadrotorParams.from_airframe(dexi, mass=0.8)   # a test mass, not the vehicle's
    assert (params.hover_thrust, params.max_thrust) == (0.22, 0.5)
    full = plant(params, position=(0.0, 0.0, -10.0)).step(cmd(1.0), DT)
    capped = plant(params, position=(0.0, 0.0, -10.0)).step(cmd(0.5), DT)
    assert full.velocity.tobytes() == capped.velocity.tobytes()
    assert full.velocity[2] == pytest.approx(-(0.5 / 0.22 - 1.0) * 9.81 * DT)


def test_legacy_control_dict_converts():
    from contract_uav_core.interfaces import AttitudeThrustCommand

    c = AttitudeThrustCommand.from_legacy({'thrust': np.float64(0.6), 'torques': np.zeros(3),
                                           'desired_attitude': np.array([0.1, -0.2, 0.3]),
                                           'desired_rates': np.zeros(3)})
    assert c.thrust == 0.6 and type(c.thrust) is float
    assert c.attitude.tolist() == [0.1, -0.2, 0.3]


def test_invalid_inputs_are_rejected(x500):
    from contract_uav_core.sim.quadrotor import QuadrotorParams

    p = plant(x500, position=(0.0, 0.0, -1.0))
    for bad_dt in (0.0, -0.01, float('nan')):
        with pytest.raises(ValueError, match='dt'):
            p.step(cmd(0.6), bad_dt)
    with pytest.raises(ValueError, match='below the ground'):
        plant(x500, position=(0.0, 0.0, 0.5))
    with pytest.raises(ValueError, match='could not hover'):
        QuadrotorParams(mass=1.0, hover_thrust=0.6, max_thrust=0.5,
                        attitude_time_constant=0.1, drag_coefficient=0.0)
