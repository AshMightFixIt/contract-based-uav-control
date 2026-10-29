"""P1.1 / P1.2 (W4-05): the thrust sign, the heading-aware tilt and the hover normalisation.

Unit tests of frames.accel_to_attitude and frames.accel_to_thrust, of the three
tiers' use of them (with and without an airframe), and of the legacy script
plants' thrust sign. NED world, z down (interfaces.py): a target above the
vehicle has a more negative z and must give thrust above hover.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.usefixtures("core_package")

G = 9.81
TIERS = ['PID', 'MPC', 'Hinf']


def tier_class(tier):
    from contract_uav_core.control.hinf import HInfinityController
    from contract_uav_core.control.mpc import MPCController
    from contract_uav_core.control.pid import PIDController
    return {'PID': PIDController, 'MPC': MPCController, 'Hinf': HInfinityController}[tier]


def make(tier, airframe=None):
    return tier_class(tier)(0.02) if airframe is None else tier_class(tier)(0.02, airframe=airframe)


def airframe(name):
    from contract_uav_core.config.airframe import load_airframe
    return load_airframe(name)


def hover_only(hover):
    """An airframe-like object with only the three thrust fields; limits [0, 1]."""
    return SimpleNamespace(hover_thrust=hover, min_thrust=0.0, max_thrust=1.0)


def at_rest(z=-5.0, yaw=0.0):
    return {'position': np.array([0.0, 0.0, z]), 'velocity': np.zeros(3),
            'attitude': np.array([0.0, 0.0, yaw]), 'rates': np.zeros(3)}


def target(x=0.0, y=0.0, z=-5.0, yaw=0.0):
    return {'position': np.array([x, y, z]), 'velocity': np.zeros(3), 'yaw': yaw}


# ---------------------------------------------------------------------------
# frames.accel_to_thrust: sign and normalisation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('hover', [0.22, 0.5, 0.6])
def test_accel_to_thrust_is_signed_and_normalised(hover):
    from contract_uav_core.frames import accel_to_thrust
    assert accel_to_thrust(0.0, hover) == hover                         # hold: exactly hover
    assert accel_to_thrust(-1.0, hover) > hover > accel_to_thrust(1.0, hover)   # climb (z down) needs more
    assert accel_to_thrust(-0.5 * G, hover) == pytest.approx(1.5 * hover)
    assert accel_to_thrust(0.25 * G, hover) == pytest.approx(0.75 * hover)
    assert accel_to_thrust(2.0 * G, hover) == 0.0                       # clipped at the default limits
    assert accel_to_thrust(-10 * G, hover) == 1.0
    assert accel_to_thrust(-10 * G, hover, 0.1, 0.5) == 0.5             # the airframe's limits
    assert accel_to_thrust(10 * G, hover, 0.1, 0.5) == 0.1


@pytest.mark.parametrize('name, mass', [('x500_sitl', None), ('dexi', 0.8)])
def test_accel_to_thrust_inverts_the_reference_plant(name, mass):
    """Level and without drag, the plant's vertical acceleration is exactly the demand."""
    import dataclasses
    from contract_uav_core.frames import accel_to_thrust
    from contract_uav_core.interfaces import AttitudeThrustCommand
    from contract_uav_core.sim.quadrotor import QuadrotorParams, QuadrotorPlant

    af = airframe(name)
    params = dataclasses.replace(QuadrotorParams.from_airframe(af, mass=mass), drag_coefficient=0.0)
    for demand in (-3.0, -1.0, 0.0, 2.0):
        plant = QuadrotorPlant(params, position=(0.0, 0.0, -10.0))
        thrust = accel_to_thrust(demand, af.hover_thrust, af.min_thrust, af.max_thrust)
        state = plant.step(AttitudeThrustCommand(attitude=np.zeros(3), thrust=float(thrust)), 0.01)
        assert state.velocity[2] / 0.01 == pytest.approx(demand, abs=1e-9), (name, demand)


# ---------------------------------------------------------------------------
# frames.accel_to_attitude: heading-aware tilt
# ---------------------------------------------------------------------------
# A north demand of 1 m/s^2 at each heading: (roll, pitch) in units of 1/g.
NORTH_DEMAND = [
    (0, (0.0, -1.0)),       # facing north: nose down
    (90, (-1.0, 0.0)),      # facing east: north is to the left, left side down
    (180, (0.0, 1.0)),      # facing south: nose up
    (270, (1.0, 0.0)),      # facing west: north is to the right, right side down
]


@pytest.mark.parametrize('yaw_deg, roll_pitch', NORTH_DEMAND)
def test_tilt_for_each_yaw(yaw_deg, roll_pitch):
    from contract_uav_core.frames import accel_to_attitude
    yaw = math.radians(yaw_deg)
    att = accel_to_attitude(np.array([1.0, 0.0, 0.0]), yaw, 0.5)
    assert att[0] == pytest.approx(roll_pitch[0] / G, abs=1e-15)
    assert att[1] == pytest.approx(roll_pitch[1] / G, abs=1e-15)
    assert att[2] == yaw                                               # the setpoint's yaw, unchanged
    # an east demand at the same heading: the same tilt turned by 90 deg
    att_e = accel_to_attitude(np.array([0.0, 1.0, 0.0]), yaw, 0.5)
    att_n = accel_to_attitude(np.array([1.0, 0.0, 0.0]), yaw - math.pi / 2, 0.5)
    assert att_e[:2] == pytest.approx(att_n[:2], abs=1e-15)


def test_tilt_at_yaw_zero_is_the_legacy_mapping():
    """Within the limit, and for a saturating demand along one axis, yaw 0 gives the legacy values."""
    from contract_uav_core.frames import accel_to_attitude
    for a in ([1.0, -0.5, 0.3], [-0.4, 0.7, 0.0], [10.0, 0.0, 0.0], [0.0, -10.0, 0.0]):
        a = np.array(a)
        expected = [np.clip(a[1] / G, -0.22, 0.22), np.clip(-a[0] / G, -0.22, 0.22), 0.0]
        assert accel_to_attitude(a, 0.0, 0.22).tolist() == expected


DIRECTIONS = [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (-0.6, 0.8), (-1.0, -0.2)]


@pytest.mark.parametrize('yaw_deg', [45, 135, 0, 90, 225, -45, 30])
@pytest.mark.parametrize('direction', DIRECTIONS)
def test_tilt_magnitude_stays_within_the_limit_at_any_heading(yaw_deg, direction):
    """N2: a saturating demand tilts by exactly max_tilt, in the demand's direction, at any heading.

    (Clipping roll and pitch separately gave up to sqrt(2) * max_tilt at 45 and 135 deg.)
    """
    from contract_uav_core.frames import accel_to_attitude
    yaw, max_tilt = math.radians(yaw_deg), 0.22
    demand = 20.0 * np.array([direction[0], direction[1], 0.0]) / math.hypot(*direction)
    att = accel_to_attitude(demand, yaw, max_tilt)
    free = accel_to_attitude(demand, yaw, 10.0)                      # the same, with no limit acting
    assert math.hypot(att[0], att[1]) == pytest.approx(max_tilt, rel=1e-12)
    assert math.hypot(free[0], free[1]) == pytest.approx(20.0 / G, rel=1e-12)
    assert att[:2] == pytest.approx(free[:2] * max_tilt / math.hypot(free[0], free[1]), rel=1e-12, abs=1e-15)
    assert att[2] == yaw
    # inside the limit nothing is scaled
    small = accel_to_attitude(0.1 * demand, yaw, max_tilt)
    assert small[:2] == pytest.approx(0.1 * free[:2], rel=1e-12, abs=1e-15)


def test_tilt_limit_applies_in_the_heading_frame():
    from contract_uav_core.frames import accel_to_attitude
    att = accel_to_attitude(np.array([0.0, 30.0, 0.0]), math.radians(90), 0.22)   # east, facing east
    assert att[1] == -0.22 and abs(att[0]) < 1e-15                    # all pitch, clipped
    att = accel_to_attitude(np.array([30.0, 0.0, 0.0]), math.radians(90), 0.22)   # north, facing east
    assert att[0] == -0.22 and abs(att[1]) < 1e-15                    # all roll, clipped


@pytest.mark.parametrize('scale', [1.0, 20.0], ids=['within-limit', 'saturating'])
@pytest.mark.parametrize('yaw_deg', [0, 30, 45, 90, 135, 180, 270, -45])
@pytest.mark.parametrize('direction', [(1.0, 0.0), (0.0, 1.0), (-0.6, 0.8)])
def test_plant_accelerates_along_the_demand_at_any_heading(x500_params, yaw_deg, direction, scale):
    """The attitude from accel_to_attitude moves sim/quadrotor.py along the NED demand (also when limited)."""
    from contract_uav_core.frames import accel_to_attitude, accel_to_thrust
    from contract_uav_core.interfaces import AttitudeThrustCommand
    from contract_uav_core.sim.quadrotor import QuadrotorPlant

    yaw = math.radians(yaw_deg)
    demand = scale * np.array([direction[0], direction[1], 0.0])
    att = accel_to_attitude(demand, yaw, 0.3)
    plant = QuadrotorPlant(x500_params, position=(0.0, 0.0, -20.0), attitude=(att[0], att[1], yaw))
    cmd = AttitudeThrustCommand(attitude=att, thrust=float(accel_to_thrust(0.0, x500_params.hover_thrust)))
    for _ in range(50):
        state = plant.step(cmd, 0.01)
    horizontal = state.velocity[:2]
    assert np.dot(horizontal, direction) / np.linalg.norm(horizontal) > 0.999, (yaw_deg, horizontal)


@pytest.fixture(scope='module')
def x500_params():
    from contract_uav_core.sim.quadrotor import QuadrotorParams
    return QuadrotorParams.from_airframe(airframe('x500_sitl'))


# ---------------------------------------------------------------------------
# The tiers: sign, rotation, normalisation, airframe
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('af', [None, 'x500_sitl', 'dexi'])
@pytest.mark.parametrize('tier', TIERS)
def test_target_above_gives_thrust_above_hover(tier, af):
    """G1.1 (unit level): a setpoint above the drone (more negative z) gives thrust above hover."""
    frame = None if af is None else airframe(af)
    hover = 0.5 if frame is None else frame.hover_thrust
    up = make(tier, frame).compute_accel_command(at_rest(-5.0), target(z=-6.0))
    down = make(tier, frame).compute_accel_command(at_rest(-5.0), target(z=-4.0))
    hold = make(tier, frame).compute_accel_command(at_rest(-5.0), target(z=-5.0))
    assert up.accel[2] < 0.0 < down.accel[2] and hold.accel[2] == 0.0
    assert up.thrust > hover > down.thrust
    assert hold.thrust == hover
    # climbing already, the rate term asks for less thrust than at rest
    climbing = dict(at_rest(-5.0), velocity=np.array([0.0, 0.0, -2.0]))
    assert make(tier, frame).compute_accel_command(climbing, target(z=-5.0)).thrust < hover


@pytest.mark.parametrize('tier', TIERS)
def test_no_airframe_defaults_to_hover_half(tier):
    ctrl = make(tier)
    assert (ctrl.hover_thrust, ctrl.min_thrust, ctrl.max_thrust, ctrl.airframe) == (0.5, 0.0, 1.0, None)
    for state, sp in [(at_rest(-5.0), target(z=-5.0)), (at_rest(-5.0), target(3.0, -2.0, -6.5, 0.4))]:
        a = make(tier).compute_control(state, sp)
        b = make(tier, hover_only(0.5)).compute_control(state, sp)
        for key in a:
            assert np.asarray(a[key]).tobytes() == np.asarray(b[key]).tobytes(), (tier, key)


@pytest.mark.parametrize('tier', TIERS)
def test_thrust_is_normalised_by_hover(tier):
    """The same demand at hover 0.22 and 0.5: thrust / hover is the same."""
    for state, sp in [(at_rest(-5.0), target(z=-5.8)), (at_rest(-5.0), target(z=-4.6)),
                      (dict(at_rest(-5.0), velocity=np.array([0.0, 0.0, 0.7])), target(z=-5.0))]:
        low = make(tier, hover_only(0.22)).compute_accel_command(state, sp)
        half = make(tier, hover_only(0.5)).compute_accel_command(state, sp)
        assert low.accel[2] == half.accel[2]                              # the demand does not depend on hover
        assert low.thrust / 0.22 == pytest.approx(half.thrust / 0.5, rel=1e-12)
        assert low.thrust != pytest.approx(half.thrust)


def test_dexi_airframe_sets_hover_and_limits():
    dexi = airframe('dexi')
    # far targets: PID saturates at the airframe's limits; the MPC (|a_z| <= 5 m/s^2) and
    # the H-inf (|a_z| <= 0.8 g) saturate their demand first, inside the limits
    expected = {'PID': (0.5, 0.025), 'MPC': (0.22 * (1 + 5.0 / G), 0.22 * (1 - 5.0 / G)),
                'Hinf': (0.22 * 1.8, 0.22 * 0.2)}
    for tier in TIERS:
        ctrl = make(tier, dexi)
        assert (ctrl.hover_thrust, ctrl.min_thrust, ctrl.max_thrust) == (0.22, 0.025, 0.5)
        assert ctrl.airframe is dexi
        far_up = ctrl.compute_accel_command(at_rest(-5.0), target(z=-60.0))
        far_down = make(tier, dexi).compute_accel_command(at_rest(-60.0), target(z=-5.0))
        assert (far_up.thrust, far_down.thrust) == pytest.approx(expected[tier]), tier


@pytest.mark.parametrize('tier', TIERS)
def test_airframe_thrust_limits_clip_every_tier(tier):
    band = SimpleNamespace(hover_thrust=0.22, min_thrust=0.2, max_thrust=0.25)
    assert make(tier, band).compute_accel_command(at_rest(-5.0), target(z=-60.0)).thrust == 0.25
    assert make(tier, band).compute_accel_command(at_rest(-60.0), target(z=-5.0)).thrust == 0.2


def test_hinf_vertical_limit_scales_with_hover():
    """The H-inf's old thrust band [0.1, 0.9] at hover 0.5 is +-0.8 g at any hover."""
    for hover in (0.22, 0.5):
        up = make('Hinf', hover_only(hover)).compute_accel_command(at_rest(-5.0), target(z=-80.0))
        down = make('Hinf', hover_only(hover)).compute_accel_command(at_rest(-80.0), target(z=-5.0))
        assert up.accel[2] == pytest.approx(-0.8 * G) and down.accel[2] == pytest.approx(0.8 * G)
        assert up.thrust == pytest.approx(1.8 * hover) and down.thrust == pytest.approx(0.2 * hover)


@pytest.mark.parametrize('bad', [dict(hover_thrust=0.0), dict(min_thrust=0.7), dict(max_thrust=0.3),
                                 dict(hover_thrust=float('nan')), dict(max_thrust=1.5)])
def test_invalid_airframe_values_are_rejected(bad):
    values = dict(hover_thrust=0.5, min_thrust=0.1, max_thrust=0.9)
    values.update(bad)
    for tier in TIERS:
        with pytest.raises(ValueError, match='hover_thrust'):
            make(tier, SimpleNamespace(**values))


@pytest.mark.parametrize('tier', ['PID', 'MPC'])
@pytest.mark.parametrize('yaw_deg, roll_pitch', NORTH_DEMAND)
def test_pid_mpc_rotate_by_the_setpoint_yaw(tier, yaw_deg, roll_pitch):
    """A target to the north, heading held at yaw: the tilt points north in the heading frame."""
    yaw = math.radians(yaw_deg)
    cmd = make(tier).compute_accel_command(at_rest(yaw=yaw), target(x=1.0, yaw=yaw))
    assert cmd.yaw == yaw and cmd.desired_attitude[2] == yaw
    roll, pitch = cmd.desired_attitude[:2]
    assert np.sign(np.round([roll, pitch], 12)).tolist() == list(roll_pitch), (roll, pitch)


@pytest.mark.parametrize('yaw_deg, roll_pitch', NORTH_DEMAND)
def test_hinf_rotates_by_the_yaw_it_commands(yaw_deg, roll_pitch):
    """The H-inf holds the current yaw, whatever the setpoint's, and rotates by that yaw."""
    yaw = math.radians(yaw_deg)
    for setpoint_yaw in (yaw, yaw + 1.0):
        cmd = make('Hinf').compute_accel_command(at_rest(yaw=yaw), target(x=1.0, yaw=setpoint_yaw))
        assert cmd.yaw == yaw and cmd.desired_attitude[2] == yaw
        roll, pitch = cmd.desired_attitude[:2]
        assert np.sign(np.round([roll, pitch], 12)).tolist() == list(roll_pitch), (setpoint_yaw, roll, pitch)


# ---------------------------------------------------------------------------
# The legacy script plants (one line each; W4-10 replaces them)
# ---------------------------------------------------------------------------
SCRIPT_PLANTS = [
    ('benchmarks/benchmark_controllers.py', 'DroneSimulation'),
    ('benchmarks/benchmark_sensor_degradation.py', 'DroneSimulation'),
    ('demos/demo_waypoint_mission.py', 'DroneSimulation'),
    ('demos/demo_sensor_degradation.py', 'DroneSimulation'),
    ('demos/simulation_demo.py', 'DroneSimulation'),
]

PLANT_PROBE = r"""
import importlib.util, json, sys
import numpy as np
out = {}
for path, cls in json.loads(sys.argv[1]):
    spec = importlib.util.spec_from_file_location('plant_probe', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for thrust in (0.6, 0.4):
        sim = getattr(module, cls)()
        z0 = float(sim.position[2])
        for _ in range(10):
            sim.update({'thrust': thrust, 'desired_attitude': np.zeros(3), 'torques': np.zeros(3)})
        out[f'{path} {thrust}'] = float(sim.position[2]) - z0
from contract_uav_core.sim.racing_scenario import RacingDroneSimulation
for thrust in (0.6, 0.4):
    sim = RacingDroneSimulation()
    z0 = float(sim.position[2])
    for _ in range(10):
        sim.update({'thrust': thrust, 'desired_attitude': np.zeros(3)})
    out[f'racing_scenario {thrust}'] = float(sim.position[2]) - z0
print(json.dumps(out))
"""


def test_legacy_plants_climb_with_thrust_above_half(repo_copy, run_python):
    """The six legacy plants: thrust above 0.5 (their hover) climbs, i.e. z decreases (K01)."""
    import json
    plants = [[str(repo_copy / path), cls] for path, cls in SCRIPT_PLANTS]
    proc = run_python(['-c', PLANT_PROBE, json.dumps(plants)], cwd=repo_copy, timeout=300)
    assert proc.returncode == 0, proc.stderr[-3000:]
    dz = json.loads(proc.stdout.strip().splitlines()[-1])
    assert len(dz) == 12
    for key, change in dz.items():
        if key.endswith(' 0.6'):
            assert change < 0.0, (key, change)        # NED: up
        else:
            assert change > 0.0, (key, change)        # down
