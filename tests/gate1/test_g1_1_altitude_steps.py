"""G1.1 (gate 1, docs/ROADMAP.md): height steps settle for every controller.

W4-05 (P1.1, P1.2). Each tier (PID, MPC, H-infinity), built with the
x500_sitl airframe, flies the reference plant sim/quadrotor.py (same airframe)
in closed loop at dt = 0.02 s, from rest at z = -5 m (5 m up, NED) with yaw 0.
The setpoint steps 1 m up (z = -6) or 1 m down (z = -4). Pass: within 10 s
the altitude is within 0.2 m of the setpoint and stays there to the end of a
15 s run, and the vehicle never touches the ground.

Before W4-05 every tier fell to the ground within 2.84 s in these steps
(thrust sign K01). Reproduced here: PID 5.38 s (45 % overshoot), MPC 4.66 s,
H-infinity 4.62 s.
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.usefixtures("core_package")

DT = 0.02
START_Z = -5.0
TOLERANCE = 0.2     # m
SETTLE_BY = 10.0    # s
RUN = 15.0          # s


def fly(tier, setpoint, seconds):
    """Closed loop of one tier on the reference plant; returns [(t, state, thrust)]."""
    from contract_uav_core.config.airframe import load_airframe
    from contract_uav_core.control.hinf import HInfinityController
    from contract_uav_core.control.mpc import MPCController
    from contract_uav_core.control.pid import PIDController
    from contract_uav_core.interfaces import AttitudeThrustCommand
    from contract_uav_core.sim.quadrotor import QuadrotorParams, QuadrotorPlant

    airframe = load_airframe('x500_sitl')
    cls = {'PID': PIDController, 'MPC': MPCController, 'Hinf': HInfinityController}[tier]
    controller = cls(DT, airframe=airframe)
    plant = QuadrotorPlant(QuadrotorParams.from_airframe(airframe), position=(0.0, 0.0, START_Z))
    state, log = plant.state, []
    for k in range(int(round(seconds / DT))):
        control = controller.compute_control(
            {'position': state.position, 'velocity': state.velocity,
             'attitude': state.attitude, 'rates': state.rates}, setpoint)
        state = plant.step(AttitudeThrustCommand.from_legacy(control), DT)
        log.append(((k + 1) * DT, state, float(control['thrust'])))
    return log


def settling_time(log, error, tolerance):
    """The time from which error(state) < tolerance to the end of the log (inf if never)."""
    last_out = 0.0
    for t, state, _ in log:
        if error(state) >= tolerance:
            last_out = t
    return float('inf') if last_out >= log[-1][0] else last_out


@pytest.mark.parametrize('step', [-1.0, 1.0], ids=['1m-up', '1m-down'])
@pytest.mark.parametrize('tier', ['PID', 'MPC', 'Hinf'])
def test_g1_1_altitude_step_settles(tier, step):
    """G1.1: a +-1 m altitude step settles within 0.2 m in 10 s, with no ground contact."""
    target_z = START_Z + step
    setpoint = {'position': np.array([0.0, 0.0, target_z]), 'velocity': np.zeros(3), 'yaw': 0.0}
    log = fly(tier, setpoint, RUN)

    assert not any(state.on_ground for _, state, _ in log), tier
    settle = settling_time(log, lambda s: abs(s.position[2] - target_z), TOLERANCE)
    assert settle <= SETTLE_BY, (tier, step, settle)
    # it moved the right way from the first second (the K01 failure went the other way)
    z_1s = next(state.position[2] for t, state, _ in log if t >= 1.0)
    assert np.sign(z_1s - START_Z) == np.sign(step), (tier, z_1s)
    # level flight, no lateral drift, and thrust inside the airframe's limits (0.12 .. 1.0)
    assert all(np.linalg.norm(state.position[:2]) < 1e-9 for _, state, _ in log)
    assert all(0.12 <= thrust <= 1.0 for _, _, thrust in log)
