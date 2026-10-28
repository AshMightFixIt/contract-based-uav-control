"""G1.2 (gate 1, docs/ROADMAP.md), controller level: sideways steps settle at any heading.

W4-05 (P1.2). Each tier (PID, MPC, H-infinity), built with the x500_sitl
airframe, flies the reference plant sim/quadrotor.py (same airframe) in closed
loop at dt = 0.02 s, from rest at (0, 0, -5) with the heading held at yaw 0,
90, 180 or 270 deg (the setpoint's yaw equals the start yaw). The setpoint is
10 m to the north or 10 m to the east, at the same altitude. Pass: within 20 s
the horizontal distance to the setpoint is within 0.5 m and stays there to the
end of a 30 s run; no ground contact; the altitude stays within 0.5 m and the
heading is held; and the settling time is the same at all four headings.

Before W4-05 the tilt ignored the heading (A-R2): at 90, 180 and 270 deg the
error grew to 46-161 m (and the thrust sign, K01, put every tier on the
ground). Reproduced here: PID 12.4 s, MPC 8.2 s, H-infinity 15.7 s, the same at
every heading.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

pytestmark = pytest.mark.usefixtures("core_package")

DT = 0.02
START = (0.0, 0.0, -5.0)
TOLERANCE = 0.5     # m, horizontal
SETTLE_BY = 20.0    # s
RUN = 30.0          # s
YAWS = (0, 90, 180, 270)


def fly(tier, yaw, setpoint, seconds):
    """Closed loop of one tier on the reference plant; returns [(t, state)]."""
    from contract_uav_core.config.airframe import load_airframe
    from contract_uav_core.control.hinf import HInfinityController
    from contract_uav_core.control.mpc import MPCController
    from contract_uav_core.control.pid import PIDController
    from contract_uav_core.interfaces import AttitudeThrustCommand
    from contract_uav_core.sim.quadrotor import QuadrotorParams, QuadrotorPlant

    airframe = load_airframe('x500_sitl')
    cls = {'PID': PIDController, 'MPC': MPCController, 'Hinf': HInfinityController}[tier]
    controller = cls(DT, airframe=airframe)
    plant = QuadrotorPlant(QuadrotorParams.from_airframe(airframe), position=START,
                           attitude=(0.0, 0.0, yaw))
    state, log = plant.state, []
    for k in range(int(round(seconds / DT))):
        control = controller.compute_control(
            {'position': state.position, 'velocity': state.velocity,
             'attitude': state.attitude, 'rates': state.rates}, setpoint)
        state = plant.step(AttitudeThrustCommand.from_legacy(control), DT)
        log.append(((k + 1) * DT, state))
    return log


def settling_time(log, error, tolerance):
    """The time from which error(state) < tolerance to the end of the log (inf if never)."""
    last_out = 0.0
    for t, state in log:
        if error(state) >= tolerance:
            last_out = t
    return float('inf') if last_out >= log[-1][0] else last_out


@pytest.mark.parametrize('direction', [(10.0, 0.0), (0.0, 10.0)], ids=['10m-north', '10m-east'])
@pytest.mark.parametrize('tier', ['PID', 'MPC', 'Hinf'])
def test_g1_2_lateral_step_settles_at_any_heading(tier, direction):
    """G1.2 (controller level): a 10 m lateral step converges within 0.5 m at yaw 0, 90, 180, 270."""
    goal = np.array([START[0] + direction[0], START[1] + direction[1]])
    settle_times = {}
    for yaw_deg in YAWS:
        yaw = math.radians(yaw_deg)
        setpoint = {'position': np.array([goal[0], goal[1], START[2]]), 'velocity': np.zeros(3),
                    'yaw': yaw}
        log = fly(tier, yaw, setpoint, RUN)
        where = (tier, direction, yaw_deg)

        assert not any(state.on_ground for _, state in log), where
        assert all(abs(state.position[2] - START[2]) < 0.5 for _, state in log), where
        assert all(abs(math.remainder(state.attitude[2] - yaw, 2 * math.pi)) < 1e-9 for _, state in log), where
        # it heads for the goal from the start (the A-R2 failure went sideways or backwards)
        p_2s = next(state.position[:2] for t, state in log if t >= 2.0)
        assert np.dot(p_2s - START[:2], goal - START[:2]) / (np.linalg.norm(p_2s - START[:2]) * 10.0) > 0.99, where
        settle = settling_time(log, lambda s: np.linalg.norm(s.position[:2] - goal), TOLERANCE)
        assert settle <= SETTLE_BY, (where, settle)
        settle_times[yaw_deg] = settle
    # heading-invariant: the same response at every heading (to within a step or two)
    assert max(settle_times.values()) - min(settle_times.values()) <= 2 * DT + 1e-9, settle_times
