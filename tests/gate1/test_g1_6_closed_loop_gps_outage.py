"""G1.6 (gate 1, docs/ROADMAP.md): GPS outages in closed loop, the core flying sim/quadrotor.py.

W4-08b. The whole core (AdaptiveDroneController.control_step) flies the reference
plant (hover thrust 0.5, attitude lag 0.1 s, drag 0.05 kg/m, 2 kg) in the wind
of the orchestrator's GPS-climb probe; the GPS is lost the way the benchmarks
lose it (valid False, 2 satellites, HDOP 10). Noise: 1 cm, 1 cm/s, 1 mrad.

Measured at base 4a06fb3 (the W4-08 estimator, which kept the last velocity
whatever the controller commanded; scratch probes in the W4-08b report):
- the probe itself (an 8 s outage while climbing): the vehicle climbed from
  -8.6 to -55.2 m at up to 12.5 m/s, and the return glided a 52 m error at 2 m/s;
- hover, 8 s: 0.85 m horizontally, 1.55 m vertically, |vz| up to 0.48 m/s;
- hover, 20 s: 14.4 m and 5.0 m, |vz| up to 0.95 m/s;
- flying at 1 m/s: 26.1 m and 7.4 m, |vz| up to 2.1 m/s; at 3 m/s: 30.2 m and
  9.1 m, |vz| up to 3.0 m/s.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

pytestmark = pytest.mark.usefixtures("core_package")

DT = 0.02
START = np.array([0.0, 0.0, -5.0])
# The probe's wind (NED, m/s), as (start, end, wind) within a 120 s cycle
WIND = [(0, 25, (0.5, 0, 0)), (25, 45, (4.5, 2, 0)), (45, 60, (9, 4, 0)), (60, 80, (5, 2, 0)),
        (80, 120, (1, 0.5, 0))]
T_START = 70.0      # the runs start here, hovering at rest at START, in the 5 m/s wind
T_LOSS = 100.0      # the GPS is lost here (as in the probe), in the 1 m/s wind
GOOD = {'gps_satellites': 12.0, 'gps_hdop': 0.8, 'imu_temperature': 25.0, 'imu_calibrated': 1.0,
        'battery_voltage': 12.4, 'motor_temperature': 30.0, 'wind_speed': 0.5, 'disturbance': 0.2}


def wind_at(t):
    c = t % 120.0
    for a, b, w in WIND:
        if a <= c < b:
            return np.array(w, dtype=float)
    return np.zeros(3)


class Run:
    """One closed-loop flight; per step: time, GPS flag, true position and velocity, the published
    position and velocity, the setpoint, and the estimator's status."""

    def __init__(self, outage, *, speed=None, gust=None, after=20.0, seed=1):
        from contract_uav_core.core import AdaptiveDroneController
        from contract_uav_core.interfaces import AttitudeThrustCommand
        from contract_uav_core.sim.quadrotor import QuadrotorParams, QuadrotorPlant
        logging.disable(logging.WARNING)
        try:
            ctrl = AdaptiveDroneController(dt=DT)
            assert ctrl.pre_flight_check(GOOD, {'target_position': START, 'waypoints': [START]})[0]
            assert ctrl.start_mission()
            ctrl.command_hover(START)
            plant = QuadrotorPlant(QuadrotorParams(mass=2.0, hover_thrust=0.5, max_thrust=1.0,
                                                   attitude_time_constant=0.1, drag_coefficient=0.05),
                                   position=START)
            rng = np.random.default_rng(seed)
            s = plant.state
            self.t_loss = None if speed is not None else T_LOSS
            rows = []
            k = 0
            while True:
                t = T_START + k * DT
                if self.t_loss is not None and t >= self.t_loss + outage + after - 1e-9:
                    break
                if speed is not None and self.t_loss is None:
                    if k == int(round((95.0 - T_START) / DT)):           # off toward a far waypoint
                        ctrl.set_waypoint_mission([np.array([80.0, 0.0, -5.0])])
                    if np.linalg.norm(s.velocity[0:2]) >= speed:       # lost when the speed is reached
                        self.t_loss = t
                gps = not (self.t_loss is not None and self.t_loss <= t < self.t_loss + outage)
                wind = wind_at(t)
                if gust is not None and self.t_loss is not None and self.t_loss + 1.0 <= t < self.t_loss + outage:
                    wind = np.array(gust, dtype=float)          # a gust the estimator cannot see
                raw = {'gps': {'position': s.position + rng.normal(0, 0.01, 3),
                               'velocity': s.velocity + rng.normal(0, 0.01, 3),
                               'valid': gps, 'satellites': 12 if gps else 2, 'hdop': 0.8 if gps else 10.0},
                       'imu': {'attitude': s.attitude + rng.normal(0, 0.001, 3),
                               'rates': s.rates + rng.normal(0, 0.001, 3),
                               'valid': True, 'calibrated': True, 'temperature': 25.0},
                       'battery': {'voltage': 12.4}, 'motors': {'temperature': 30.0},
                       'wind': {'speed': float(np.linalg.norm(wind[:2])), 'direction': 0.0}}
                control, tel = ctrl.control_step(raw)
                status = ctrl.ekf.get_status()
                rows.append((t, gps, s.position, s.velocity, tel['position'], tel['velocity'],
                             tel['setpoint']['position'], status))
                s = plant.step(AttitudeThrustCommand.from_legacy(control), DT, wind=wind)
                k += 1
        finally:
            logging.disable(logging.NOTSET)
        self.outage = outage
        self.t = np.array([r[0] for r in rows])
        self.gps = np.array([r[1] for r in rows])
        self.true_pos = np.array([r[2] for r in rows])
        self.true_vel = np.array([r[3] for r in rows])
        self.est_pos = np.array([r[4] for r in rows])
        self.est_vel = np.array([r[5] for r in rows])
        self.setpoint = np.array([r[6] for r in rows])
        self.status = [r[7] for r in rows]
        self.lost = ~self.gps
        self.back = np.flatnonzero(self.gps & (self.t >= self.t_loss))[0]     # first step with GPS again

    def deviation(self):
        """True position minus the setpoint (the HOVER setpoint captured at the loss)."""
        return self.true_pos - self.setpoint

    def settled_after_return(self, tolerance=0.5):
        """Seconds from the return until the true position stays within tolerance of the setpoint."""
        far = np.linalg.norm(self.deviation(), axis=1) > tolerance
        far[:self.back] = True
        last_far = np.flatnonzero(far)[-1]
        assert last_far < len(self.t) - 1, "never settled"
        return self.t[last_far + 1] - self.t[self.back]


@pytest.fixture(scope="module")
def hover8():
    return Run(8.0)


@pytest.fixture(scope="module")
def hover20():
    return Run(20.0)


@pytest.mark.parametrize("name, bound, base", [("hover8", 0.5, (0.85, 1.55)), ("hover20", 1.0, (14.4, 5.0))])
def test_g1_6_hover_outage_stays_put_without_runaway(name, bound, base, request):
    """G1.6: an 8 s and a 20 s GPS outage in hover, with the probe's wind: the true position stays
    within `bound` of the hover setpoint horizontally and vertically (measured 0.14 / 0.02 m and
    0.65 / 0.04 m; base: `base` m), |true vz| < 1 m/s throughout (measured 0.004 m/s; base 0.48
    and 0.95 m/s), and it is within 0.5 m of the setpoint 2 s after the return (measured 0 and
    0.8 s). The 20 s drift is the noise of the learned bias (0.65 m was the worst of 6 seeds)."""
    run = request.getfixturevalue(name)
    dev = run.deviation()[run.lost]
    assert np.max(np.linalg.norm(dev[:, 0:2], axis=1)) < bound
    assert np.max(np.abs(dev[:, 2])) < bound
    assert np.max(np.abs(run.true_vel[run.lost, 2])) < 1.0
    assert run.settled_after_return() <= 2.0
    # the covariance grows honestly: untrusted (eph > 5 m) by the end of the outage, trusted after
    assert not run.status[run.back - 1]['position_trusted'] and run.status[-1]['position_trusted']


@pytest.mark.parametrize("speed, braking, base", [(1.0, 2.0, 26.1), (3.0, 4.0, 30.2)])
def test_g1_6_outage_while_flying_does_not_run_away(speed, braking, base):
    """G1.6: GPS lost while flying at 1 m/s and at 3 m/s: the supervisor's HOVER brakes, and the
    vehicle stops and holds. The true position stays within `braking` m of the point where GPS
    was lost (the braking distance; measured 0.8 and 2.0 m; base `base` m horizontally), within
    0.5 m vertically (measured 0.18 m), |true vz| < 1 m/s (measured 0.10 and 0.15 m/s; base 2.1
    and 3.0 m/s), and it is back within 0.5 m of the setpoint 5 s after the return (measured
    2.3 and 2.7 s)."""
    run = Run(8.0, speed=speed)
    assert np.linalg.norm(run.true_vel[run.lost][0, 0:2]) == pytest.approx(speed, abs=0.1)
    dev = run.deviation()[run.lost]
    assert np.max(np.linalg.norm(dev[:, 0:2], axis=1)) < braking
    assert np.max(np.abs(dev[:, 2])) < 0.5
    assert np.max(np.abs(run.true_vel[run.lost, 2])) < 1.0
    assert run.settled_after_return() <= 5.0


def correction_rates(run):
    """Per step from the return: how fast the published position moves beyond the published
    velocity, |p(k) - p(k-1) - v(k-1) dt| / dt (the glide's measure, m/s)."""
    pub, vel = run.est_pos[run.back - 1:], run.est_vel[run.back - 1:]
    return np.linalg.norm(pub[1:] - pub[:-1] - vel[:-1] * DT, axis=1) / DT


def test_g1_6_large_offset_at_return_is_reset_and_reaches_the_controller_at_once():
    """G1.6: a 6 m/s gust that the estimator cannot see blows the vehicle 12.9 m off during an 8 s
    outage. At the return the position is reset, not glided: the published position is the true
    one at that very step, with one reset event (xy_reset_counter, delta_xy), and the vehicle is
    back within 0.5 m of the setpoint 15 s later (measured 12.3 s). W4-08 glided it at 2 m/s,
    so the controller saw the whole error only after 6.5 s (a 52 m one in the probe: 26 s)."""
    run = Run(8.0, gust=(6.0, 0.0, 0.0))
    offset = run.true_pos[run.back - 1] - run.est_pos[run.back - 1]
    assert np.linalg.norm(offset[0:2]) > 10.0 and abs(offset[2]) < 0.5
    status = run.status[run.back]
    assert status['xy_reset_counter'] == 1 and status['z_reset_counter'] == 0
    assert np.linalg.norm(status['delta_xy'] - offset[0:2]) < 0.2
    assert np.linalg.norm(run.est_pos[run.back] - run.true_pos[run.back]) < 0.05
    assert run.status[-1]['xy_reset_counter'] == 1                 # one event, no more
    assert run.settled_after_return() <= 15.0
    assert np.max(np.abs(run.true_vel[:, 2])) < 1.0


def test_g1_6_small_offset_at_return_still_glides_at_2_m_s():
    """G1.6: a 3 m/s gust leaves 3.7 m at the return, within RESET_OFFSET: it glides at <= 2 m/s
    (at the limit), with no reset event, and the vehicle is back within 0.5 m of the setpoint
    8 s later (measured 6.5 s)."""
    run = Run(8.0, gust=(3.0, 0.0, 0.0))
    offset = run.true_pos[run.back - 1] - run.est_pos[run.back - 1]
    assert 3.0 < np.linalg.norm(offset) < 5.0
    rates = correction_rates(run)
    # 1 mm/s: the first return step was predicted with the command, p += v dt + a dt^2 / 2, which
    # the measure (v dt only) does not include
    assert np.max(rates) <= 2.0 + 1e-3
    assert np.max(rates) > 0.99 * 2.0
    assert run.status[run.back]['gliding']
    assert all(s['xy_reset_counter'] == 0 and s['z_reset_counter'] == 0 for s in run.status)
    assert run.settled_after_return() <= 8.0


def test_g1_6_the_gps_climb_probe_no_longer_runs_away():
    """G1.6: the orchestrator's probe exactly: the core patrols four waypoints from t = 0 and loses
    GPS for 8 s at t = 100 s while climbing toward -8 m. At base the vehicle then climbed from
    -8.6 to -55.2 m (vz -12.5 m/s) with the estimate saying -4.0 m. Now the true height stays
    within 1.5 m of the hover setpoint (measured 1.2 m), |true vz| < 1 m/s throughout (0.71 m/s,
    the climb being stopped at the loss), the horizontal position within 4 m (2.7 m: it was
    flying at 2.4 m/s), and the estimate within RESET_OFFSET of the truth at the end of the
    outage (4.0 m), so the return glides for less than 2.5 s. The horizontal error is the
    learned bias holding the drag of the flight before the loss, which vanishes once the vehicle
    has stopped (W4-08b report, section 5)."""
    from contract_uav_core.core import AdaptiveDroneController
    from contract_uav_core.interfaces import AttitudeThrustCommand
    from contract_uav_core.sim.quadrotor import QuadrotorParams, QuadrotorPlant
    waypoints = [np.array([10.0, 0.0, -5.0]), np.array([10.0, 10.0, -8.0]),
                 np.array([0.0, 10.0, -6.0]), np.array([0.0, 0.0, -5.0])]
    logging.disable(logging.WARNING)
    try:
        ctrl = AdaptiveDroneController(dt=DT)
        assert ctrl.pre_flight_check(GOOD, {'target_position': waypoints[0], 'waypoints': list(waypoints)})[0]
        ctrl.start_mission()
        plant = QuadrotorPlant(QuadrotorParams(mass=2.0, hover_thrust=0.5, max_thrust=1.0,
                                               attitude_time_constant=0.1, drag_coefficient=0.05),
                               position=(0, 0, -5))
        rng = np.random.default_rng(1)
        s = plant.state
        lost = []
        for k in range(int(112 / DT)):
            t = k * DT
            wind = wind_at(t)
            gps = not (100.0 <= t < 108.0)
            raw = {'gps': {'position': s.position + rng.normal(0, 0.01, 3),
                           'velocity': s.velocity + rng.normal(0, 0.01, 3),
                           'valid': gps, 'satellites': 12 if gps else 2, 'hdop': 0.8 if gps else 10.0},
                   'imu': {'attitude': s.attitude + rng.normal(0, 0.001, 3), 'rates': s.rates + rng.normal(0, 0.001, 3),
                           'valid': True, 'calibrated': True, 'temperature': 25.0},
                   'battery': {'voltage': 12.4}, 'motors': {'temperature': 30.0},
                   'wind': {'speed': float(np.linalg.norm(wind[:2])), 'direction': 0.0}}
            control, tel = ctrl.control_step(raw)
            if not gps:
                lost.append((s.position - tel['setpoint']['position'], s.velocity[2],
                             np.linalg.norm(tel['position'] - s.position)))
            s = plant.step(AttitudeThrustCommand.from_legacy(control), DT, wind=wind)
            if tel['flight_mode'] == 'hover' and gps and ctrl.supervisor.current_waypoint_idx >= 3:
                ctrl.set_waypoint_mission(waypoints)
    finally:
        logging.disable(logging.NOTSET)
    dev = np.array([d for d, _, _ in lost])
    assert len(lost) == int(8.0 / DT)
    assert np.max(np.abs(dev[:, 2])) < 1.5
    assert np.max(np.abs([vz for _, vz, _ in lost])) < 1.0
    assert np.max(np.linalg.norm(dev[:, 0:2], axis=1)) < 4.0
    assert lost[-1][2] < 5.0
