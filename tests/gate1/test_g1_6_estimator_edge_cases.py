"""G1.6 (gate 1, docs/ROADMAP.md): the estimator handles its edge cases.

W4-08 (K17, A-F7 core). The estimator is ContractAwareEKF (estimation/ekf.py),
here at 50 Hz on synthetic measurements of a known truth (noise 1 cm, 1 cm/s,
1 mrad), except the first check, which goes through the whole core
(AdaptiveDroneController.control_step) the way the ROS node feeds it.

Measured before W4-08 (base 79a91eb, scratch probe in the W4-08 report): a GPS
flagged invalid with zeros, or a missing GPS position, pulled the estimate from
(10, -4, -5) to (0.07, -0.03, -0.03) in 1 s; the yaw error reached 137 degrees
crossing +-180; 20 s without GPS grew sqrt(P_xx) from 0.3 to 115.5 m with no
untrusted flag; GPS recovery snapped back in one step; nothing was gated; there
was no pass-through mode.
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.usefixtures("core_package")

DT = 0.02
GPS = {'satellites': 12, 'hdop': 0.8, 'valid': True}
NO_GPS = {'satellites': 0, 'hdop': 99.9, 'valid': False}
IMU = {'calibrated': True, 'temperature': 25.0, 'valid': True}


def make_ekf(**kw):
    from contract_uav_core.contracts.monitor import HierarchicalContractMonitor
    from contract_uav_core.estimation.ekf import ContractAwareEKF
    monitor = HierarchicalContractMonitor()
    monitor.define_contracts()
    return ContractAwareEKF(monitor, **kw)


class Truth:
    """A vehicle at constant velocity and yaw rate, and the estimator fed from it."""

    def __init__(self, pos=(10.0, -4.0, -5.0), vel=(0.0, 0.0, 0.0), yaw=0.0, yaw_rate=0.0, seed=0, **kw):
        self.ekf = make_ekf(**kw)
        self.pos, self.vel = np.array(pos, dtype=float), np.array(vel, dtype=float)
        self.yaw, self.yaw_rate = yaw, yaw_rate
        self.rng = np.random.default_rng(seed)
        self.t = 0.0

    def measurements(self):
        n = self.rng.standard_normal
        yaw = np.arctan2(np.sin(self.yaw), np.cos(self.yaw))       # as an IMU reports it
        return {'gps_position': self.pos + n(3) * 0.01, 'gps_velocity': self.vel + n(3) * 0.01,
                'imu_attitude': np.array([0.0, 0.0, yaw]) + n(3) * 0.001,
                'imu_rates': np.array([0.0, 0.0, self.yaw_rate]) + n(3) * 0.001}

    def step(self, edit=None, gps=GPS):
        self.t += DT
        self.pos = self.pos + self.vel * DT
        self.yaw += self.yaw_rate * DT
        meas = self.measurements()
        if edit:
            edit(meas)
        self.ekf.predict(DT)
        self.ekf.update(meas, {'gps': dict(gps), 'imu': dict(IMU)}, self.t)

    def fly(self, seconds, **kw):
        for _ in range(int(round(seconds / DT))):
            self.step(**kw)

    def position_error(self):
        return self.ekf.get_state()['position'] - self.pos


def test_g1_6_missing_gps_position_is_not_fused():
    """G1.6: a missing GPS position is not fused (before: fused as zeros, estimate to the origin)."""
    truth = Truth()
    truth.fly(5.0)
    truth.fly(1.0, edit=lambda m: m.pop('gps_position'))
    assert np.linalg.norm(truth.position_error()) < 0.01
    assert 'gps_position' not in truth.ekf.last_fused


def test_g1_6_invalid_flagged_gps_with_zeros_is_not_fused_through_the_core():
    """G1.6: an invalid-flagged GPS carrying zeros (A-F7) is not fused, through the whole core.

    The raw sensor dict is the one the ROS node builds when PX4 marks the position
    invalid but still reports satellites and HDOP (px4_conversions.build_raw_sensors).
    """
    from contract_uav_core.core import AdaptiveDroneController
    ctrl = AdaptiveDroneController(dt=DT, use_horizon_planner=False)
    target = np.array([10.0, -4.0, -5.0])
    conditions = {'gps_satellites': 12.0, 'gps_hdop': 1.0, 'imu_temperature': 25.0, 'imu_calibrated': 1.0,
                  'battery_voltage': 12.4, 'motor_temperature': 25.0, 'wind_speed': 1.0, 'disturbance': 0.2}
    assert ctrl.pre_flight_check(conditions, {'target_position': target, 'waypoints': [target]})[0]
    assert ctrl.start_mission()

    def raw(valid):
        pos = target if valid else np.zeros(3)
        return {'gps': {'position': pos.copy(), 'velocity': np.zeros(3), 'valid': valid,
                        'satellites': 12, 'hdop': 1.0},
                'imu': {'attitude': np.zeros(3), 'rates': np.zeros(3), 'valid': True,
                        'calibrated': True, 'temperature': 25.0},
                'battery': {'voltage': 12.4}, 'motors': {'temperature': 25.0}}

    for _ in range(250):
        _, telemetry = ctrl.control_step(raw(True))
    assert np.linalg.norm(telemetry['position'] - target) < 0.01
    for _ in range(50):
        _, telemetry = ctrl.control_step(raw(False))
    assert np.linalg.norm(telemetry['position'] - target) < 0.05     # base: about 11.5 m off, near the origin


def test_g1_6_yaw_error_below_1_degree_crossing_180():
    """G1.6: the yaw error stays below 1 degree while the heading crosses +-180 (before: 137 degrees)."""
    truth = Truth(yaw=2.5, yaw_rate=0.5)
    truth.fly(2.0)
    worst = 0.0
    for _ in range(int(20.0 / DT)):          # 10 rad: crosses +-pi more than once
        truth.step()
        est = truth.ekf.get_state()['attitude'][2]
        worst = max(worst, abs(np.arctan2(np.sin(est - truth.yaw), np.cos(est - truth.yaw))))
        assert -np.pi < est <= np.pi
    assert np.degrees(worst) < 1.0, np.degrees(worst)


def test_g1_6_twenty_second_outage_bounded_honest_and_untrusted():
    """G1.6: a 20 s GPS outage at constant velocity: the position error stays bounded, the
    3-sigma bound covers the true error, and the untrusted flag is set (before: sqrt(P_xx)
    115.5 m, no flag). The covariance covers any manoeuvre within the speed envelope: see
    tests/estimation/test_ekf_outage.py for braking and accelerating truths."""
    truth = Truth(vel=(1.0, 0.5, 0.0))
    truth.fly(10.0)
    assert truth.ekf.position_trusted
    for _ in range(int(20.0 / DT)):
        truth.step(gps=NO_GPS)
        P, err = truth.ekf.get_covariance(), truth.position_error()
        assert all(abs(err[i]) <= 3.0 * np.sqrt(P[i, i]) for i in range(3))
    assert np.linalg.norm(truth.position_error()) < 1.0
    assert np.sqrt(truth.ekf.get_covariance()[0, 0]) < 30.0     # 26.9 m (base 115.5 m)
    assert truth.ekf.position_trusted is False


def test_g1_6_recovery_correction_rate_at_most_2_m_s():
    """G1.6: on GPS recovery the correction rate is <= 2 m/s (before: one-step snap)."""
    truth = Truth(vel=(1.0, 0.0, 0.0))
    truth.fly(10.0)
    truth.vel = np.zeros(3)                 # the vehicle stops; dead reckoning keeps going
    truth.fly(10.0, gps=NO_GPS)
    assert np.linalg.norm(truth.position_error()) > 9.0
    rates = []
    for _ in range(int(8.0 / DT)):
        before = truth.ekf.get_state()
        truth.step()
        after = truth.ekf.get_state()
        rates.append(np.linalg.norm(after['position'] - before['position'] - before['velocity'] * DT) / DT)
    assert max(rates) <= 2.0 + 1e-9
    assert np.linalg.norm(truth.position_error()) < 0.1


def test_g1_6_outlier_rejected_real_jump_accepted_after_10_rejects():
    """G1.6: an outlier jump is rejected, and a real jump is accepted after 10 rejects."""
    truth = Truth()
    truth.fly(5.0)
    outlier = lambda m: m.__setitem__('gps_position', m['gps_position'] + [25.0, 0.0, 0.0])
    truth.step(edit=outlier)
    assert 'gps_position' in truth.ekf.last_rejected
    truth.fly(1.0)
    assert np.linalg.norm(truth.position_error()) < 0.01

    truth.pos = truth.pos + [25.0, -10.0, 0.0]   # a real jump (e.g. a GPS reference change)
    for k in range(9):
        truth.step()
        assert 'gps_position' in truth.ekf.last_rejected, k
    truth.step()
    assert 'gps_position' in truth.ekf.last_reset
    assert np.linalg.norm(truth.ekf.x[0:3] - truth.pos) < 0.1      # accepted by the filter
    truth.fly(20.0)
    assert np.linalg.norm(truth.position_error()) < 0.05             # and published, after the glide


def test_g1_6_passthrough_returns_px4_estimate_with_eph_epv_covariance():
    """G1.6: pass-through returns PX4's estimate with the eph/epv covariance."""
    ekf = make_ekf(mode='passthrough')
    pos, vel = np.array([12.5, -3.25, -7.0]), np.array([0.75, -0.5, 0.125])
    meas = {'gps_position': pos, 'gps_velocity': vel, 'imu_attitude': np.zeros(3), 'imu_rates': np.zeros(3)}
    gps = dict(GPS, eph=1.2, epv=2.0, evh=0.4, evv=0.3)
    ekf.predict(DT)
    state, _ = ekf.update(meas, {'gps': gps, 'imu': dict(IMU)}, 0.0)
    assert np.array_equal(state[0:3], pos) and np.array_equal(state[3:6], vel)
    P = ekf.get_covariance()
    assert np.allclose(P[0:6, 0:6], np.diag([0.72, 0.72, 4.0, 0.08, 0.08, 0.09]))
    assert ekf.get_status()['eph'] == pytest.approx(1.2) and ekf.get_status()['epv'] == pytest.approx(2.0)
