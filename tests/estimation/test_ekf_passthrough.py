"""ContractAwareEKF(mode='passthrough'): PX4's own estimate as the state (W4-08 T7, decision D7).

In pass-through mode the GPS measurements are PX4's position and velocity
estimate (the node sends vehicle_local_position). They are taken as they are,
when flagged valid, with a covariance from eph / epv (evh / evv for the
velocity); the IMU is still fused. The default mode stays 'fusion'.
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.usefixtures("core_package")

DT = 0.02
IMU = {'calibrated': True, 'temperature': 25.0, 'valid': True}


def make_ekf(**kw):
    from contract_uav_core.contracts.monitor import HierarchicalContractMonitor
    from contract_uav_core.estimation.ekf import ContractAwareEKF
    monitor = HierarchicalContractMonitor()
    monitor.define_contracts()
    return ContractAwareEKF(monitor, **kw)


def px4(pos, vel, **gps):
    meas = {'gps_position': np.array(pos, dtype=float), 'gps_velocity': np.array(vel, dtype=float),
            'imu_attitude': np.array([0.01, -0.02, 2.0]), 'imu_rates': np.zeros(3)}
    sensors = {'gps': {'satellites': 12, 'hdop': 0.8, 'valid': True, **gps}, 'imu': dict(IMU)}
    return meas, sensors


def step(ekf, meas, sensors, t=1.0):
    ekf.predict(DT)
    return ekf.update(meas, sensors, t)


def test_default_mode_is_fusion():
    assert make_ekf().mode == 'fusion'
    with pytest.raises(ValueError, match="mode"):
        make_ekf(mode='px4')


def test_mode_is_keyword_only():
    from contract_uav_core.contracts.monitor import HierarchicalContractMonitor
    from contract_uav_core.estimation.ekf import ContractAwareEKF
    with pytest.raises(TypeError):
        ContractAwareEKF(HierarchicalContractMonitor(), 'passthrough')


def test_passthrough_returns_px4_estimate_with_its_covariance():
    ekf = make_ekf(mode='passthrough')
    pos, vel = [12.5, -3.25, -7.0], [0.75, -0.5, 0.125]
    meas, sensors = px4(pos, vel, eph=1.5, epv=2.5, evh=0.3, evv=0.2)
    state, _ = step(ekf, meas, sensors)
    assert np.array_equal(state[0:3], pos) and np.array_equal(state[3:6], vel)
    assert np.array_equal(ekf.get_state()['position'], pos)
    assert np.array_equal(ekf.get_state()['velocity'], vel)
    P = ekf.get_covariance()
    assert np.allclose(np.diag(P)[0:6], [1.5 ** 2 / 2, 1.5 ** 2 / 2, 2.5 ** 2, 0.3 ** 2 / 2, 0.3 ** 2 / 2, 0.2 ** 2])
    assert not P[0:6, 6:].any() and not P[6:, 0:6].any()
    off = P[0:6, 0:6] - np.diag(np.diag(P[0:6, 0:6]))
    assert not off.any()
    status = ekf.get_status()
    assert status['eph'] == pytest.approx(1.5) and status['epv'] == pytest.approx(2.5)
    assert status['position_trusted']


def test_passthrough_still_fuses_the_imu():
    ekf = make_ekf(mode='passthrough')
    meas, sensors = px4([1.0, 2.0, -3.0], [0.0, 0.0, 0.0], eph=1.0, epv=1.0)
    for k in range(200):
        step(ekf, meas, sensors, k * DT)
    assert np.allclose(ekf.get_state()['attitude'], [0.01, -0.02, 2.0], atol=1e-3)
    assert set(ekf.last_fused) == {'gps_position', 'gps_velocity', 'imu_attitude', 'imu_rates'}


def test_missing_accuracies_fall_back_to_nominal_noise():
    ekf = make_ekf(mode='passthrough')
    meas, sensors = px4([1.0, 2.0, -3.0], [0.1, 0.2, 0.3], eph=float('nan'), evv=-1.0)
    step(ekf, meas, sensors)
    P = ekf.get_covariance()
    assert np.allclose(np.diag(P)[0:6], np.diag(ekf.R_nominal)[0:6])


def test_passthrough_does_not_use_an_invalid_estimate():
    """valid=False with zeros (A-F7): not used; the state is predicted on from the last estimate."""
    ekf = make_ekf(mode='passthrough')
    meas, sensors = px4([10.0, -4.0, -5.0], [1.0, 0.0, 0.0], eph=0.5, epv=0.5, evh=0.1, evv=0.1)
    step(ekf, meas, sensors)
    bad, bad_sensors = px4([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], eph=0.5, epv=0.5)
    bad_sensors['gps']['valid'] = False
    for k in range(50):
        step(ekf, bad, bad_sensors)
    assert np.allclose(ekf.get_state()['position'], [10.0 + 50 * DT, -4.0, -5.0])
    assert 'gps_position' not in ekf.last_fused
    assert ekf.get_covariance()[0, 0] > 0.5 ** 2 / 2


def test_passthrough_axis_flags():
    ekf = make_ekf(mode='passthrough')
    meas, sensors = px4([10.0, -4.0, -5.0], [0.0, 0.0, 0.0], eph=0.5, epv=0.5)
    step(ekf, meas, sensors)
    meas, sensors = px4([0.0, 0.0, -6.0], [0.0, 0.0, 0.0], eph=0.5, epv=0.5, xy_valid=False)
    step(ekf, meas, sensors)
    assert ekf.last_fused['gps_position'] == (2,)
    assert np.allclose(ekf.get_state()['position'], [10.0, -4.0, -6.0])


def test_passthrough_ignores_the_satellite_contract_but_reports_it():
    """PX4's estimate can be valid without a good GPS (e.g. vision); gps_ok is still reported."""
    ekf = make_ekf(mode='passthrough')
    meas, sensors = px4([3.0, 4.0, -5.0], [0.0, 0.0, 0.0], eph=0.8, epv=0.8)
    sensors['gps']['satellites'] = 3
    step(ekf, meas, sensors)
    assert np.array_equal(ekf.get_state()['position'], [3.0, 4.0, -5.0])
    assert ekf.check_sensor_contracts(1.0) == (False, True)


def test_passthrough_flags_a_poor_estimate_untrusted():
    ekf = make_ekf(mode='passthrough')
    meas, sensors = px4([3.0, 4.0, -5.0], [0.0, 0.0, 0.0], eph=6.0, epv=1.0)
    step(ekf, meas, sensors)
    assert ekf.position_trusted is False


def test_passthrough_glides_back_after_an_outage():
    ekf = make_ekf(mode='passthrough')
    meas, sensors = px4([0.0, 0.0, -5.0], [1.0, 0.0, 0.0], eph=0.5, epv=0.5, evh=0.1, evv=0.1)
    step(ekf, meas, sensors)
    lost = dict(sensors['gps'], valid=False)
    for k in range(250):                       # 5 s dead-reckoned at 1 m/s: x = 5 m
        step(ekf, meas, {'gps': lost, 'imu': IMU})
    back, back_sensors = px4([0.0, 0.0, -5.0], [0.0, 0.0, 0.0], eph=0.5, epv=0.5, evh=0.1, evv=0.1)
    before = ekf.get_state()['position']
    step(ekf, back, back_sensors)
    assert np.array_equal(ekf.x[0:3], [0.0, 0.0, -5.0])          # the state is PX4's again ...
    moved = ekf.get_state()['position'] - before - np.array([1.0, 0.0, 0.0]) * DT
    assert np.linalg.norm(moved) == pytest.approx(ekf.MAX_RECOVERY_SPEED * DT)   # ... published: glides
    assert ekf.get_status()['gliding']
