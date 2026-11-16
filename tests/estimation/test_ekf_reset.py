"""ContractAwareEKF resets, instead of gliding, a GPS return that is far off (W4-08b T2).

W4-08 glided every GPS return at <= 2 m/s, so in closed loop a 52 m height error
reached the controller over 26 s. Now a channel (horizontal, vertical) whose
published position is more than RESET_OFFSET (5 m) from the estimate is reset to
the measurement, with a PX4-style reset event in get_status() (xy_reset_counter,
delta_xy, z_reset_counter, delta_z); smaller offsets still glide. In pass-through
mode PX4's own reset counters reset too.
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.usefixtures("core_package")

DT = 0.02
START = np.array([0.0, 0.0, -5.0])
GPS = {'satellites': 12, 'hdop': 0.8, 'valid': True}
NO_GPS = {'satellites': 0, 'hdop': 99.9, 'valid': False}
IMU = {'calibrated': True, 'temperature': 25.0, 'valid': True}
PX4 = {'eph': 0.5, 'epv': 0.5, 'evh': 0.1, 'evv': 0.1}


def make_ekf(**kw):
    from contract_uav_core.contracts.monitor import HierarchicalContractMonitor
    from contract_uav_core.estimation.ekf import ContractAwareEKF
    monitor = HierarchicalContractMonitor()
    monitor.define_contracts()
    return ContractAwareEKF(monitor, **kw)


def step(ekf, pos, vel, gps, t, flags=None):
    meas = {'imu_attitude': np.zeros(3), 'imu_rates': np.zeros(3)}
    if gps is not None:
        meas.update(gps_position=np.asarray(pos, float), gps_velocity=np.asarray(vel, float))
    ekf.predict(DT)
    ekf.update(meas, {'gps': {**(gps or NO_GPS), **(flags or {})}, 'imu': IMU}, t)


def outage(mode='fusion', seconds=20.0):
    """10 s of GPS hovering at START, then an outage (hovering truth, kept zero velocity)."""
    ekf = make_ekf(mode=mode)
    gps = {**GPS, **PX4}
    t = 0.0
    for _ in range(500):
        t += DT
        step(ekf, START, np.zeros(3), gps, t)
    for _ in range(int(round(seconds / DT))):
        t += DT
        step(ekf, START, np.zeros(3), None, t)
    return ekf, gps, t


def returned_at(offset, mode='fusion'):
    """The estimator after GPS comes back with the truth `offset` from where it was."""
    ekf, gps, t = outage(mode)
    before = ekf.get_state()['position']
    step(ekf, START + offset, np.zeros(3), gps, t + DT)
    return ekf, before


@pytest.mark.parametrize("mode", ['fusion', 'passthrough'])
@pytest.mark.parametrize("distance, reset", [(4.8, False), (5.3, True)])
def test_reset_offset_is_the_threshold(mode, distance, reset):
    """RESET_OFFSET (5 m) separates a glide from a reset, in both modes. (In fusion mode the
    offset to remove is about 2 % less than the measured one: after a 20 s outage the position
    gain is 0.998, and the GPS velocity fused next pulls back through the correlation.)"""
    ekf, before = returned_at(np.array([0.6, 0.8, 0.0]) * distance, mode)
    status = ekf.get_status()
    published = ekf.get_state()['position']
    assert status['xy_reset_counter'] == int(reset)
    assert status['gliding'] is (not reset)
    if reset:
        assert np.allclose(published, START + np.array([0.6, 0.8, 0.0]) * distance)
    else:
        moved = np.linalg.norm(published - before)
        assert moved == pytest.approx(ekf.MAX_RECOVERY_SPEED * DT, rel=1e-6)


def test_reset_snaps_position_and_velocity_to_the_measurement():
    ekf, gps, t = outage()
    vel = np.array([0.4, -0.3, 0.1])
    before = ekf.get_state()['position']
    step(ekf, START + [12.0, -9.0, 0.0], vel, gps, t + DT)
    R = ekf.gps_noise(gps, ekf.R_nominal.diagonal()[0:6].tolist())
    assert np.array_equal(ekf.x[0:2], (START + [12.0, -9.0, 0.0])[0:2])     # exactly the measurement
    assert np.array_equal(ekf.x[3:5], vel[0:2])
    P = ekf.P
    assert np.allclose([P[0, 0], P[1, 1], P[3, 3], P[4, 4]], [R[0], R[1], R[3], R[4]])
    assert P[0, 3] == 0.0 and P[1, 4] == 0.0
    status = ekf.get_status()
    assert status['xy_reset_counter'] == 1
    assert np.allclose(status['delta_xy'], ekf.x[0:2] - (before[0:2]), atol=1e-9)   # zero velocity: no prediction
    assert status['z_reset_counter'] == 0 and status['delta_z'] == 0.0
    assert not ekf.recovering and not status['gliding']


def test_channels_are_reset_separately():
    """A far horizontal return resets xy while a small vertical one glides; and the other way."""
    ekf, _ = returned_at(np.array([8.0, 0.0, -1.0]))
    status = ekf.get_status()
    assert status['xy_reset_counter'] == 1 and status['z_reset_counter'] == 0
    assert status['gliding'] and status['glide_offset'][0:2] == pytest.approx([0.0, 0.0])
    assert status['glide_offset'][2] > 0.9                      # z glides from 1 m below
    ekf, _ = returned_at(np.array([1.0, 0.0, -7.0]))
    status = ekf.get_status()
    assert status['z_reset_counter'] == 1 and status['xy_reset_counter'] == 0
    assert status['delta_z'] == pytest.approx(-7.0, abs=0.05)
    assert status['glide_offset'][2] == 0.0 and abs(status['glide_offset'][0]) > 0.9


def test_a_reset_needs_the_channels_gps_position():
    """Only the GPS position of a channel resets it: a returning velocity alone, while the
    position stays invalid, glides (S-2), and nothing is reset."""
    ekf, gps, t = outage(seconds=10.0)
    for k in range(5):
        step(ekf, START, np.array([0.8, 0.0, 0.0]), gps, t + (k + 1) * DT, flags={'xy_valid': False})
    assert ekf.get_status()['xy_reset_counter'] == 0


def test_no_reset_event_at_start_up():
    """The first fix after start-up is not a return: it converges as before, with no event."""
    ekf = make_ekf()
    for k in range(50):
        step(ekf, [30.0, -20.0, -5.0], np.zeros(3), GPS, (k + 1) * DT)
    status = ekf.get_status()
    assert status['xy_reset_counter'] == 0 and status['z_reset_counter'] == 0


def test_passthrough_resets_when_px4_reports_one():
    """Pass-through: a change of PX4's xy_reset_counter / z_reset_counter is a reset of that
    channel at any size, also while the position stays valid; the first counter seen, or an
    unchanged one, is not."""
    ekf = make_ekf(mode='passthrough')
    gps = {**GPS, **PX4, 'xy_reset_counter': 3, 'z_reset_counter': 7}
    t = 0.0
    for _ in range(50):
        t += DT
        step(ekf, START, np.zeros(3), gps, t)
    status = ekf.get_status()
    assert status['xy_reset_counter'] == 0 and status['z_reset_counter'] == 0
    t += DT
    step(ekf, START + [0.8, -0.6, 0.0], np.zeros(3), {**gps, 'xy_reset_counter': 4}, t)
    status = ekf.get_status()
    assert status['xy_reset_counter'] == 1 and status['z_reset_counter'] == 0
    assert np.allclose(status['delta_xy'], [0.8, -0.6])
    t += DT
    step(ekf, START + [0.8, -0.6, 0.5], np.zeros(3), {**gps, 'xy_reset_counter': 4, 'z_reset_counter': 8}, t)
    status = ekf.get_status()
    assert status['xy_reset_counter'] == 1 and status['z_reset_counter'] == 1
    assert status['delta_z'] == pytest.approx(0.5)
    assert np.array_equal(ekf.get_state()['position'], START + [0.8, -0.6, 0.5])


def test_passthrough_px4_reset_during_an_outage_is_a_reset_at_the_return():
    """PX4 resets while its position is invalid (the counter changes then): the small offset at
    the return is not glided but reset, as PX4 did."""
    ekf = make_ekf(mode='passthrough')
    gps = {**GPS, **PX4, 'xy_reset_counter': 0}
    t = 0.0
    for _ in range(50):
        t += DT
        step(ekf, START, np.zeros(3), gps, t)
    for _ in range(100):
        t += DT
        step(ekf, START, np.zeros(3), {**gps, 'xy_reset_counter': 1}, t, flags={'xy_valid': False})
    t += DT
    step(ekf, START + [2.0, 0.0, 0.0], np.zeros(3), {**gps, 'xy_reset_counter': 1}, t)
    status = ekf.get_status()
    assert status['xy_reset_counter'] == 1 and not status['gliding']
    assert np.array_equal(ekf.get_state()['position'], START + [2.0, 0.0, 0.0])
