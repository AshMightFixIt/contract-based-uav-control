"""ContractAwareEKF without GPS, and back (W4-08 T4, T5).

Before W4-08 the process noise was added per call whatever the step, a velocity
without GPS was frozen while its variance grew by 0.1 per step, so sqrt(P_xx)
went from 0.3 to about 116 m in 20 s of IMU-only flight, and on GPS return the
estimate snapped back. Now: per-second process noise, the last velocity kept
with a variance that grows by VEL_RANDOM_WALK_NO_GPS per second, an untrusted
flag, and a published position that glides back at <= MAX_RECOVERY_SPEED.
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


def make_ekf():
    from contract_uav_core.contracts.monitor import HierarchicalContractMonitor
    from contract_uav_core.estimation.ekf import ContractAwareEKF
    monitor = HierarchicalContractMonitor()
    monitor.define_contracts()
    return ContractAwareEKF(monitor)


class Flight:
    """Truth moving at a constant velocity (or any velocity set on it), with measurements."""

    def __init__(self, velocity, seed=0):
        self.ekf = make_ekf()
        self.pos = START.copy()
        self.vel = np.array(velocity, dtype=float)
        self.t = 0.0
        self.rng = np.random.default_rng(seed)

    def step(self, gps=True):
        self.t += DT
        self.pos = self.pos + self.vel * DT
        meas = {'imu_attitude': self.rng.standard_normal(3) * 0.001,
                'imu_rates': self.rng.standard_normal(3) * 0.001}
        if gps:
            meas['gps_position'] = self.pos + self.rng.standard_normal(3) * 0.01
            meas['gps_velocity'] = self.vel + self.rng.standard_normal(3) * 0.01
        self.ekf.predict(DT)
        return self.ekf.update(meas, {'gps': GPS if gps else NO_GPS, 'imu': IMU}, self.t)

    def fly(self, seconds, gps=True):
        for _ in range(int(round(seconds / DT))):
            self.step(gps)

    def error(self):
        return self.ekf.get_state()['position'] - self.pos


# --- T4: per-second process noise --------------------------------------------------------------

def test_process_noise_is_per_second():
    ekf = make_ekf()
    assert np.array_equal(ekf.process_noise(0.02), ekf.Q)          # unchanged at the 50 Hz step
    assert np.allclose(ekf.process_noise(0.01), ekf.Q / 2)
    assert np.allclose(ekf.process_noise(0.04), ekf.Q * 2)


@pytest.mark.parametrize("dt", [0.01, 0.04])
def test_covariance_growth_does_not_depend_on_the_step(dt):
    """Predict-only for 2 s: the same position variance at 100 Hz, 50 Hz or 25 Hz (within 5 %)."""
    def grown(step):
        ekf = make_ekf()
        for _ in range(int(round(2.0 / step))):
            ekf.predict(step)
        return ekf.P[0, 0]
    assert grown(dt) == pytest.approx(grown(0.02), rel=0.05)


# --- T4: IMU-only: keep the velocity, bounded and honest covariance ----------------------------

@pytest.mark.parametrize("speed", [1.0, 3.0])
def test_ten_second_outage_keeps_the_velocity(speed):
    """The last velocity is kept (not decayed to hover): after 10 s the error is centimetres."""
    flight = Flight([speed, 0.0, 0.0])
    flight.fly(10.0)
    v_before = flight.ekf.get_state()['velocity'].copy()
    flight.fly(10.0, gps=False)
    assert np.array_equal(flight.ekf.get_state()['velocity'], v_before)
    assert np.linalg.norm(flight.error()) < 0.3
    assert flight.ekf.estimation_quality == 'degraded'


def test_velocity_variance_grows_at_the_no_gps_rate():
    flight = Flight([1.0, 0.0, 0.0])
    flight.fly(10.0)
    flight.step(gps=False)          # the first predict after the loss still had GPS velocity
    p0 = flight.ekf.P[3, 3]
    flight.fly(10.0, gps=False)
    rate = (flight.ekf.P[3, 3] - p0) / 10.0
    assert rate == pytest.approx(flight.ekf.VEL_RANDOM_WALK_NO_GPS, rel=1e-6)
    assert not flight.ekf.velocity_observed.any()


def test_twenty_second_outage_is_bounded_honest_and_untrusted():
    flight = Flight([1.0, 0.5, 0.0])
    flight.fly(10.0)
    assert flight.ekf.position_trusted
    sigma0 = np.sqrt(flight.ekf.get_covariance()[0, 0])
    worst, flagged_at = 0.0, None
    for k in range(1000):
        flight.step(gps=False)
        P = flight.ekf.get_covariance()
        err = flight.error()
        worst = max(worst, max(abs(err[i]) / (3 * np.sqrt(P[i, i])) for i in range(3)))
        if flagged_at is None and not flight.ekf.position_trusted:
            flagged_at = (k + 1) * DT
    sigma = np.sqrt(flight.ekf.get_covariance()[0, 0])
    assert sigma0 < 0.5
    assert 5.0 < sigma < 20.0          # base: 115.5 m
    assert worst < 1.0                 # the 3-sigma bound covers the true error throughout
    assert np.linalg.norm(flight.error()) < 1.0
    assert flagged_at is not None and 5.0 < flagged_at < 15.0
    status = flight.ekf.get_status()
    assert status['position_trusted'] is False and status['eph'] > flight.ekf.POSITION_UNTRUSTED_EPH


def test_untrusted_flag_follows_eph():
    ekf = make_ekf()
    for eph, trusted in [(4.99, True), (5.01, False)]:
        P = np.eye(12) * 0.01
        P[0, 0] = P[1, 1] = eph ** 2 / 2
        ekf.P = P
        ekf.update({}, {'gps': NO_GPS, 'imu': IMU}, 0.0)
        assert ekf.position_trusted is trusted
        assert ekf.get_status()['eph'] == pytest.approx(eph)


def test_dead_reckoning_adds_the_extra_process_noise():
    flight = Flight([0.0, 0.0, 0.0])
    flight.fly(2.0)
    ekf = flight.ekf
    ekf.predict(DT)
    P_pred, Q = ekf.P.copy(), ekf.last_Q.copy()
    ekf.update({}, {'gps': NO_GPS, 'imu': {**IMU, 'calibrated': False}}, 3.0)
    assert ekf.estimation_quality == 'emergency'
    assert np.allclose(ekf.P, P_pred + ekf.DEAD_RECKONING_EXTRA_Q * Q)


# --- T5: smooth recovery -----------------------------------------------------------------------

def outage_then_stop():
    """10 s of GPS at 1 m/s, then a 10 s outage in which the truth stops: the dead-reckoned
    estimate keeps going and ends about 10 m off."""
    flight = Flight([1.0, 0.0, 0.0])
    flight.fly(10.0)
    flight.vel = np.zeros(3)
    flight.fly(10.0, gps=False)
    return flight


def test_gps_return_glides_at_most_2_m_s():
    flight = outage_then_stop()
    ekf = flight.ekf
    assert np.linalg.norm(flight.error()) > 9.0
    published = ekf.get_state()['position']
    rates, covered = [], []
    for k in range(int(8.0 / DT)):
        v = ekf.get_state()['velocity']
        flight.step(gps=True)
        now = ekf.get_state()['position']
        rates.append(np.linalg.norm(now - published - v * DT) / DT)   # beyond the estimated velocity
        P = ekf.get_covariance()
        covered.append(all(abs(flight.error()[i]) <= 3 * np.sqrt(P[i, i]) for i in range(3)))
        published = now
    assert max(rates) <= ekf.MAX_RECOVERY_SPEED + 1e-9
    assert max(rates) > 0.99 * ekf.MAX_RECOVERY_SPEED     # it did glide at the limit
    assert all(covered)                                    # the covariance includes the glide offset
    assert not ekf.get_status()['gliding']
    assert np.linalg.norm(flight.error()) < 0.1


def test_filter_takes_the_measurement_in_full_while_publishing_the_glide():
    flight = outage_then_stop()
    ekf = flight.ekf
    flight.step(gps=True)
    status = ekf.get_status()
    assert status['gliding']
    assert np.allclose(ekf.get_state()['position'], ekf.x[0:3] + status['glide_offset'])
    assert np.linalg.norm(ekf.x[0:3] - flight.pos) < 1.0        # the filter itself snapped back
    assert np.linalg.norm(status['glide_offset']) > 8.0
    P, Pf = ekf.get_covariance(), ekf.P
    assert np.allclose(P[0:3, 0:3], Pf[0:3, 0:3] + np.outer(status['glide_offset'], status['glide_offset']))


def test_no_glide_at_the_first_fix():
    """Start-up from the initial guess (the origin) converges as before: nothing glides."""
    flight = Flight([0.0, 0.0, 0.0])
    flight.pos = np.array([30.0, -20.0, -5.0])
    for _ in range(50):
        flight.step()
        assert not flight.ekf.get_status()['gliding']
    assert np.linalg.norm(flight.error()) < 0.1


def test_short_dropout_does_not_glide():
    flight = Flight([1.0, 0.0, 0.0])
    flight.fly(5.0)
    flight.step(gps=False)
    flight.step(gps=True)
    assert flight.ekf.glide_offset is None
    assert np.linalg.norm(flight.error()) < 0.05
