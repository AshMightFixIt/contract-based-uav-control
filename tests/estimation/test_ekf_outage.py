"""ContractAwareEKF without GPS, and back (W4-08 T4, T5).

Before W4-08 the process noise was added per call whatever the step, a velocity
without GPS was frozen while its variance grew by 0.1 per step, so sqrt(P_xx)
went from 0.3 to about 116 m in 20 s of IMU-only flight, and on GPS return the
estimate snapped back. Now: per-second process noise, the last velocity kept
with a variance that covers any velocity within NO_GPS_SPEED_LIMIT (fix round,
S-1; first a 0.2 m/s^2 random walk, which a braking vehicle left), an untrusted
flag, and a published position that glides back at <= MAX_RECOVERY_SPEED.

W4-08b (T2): a return more than RESET_OFFSET (5 m) away is reset instead, with a
reset event. The glide tests below therefore use outages that leave about 4 m
(they used about 10 m, which is now reset: see test_ekf_reset.py).
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


def make_ekf(**kw):
    from contract_uav_core.contracts.monitor import HierarchicalContractMonitor
    from contract_uav_core.estimation.ekf import ContractAwareEKF
    monitor = HierarchicalContractMonitor()
    monitor.define_contracts()
    return ContractAwareEKF(monitor, **kw)


class Flight:
    """Truth moving at a constant velocity (or any velocity set on it), with measurements."""

    def __init__(self, velocity, seed=0, dt=DT):
        self.ekf = make_ekf()
        self.pos = START.copy()
        self.vel = np.array(velocity, dtype=float)
        self.t = 0.0
        self.dt = dt
        self.rng = np.random.default_rng(seed)

    def step(self, gps=True):
        self.t += self.dt
        self.pos = self.pos + self.vel * self.dt
        meas = {'imu_attitude': self.rng.standard_normal(3) * 0.001,
                'imu_rates': self.rng.standard_normal(3) * 0.001}
        if gps:
            meas['gps_position'] = self.pos + self.rng.standard_normal(3) * 0.01
            meas['gps_velocity'] = self.vel + self.rng.standard_normal(3) * 0.01
        self.ekf.predict(self.dt)
        return self.ekf.update(meas, {'gps': GPS if gps else NO_GPS, 'imu': IMU}, self.t)

    def fly(self, seconds, gps=True):
        for _ in range(int(round(seconds / self.dt))):
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


def test_velocity_variance_is_the_speed_envelope():
    """S-1: from the first update without GPS velocity, each velocity variance is
    ((|v_i| + limit_i) / 3)^2, and it stays there for the whole outage."""
    flight = Flight([1.0, -0.5, 0.0])
    flight.fly(10.0)
    ekf = flight.ekf
    flight.step(gps=False)
    v = ekf.get_state()['velocity']
    expected = [((abs(v[i]) + ekf.NO_GPS_SPEED_LIMIT[i]) / 3.0) ** 2 for i in range(3)]
    assert np.diag(ekf.P)[3:6] == pytest.approx(expected)
    flight.fly(10.0, gps=False)
    assert np.diag(ekf.P)[3:6] == pytest.approx(expected)
    assert np.array_equal(ekf.get_state()['velocity'], v)       # the velocity is kept
    assert not ekf.velocity_observed.any()


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
    assert 20.0 < sigma < 30.0         # 26.9 m: the 4/3 m/s envelope sigma over 20 s (base: 115.5 m)
    assert worst < 1.0                 # the 3-sigma bound covers the true error throughout
    assert np.linalg.norm(flight.error()) < 1.0
    assert flagged_at is not None and 1.0 < flagged_at < 5.0     # eph > 5 m after 2.7 s
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
    expected = P_pred + ekf.DEAD_RECKONING_EXTRA_Q * Q
    for i in range(3):                 # the unmeasured velocities: the speed envelope (S-1)
        bound = (abs(ekf.x[3 + i]) + ekf.NO_GPS_SPEED_LIMIT[i]) / 3.0
        expected[3 + i, 3 + i] = max(expected[3 + i, 3 + i], bound ** 2)
    assert np.allclose(ekf.P, expected)


# --- T5: smooth recovery -----------------------------------------------------------------------

# An outage that leaves the estimate about 4 m off: inside RESET_OFFSET, so it glides (W4-08b)
GLIDE_OUTAGE = 4.0  # s


def outage_then_stop(dt=DT, outage=GLIDE_OUTAGE):
    """10 s of GPS at 1 m/s, then an outage in which the truth stops: the dead-reckoned
    estimate keeps going and ends about 1 m per second of outage off. dt: the estimator's step."""
    flight = Flight([1.0, 0.0, 0.0], dt=dt)
    flight.fly(10.0)
    flight.vel = np.zeros(3)
    flight.fly(outage, gps=False)
    return flight


@pytest.mark.parametrize("dt", [0.01, 0.02, 0.04])
def test_gps_return_glides_at_most_2_m_s(dt):
    """T5, at the step the estimator is given (R-08 re-check R-1: the node runs the core on the
    measured step, so the limit must use it, not the nominal 0.02 s; with the nominal step a
    0.01 s loop would glide at 4 m/s)."""
    flight = outage_then_stop(dt)
    ekf = flight.ekf
    assert 3.5 < np.linalg.norm(flight.error()) < ekf.RESET_OFFSET
    published = ekf.get_state()['position']
    rates, covered = [], []
    for k in range(int(round(8.0 / dt))):
        v = ekf.get_state()['velocity']
        flight.step(gps=True)
        now = ekf.get_state()['position']
        rates.append(np.linalg.norm(now - published - v * dt) / dt)   # beyond the estimated velocity
        P = ekf.get_covariance()
        covered.append(all(abs(flight.error()[i]) <= 3 * np.sqrt(P[i, i]) for i in range(3)))
        published = now
    assert max(rates) <= ekf.MAX_RECOVERY_SPEED + 1e-9
    assert max(rates) > 0.99 * ekf.MAX_RECOVERY_SPEED     # it did glide at the limit
    assert all(covered)                                    # the covariance includes the glide offset
    assert not ekf.get_status()['gliding']
    assert np.linalg.norm(flight.error()) < 0.1
    assert ekf.get_status()['xy_reset_counter'] == 0                  # glided, not reset


def test_filter_takes_the_measurement_in_full_while_publishing_the_glide():
    flight = outage_then_stop()
    ekf = flight.ekf
    flight.step(gps=True)
    status = ekf.get_status()
    assert status['gliding']
    assert np.allclose(ekf.get_state()['position'], ekf.x[0:3] + status['glide_offset'])
    assert np.linalg.norm(ekf.x[0:3] - flight.pos) < 1.0        # the filter itself snapped back
    assert np.linalg.norm(status['glide_offset']) > 3.0
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


# --- T5, per-axis and gated returns (review R-08, B-1 and S-2) ---------------------------------

def per_axis_outage(mode, dt=DT):
    """10 s of GPS at 1 m/s; then GLIDE_OUTAGE s in which the truth stops and PX4 marks the
    horizontal position and velocity invalid (xy_valid, v_xy_valid False) while z stays valid
    (baro), as PX4 does when it loses GPS; then everything valid again, for 8 s; all at the step dt.
    Returns the published correction rate beyond the estimated velocity at each step after
    the return, whether the estimator was recovering then, and the estimator."""
    ekf = make_ekf(mode=mode)
    pos, vel = START.copy(), np.array([1.0, 0.0, 0.0])
    accuracy = {'eph': 0.5, 'epv': 0.5, 'evh': 0.1, 'evv': 0.1}

    def step(t, flags):
        meas = {'gps_position': pos.copy(), 'gps_velocity': vel.copy(),
                'imu_attitude': np.zeros(3), 'imu_rates': np.zeros(3)}
        ekf.predict(dt)
        ekf.update(meas, {'gps': {**GPS, **accuracy, **flags}, 'imu': IMU}, t)

    t, steps = 0.0, int(round(10.0 / dt))
    for _ in range(steps):
        t += dt
        pos = pos + vel * dt
        step(t, {})
    vel = np.zeros(3)
    for _ in range(int(round(GLIDE_OUTAGE / dt))):
        t += dt
        step(t, {'xy_valid': False, 'v_xy_valid': False})
        assert ekf.last_fused['gps_position'] == (2,)          # z was fused all along
    assert 3.5 < np.linalg.norm(ekf.get_state()['position'] - pos) < ekf.RESET_OFFSET
    rates, recovering = [], []
    for _ in range(int(round(8.0 / dt))):
        before = ekf.get_state()
        t += dt
        step(t, {})
        after = ekf.get_state()
        rates.append(np.linalg.norm(after['position'] - before['position'] - before['velocity'] * dt) / dt)
        recovering.append(ekf.recovering or ekf.glide_offset is not None)
    return np.array(rates), np.array(recovering), ekf, pos


@pytest.mark.parametrize("dt", [0.01, 0.02, 0.04])
@pytest.mark.parametrize("mode", ['fusion', 'passthrough'])
def test_per_axis_gps_return_glides(mode, dt):
    """B-1: a horizontal return while z stayed valid glides at <= 2 m/s (it snapped at 481 m/s),
    at the step the estimator is given (R-08 re-check R-1)."""
    rates, recovering, ekf, pos = per_axis_outage(mode, dt)
    assert recovering[0]
    assert rates[recovering].max() <= ekf.MAX_RECOVERY_SPEED + 1e-9
    assert rates[recovering].max() > 0.99 * ekf.MAX_RECOVERY_SPEED    # it glided at the limit
    assert not recovering[-1]
    assert np.linalg.norm(ekf.get_state()['position'] - pos) < 0.05
    if mode == 'passthrough':                    # after the recovery: PX4's position as it is
        assert np.array_equal(ekf.get_state()['position'], pos)


def test_gated_gps_return_does_not_move_the_published_position_unglided():
    """S-2: GPS back 200 m away after 20 s (beyond the gate even with the no-GPS covariance):
    the position is gated for 9 steps while the velocity is fused, and the velocity must not
    move the published position faster than the glide (it did at 13.6 m/s, 264 m/s with the
    kept velocity 0.5 m/s off, for a 100 m return). W4-08b (T2): at the 10th step the gate
    takes the position, and 200 m is beyond RESET_OFFSET, so it is reset with a reset event
    (before: glided for 100 s); after it, no more jumps."""
    flight = Flight([0.0, 0.0, 0.0])
    flight.fly(10.0)
    ekf = flight.ekf
    ekf.x[3] += 0.5                              # the kept velocity is 0.5 m/s off
    flight.fly(20.0, gps=False)
    flight.pos = flight.pos + [200.0, 0.0, 0.0]  # the GPS fix comes back 200 m away
    rates = []
    for k in range(int(5.0 / DT)):
        before = ekf.get_state()
        flight.step(gps=True)
        after = ekf.get_state()
        rates.append(np.linalg.norm(after['position'] - before['position'] - before['velocity'] * DT) / DT)
        if k < 9:
            assert 'gps_position' in ekf.last_rejected and 'gps_velocity' in ekf.last_fused, k
        if k == 9:
            assert 'gps_position' in ekf.last_reset
            status = ekf.get_status()
            assert status['xy_reset_counter'] == 1 and status['z_reset_counter'] == 0
            assert status['delta_xy'][0] > 190.0 and not status['gliding']
            assert np.allclose(after['position'], flight.pos, atol=0.1)     # the controller sees it now
    assert max(rates[:9]) <= ekf.MAX_RECOVERY_SPEED + 1e-9
    assert max(rates[10:]) <= ekf.MAX_RECOVERY_SPEED + 1e-9
    assert not ekf.recovering and ekf.get_status()['xy_reset_counter'] == 1


# --- S-1: honest without GPS for manoeuvres, not only at constant velocity ----------------------

LIMIT = np.array([3.0, 3.0, 2.0])


def brake(a):
    return lambda v, rng: -np.sign(v) * np.minimum(np.abs(v) / DT, a)


def to_velocity(target, a):
    return lambda v, rng: np.clip((np.asarray(target, dtype=float) - v) / DT, -a, a)


def random_within_envelope(sigma):
    def accel(v, rng):
        a = np.r_[rng.standard_normal(2) * sigma / np.sqrt(DT), 0.0]
        return (np.clip(v + a * DT, -LIMIT, LIMIT) - v) / DT
    return accel


def outage_coverage(v0, accel, seed, seconds=20.0):
    """Share of axis-steps of a GPS outage in which the published position is within 3 sigma."""
    flight = Flight(v0, seed=seed)
    flight.fly(10.0)
    inside = total = 0
    for _ in range(int(seconds / DT)):
        flight.vel = flight.vel + accel(flight.vel, flight.rng) * DT
        flight.step(gps=False)
        sigma = np.sqrt(np.diag(flight.ekf.get_covariance())[0:3])
        inside += int(np.sum(np.abs(flight.error()) <= 3.0 * sigma))
        total += 3
    return inside / total, flight


@pytest.mark.parametrize("name, v0, accel, seeds", [
    ('brake 3 m/s to hover at 2 m/s^2', [3.0, 0.0, 0.0], brake(2.0), 2),
    ('stop by the sim drag from 3 m/s', [3.0, 0.0, 0.0], lambda v, rng: -v * (0.03 / DT), 2),
    ('hover to 3 m/s at 1 m/s^2', [0.0, 0.0, 0.0], to_velocity([3.0, 0.0, 0.0], 1.0), 2),
    ('reverse 3 m/s to -3 m/s at 1 m/s^2', [3.0, 0.0, 0.0], to_velocity([-3.0, 0.0, 0.0], 1.0), 2),
    ('turn 3 m/s x to 3 m/s y at 1 m/s^2', [3.0, 0.0, 0.0], to_velocity([0.0, 3.0, 0.0], 1.0), 2),
    ('random 1 m/s^2 within the speed envelope', [1.0, 0.0, 0.0], random_within_envelope(1.0), 8),
])
def test_outage_covariance_covers_manoeuvres_within_the_envelope(name, v0, accel, seeds):
    """S-1: braking (what the supervisor's GPS-loss HOVER commands), accelerating, reversing,
    turning or random accelerations of 1 m/s^2, with the speed inside NO_GPS_SPEED_LIMIT:
    the 3-sigma bound covers >= 95 % of axis-steps over seeded runs (measured: 100 %;
    with the earlier 0.2 m/s^2 random walk 46-91 %)."""
    coverage = [outage_coverage(v0, accel, seed)[0] for seed in range(seeds)]
    assert min(coverage) >= 0.95, coverage


def test_outage_at_constant_velocity_is_no_worse_than_base():
    """The velocity is still kept, so at constant velocity the error stays centimetres (base: the
    same 0.15 m after 20 s for this seed), while the bound is wider."""
    coverage, flight = outage_coverage([3.0, 0.0, 0.0], lambda v, rng: np.zeros(3), seed=0)
    assert coverage == 1.0
    assert np.linalg.norm(flight.error()) < 0.3


def test_position_is_untrusted_while_a_far_glide_is_in_progress():
    """S-4 (R-08 mutation M3): the untrusted flag uses the published covariance (P + d d').
    Right after the return the filter's own eph is small, but the published position is still
    about 4 m from the estimate, so it is not trusted until the glide has brought it within the
    threshold. W4-08b: a glide is now at most RESET_OFFSET = POSITION_UNTRUSTED_EPH = 5 m (a
    farther return is reset), so this instance's threshold is lowered to 2.5 m to see the flag
    follow the glide (before: a 10 m glide against 5 m)."""
    flight = outage_then_stop()
    ekf = flight.ekf
    ekf.POSITION_UNTRUSTED_EPH = 2.5
    flight.step(gps=True)
    own_eph = np.sqrt(ekf.P[0, 0] + ekf.P[1, 1])
    assert own_eph < ekf.POSITION_UNTRUSTED_EPH
    assert np.linalg.norm(ekf.glide_offset[0:2]) > 3.0
    assert ekf.position_trusted is False
    trusted_at = None
    for k in range(int(8.0 / DT)):
        flight.step(gps=True)
        offset = ekf.get_status()['glide_offset']
        published_eph = np.sqrt(ekf.P[0, 0] + ekf.P[1, 1] + offset[0] ** 2 + offset[1] ** 2)
        assert ekf.position_trusted is bool(published_eph <= ekf.POSITION_UNTRUSTED_EPH), k
        if trusted_at is None and ekf.position_trusted:
            trusted_at = k
    assert trusted_at is not None and ekf.position_trusted


def test_recovery_lasts_until_the_position_is_fused():
    """S-2: while the returning GPS position is still gated, recovery goes on even if the
    published position has caught up in between: a velocity fix that then moves the estimate
    (through the position-velocity correlation of the outage) is glided too."""
    ekf = make_ekf()
    pos, t = START.copy(), 0.0

    def step(gps, velocity=np.zeros(3)):
        nonlocal t
        t += DT
        meas = {'imu_attitude': np.zeros(3), 'imu_rates': np.zeros(3)}
        if gps:
            meas.update(gps_position=pos + np.array([200.0, 0.0, 0.0]) * (t > 30.0), gps_velocity=velocity)
        ekf.predict(DT)
        ekf.update(meas, {'gps': GPS if gps else NO_GPS, 'imu': IMU}, t)

    for _ in range(500):
        step(True)
    for _ in range(1000):
        step(False)
    step(True)                                   # back: position gated, velocity exact (0)
    assert 'gps_position' in ekf.last_rejected and ekf.glide_offset is None
    assert ekf.recovering                        # the position is not fused yet
    for k in range(8):                           # now the GPS velocity reads 0.5 m/s
        before = ekf.get_state()
        step(True, velocity=np.array([0.5, 0.0, 0.0]))
        after = ekf.get_state()
        rate = np.linalg.norm(after['position'] - before['position'] - before['velocity'] * DT) / DT
        assert 'gps_position' in ekf.last_rejected, k
        assert rate <= ekf.MAX_RECOVERY_SPEED + 1e-9, (k, rate)
