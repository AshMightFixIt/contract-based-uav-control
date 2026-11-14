"""ContractAwareEKF: fuse only present and valid data, gate, wrap the heading (W4-08 T1-T3).

Before W4-08, update_full started from z = zeros(12) with H = I, so a missing
measurement was fused as zeros and pulled the estimate to the origin (K17); a
GPS flagged invalid that still carried zeros was fused the same way (A-F7, core
half). The heading was not wrapped, and nothing was gated.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

pytestmark = pytest.mark.usefixtures("core_package")

DT = 0.02
TRUTH = np.array([10.0, -4.0, -5.0])


def make_ekf():
    from contract_uav_core.contracts.monitor import HierarchicalContractMonitor
    from contract_uav_core.estimation.ekf import ContractAwareEKF
    monitor = HierarchicalContractMonitor()
    monitor.define_contracts()
    return ContractAwareEKF(monitor)


def good_sensors(**gps_extra):
    return {'gps': {'satellites': 12, 'hdop': 0.8, 'valid': True, **gps_extra},
            'imu': {'calibrated': True, 'temperature': 25.0, 'valid': True}}


def measurements(pos=TRUTH, vel=np.zeros(3), att=np.zeros(3), rates=np.zeros(3)):
    return {'gps_position': np.array(pos, dtype=float), 'gps_velocity': np.array(vel, dtype=float),
            'imu_attitude': np.array(att, dtype=float), 'imu_rates': np.array(rates, dtype=float)}


def converged(seconds=5.0):
    """An estimator that has fused exact measurements of TRUTH (hovering) for `seconds`."""
    ekf = make_ekf()
    for k in range(int(seconds / DT)):
        ekf.predict(DT)
        ekf.update(measurements(), good_sensors(), k * DT)
    return ekf


def step(ekf, meas, sensors, t=10.0):
    ekf.predict(DT)
    return ekf.update(meas, sensors, t)


# --- T1: fuse only what is present, finite and not flagged invalid -------------------------

def test_converged_estimate_is_the_truth():
    ekf = converged()
    assert np.allclose(ekf.get_state()['position'], TRUTH, atol=1e-3)
    assert ekf.last_fused == {'gps_position': (0, 1, 2), 'gps_velocity': (3, 4, 5),
                              'imu_attitude': (6, 7, 8), 'imu_rates': (9, 10, 11)}


def test_missing_gps_position_is_not_fused():
    ekf = converged()
    meas = measurements()
    del meas['gps_position']
    for k in range(50):
        step(ekf, meas, good_sensors())
    assert np.allclose(ekf.get_state()['position'], TRUTH, atol=1e-3)
    assert 'gps_position' not in ekf.last_fused
    assert ekf.last_fused['gps_velocity'] == (3, 4, 5)


def test_invalid_gps_with_zeros_is_not_fused():
    """A-F7: valid=False, zeros in the measurements, satellites and HDOP still good."""
    ekf = converged()
    sensors = good_sensors()
    sensors['gps']['valid'] = False
    meas = measurements(pos=np.zeros(3), vel=np.zeros(3))
    for k in range(50):
        step(ekf, meas, sensors)
    assert np.allclose(ekf.get_state()['position'], TRUTH, atol=1e-3)
    assert set(ekf.last_fused) == {'imu_attitude', 'imu_rates'}


@pytest.mark.parametrize("flag, key, fused", [
    ('xy_valid', 'gps_position', (2,)),
    ('z_valid', 'gps_position', (0, 1)),
    ('v_xy_valid', 'gps_velocity', (5,)),
    ('v_z_valid', 'gps_velocity', (3, 4)),
])
def test_axis_flags_drop_their_components(flag, key, fused):
    """PX4-style per-axis flags: a False flag drops only the components it covers.

    The components it covers carry zeros (what the bridge sends when PX4 marks them
    invalid); they are not fused, so the estimate stays at the truth."""
    ekf = converged()
    first = 0 if key == 'gps_position' else 3   # state row of the measurement's x component
    dropped = [i for i in range(3) if first + i not in fused]
    pos, vel = TRUTH.copy(), np.zeros(3)
    if key == 'gps_position':
        pos[dropped] = 0.0
    else:
        vel[dropped] = 2.0
    for k in range(50):
        step(ekf, measurements(pos=pos, vel=vel), good_sensors(**{flag: False}))
        assert ekf.last_fused[key] == fused
        assert not ekf.last_rejected and not ekf.last_reset
    assert np.allclose(ekf.get_state()['position'], TRUTH, atol=1e-3)
    assert np.allclose(ekf.get_state()['velocity'], 0.0, atol=1e-3)


def test_nonfinite_component_is_not_fused():
    ekf = converged()
    meas = measurements(pos=[np.nan, -4.0, np.inf])
    step(ekf, meas, good_sensors())
    assert ekf.last_fused['gps_position'] == (1,)
    assert np.all(np.isfinite(ekf.get_state()['position']))
    assert np.all(np.isfinite(ekf.get_covariance()))


def test_missing_imu_attitude_is_not_fused_without_gps():
    """IMU-only mode (GPS contract violated): a missing attitude is not fused as zeros either."""
    ekf = converged()
    yaw = 1.0
    for k in range(100):
        step(ekf, measurements(att=[0.0, 0.0, yaw]), good_sensors())
    sensors = good_sensors()
    sensors['gps'].update(satellites=3)
    meas = measurements(att=[0.0, 0.0, yaw])
    del meas['imu_attitude']
    for k in range(50):
        step(ekf, meas, sensors)
    assert ekf.estimation_quality == 'degraded'
    assert set(ekf.last_fused) == {'imu_rates'}
    assert ekf.get_state()['attitude'][2] == pytest.approx(yaw, abs=1e-3)


def test_wrong_shape_raises():
    ekf = make_ekf()
    meas = measurements()
    meas['gps_position'] = np.zeros(2)
    with pytest.raises(ValueError, match="gps_position"):
        step(ekf, meas, good_sensors())


def reference_update(x, P, z, rows, R):
    """The pre-W4-08 update (update_full / update_imu_only), written out."""
    H = np.eye(12)[list(rows)]
    y = z - H @ x
    S = H @ P @ H.T + R[np.ix_(rows, rows)]
    K = P @ H.T @ np.linalg.inv(S)
    return x + K @ y, (np.eye(12) - K @ H) @ P


@pytest.mark.parametrize("satellites, rows", [(12, list(range(12))), (3, list(range(6, 12)))],
                         ids=['full', 'imu-only'])
def test_update_is_the_old_update_when_everything_is_valid(satellites, rows):
    """With every measurement valid and inside the gate, the numbers are the old 12x12 update's,
    to rounding (the filter now works per pair, with scalar updates)."""
    ekf = converged(1.0)
    rng = np.random.default_rng(0)
    sensors = good_sensors()
    sensors['gps']['satellites'] = satellites
    for k in range(20):
        meas = measurements(pos=TRUTH + rng.standard_normal(3) * 0.3, vel=rng.standard_normal(3) * 0.2,
                            att=rng.standard_normal(3) * 0.05, rates=rng.standard_normal(3) * 0.05)
        ekf.predict(DT)
        x0, P0 = ekf.x.copy(), ekf.P.copy()
        z = np.concatenate([meas['gps_position'], meas['gps_velocity'],
                            meas['imu_attitude'], meas['imu_rates']])[rows]
        R = ekf.R_nominal if satellites == 12 else ekf.R_degraded
        x_ref, P_ref = reference_update(x0, P0, z, rows, R)
        ekf.update(meas, sensors, k * DT)
        assert np.allclose(ekf.x, x_ref, rtol=1e-12, atol=1e-14), k
        assert np.allclose(ekf.P, P_ref, rtol=1e-11, atol=1e-15), k


# --- T2: innovation gate ----------------------------------------------------------------------

def test_outlier_is_rejected():
    ekf = converged()
    step(ekf, measurements(pos=TRUTH + [20.0, 0.0, 0.0]), good_sensors())
    assert ekf.last_rejected == {'gps_position': (0, 1, 2)}
    assert 'gps_position' not in ekf.last_fused
    assert ekf.last_fused['gps_velocity'] == (3, 4, 5)
    assert np.allclose(ekf.get_state()['position'], TRUTH, atol=1e-3)
    assert ekf.reject_count['gps_position'] == 1
    step(ekf, measurements(), good_sensors())
    assert ekf.last_fused['gps_position'] == (0, 1, 2)
    assert ekf.reject_count['gps_position'] == 0


def test_real_jump_is_accepted_after_ten_rejects(caplog):
    ekf = converged()
    jumped = TRUTH + [20.0, -15.0, 0.0]
    caplog.set_level(logging.WARNING, logger='contract_uav_core.estimation.ekf')
    for k in range(9):
        step(ekf, measurements(pos=jumped), good_sensors())
        assert 'gps_position' in ekf.last_rejected, k
        assert np.allclose(ekf.get_state()['position'], TRUTH, atol=1e-3)
    assert not caplog.records
    step(ekf, measurements(pos=jumped), good_sensors())   # the 10th reject: reset
    assert ekf.last_reset == {'gps_position': (0, 1, 2)}
    assert np.array_equal(ekf.x[0:3], jumped)
    P = ekf.P   # the filter's own covariance (get_covariance adds the glide offset, T5)
    assert np.allclose(np.diag(P)[0:3], 10.0 * np.diag(ekf.R_nominal)[0:3])
    assert not P[0:3, 3:].any() and not P[3:, 0:3].any()
    assert len(caplog.records) == 1 and 'reset' in caplog.records[0].getMessage()
    for k in range(20):
        step(ekf, measurements(pos=jumped), good_sensors())
        assert ekf.last_fused['gps_position'] == (0, 1, 2)
    assert np.allclose(ekf.x[0:3], jumped, atol=1e-2)   # the filter took the jump ...
    assert ekf.get_status()['gliding']                   # ... and the published position glides there (T5)
    assert len(caplog.records) == 1


def test_reject_count_restarts_when_the_measurement_is_absent():
    ekf = converged()
    for k in range(5):
        step(ekf, measurements(pos=TRUTH + [20.0, 0.0, 0.0]), good_sensors())
    assert ekf.reject_count['gps_position'] == 5
    meas = measurements()
    del meas['gps_position']
    step(ekf, meas, good_sensors())
    assert ekf.reject_count['gps_position'] == 0


def test_no_gate_while_converging_from_the_initial_guess():
    """The initial state (zeros, variance 1) is no estimate: a fix however far is fused, without
    gating, until its innovation has been inside the gate once; from then on it is gated."""
    ekf = make_ekf()
    far = np.array([300.0, -200.0, -50.0])
    for k in range(200):
        step(ekf, measurements(pos=far), good_sensors(), t=k * DT)
        assert ekf.last_fused['gps_position'] == (0, 1, 2), k
        assert not ekf.last_rejected and not ekf.last_reset, k
    assert np.allclose(ekf.get_state()['position'], far, atol=1e-2)
    assert ekf.initialized['gps_position']
    step(ekf, measurements(pos=far + [30.0, 0.0, 0.0]), good_sensors())
    assert ekf.last_rejected == {'gps_position': (0, 1, 2)}


# --- T3: heading wrap -------------------------------------------------------------------------

def test_wrap_angle():
    from contract_uav_core.estimation.ekf import wrap_angle
    for a in (0.0, 1.0, -1.0, 3.0, -3.0, np.pi, np.nextafter(-np.pi, 0.0)):
        assert wrap_angle(a) == a
    assert wrap_angle(-np.pi) == np.pi
    assert wrap_angle(1.5 * np.pi) == pytest.approx(-0.5 * np.pi)
    assert wrap_angle(-1.5 * np.pi) == pytest.approx(0.5 * np.pi)
    assert wrap_angle(3.0 * np.pi) == pytest.approx(np.pi)
    assert wrap_angle(101.0) == pytest.approx(np.arctan2(np.sin(101.0), np.cos(101.0)))
    for a in np.linspace(-50.0, 50.0, 2001):
        w = wrap_angle(a)
        assert -np.pi < w <= np.pi
        assert np.cos(w) == pytest.approx(np.cos(a)) and np.sin(w) == pytest.approx(np.sin(a), abs=1e-9)


def test_yaw_innovation_is_wrapped():
    """Estimate near +pi, measurement near -pi: the update moves across +-pi, not through 0."""
    ekf = converged(1.0)
    for k in range(100):
        step(ekf, measurements(att=[0.0, 0.0, 3.1]), good_sensors())
    step(ekf, measurements(att=[0.0, 0.0, -3.1]), good_sensors())
    yaw = ekf.get_state()['attitude'][2]
    assert ekf.last_fused['imu_attitude'] == (6, 7, 8)
    assert abs(yaw) > 3.1 - 1e-9          # moved toward +-pi, not toward 0
    assert -np.pi < yaw <= np.pi


def test_yaw_state_is_wrapped_by_predict():
    ekf = make_ekf()
    ekf.x[8], ekf.x[11] = 3.1, 1.0      # yaw near +pi, turning at 1 rad/s
    for k in range(5):
        ekf.predict(DT)
    assert ekf.x[8] == pytest.approx(3.2 - 2 * np.pi)


# --- the covariance as six 2x2 blocks (S-5) ---------------------------------------------------

def test_covariance_keeps_the_pair_structure():
    """P only ever couples a quantity with its own rate (the pairs), so it is kept as 2x2 blocks."""
    ekf = converged(1.0)
    P = ekf.get_covariance()
    pattern = np.zeros((12, 12), dtype=bool)
    for a, b in ekf.PAIRS:
        pattern[np.ix_([a, b], [a, b])] = True
    assert not P[~pattern].any()
    assert np.array_equal(P, P.T)


def test_covariance_setter_round_trips_and_refuses_other_couplings():
    ekf = make_ekf()
    P = np.diag(np.arange(1.0, 13.0))
    P[0, 3] = P[3, 0] = 0.25
    ekf.P = P
    assert np.array_equal(ekf.P, P)
    bad = P.copy()
    bad[0, 1] = bad[1, 0] = 0.1          # x with y: not a pair
    with pytest.raises(ValueError, match="pair"):
        ekf.P = bad
    asymmetric = P.copy()
    asymmetric[3, 0] = 0.3
    with pytest.raises(ValueError):
        ekf.P = asymmetric
    assert np.array_equal(ekf.P, P)      # a refused matrix changes nothing
