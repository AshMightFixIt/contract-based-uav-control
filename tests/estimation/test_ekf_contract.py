"""The estimator contract check in ContractAwareEKF.update (W4-08 T8).

Before W4-08 the result was False on every step, even in nominal flight, for
two reasons in ekf.py: the outputs were the Frobenius norms of the covariance
blocks (variances, m^2, against bounds in m), and none of the contract's inputs
(position_meas_error, ...) was passed, so its assumptions never held and its
relational bounds lost their input term (0.7 * meas_error + 0.1 became 0.1).
Both are fixed. In nominal flight the velocity and attitude guarantees still
fail, on the contract's values (handed to W4-11), so these tests check that the
estimator passes the right quantities, not that the contract holds.
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.usefixtures("core_package")

DT = 0.02
SENSORS = {'gps': {'satellites': 12, 'hdop': 0.8, 'valid': True},
           'imu': {'calibrated': True, 'temperature': 25.0, 'valid': True}}


def make_ekf():
    from contract_uav_core.contracts.monitor import HierarchicalContractMonitor
    from contract_uav_core.estimation.ekf import ContractAwareEKF
    monitor = HierarchicalContractMonitor()
    monitor.define_contracts()
    return ContractAwareEKF(monitor)


def hover(ekf, seconds=5.0, sensors=SENSORS):
    meas = {'gps_position': np.array([0.0, 0.0, -5.0]), 'gps_velocity': np.zeros(3),
            'imu_attitude': np.zeros(3), 'imu_rates': np.zeros(3)}
    ok = None
    for k in range(int(seconds / DT)):
        ekf.predict(DT)
        _, ok = ekf.update(meas, sensors, k * DT)
    return ok


def test_outputs_are_rms_errors_of_the_published_covariance():
    ekf = make_ekf()
    hover(ekf)
    values = ekf.estimator_metrics.assumption_values
    P = ekf.get_covariance()
    assert values['position_error'] == pytest.approx(np.sqrt(np.trace(P[0:3, 0:3])))
    assert values['velocity_error'] == pytest.approx(np.sqrt(np.trace(P[3:6, 3:6])))
    assert values['attitude_error'] == pytest.approx(np.sqrt(np.trace(P[6:9, 6:9])))


def test_outputs_scale_as_a_standard_deviation():
    """4 x the covariance gives 2 x the error (a variance, as before W4-08, would give 4 x)."""
    ekf = make_ekf()
    hover(ekf)
    before = ekf.estimator_contract_values()
    ekf.P = ekf.P * 4.0
    after = ekf.estimator_contract_values()
    for name in ('position_error', 'velocity_error', 'attitude_error'):
        assert after[name] == pytest.approx(2.0 * before[name])


def test_inputs_are_the_sensor_contracts_predicted_bounds():
    ekf = make_ekf()
    hover(ekf)
    values = ekf.estimator_metrics.assumption_values
    expected = ekf.contract_monitor.sensor_contract.predict_guarantees(
        {'gps_satellites': 12.0, 'gps_hdop': 0.8, 'imu_temperature': 25.0, 'imu_calibrated': 1.0})
    for name in ('position_meas_error', 'velocity_meas_error', 'rate_meas_error', 'attitude_meas_error'):
        assert values[name] == pytest.approx(expected[f'{name}_max'])
    assert values['position_meas_error'] == pytest.approx(1.5 * 0.8)    # HDOP x UERE


def test_assumptions_hold_in_nominal_flight():
    ekf = make_ekf()
    hover(ekf)
    assert ekf.estimator_metrics.assumptions_met is True


def test_assumptions_fail_with_a_poor_gps():
    """HDOP 6: position_meas_error 9 m, beyond the estimator contract's 7.5 m."""
    ekf = make_ekf()
    sensors = {'gps': {'satellites': 12, 'hdop': 6.0, 'valid': True}, 'imu': SENSORS['imu']}
    hover(ekf, 1.0, sensors)
    assert ekf.estimator_metrics.assumptions_met is False


@pytest.mark.parametrize("hdop", [0.8, 2.0])
def test_relational_bound_uses_the_measurement_error(hdop):
    ekf = make_ekf()
    sensors = {'gps': {'satellites': 12, 'hdop': hdop, 'valid': True}, 'imu': SENSORS['imu']}
    hover(ekf, 1.0, sensors)
    assert ekf.estimator_metrics.predicted_bounds['position_error_max'] == pytest.approx(0.7 * 1.5 * hdop + 0.1)


def test_result_is_the_contracts_own_verdict():
    ekf = make_ekf()
    ok = hover(ekf)
    values = ekf.estimator_metrics.assumption_values
    met, margins = ekf.contract_monitor.estimator_contract.check_guarantees(values)
    assert ok is met
    assert margins['position_error'] > 0     # in nominal hover the position guarantee holds
